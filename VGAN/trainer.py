import os, time, glob
from itertools import chain

import numpy as np

import itertools, time, os
import torch
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.optim as optim

from models import vgan as vgan

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

class Trainer(object):
    def __init__(self, config, image_loader, video_loader):
        self.config = config
        self.image_loader = image_loader
        self.video_loader = video_loader

        self.image_size = int(config.image_size)
        self.n_channels = int(config.n_channels)
        self.dim_z_content = int(config.dim_z_content)
        self.dim_z_category = int(config.dim_z_category)
        self.dim_z_motion = int(config.dim_z_motion)
        self.video_length = int(config.video_length)

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay

        self.image_discriminator = config.image_discriminator
        self.video_discriminator = config.video_discriminator

        self.use_noise = config.use_noise
        self.noise_sigma = float(config.noise_sigma)

        self.use_infogan = config.use_infogan
        self.use_categories = config.use_categories

        self.image_batch_size = self.image_loader.batch_size
        self.video_batch_size = self.video_loader.batch_size

        self.log_interval = int(config.log_interval)
        self.checkpoint_step = int(config.checkpoint_step)
        self.train_batches = int(config.batches)

        self.use_cuda = config.cuda

        self.outf = config.outf

        self.build_model()

        if self.use_cuda:
            self.generator.cuda()
            self.discriminator.cuda()


    def load_model(self):
        print("[*] Load models from {}...".format(self.outf))

        paths = glob.glob(os.path.join(self.outf, '*.pth'))
        paths.sort()
        self.start_epoch = 0

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.outf))
            return

        epochs = [int(path.split('.')[-2].split('_')[-2].split('-')[-1]) for path in paths]
        self.start_epoch = str(max(epochs))
        steps = [int(path.split('.')[-2].split('_')[-1].split('-')[-1]) for path in paths]
        self.start_step = str(max(steps))

        G_filename = '{}/netG_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)
        D_filename = '{}/netD_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)


        self.generator.load_state_dict(torch.load(G_filename))
        self.discriminator.load_state_dict(torch.load(D_filename))


        print("[*] Model loaded: {}".format(G_filename))
        print("[*] Model loaded: {}".format(D_filename))


    def build_model(self):
        self.generator = vgan.VideoGenerator(self.n_channels, self.video_length)
        self.discriminator = vgan.VideoDiscriminator(n_channels=self.n_channels)

        if self.outf != None:
            self.load_model()

    def train(self):
        self.gan_criterion = nn.BCEWithLogitsLoss()

        if self.use_cuda:
            self.generator.cuda()
            self.discriminator.cuda()

        # create optimizers
        opt_generator = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2),
                                             weight_decay=self.weight_decay)

        A_loader, B_loader = iter(self.image_loader), iter(self.video_loader)
        valid_x_A, valid_x_B = A_loader.next(), B_loader.next()
        valid_x_A, valid_x_B = valid_x_A["images"], valid_x_B["images"]
        valid_x_B = valid_x_B.permute(0,2,1,3,4)

        valid_x_A = self._get_variable(valid_x_A).resize(self.image_batch_size, self.n_channels, self.image_size, self.image_size)
        vutils.save_image(valid_x_A.data, '{}/valid_im.png'.format(self.outf), nrow=1, normalize=True)

        valid_x_B = self._get_variable(valid_x_B).resize(self.video_batch_size * self.video_length, self.n_channels, self.image_size, self.image_size)
        vutils.save_image(valid_x_B.data, '{}/valid_gif.png'.format(self.outf), nrow=self.video_length, normalize=True)

        start_time = time.time()

        for epoch in range(self.train_batches):

            for step in range(len(self.video_loader)):
                try:
                    realIm, realGif = A_loader.next(), B_loader.next()
                    realGif, realIm = realGif["images"], realIm["images"]

                except StopIteration:
                    A_loader, B_loader = iter(self.image_loader), iter(self.video_loader)
                    realIm, realGif = A_loader.next(), B_loader.next()
                    realGif, realIm = realGif["images"], realIm["images"]

                if realIm.size(0) != realGif.size(0):
                    print("[!] Sampled dataset from A and B have different # of data. Try resampling...")
                    continue


                realIm, realGif = Variable(realIm.cuda(), requires_grad=False), Variable(realGif.cuda(), requires_grad=False)

                ############################
                # (1) Update G network: minimize Lgan(MSE) + Lcycle(L1)
                ###########################
                self.generator.zero_grad()

                # GAN loss: D_A(G_A(A))
                fakeGif = self.generator(realIm)

                output = self.discriminator(fakeGif)
                loss_G = self.gan_criterion(output, Variable(torch.ones(output.size()).cuda()))

                loss_G += torch.mean(torch.abs(fakeGif[:, :, 0, :, :] - realIm))

                loss_G.backward()
                opt_generator.step()

                ############################
                # (2) Update D network: minimize LSGAN loss
                ###########################

                #### train D ####
                # train with real
                self.discriminator.zero_grad()

                D_real = self.discriminator(realGif)
                loss_D_real = self.gan_criterion(D_real, Variable(torch.ones(D_real.size()).cuda()))

                # train with fake
                D_fake = self.discriminator(fakeGif.detach())
                loss_D_fake = self.gan_criterion(D_fake, Variable(torch.zeros(D_fake.size()).cuda()))

                loss_D = loss_D_real + loss_D_fake

                loss_D.backward()
                opt_discriminator.step()

                step_end_time = time.time()


                print('[%d/%d][%d/%d] - time: %.2f, loss_D: %.3f, '
                      'loss_G: %.3f'
                      % (epoch, self.train_batches, step, len(self.video_loader), step_end_time - start_time,
                         loss_D, loss_G))


                if step % self.log_interval == 0:
                    fakeGif = self.generator(valid_x_A)
                    fakeGif = fakeGif.permute(0, 2, 1, 3, 4)
                    fakeGif = fakeGif.resize(self.video_batch_size * self.video_length, self.n_channels, self.image_size, self.image_size)
                    vutils.save_image(denorm(fakeGif.data), '%s/fakeGif_AB_%03d_%d.png' % (self.outf, epoch, step), nrow=self.video_length)


            if epoch % self.checkpoint_step == 0:
                torch.save(self.generator.state_dict(), '%s/netG_epoch-%d_step-%s.pth' % (self.outf, epoch, step))
                torch.save(self.discriminator.state_dict(), '%s/netD_epoch-%d_step-%s.pth' % (self.outf, epoch, step))

                print("Saved checkpoint")

    def _get_variable(self, inputs):
        out = Variable(inputs.cuda())
        return out
