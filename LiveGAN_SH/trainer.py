import os, time, glob
from itertools import chain

import numpy as np

import itertools, time, os
import torch
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.optim as optim

from models import mocogan as mocogan

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
        self.lambda_seq = config.lambda_seq
        self.lambda_l1 = config.lambda_l1

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
            self.image_discriminator.cuda()
            self.video_discriminator.cuda()

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
        D_S_filename = '{}/netD_S_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)
        D_V_filename = '{}/netD_V_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)
        D_I_filename = '{}/netD_I_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)
        Im_Recon_filename = '{}/ImageRecon_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)
        Video_Recon_filename = '{}/VideoRecon_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)

        self.generator.load_state_dict(torch.load(G_filename))
        self.seq_discriminator.load_state_dict(torch.load(D_S_filename))
        self.video_discriminator.load_state_dict(torch.load(D_V_filename))
        self.image_discriminator.load_state_dict(torch.load(D_I_filename))
        self.image_reconstructor.load_state_dict(torch.load(Im_Recon_filename))
        self.video_reconstructor.load_state_dict(torch.load(Video_Recon_filename))

        print("[*] Model loaded: {}".format(G_filename))
        print("[*] Model loaded: {}".format(D_S_filename))
        print("[*] Model loaded: {}".format(D_V_filename))
        print("[*] Model loaded: {}".format(D_I_filename))
        print("[*] Model loaded: {}".format(Im_Recon_filename))
        print("[*] Model loaded: {}".format(Video_Recon_filename))


    def build_model(self):
        self.generator = mocogan.VideoGenerator(self.n_channels, self.dim_z_content,
                                                self.dim_z_category, self.dim_z_motion, self.video_length)
        self.seq_discriminator = mocogan.SequenceDiscriminator(video_len=self.video_length,
                                                          dim_z_motion=self.dim_z_motion,
                                                          n_channels=self.n_channels, use_noise=self.use_noise,
                                                          noise_sigma=self.noise_sigma)

        self.image_discriminator = self.build_discriminator(self.image_discriminator, n_channels=self.n_channels,
                                                        use_noise=self.use_noise, noise_sigma=self.noise_sigma)
        
        self.video_discriminator = self.build_discriminator(self.video_discriminator, dim_categorical=self.dim_z_category,
                                                        n_channels=self.n_channels, use_noise=self.use_noise,
                                                        noise_sigma=self.noise_sigma)

        self.image_reconstructor = mocogan.ImageReconstructor(self.n_channels,
                                                              self.dim_z_content + self.dim_z_category + self.dim_z_motion)

        self.video_reconstructor = mocogan.VideoReconstructor(self.n_channels, self.video_length,
                                                              self.dim_z_content + self.dim_z_category,
                                                              self.dim_z_motion)

        if self.outf != None:
            self.load_model()


    def build_discriminator(self, type, **kwargs):
        discriminator_type = getattr(mocogan, type)

        if 'Categorical' not in type and 'dim_categorical' in kwargs:
            kwargs.pop('dim_categorical')

        return discriminator_type(**kwargs)


    def train(self):
        self.gan_criterion = nn.BCEWithLogitsLoss()
        self.category_criterion = nn.CrossEntropyLoss()

        if self.use_cuda:
            self.generator.cuda()
            self.seq_discriminator.cuda()
            self.image_discriminator.cuda()
            self.video_discriminator.cuda()
            self.image_reconstructor.cuda()
            self.video_reconstructor.cuda()

        # create optimizers
        opt_generator = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        opt_seq_discriminator = optim.Adam(self.seq_discriminator.parameters(), lr=self.lr,
                                           betas=(self.beta1, self.beta2),
                                           weight_decay=self.weight_decay)
        opt_image_discriminator = optim.Adam(self.image_discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2),
                                             weight_decay=self.weight_decay)
        opt_video_discriminator = optim.Adam(self.video_discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2),
                                             weight_decay=self.weight_decay)


        A_loader, B_loader = iter(self.image_loader), iter(self.video_loader)
        valid_x_A, valid_x_B = A_loader.next(), B_loader.next()
        valid_x_A_categ = valid_x_A["categories"]
        valid_x_A, valid_x_B = valid_x_A["images"], valid_x_B["images"]
        valid_x_B = valid_x_B.permute(0,2,1,3,4)

        valid_x_A = self._get_variable(valid_x_A).resize(self.image_batch_size, self.n_channels, self.image_size, self.image_size)
        vutils.save_image(valid_x_A.data, '{}/valid_im.png'.format(self.outf), nrow=1, normalize=True)

        valid_x_B = self._get_variable(valid_x_B).resize(self.video_batch_size * self.video_length, self.n_channels, self.image_size, self.image_size)
        vutils.save_image(valid_x_B.data, '{}/valid_gif.png'.format(self.outf), nrow=self.video_length, normalize=True)

        start_time = time.time()

        for epoch in range(self.start_epoch,self.train_batches):

            for step in range(len(self.video_loader)):
                try:
                    realIm, realGif = A_loader.next(), B_loader.next()
                    realGifCateg, realImCateg = realGif["categories"], realIm["categories"]
                    realGif, realIm = realGif["images"], realIm["images"]

                except StopIteration:
                    A_loader, B_loader = iter(self.image_loader), iter(self.video_loader)
                    realIm, realGif = A_loader.next(), B_loader.next()
                    realGifCateg, realImCateg = realGif["categories"], realIm["categories"]
                    realGif, realIm = realGif["images"], realIm["images"]

                if realIm.size(0) != realGif.size(0):
                    print("[!] Sampled dataset from A and B have different # of data. Try resampling...")
                    continue


                realIm, realGif = Variable(realIm.cuda(), requires_grad=False), Variable(realGif.cuda(), requires_grad=False)
                image_batch_size = realIm.size(0)

                ############################
                # (1) Update G network: minimize Lgan(MSE) + Lcycle(L1)
                ###########################
                self.generator.zero_grad()

                #### train with gif
                # GAN loss: D_A(G_A(A))
                fake = (self.generator.sample_videos(realIm, realImCateg, image_batch_size), self.generator.sample_images(realIm, realImCateg, image_batch_size))
                fakeGif, generated_categ = fake[0][0], fake[0][1]

                output, fake_categ = self.video_discriminator(fakeGif)
                # loss_G = self.gan_criterion(output, Variable(torch.ones(output.size()).cuda()))

                output, _ = self.seq_discriminator(fakeGif)
                loss_G = self.lambda_seq * self.gan_criterion(output, Variable(torch.ones(output.size()).cuda()))


                loss_G += self.lambda_l1 * torch.mean(torch.abs(fakeGif[:, :, 0, :, :] - realIm))


                if self.config.use_reconstruct:
                    recon = self.video_reconstructor(fakeGif)
                    loss_G += torch.mean(torch.abs(recon - realIm))
                if self.config.use_infogan:
                    loss_G += self.category_criterion(fake_categ.squeeze(), generated_categ)


                #### train with image
                fakeIm = fake[1][0]

                output, fake_categ = self.image_discriminator(fakeIm)
                loss_G += self.gan_criterion(output, Variable(torch.ones(output.size()).cuda()))

                if self.config.use_reconstruct:
                    recon = self.image_reconstructor(fakeIm)
                    loss_G += torch.mean(torch.abs(recon - realIm))

                loss_G.backward()
                opt_generator.step()

                ############################
                # (2) Update D network: minimize LSGAN loss
                ###########################

                #### train D_S ####
                # train with real
                self.seq_discriminator.zero_grad()

                D_real = self.seq_discriminator(realGif)[0]
                loss_D_real = self.gan_criterion(D_real, Variable(torch.ones(D_real.size()).cuda()))

                # train with fake
                D_fake = self.seq_discriminator(fakeGif.detach())[0]
                loss_D_fake = self.gan_criterion(D_fake, Variable(torch.zeros(D_fake.size()).cuda()))

                loss_D_S = loss_D_real + loss_D_fake

                loss_D_S.backward()
                opt_seq_discriminator.step()


                #### train D_V ####
                # train with real
                self.video_discriminator.zero_grad()

                D_real, real_categ = self.video_discriminator(realGif)
                # loss_D_real = self.gan_criterion(D_real, Variable(torch.ones(D_real.size()).cuda()))
                #
                # # train with fake
                # D_fake, fake_categ = self.video_discriminator(fakeGif.detach())
                # loss_D_fake = self.gan_criterion(D_fake, Variable(torch.zeros(D_fake.size()).cuda()))
                #
                # loss_D_V = loss_D_real + loss_D_fake

                if self.config.use_categories:
                    categories_gt = Variable(torch.squeeze(realGifCateg.long()), requires_grad=False).cuda()
                    loss_D_V = self.category_criterion(real_categ.squeeze(), categories_gt)

                loss_D_V.backward()
                opt_video_discriminator.step()


                #### train D_I ####
                # train with real
                self.image_discriminator.zero_grad()

                D_real = self.image_discriminator(realIm)[0]
                loss_D_real = self.gan_criterion(D_real, Variable(torch.ones(D_real.size()).cuda()))

                # train with fake
                D_fake = self.image_discriminator(fakeIm.detach())[0]
                loss_D_fake = self.gan_criterion(D_fake, Variable(torch.zeros(D_fake.size()).cuda()))

                loss_D_I = loss_D_real + loss_D_fake

                loss_D_I.backward()
                opt_image_discriminator.step()

                step_end_time = time.time()


                print('[%d/%d][%d/%d] - time: %.2f, loss_D_V: %.3f, loss_D_I: %.3f, '
                      'loss_G: %.3f'
                      % (epoch, self.train_batches, step, len(self.video_loader), step_end_time - start_time,
                         loss_D_V, loss_D_I, loss_G))


                if step % self.log_interval == 0:
                    fake = (self.generator.sample_videos(valid_x_A, valid_x_A_categ, self.image_batch_size), self.generator.sample_images(valid_x_A, valid_x_A_categ, self.image_batch_size))
                    fakeGif = fake[0][0]
                    fakeGif = fakeGif.permute(0, 2, 1, 3, 4)
                    fakeGif = fakeGif.resize(self.video_batch_size * self.video_length, self.n_channels, self.image_size, self.image_size)
                    vutils.save_image(denorm(fakeGif.data), '%s/fakeGif_AB_%03d_%d.png' % (self.outf, epoch, step), nrow=self.video_length)

                    #fakeIm = fake[1][0].resize(self.image_batch_size, self.n_channels, self.image_size,
                    #                            self.image_size)
                    #vutils.save_image(denorm(fakeIm.data), '%s/fakeIm_AB_%03d_%d.png' % (self.outf, epoch, step),
                    #                  nrow=1)

            if epoch % self.checkpoint_step == 0:
                torch.save(self.generator.state_dict(), '%s/netG_epoch-%d_step-%s.pth' % (self.outf, epoch, step))
                torch.save(self.seq_discriminator.state_dict(), '%s/netD_S_epoch-%d_step-%s.pth' % (self.outf, epoch, step))
                torch.save(self.video_discriminator.state_dict(), '%s/netD_V_epoch-%d_step-%s.pth' % (self.outf, epoch, step))
                torch.save(self.image_reconstructor.state_dict(), '%s/ImageRecon_epoch-%d_step-%s.pth' % (self.outf, epoch, step))
                torch.save(self.video_reconstructor.state_dict(), '%s/VideoRecon_epoch-%d_step-%s.pth' % (self.outf, epoch, step))
                torch.save(self.image_discriminator.state_dict(), '%s/netD_I_epoch-%d_step-%s.pth' % (self.outf, epoch, step))

                print("Saved checkpoint")

    def _get_variable(self, inputs):
        out = Variable(inputs.cuda())
        return out
