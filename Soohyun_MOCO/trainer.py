import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import itertools, time, os
from glob import glob
import numpy as np
import torchvision.utils as vutils
import models.soohyun as cyclegan


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Trainer(object):
    def __init__(self, config, dataloader):
        self.config = config
        self.train_loader_A = dataloader[0]
        self.train_loader_B = dataloader[1]

        self.ngpu = int(config.ngpu)
        self.input_nc = int(config.input_nc)
        self.output_nc = int(config.output_nc)

        self.dim_z_content = int(config.dim_z_content)
        self.dim_z_category = int(config.dim_z_category)
        self.dim_z_motion = int(config.dim_z_motion)

        self.nz = config.nz
        self.ngf = int(config.ngf)
        self.ndf = int(config.ndf)
        self.cuda = config.cuda

        self.batch_size = config.batch_size
        self.image_size = config.image_size

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.niter = config.niter
        self.num_steps = len(self.train_loader_A)
        self.decay_epoch = config.decay_epoch
        self.cycle_lambda = config.cycle_lambda

        self.outf = config.outf
        self.sample_step = config.sample_step
        self.checkpoint_step = config.checkpoint_step

        self.build_model()

        if self.cuda:
            self.netG.cuda()
            self.netD_V.cuda()
            self.netD_I.cuda()
            self.image_reconstructor.cuda()
            self.video_reconstructor.cuda()


    def load_model(self):
        print("[*] Load models from {}...".format(self.outf))

        paths = glob(os.path.join(self.outf, 'net*.pth'))
        paths.sort()
        self.start_epoch = 0

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.outf))
            return

        epochs = [int(os.path.basename(path.split('.')[0].split('_')[-2].split('-')[-1])) for path in paths]
        self.start_epoch = str(max(epochs))
        steps = [int(os.path.basename(path.split('.')[0].split('_')[-1].split('-')[-1])) for path in paths]
        self.start_step = str(max(steps))


        G_filename = '{}/netG_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)
        D_V_filename = '{}/netD_V_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)
        D_I_filename = '{}/netD_I_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)
        Im_Recon_filename = '{}/ImageRecon_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)
        Video_Recon_filename = '{}/VideoRecon_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)

        self.netG.load_state_dict(torch.load(G_filename))
        self.netD_V.load_state_dict(torch.load(D_V_filename))
        self.netD_I.load_state_dict(torch.load(D_I_filename))
        self.image_reconstructor.load_state_dict(torch.load(Im_Recon_filename))
        self.video_reconstructor.load_state_dict(torch.load(Video_Recon_filename))


        print("[*] Model loaded: {}".format(G_filename))
        print("[*] Model loaded: {}".format(D_V_filename))
        print("[*] Model loaded: {}".format(D_I_filename))
        print("[*] Model loaded: {}".format(Im_Recon_filename))
        print("[*] Model loaded: {}".format(Video_Recon_filename))


    def build_model(self):
        # self, ngpu, ngf, input_nc, dim_z_content, dim_z_category, dim_z_motion):
        self.netG = cyclegan._netG(self.ngpu, self.ngf, self.input_nc, self.dim_z_content,
                                   self.dim_z_category, self.dim_z_motion, self.config.ntimestep)
        self.netG.apply(weights_init)

        # self, n_channels, n_output_neurons = 1, ndf = 64):
        self.netD_V = cyclegan._netD_V(self.ngpu, self.input_nc)
        self.netD_V.apply(weights_init)

        self.netD_I = cyclegan._netD_I(self.input_nc)
        self.netD_I.apply(weights_init)

        self.image_reconstructor = cyclegan.ImageReconstructor(self.input_nc,
                                                              self.dim_z_content + self.dim_z_category + self.dim_z_motion)
        self.image_reconstructor.apply(weights_init)

        self.video_reconstructor = cyclegan.VideoReconstructor(self.input_nc, self.config.ntimestep,
                                                              self.dim_z_content + self.dim_z_category + self.dim_z_motion)
        self.video_reconstructor.apply(weights_init)

        self.load_model()


    def train(self):
        BCELoss = nn.BCEWithLogitsLoss()
        L1loss = nn.L1Loss()

        if self.cuda:
            BCELoss.cuda()
            L1loss.cuda()

        # setup optimizer
        optimizerD_V = optim.Adam(self.netD_V.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        optimizerD_I = optim.Adam(self.netD_I.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        optimizerG = optim.Adam(itertools.chain(self.netG.parameters(), self.image_reconstructor.parameters(), self.video_reconstructor.parameters()), lr=self.lr, betas=(self.beta1, self.beta2))
        # optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        A_loader, B_loader = iter(self.train_loader_A), iter(self.train_loader_B)
        valid_x_A, valid_x_B = A_loader.next(), B_loader.next()

        valid_x_A = self._get_variable(valid_x_A).resize(self.batch_size, self.input_nc, self.image_size, self.image_size )
        vutils.save_image(valid_x_A.data, '{}/valid_im.png'.format(self.outf), nrow=1, normalize=True)

        valid_x_B = self._get_variable(valid_x_B).resize(self.batch_size * 10, self.input_nc, self.image_size, self.image_size)
        vutils.save_image(valid_x_B.data, '{}/valid_gif.png'.format(self.outf), nrow=10, normalize=True)

        start_time = time.time()

        for epoch in range(int(self.start_epoch), self.niter):
            if (epoch+1) > self.decay_epoch:
                optimizerD_V.param_groups[0]['lr'] -= self.lr / (self.niter - self.decay_epoch)
                optimizerD_I.param_groups[0]['lr'] -= self.lr / (self.niter - self.decay_epoch)
                optimizerG.param_groups[0]['lr'] -= self.lr / (self.niter - self.decay_epoch)

            for step, (realIm, realGif) in enumerate(itertools.izip(self.train_loader_A, self.train_loader_B)):
                # realGif size. 8, 30, 128, 128
                # realIm size. 8, 3, 128, 128
                realGif = realGif.view(-1, self.config.ntimestep, self.input_nc, self.image_size, self.image_size)
                realGif = realGif.permute(0,2,1,3,4)
                # print(realGif.shape)

                vutils.save_image(realGif[0,:,0,:,:], '%s/test.png' % (self.outf), nrow=1, normalize=True)
                realIm, realGif = Variable(realIm.cuda()), Variable(realGif.cuda())
                # if step % self.sample_step == 0:
                #     a = realGif.resize(self.batch_size * self.config.ntimestep, self.input_nc, self.image_size,
                #                                 self.image_size)
                #     vutils.save_image(a.data, '%s/realGif_AB_%03d_%d.png' % (self.outf, epoch, step), nrow=self.config.ntimestep, normalize=True)
                #
                #     vutils.save_image(realIm.data, '%s/realIm_AB_%03d_%d.png' % (self.outf, epoch, step),
                #                       nrow=1, normalize=True)

                ############################
                # (1) Update G network: minimize Lgan(MSE) + Lcycle(L1)
                ###########################
                for p in self.netD_V.parameters():
                    p.requires_grad = False
                for p in self.netD_I.parameters():
                    p.requires_grad = False

                self.netG.zero_grad()
                self.image_reconstructor.zero_grad()
                self.video_reconstructor.zero_grad()

                #### train with gif
                # GAN loss: D_A(G_A(A))
                fake = self.netG(realIm)
                fakeGif, fake_categ = fake[0][0], fake[0][1]
                # print("fakeM",fakeGif.shape)
                # print("fakecateg", fake_categ.shape)
                # fakeM size. (8, 3, 10, 128, 128)
                # fakeCateg size. (8)

                output = self.netD_V(fakeGif) # output size. (8, 1, 1, )
                loss_G = BCELoss(output, Variable(torch.ones(output.size()).cuda()))

                recon = self.video_reconstructor(fakeGif)
                # print("recon", recon.shape)
                loss_G += torch.mean(torch.abs(recon - realIm))

                #### train with image
                fakeIm = fake[1][0]
                # print("fakeim", fakeIm.shape)

                output = self.netD_I(fakeIm)
                loss_G += BCELoss(output, Variable(torch.ones(output.size()).cuda()))

                recon = self.image_reconstructor(fakeIm)
                loss_G += torch.mean(torch.abs(recon - realIm))

                loss_G.backward()
                optimizerG.step()


                ############################
                # (2) Update D network: minimize LSGAN loss
                ###########################
                for p in self.netD_V.parameters():
                    p.requires_grad = True
                for p in self.netD_I.parameters():
                    p.requires_grad = True

                #### train D_V ####
                # train with real
                self.netD_V.zero_grad()

                D_real = self.netD_V(realGif)
                loss_D_real = BCELoss(D_real, Variable(torch.ones(D_real.size()).cuda()))

                # train with fake
                D_fake = self.netD_V(fakeGif.detach())
                loss_D_fake = BCELoss(D_fake, Variable(torch.zeros(D_fake.size()).cuda()))

                loss_D_V = loss_D_real + loss_D_fake
                loss_D_V.backward()
                optimizerD_V.step()

                #### train G_I ####
                # train with real
                self.netD_I.zero_grad()

                D_real = self.netD_I(realIm)
                loss_D_real = BCELoss(D_real, Variable(torch.ones(D_real.size()).cuda()))

                # train with fake
                D_fake = self.netD_I(fakeIm.detach())
                loss_D_fake = BCELoss(D_fake, Variable(torch.zeros(D_fake.size()).cuda()))

                loss_D_I = loss_D_real + loss_D_fake
                loss_D_I.backward()
                optimizerD_I.step()


                step_end_time = time.time()

                print('[%d/%d][%d/%d] - time_passed: %.2f, loss_D_V: %.3f, loss_D_I: %.3f, '
                      'loss_G: %.3f'
                      % (epoch, self.niter, step, self.num_steps, step_end_time - start_time,
                         loss_D_V, loss_D_I, loss_G))


                if step % self.sample_step == 0:
                    fake = self.netG(valid_x_A)

                    fakeGif = fake[0][0].resize(self.batch_size*self.config.ntimestep, self.input_nc, self.image_size, self.image_size)
                    vutils.save_image(fakeGif.data, '%s/fakeGif_AB_%03d_%d.png' % (self.outf, epoch, step), nrow=self.config.ntimestep, normalize=True)

                    fakeIm = fake[1][0].resize(self.batch_size, self.input_nc, self.image_size,
                                                self.image_size)
                    vutils.save_image(fakeIm.data, '%s/fakeIm_AB_%03d_%d.png' % (self.outf, epoch, step),
                                      nrow=1, normalize=True)

                if step% self.checkpoint_step == 0 and step != 0:
                    torch.save(self.netG.state_dict(), '%s/netG_epoch-%d_step-%s.pth' % (self.outf, epoch, step))
                    torch.save(self.netD_V.state_dict(), '%s/netD_V_epoch-%d_step-%s.pth' % (self.outf, epoch, step))
                    torch.save(self.netD_I.state_dict(), '%s/netD_I_epoch-%d_step-%s.pth' % (self.outf, epoch, step))
                    torch.save(self.image_reconstructor.state_dict(), '%s/ImageRecon_epoch-%d_step-%s.pth' % (self.outf, epoch, step))
                    torch.save(self.video_reconstructor.state_dict(), '%s/VideoRecon_epoch-%d_step-%s.pth' % (self.outf, epoch, step))

                    print("Saved checkpoint")



    def _get_variable(self, inputs):
        if self.ngpu > 0:
            out = Variable(inputs.cuda())
        else:
            out = Variable(inputs)
        return out
