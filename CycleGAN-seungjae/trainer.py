import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

import models.cyclegan as cyclegan

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Trainer(object):
    def __init__(self, config, train_loader_A, train_loader_B):
        self.config = config
        self.train_loader_A = train_loader_A
        self.train_loader_B = train_loader_B
        self.cuda = config.cuda
        self.ngpu = int(config.ngpu)

        self.nc = int(config.nc)
        self.ngf = int(config.ngf)
        self.ndf = int(config.ndf)

        self.niter = config.niter
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.beta1 = config.beta1

        self.outf = config.outf
        self.sample_step = config.sample_step
        self.checkpoint_step = config.checkpoint_step

        self.build_model()

    def build_model(self):
        self.netG_AB = cyclegan._netG(self.ngpu, self.nc, self.ngf)
        self.netG_AB.apply(weights_init)
        if self.config.netG_AB != '':
            self.netG_AB.load_state_dict(torch.load(self.config.netG_AB))

        self.netG_BA = cyclegan._netG(self.ngpu, self.nc, self.ngf)
        self.netG_BA.apply(weights_init)
        if self.config.netG_BA != '':
            self.netG_BA.load_state_dict(torch.load(self.config.netG_BA))

        self.netD_A = cyclegan._netD(self.ngpu, self.nc, self.ndf)
        self.netD_A.apply(weights_init)
        if self.config.netD_A != '':
            self.netD_A.load_state_dict(torch.load(self.config.netD_A))

        self.netD_B = cyclegan._netD(self.ngpu, self.nc, self.ndf)
        self.netD_B.apply(weights_init)
        if self.config.netD_B != '':
            self.netD_B.load_state_dict(torch.load(self.config.netD_B))

        self.optimizerG = optim.Adam(itertools.chain(self.netG_AB.parameters(), self.netG_BA.parameters()),
                                     lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerD_A = optim.Adam(self.netD_A.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerD_B = optim.Adam(self.netD_B.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        if self.cuda:
            self.netG_AB.cuda()
            self.netG_BA.cuda()
            self.netD_A.cuda()
            self.netD_B.cuda()

    def to_var(self, x, volatile=False):
        if self.cuda:
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self):
        MSE_loss = nn.MSELoss()
        L1_loss = nn.L1Loss()

        if self.cuda:
            MSE_loss.cuda(), L1_loss.cuda()

        fixed_a = []
        fixed_b = []
        for i, (real_a, real_b) in enumerate(zip(self.train_loader_A, self.train_loader_B)):
            fixed_a.append(real_a)
            fixed_b.append(real_b)
            if i == 10:
                break
        fixed_a = torch.cat(fixed_a, dim=0)
        fixed_a = self.to_var(fixed_a, volatile=True)
        fixed_b = torch.cat(fixed_b, dim=0)
        fixed_b = self.to_var(fixed_b, volatile=True)

        for epoch in range(self.niter):
            for i, (real_a, real_b) in enumerate(zip(self.train_loader_A, self.train_loader_B)):

                real_a = self.to_var(real_a)
                real_b = self.to_var(real_b)

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                for p in self.netD_A.parameters(): # reset requires_grad
                    p.requires_grad = True # they are set to False below in netG update
                for p in self.netD_B.parameters(): # reset requires_grad
                    p.requires_grad = True # they are set to False below in netG update

                # train with real
                D_A_real = self.netD_A(real_b)
                errD_A_real = MSE_loss(D_A_real, self.to_var(torch.ones(D_A_real.size())))

                fake_b = self.netG_AB(real_a)
                D_A_fake = self.netD_A(fake_b)
                errD_A_fake = MSE_loss(D_A_fake, self.to_var(torch.zeros(D_A_fake.size())))

                # optimize
                errD_A = (errD_A_real + errD_A_fake) * 0.5
                self.optimizerD_A.zero_grad()
                errD_A.backward()
                self.optimizerD_A.step()

                # train with fake
                D_B_real = self.netD_B(real_a)
                errD_B_real = MSE_loss(D_B_real, self.to_var(torch.ones(D_B_real.size())))

                fake_a = self.netG_BA(real_b)
                D_B_fake = self.netD_B(fake_a)
                errD_B_fake = MSE_loss(D_B_fake, self.to_var(torch.zeros(D_B_fake.size())))

                # optimize
                errD_B = (errD_B_real + errD_B_fake) * 0.5
                self.optimizerD_B.zero_grad()
                errD_B.backward()
                self.optimizerD_B.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                for p in self.netD_A.parameters():
                    p.requires_grad = False # to avoid computation
                for p in self.netD_B.parameters():
                    p.requires_grad = False # to avoid computation

                fake_b = self.netG_AB(real_a)
                D_A_result = self.netD_A(fake_b)
                errG_A = MSE_loss(D_A_result, self.to_var(torch.ones(D_A_result.size())))

                rec_a = self.netG_BA(fake_b)
                cycle_loss_A = L1_loss(rec_a, real_a)

                fake_a = self.netG_BA(real_b)
                D_B_result = self.netD_B(fake_a)
                errG_B = MSE_loss(D_B_result, self.to_var(torch.ones(D_B_result.size())))

                rec_b = self.netG_AB(fake_a)
                cycle_loss_B = L1_loss(rec_b, real_b)

                errG = errG_A + errG_B + 10 * cycle_loss_A + 10 * cycle_loss_B
                self.optimizerG.zero_grad()
                errG.backward()
                self.optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D_A: %.4f Loss_D_B: %.4f Loss_G_A: %.4f Loss_G_B: %.4f'
                      'Loss_A_cycle: %.4f Loss_B_cycle: %.4f'
                      % (epoch + 1, self.niter, i + 1, len(self.train_loader_A),
                         errD_A.data[0], errD_B.data[0], errG_A.data[0], errG_B.data[0],
                         cycle_loss_A.data[0], cycle_loss_B.data[0]))

                if (i + 1) % self.sample_step == 0:
                    # a to b
                    fake_image_list_a = [fixed_a]
                    fake_image_list_a.append(self.netG_AB(fixed_a))
                    fake_image_list_a.append(self.netG_BA(self.netG_AB(fixed_a)))
                    fake_images_a = torch.cat(fake_image_list_a, dim=3)
                    vutils.save_image(fake_images_a.data,
                                      '%s/fake_samples_A_epoch_%03d_step_%03d.png'
                                      % (self.outf, epoch + 1, i + 1),
                                      normalize=True, nrow=1, padding=0)
                    # b to a
                    fake_image_list_b = [fixed_b]
                    fake_image_list_b.append(self.netG_BA(fixed_b))
                    fake_image_list_b.append(self.netG_AB(self.netG_BA(fixed_b)))
                    fake_images_b = torch.cat(fake_image_list_b, dim=3)
                    vutils.save_image(fake_images_b.data,
                                      '%s/fake_samples_B_epoch_%03d_step_%03d.png'
                                      % (self.outf, epoch + 1, i + 1),
                                      normalize=True, nrow=1, padding=0)

                if (i + 1) % self.checkpoint_step == 0:
                    torch.save(self.netG_AB.state_dict(), '%s/netG_AB_epoch_%03d_step_%03d.pth' % (self.outf, epoch + 1, i + 1))
                    torch.save(self.netG_BA.state_dict(), '%s/netG_BA_epoch_%03d_step_%03d.pth' % (self.outf, epoch + 1, i + 1))
                    torch.save(self.netD_A.state_dict(), '%s/netD_A_epoch_%03d_step_%03d.pth' % (self.outf, epoch + 1, i + 1))
                    torch.save(self.netD_B.state_dict(), '%s/netD_B_epoch_%03d_step_%03d.pth' % (self.outf, epoch + 1, i + 1))