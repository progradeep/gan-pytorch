import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

import models.began as began

def L1Loss(a, b):
    return torch.mean(torch.abs(a-b))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

        self.ngpu = int(config.ngpu)
        self.nc = int(config.nc)
        self.nz = int(config.nz)
        self.n_hidden = int(config.n_hidden)
        self.cuda = config.cuda

        self.batch_size = config.batch_size
        self.image_size = config.image_size

        self.conv_hidden_num = config.conv_hidden_num

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        
        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.lr_update_step = config.lr_update_step

        self.niter = config.niter

        self.outf = config.outf

        self.build_model()

        if self.cuda:
            self.netD.cuda()
            self.netG.cuda()

    def build_model(self):
        self.netD = began._netD(self.ngpu, self.nz, self.n_hidden, self.nc)
        self.netD.apply(weights_init)
        if self.config.netD != '':
            self.netD.load_state_dict(torch.load(self.config.netD))
        self.netG = began._netG(self.ngpu, self.nz, self.n_hidden, self.nc)
        self.netG.apply(weights_init)
        if self.config.netG != '':
            self.netG.load_state_dict(torch.load(self.config.netG))
       
    def train(self):
        l1=L1Loss

        input = torch.FloatTensor(self.batch_size, 3, self.image_size, self.image_size)
        D_noise = torch.FloatTensor(self.batch_size, self.nz)
        G_noise = torch.FloatTensor(self.batch_size, self.nz)
        fixed_noise = torch.FloatTensor(self.batch_size, self.nz).normal_(0, 1)

        if self.cuda:
            input = input.cuda()
            D_noise, G_noise, fixed_noise = D_noise.cuda(), G_noise.cuda(), fixed_noise.cuda()

        fixed_noise = Variable(fixed_noise)

        # setup optimizer
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        k_t = 0

        for epoch in range(self.niter):
            for i, data in enumerate(self.data_loader, 0):

                # train D network
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                if self.cuda:
                    real_cpu = real_cpu.cuda()
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)

                AE_x = self.netD(inputv)

                D_noise.resize_(self.batch_size, self.nz).normal_(0, 1)
                D_noisev = Variable(D_noise)
                D_fake = self.netG(D_noisev)
                AE_G_d = self.netD(D_fake.detach())

                d_loss_real = l1(AE_x, inputv)
                d_loss_fake = l1(AE_G_d, D_fake.detach())
                d_loss = d_loss_real - k_t * d_loss_fake

                self.netD.zero_grad()
                d_loss.backward()
                optimizerD.step()

                #train G network
                G_noise.resize_(self.batch_size, self.nz).normal_(0, 1)
                G_noisev = Variable(G_noise)
                G_fake = self.netG(G_noisev)
                AE_G_g = self.netD(G_fake)

                g_loss = l1(G_fake, AE_G_g)

                self.netG.zero_grad()
                g_loss.backward()
                optimizerG.step()

                g_d_balance = (self.gamma * d_loss_real - d_loss_fake).data[0]
                k_t += self.lambda_k * g_d_balance
                k_t = max(min(1, k_t), 0)

                measure = d_loss_real.data[0] + abs(g_d_balance)

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Measure: %.4f'
                      % (epoch, self.niter, i, len(self.data_loader),
                         d_loss_fake.data[0], g_loss.data[0], measure))
                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                            '%s/real_samples.png' % self.outf,
                            normalize=True)
                    fake = self.netG(fixed_noise)
                    vutils.save_image(fake.data,
                            '%s/fake_samples_epoch_%03d.png' % (self.outf, epoch),
                            normalize=True)

            # do checkpointing
            torch.save(self.netG.state_dict(), '%s/netG_epoch_%03d.pth' % (self.outf, epoch))
            torch.save(self.netD.state_dict(), '%s/netD_epoch_%03d.pth' % (self.outf, epoch)) 




