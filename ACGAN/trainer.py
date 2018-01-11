import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
from torch.autograd import Variable

import models.acgan as acgan


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
        self.ngf = int(config.ngf)
        self.ndf = int(config.ndf)
        self.cuda = config.cuda

        self.batch_size = config.batch_size
        self.image_size = config.image_size

        self.lr = config.lr
        self.beta1 = config.beta1

        self.niter = config.niter

        self.outf = config.outf

        self.nl = config.nl  # add nl

        self.build_model()

        if self.cuda:
            self.netD.cuda()
            self.netG.cuda()

    def build_model(self):
        self.netG = acgan._netG(self.ngpu, self.nz, self.ngf, self.nc)
        self.netG.apply(weights_init)
        if self.config.netG != '':
            self.netG.load_state_dict(torch.load(self.config.netG))
        self.netD = acgan._netD(self.ngpu, self.nl, self.ndf, self.nc)
        self.netD.apply(weights_init)
        if self.config.netD != '':
            self.netD.load_state_dict(torch.load(self.config.netD))

    def train(self):
        dis_criterion = nn.BCELoss()
        aux_criterion = nn.NLLLoss()  # add class loss

        input = torch.FloatTensor(self.batch_size, 3, self.image_size, self.image_size)
        noise = torch.FloatTensor(self.batch_size, self.nz, 1, 1)
        fixed_noise = torch.FloatTensor(self.batch_size, self.nz, 1, 1).normal_(0, 1)
        dis_label = torch.FloatTensor(self.batch_size)
        aux_label = torch.LongTensor(self.batch_size) # add class label
        real_label = 1
        fake_label = 0

        if self.cuda:
            dis_criterion.cuda()
            aux_criterion.cuda()
            input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()  # add class label
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        inputv = Variable(input)
        noisev = Variable(noise)
        fixed_noisev = Variable(fixed_noise)
        dis_labelv = Variable(dis_label)
        aux_labelv = Variable(aux_label)

        # setup optimizer
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        for epoch in range(self.niter):
            for i, data in enumerate(self.data_loader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                for p in self.netD.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update

                # train with real
                self.netD.zero_grad()
                real_cpu, c_label = data  # add c_label
                batch_size = real_cpu.size(0)
                if self.cuda:
                    real_cpu = real_cpu.cuda()
                    c_label = c_label.cuda()  # add c label
                inputv.data.resize_as_(real_cpu).copy_(real_cpu)
                dis_labelv.data.resize_(batch_size).fill_(real_label)
                aux_labelv.data.resize_(batch_size).copy_(c_label)

                dis_out, aux_out = self.netD(inputv)
                dis_errD_real = dis_criterion(dis_out, dis_labelv)
                aux_errD_real = aux_criterion(aux_out, aux_labelv)
                errD_real = dis_errD_real + aux_errD_real
                errD_real.backward()

                D_x = dis_out.data.mean()

                # train with fake
                noisev.data.resize_(batch_size, self.nz, 1, 1).normal_(0, 1)
                c_label = np.random.randint(0, self.nl, batch_size)
                noisev_ = np.random.normal(0, 1, (batch_size, self.nz))
                class_onehot = np.zeros((batch_size, self.nl))
                class_onehot[np.arange(batch_size), c_label] = 1
                noisev_[np.arange(batch_size), :self.nl] = class_onehot[np.arange(batch_size)]
                noisev_ = (torch.from_numpy(noisev_))
                noisev.data.copy_(noisev_.view(batch_size, self.nz, 1, 1))
                aux_labelv.data.resize_(batch_size).copy_(torch.from_numpy(c_label))

                fake = self.netG(noisev)
                dis_labelv.data.fill_(fake_label)
                dis_out, aux_out = self.netD(fake.detach())
                dis_errD_fake = dis_criterion(dis_out, dis_labelv)
                aux_errD_fake = aux_criterion(aux_out, aux_labelv)
                errD_fake = dis_errD_fake + aux_errD_fake
                errD_fake.backward()

                D_G_z1 = dis_out.data.mean()

                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                for p in self.netD.parameters():
                    p.requires_grad = False  # to avoid computation
                self.netG.zero_grad()
                dis_labelv.data.fill_(real_label) # fake labels are real for generator cost
                dis_out, aux_out = self.netD(fake)

                dis_errG = dis_criterion(dis_out, dis_labelv)
                aux_errG = aux_criterion(aux_out, aux_labelv)
                errG = dis_errG + aux_errG
                errG.backward()

                D_G_z2 = dis_out.data.mean()
                optimizerG.step()

                print(
                    '[%d/%d][%d/%d] Loss_D_real: %.4f Loss_D_fake: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch+1, self.niter, i+1, len(self.data_loader),
                       errD_real.data[0], errD_fake.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                                      '%s/real_samples.png' % self.outf,
                                      normalize=True)
                    fake = self.netG(fixed_noisev)
                    vutils.save_image(fake.data,
                                      '%s/fake_samples_epoch_%03d.png' % (self.outf, epoch+1),
                                      normalize=True)

            # do checkpointing
            torch.save(self.netG.state_dict(), '%s/netG_epoch_%03d.pth' % (self.outf, epoch+1))
            torch.save(self.netD.state_dict(), '%s/netD_epoch_%03d.pth' % (self.outf, epoch+1))
