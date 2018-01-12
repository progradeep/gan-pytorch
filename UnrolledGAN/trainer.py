import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import models.dcgan as dcgan
from utils import circle

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, gain=0.8)

class Trainer(object):
    def __init__(self, config):
        self.config = config

        self.ngpu = int(config.ngpu)
        self.nc = int(config.nc)
        self.nz = int(config.nz)
        self.ngf = int(config.ngf)
        self.ndf = int(config.ndf)
        self.cuda = config.cuda

        self.batch_size = config.batch_size

        self.lrG = config.lrG
        self.lrD = config.lrD
        self.beta1 = config.beta1

        self.niter = config.niter

        self.outf = config.outf

        self.unrolling_steps = config.unrolling_steps

        self.build_model()

        if self.cuda:
            self.netD.cuda()
            self.netG.cuda()

    def build_model(self):
        self.netG = dcgan._netG(self.ngpu, self.nz, self.ngf, self.nc)
        self.netG.apply(weights_init)
        self.netD = dcgan._netD(self.ngpu, self.nc, self.ndf)
        self.netD.apply(weights_init)
        
    def train(self):
        criterion = nn.BCELoss()

        input = torch.FloatTensor(self.batch_size, self.nc)
        noise = torch.FloatTensor(self.batch_size, self.nz)
        fixed_noise = torch.FloatTensor(self.batch_size, self.nz).normal_(0, 1)
        label = torch.FloatTensor(self.batch_size)
        real_label = 1
        fake_label = 0

        if self.cuda:
            criterion.cuda()
            input, label = input.cuda(), label.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        fixed_noise = Variable(fixed_noise)

        # setup optimizer
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.lrD, betas=(self.beta1, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=self.lrG, betas=(self.beta1, 0.999))

        for epoch in range(self.niter):

            data, _ = circle(self.batch_size)
            if self.cuda:
                data = data.cuda()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # train with real
            self.netD.zero_grad()
            
            label.fill_(real_label)
            inputv = Variable(data)
            labelv = Variable(label)

            output = self.netD(inputv)
            errD_real = criterion(output, labelv)
            D_x = output.data.mean()
            errD_real.backward()

            # train with fake
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = self.netG(noisev)
            
            labelv = Variable(label.fill_(fake_label))
            output = self.netD(fake.detach())
            errD_fake = criterion(output, labelv)
            D_G_z1 = output.data.mean()
            errD_fake.backward()

            errD = errD_real + errD_fake

            optimizerD.step()

            ############################
            # (2) Update G network: minimize log(1 - D(G(z)))
            ###########################
            self.netG.zero_grad()
            
            labelv = Variable(label.fill_(fake_label))
            output = self.netD(fake)
            errG = -criterion(output, labelv)
            D_G_z2 = output.data.mean()
            
            errG.backward()
            
            optimizerG.step()

            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, self.niter,
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            if epoch % 1000 == 0:
                plt.scatter(data[:,0], data[:,1], s=10)
                plt.savefig(
                        '%s/real_samples.png' % self.outf)
                fake = self.netG(fixed_noise)
                plt.scatter(fake[:,0], fake[:,1], s=10)
                plt.savefig(
                        '%s/fake_samples_epoch_%03d.png' % (self.outf, epoch))
                plt.close()
