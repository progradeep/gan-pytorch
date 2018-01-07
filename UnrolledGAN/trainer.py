import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

import models.dcgan as dcgan

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

        self.build_model()

        if self.cuda:
            self.netD.cuda()
            self.netG.cuda()

    def build_model(self):
        self.netG = dcgan._netG(self.ngpu, self.nz, self.ngf, self.nc)
        self.netG.apply(weights_init)
        if self.config.netG != '':
            self.netG.load_state_dict(torch.load(self.config.netG))
        self.netD = dcgan._netD(self.ngpu, self.nc, self.ndf)
        self.netD.apply(weights_init)
        if self.config.netD != '':
            self.netD.load_state_dict(torch.load(self.config.netD))
        
    def train(self):
        criterion = nn.BCELoss()

        input = torch.FloatTensor(self.batch_size, 3, self.image_size, self.image_size)
        noise = torch.FloatTensor(self.batch_size, self.nz, 1, 1)
        fixed_noise = torch.FloatTensor(self.batch_size, self.nz, 1, 1).normal_(0, 1)
        label = torch.FloatTensor(self.batch_size)
        real_label = 1
        fake_label = 0

        if self.cuda:
            criterion.cuda()
            input, label = input.cuda(), label.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        fixed_noise = Variable(fixed_noise)

        # setup optimizer
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        for epoch in range(self.niter):
            for i, data in enumerate(self.data_loader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                for p in self.netD.parameters(): # reset requires_grad
                    p.requires_grad = True # they are set to False below in netG update

                # train with real
                self.netD.zero_grad()
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                if self.cuda:
                    real_cpu = real_cpu.cuda()
                input.resize_as_(real_cpu).copy_(real_cpu)
                label.resize_(batch_size).fill_(real_label)
                inputv = Variable(input)
                labelv = Variable(label)

                output = self.netD(inputv)
                errD_real = criterion(output, labelv)
                errD_real.backward()
                D_x = output.data.mean()

                # train with fake
                noise.resize_(self.batch_size, self.nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise)
                fake = self.netG(noisev)
                labelv = Variable(label.fill_(fake_label))
                output = self.netD(fake.detach())
                errD_fake = criterion(output, labelv)
                errD_fake.backward()
                D_G_z1 = output.data.mean()
                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                for p in self.netD.parameters():
                    p.requires_grad = False # to avoid computation
                self.netG.zero_grad()
                labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
                output = self.netD(fake)
                errG = criterion(output, labelv)
                errG.backward()
                D_G_z2 = output.data.mean()
                optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, self.niter, i, len(self.data_loader),
                         errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
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




