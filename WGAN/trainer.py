import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

import models.dcgan as dcgan
import models.mlp as mlp

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
        self.lrD = config.lr_d
        self.lrG = config.lr_g
        self.beta1 = config.beta1
        self.adam = config.adam

        self.niter = config.niter
        self.diters = config.diters

        self.clamp_upper = config.clamp_upper
        self.clamp_lower = config.clamp_lower

        self.outf = config.outf

        self.build_model()

        if self.cuda:
            self.netD.cuda()
            self.netG.cuda()

    def build_model(self):
        if self.config.no_bn:
            self.netG = dcgan.DCGAN_G_nobn(self.image_size, self.nz, self.nc, self.ngf, self.ngpu, self.config.n_extra_layers)
        elif self.config.mlp_G:
            self.netG = mlp.MLP_G(self.image_size, self.nz, self.nc, self.ngf, self.ngpu)
        else:
            self.netG = dcgan.DCGAN_G(self.image_size, self.nz, self.nc, self.ngf, self.ngpu, self.config.n_extra_layers)
            self.netG.apply(weights_init)

        if self.config.netG != '': # load checkpoint if needed
            self.netG.load_state_dict(torch.load(self.config.netG))

        if self.config.mlp_D:
            self.netD = mlp.MLP_D(self.image_size, self.nz, self.nc, self.ndf, self.ngpu)
        else:
            self.netD = dcgan.DCGAN_D(self.image_size, self.nz, self.nc, self.ndf, self.ngpu, self.config.n_extra_layers)
            self.netD.apply(weights_init)

        if self.config.netD != '':
            self.netD.load_state_dict(torch.load(self.config.netD))


    def train(self):

        input = torch.FloatTensor(self.batch_size, 3, self.image_size, self.image_size)
        noise = torch.FloatTensor(self.batch_size, self.nz, 1, 1)
        fixed_noise = torch.FloatTensor(self.batch_size, self.nz, 1, 1).normal_(0, 1)
        one = torch.FloatTensor([1])
        mone = one * -1

        if self.cuda:
            input = input.cuda()
            one, mone = one.cuda(), mone.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        fixed_noise = Variable(fixed_noise, volatile=True)

        # setup optimizer
        if self.adam:
            optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
            optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        else:
            optimizerD = optim.RMSprop(self.netD.parameters(), lr = self.lrD)
            optimizerG = optim.RMSprop(self.netG.parameters(), lr = self.lrG)

        gen_iterations = 0
        for epoch in range(self.niter):
            data_iter = iter(self.data_loader)
            i = 0
            while i < len(self.data_loader):
                ############################
                # (1) Update D network
                ###########################
                for p in self.netD.parameters(): # reset requires_grad
                    p.requires_grad = True # they are set to False below in netG update

                # train the discriminator Diters times
                if gen_iterations < 25 or gen_iterations % 500 == 0:
                    diters = 100
                else:
                    diters = self.diters
                j = 0
                while j < diters and i < len(self.data_loader):
                    j += 1

                    # clamp parameters to a cube
                    for p in self.netD.parameters():
                        p.data.clamp_(self.clamp_lower, self.clamp_upper)

                    data = data_iter.next()
                    i += 1

                    # train with real
                    real_cpu, _ = data
                    self.netD.zero_grad()
                    self.batch_size = real_cpu.size(0)

                    if self.cuda:
                        real_cpu = real_cpu.cuda()
                    input.resize_as_(real_cpu).copy_(real_cpu)
                    inputv = Variable(input)

                    errD_real = self.netD(inputv)
                    errD_real.backward(one)

                    # train with fake
                    noise.resize_(self.batch_size, self.nz, 1, 1).normal_(0, 1)
                    noisev = Variable(noise)
                    fake = self.netG(noisev)
                    errD_fake = self.netD(fake.detach())
                    errD_fake.backward(mone)
                    errD = errD_real - errD_fake
                    optimizerD.step()

                ############################
                # (2) Update G network
                ###########################
                for p in self.netD.parameters():
                    p.requires_grad = False # to avoid computation
                self.netG.zero_grad()
                # in case our last batch was the tail batch of the data_loader,
                # make sure we feed a full batch of noise
                noise.resize_(self.batch_size, self.nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise)
                fake = self.netG(noisev)
                errG = self.netD(fake)
                errG.backward(one)
                optimizerG.step()
                gen_iterations += 1

                print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                    % (epoch, self.niter, i, len(self.data_loader), gen_iterations,
                    errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
                if gen_iterations % 500 == 0:
                    vutils.save_image(real_cpu, 
                            '%s/real_samples.png' % self.outf, 
                            normalize=True)
                    fake = self.netG(fixed_noise)
                    vutils.save_image(fake.data, 
                            '%s/fake_samples_%03d.png' % (self.outf, gen_iterations), 
                            normalize=True)

            # do checkpointing
            torch.save(self.netG.state_dict(), '%s/netG_epoch_%03d.pth' % (self.outf, epoch))
            torch.save(self.netD.state_dict(), '%s/netD_epoch_%03d.pth' % (self.outf, epoch))
