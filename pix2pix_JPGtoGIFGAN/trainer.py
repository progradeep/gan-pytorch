import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from data_loader import get_loader

import models.pix2pix as pix2pix

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.dataroot = config.dataroot
        self.workers = config.workers

        self.ngpu = int(config.ngpu)
        self.nc = int(config.nc)
        self.input_nc = int(config.input_nc)
        self.output_nc = int(config.output_nc)
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

        self.lamb = config.lamb

        self.build_model()

        if self.cuda:
            self.netD.cuda()
            self.netG.cuda()

    def build_model(self):
        self.netG = pix2pix.define_G(self.input_nc, self.output_nc * 10, self.ngf, 'batch', False, [])
        self.netD = pix2pix.define_D(self.input_nc + self.output_nc * 10, self.ndf, 'batch', False, [])
        if self.config.netG != '':
            self.netG.load_state_dict(torch.load(self.config.netG))
        if self.config.netD != '':
            self.netD.load_state_dict(torch.load(self.config.netD))
        print('---------- Networks initialized -------------')
        pix2pix.print_network(self.netG)
        pix2pix.print_network(self.netD)
        print('---------------------------------------------')
        
    def train(self):
        self.data_loader = get_loader(dataroot=self.dataroot, batch_size=self.batch_size, image_size = self.image_size
                                 ,num_workers=int(self.workers), shuffle = True)

        criterionGAN = pix2pix.GANLoss()
        criterionL1 = nn.L1Loss()
        criterionMSE = nn.MSELoss()

        real_a = torch.FloatTensor(self.batch_size, self.input_nc, 256, 256)
        real_b = torch.FloatTensor(self.batch_size, self.output_nc * 10, 256, 256)

        if self.cuda:
            criterionGAN = criterionGAN.cuda()
            criterionL1 = criterionL1.cuda()
            criterionMSE = criterionMSE.cuda()
            real_a = real_a.cuda()
            real_b = real_b.cuda()

        real_a = Variable(real_a)
        real_b = Variable(real_b)

        # setup optimizer
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        for epoch in range(self.niter):
            for i, data in enumerate(self.data_loader, 0):
                # forward
                real_a_cpu, real_b_cpu = data[0], torch.cat(data[1], 1)
                batch_size = real_a_cpu.size(0)
                real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
                real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
                fake_b = self.netG(real_a)

                ############################
                # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
                ###########################

                optimizerD.zero_grad()

                # train with fake
                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = self.netD.forward(fake_ab.detach())
                loss_d_fake = criterionGAN(pred_fake, False)

                # train with real
                real_ab = torch.cat((real_a, real_b), 1)
                pred_real = self.netD.forward(real_ab)
                loss_d_real = criterionGAN(pred_real, True)

                # Combined loss
                loss_d = (loss_d_fake + loss_d_real) * 0.5

                loss_d.backward()

                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
                ##########################
                optimizerG.zero_grad()
                # First, G(A) should fake the discriminator
                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = self.netD.forward(fake_ab)
                loss_g_gan = criterionGAN(pred_fake, True)

                # Second, G(A) = B
                loss_g_l1 = criterionL1(fake_b, real_b) * self.lamb

                loss_g = loss_g_gan + loss_g_l1

                loss_g.backward()

                optimizerG.step()

                print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                    epoch, i, len(self.data_loader), loss_d.data[0], loss_g.data[0]))

                if i % 100 == 0:
                    real_a_cpu.resize_(batch_size, self.nc, self.image_size, self.image_size)
                    vutils.save_image(real_a_cpu,
                            '%s/real_samples_A.png' % self.outf,
                            normalize=True, nrow=1)
                    real_b_cpu.resize_(batch_size * 10, self.nc, self.image_size, self.image_size)
                    vutils.save_image(real_b_cpu,
                            '%s/real_samples_B.png' % self.outf,
                            normalize=True, nrow=10)
                    fake_b = self.netG(real_a)
                    fake_b.data.resize_(batch_size * 10, self.nc, self.image_size, self.image_size)
                    vutils.save_image(fake_b.data,
                            '%s/fake_samples_B_epoch_%03d.png' % (self.outf, epoch),
                            normalize=True, nrow=10)


            # do checkpointing
            torch.save(self.netG.state_dict(), '%s/netG_epoch_%03d.pth' % (self.outf, epoch))
            torch.save(self.netD.state_dict(), '%s/netD_epoch_%03d.pth' % (self.outf, epoch)) 




