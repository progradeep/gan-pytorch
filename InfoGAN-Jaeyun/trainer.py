import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

import models.infogan as infogan

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def noise_sample(dis_c, con_c, noise, batch_size, nz):

    idx = np.random.randint(10, size=batch_size)
    c = np.zeros((batch_size, 10))
    c[range(batch_size), idx] = 1.0

    dis_c.data.copy_(torch.Tensor(c))
    con_c.data.uniform_(-1.0, 1.0)
    noise.data.normal_(0, 1)
    z = torch.cat([noise, dis_c, con_c], 1).view(-1, nz + 12, 1, 1)

    return z, idx

class log_gaussian:
	def __call__(self, x, mu, var):

		logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
				(x-mu).pow(2).div(var.mul(2.0)+1e-6)
    
		return logli.sum(1).mean().mul(-1)

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
            self.netQ.cuda()
            self.netShareDQ.cuda()

    def build_model(self):
        self.netG = infogan._netG(self.ngpu, self.nz, self.ngf, self.nc)
        self.netG.apply(weights_init)
        if self.config.netG != '':
            self.netG.load_state_dict(torch.load(self.config.netG))
        self.netD = infogan._netD(self.ngpu, self.nc, self.ndf)
        self.netD.apply(weights_init)
        if self.config.netD != '':
            self.netD.load_state_dict(torch.load(self.config.netD))
        self.netQ = infogan._netQ(self.ngpu, self.nc, self.ndf)
        self.netQ.apply(weights_init)
        if self.config.netQ != '':
            self.netQ.load_state_dict(torch.load(self.config.netQ))
        self.netShareDQ = infogan._netShareDQ(self.ngpu, self.nc, self.ndf)
        self.netShareDQ.apply(weights_init)
        if self.config.netShareDQ != '':
            self.netShareDQ.load_state_dict(torch.load(self.config.netShareDQ))
       
    def train(self):
        criterion = nn.BCELoss()
        criterionQ_dis = nn.CrossEntropyLoss()
        criterionQ_con = log_gaussian()

        input = torch.FloatTensor(self.batch_size, self.nc, self.image_size, self.image_size)
        noise = torch.FloatTensor(self.batch_size, self.nz, 1, 1)
        dis_c = torch.FloatTensor(self.batch_size, 10)
        con_c = torch.FloatTensor(self.batch_size, 2)
        fixed_noise = torch.FloatTensor(100, self.nz, 1, 1).normal_(0, 1)
        label = torch.FloatTensor(self.batch_size)
        real_label = 1
        fake_label = 0

        if self.cuda:
            criterion.cuda()
            criterionQ_dis.cuda()
            criterionQ_con
            input, label = input.cuda(), label.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            dis_c, con_c = dis_c.cuda(), con_c.cuda()

        fixed_noise = Variable(fixed_noise)

        c = np.linspace(-1, 1, 10).reshape(1, -1)
        c = np.repeat(c, 10, 0).reshape(-1, 1)
        c1 = np.hstack([c, np.zeros_like(c)])
        c2 = np.hstack([np.zeros_like(c), c])

        c1 = torch.Tensor(c1)
        c2 = torch.Tensor(c2)

        idx = np.arange(10).repeat(10)
        one_hot = np.zeros((100, 10))
        one_hot[range(100), idx] = 1
        one_hot = torch.Tensor(one_hot)

        if self.cuda:
            c1 = Variable(c1.cuda())
            c2 = Variable(c2.cuda())
            one_hot = Variable(one_hot.cuda())

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
                noise.resize_(batch_size, self.nz)
                dis_c.resize_(batch_size, 10)
                con_c.resize_(batch_size, 2)
                inputv = Variable(input)
                labelv = Variable(label)
                noisev = Variable(noise)
                dis_cv = Variable(dis_c)
                con_cv = Variable(con_c)

                output = self.netD(self.netShareDQ(inputv))
                errD_real = criterion(output, labelv)
                errD_real.backward()
                D_x = output.data.mean()

                # train with fake
                z, idx = noise_sample(dis_cv, con_cv, noisev, batch_size, self.nz)
                fake = self.netG(z)
                labelv = Variable(label.fill_(fake_label))
                output = self.netD(self.netShareDQ(fake.detach()))
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
                shared_output = self.netShareDQ(fake)
                output = self.netD(shared_output)
                errG = criterion(output, labelv)
                q_digit, q_mu, q_var = self.netQ(shared_output)
                class_ = torch.LongTensor(idx).cuda()
                target = Variable(class_)
                err_dis = criterionQ_dis(q_digit, target)
                err_con = criterionQ_con(con_c, q_mu, q_var)*0.1
                errG += (err_dis + err_con)
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
                    z = torch.cat([fixed_noise, one_hot, c1], 1).view(-1, self.nz + 12, 1, 1)
                    fake = self.netG(z)
                    vutils.save_image(fake.data,
                            '%s/fake_samples_c1_epoch_%03d.png' % (self.outf, epoch),
                            normalize=True, nrow=10)
                    z = torch.cat([fixed_noise, one_hot, c2], 1).view(-1, self.nz + 12, 1, 1)
                    fake = self.netG(z)
                    vutils.save_image(fake.data,
                            '%s/fake_samples_c1_epoch_%03d.png' % (self.outf, epoch),
                            normalize=True, nrow=10)

            # do checkpointing
            torch.save(self.netG.state_dict(), '%s/netG_epoch_%03d.pth' % (self.outf, epoch))
            torch.save(self.netD.state_dict(), '%s/netD_epoch_%03d.pth' % (self.outf, epoch)) 




