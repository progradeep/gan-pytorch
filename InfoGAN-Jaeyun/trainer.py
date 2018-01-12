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

def noise_sample(c, noise, bs, nz):
    idx = []
    z_list = [noise]
    for i in range(10):
        idx.append(np.random.randint(10, size=bs))
        c = np.zeros((bs, 10))
        c[range(bs),idx] = 1.0
        c = torch.Tensor(c).cuda()
        z_list.append(Variable(c))
    
    noise.data.uniform_(-1.0, 1.0)
    z = torch.cat(z_list, 1).view(-1, nz + 100, 1, 1)

    return z, idx

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

        self.lrD = config.lrD
        self.lrG = config.lrG
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
        criterionQ = nn.CrossEntropyLoss()

        input = torch.FloatTensor(self.batch_size, self.nc, self.image_size, self.image_size)
        noise = torch.FloatTensor(self.batch_size, self.nz)
        c = torch.FloatTensor(self.batch_size, 100)

        fixed_noise = np.zeros((100, self.nz))

        for i in range(10):
            row = np.random.randn(1, self.nz)
            row = np.tile(row, (10, 1))
            fixed_noise[10*i:10*(i+1), :] = row

        fixed_noise = torch.FloatTensor(fixed_noise)
        
        label = torch.FloatTensor(self.batch_size)
        real_label = 1
        fake_label = 0
        
        noisev = Variable(noise)
        cv = Variable(c)
        fixed_noise = Variable(fixed_noise)

        if self.cuda:
            criterion.cuda()
            input, label = input.cuda(), label.cuda()
            noisev.data, cv.data, fixed_noise = noisev.data.cuda(), cv.data.cuda(), fixed_noise.cuda()

        idx = np.arange(10).repeat(10)
        one_hot = np.zeros((100, 10))
        one_hot[range(100), idx] = 1

        c_fixed_list = []
        for i in range(10):
            c_fixed_i = np.zeros((100, 100))
            c_fixed_i[:, i*10:(i+1)*10] = one_hot
            c_fixed_i = torch.FloatTensor(c_fixed_i).cuda()
            c_fixed_list.append(Variable(c_fixed_i))

        # setup optimizer
        optimizerD = optim.Adam([{'params':self.netShareDQ.parameters()}, {'params':self.netD.parameters()}], lr=self.lrD, betas=(self.beta1, 0.999))
        optimizerG = optim.Adam([{'params':self.netG.parameters()}, {'params': self.netQ.parameters()}], lr=self.lrG, betas=(self.beta1, 0.999))

        for epoch in range(self.niter):
            for i, data in enumerate(self.data_loader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                for p in self.netD.parameters(): # reset requires_grad
                    p.requires_grad = True # they are set to False below in netG update

                # train with real
                optimizerD.zero_grad()
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                if self.cuda:
                    real_cpu = real_cpu.cuda()
                input.resize_as_(real_cpu).copy_(real_cpu)
                label.resize_(batch_size).fill_(real_label)

                noisev.data.resize_(batch_size, self.nz)
                cv.data.resize_(batch_size, 100)

                inputv = Variable(input)
                labelv = Variable(label)

                output = self.netD(self.netShareDQ(inputv))
                errD_real = criterion(output, labelv)
                errD_real.backward()
                D_x = output.data.mean()

                noisev.data = noisev.data.normal_(0, 1)
                z, idx = noise_sample(cv, noisev, batch_size, self.nz)

                # train with fake
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
                optimizerG.zero_grad()
                labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
                shared_output = self.netShareDQ(fake)
                output = self.netD(shared_output)
                err = criterion(output, labelv)
                
                q_list = self.netQ(shared_output)
                
                class_ = torch.LongTensor(idx).cuda()
                target = Variable(class_)
                err_c_list = []
                for k in range(10):
                    err_c = criterionQ(q_list[k], target[k])
                    err_c_list.append(err_c)

                errG = err + torch.mean(err_c)

                errG.backward()
                D_G_z2 = output.data.mean()
                optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, self.niter, i, len(self.data_loader),
                         errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                            '%s/real_samples.png' % self.outf,
                            nrow=10, normalize=True)
                    for idx, c_fixed in enumerate(c_fixed_list):
                        z = torch.cat([fixed_noise, c_fixed], 1).view(-1, self.nz + 100, 1, 1)
                        fake = self.netG(z)
                        vutils.save_image(fake.data,
                                '%s/fake_samples_c%i_epoch_%03d.png' % (self.outf, idx, epoch),
                                nrow=10, normalize=True)

            # do checkpointing
            torch.save(self.netG.state_dict(), '%s/netG_epoch_%03d.pth' % (self.outf, epoch))
            torch.save(self.netD.state_dict(), '%s/netD_epoch_%03d.pth' % (self.outf, epoch)) 




