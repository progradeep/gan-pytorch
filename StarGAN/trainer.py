import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

import models.stargan as stargan

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('InstanceNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.cuda = config.cuda
        self.ngpu = int(config.ngpu)

        self.na = int(config.na)
        self.ngf = int(config.ngf)
        self.ndf = int(config.ndf)
        self.crop_size = config.crop_size
        self.image_size = config.image_size

        self.niter = config.niter
        self.niter_decay = config.niter_decay
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.beta1 = config.beta1

        self.outf = config.outf
        self.sample_step = config.sample_step
        self.checkpoint_step = config.checkpoint_step

        self.build_model()

    def build_model(self):
        self.netG = stargan._netG(self.ngpu, self.na, self.ngf)
        self.netG.apply(weights_init)
        if self.config.netG != '':
            self.netG.load_state_dict(torch.load(self.config.netG))
            print('restored {0}'.format(self.config.netG))
        self.netD = stargan._netD(self.ngpu, self.na, self.ndf)
        self.netD.apply(weights_init)
        if self.config.netD != '':
            self.netD.load_state_dict(torch.load(self.config.netD))
            print('restored {0}'.format(self.config.netD))

        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        if self.cuda:
            self.netG.cuda()
            self.netD.cuda()

    def to_var(self, x, volatile=False):
        if self.cuda:
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def make_celeb_labels(self, real_c):
        y = [torch.FloatTensor([1, 0, 0]),  # black hair
             torch.FloatTensor([0, 1, 0]),  # blond hair
             torch.FloatTensor([0, 0, 1])]  # brown hair

        fixed_c_list = []

        # single attribute transfer
        for i in range(self.na):
            fixed_c = real_c.clone()
            for c in fixed_c:
                if i < 3:
                    c[:3] = y[i]
                else:
                    c[i] = 0 if c[i] == 1 else 1  # opposite value
            fixed_c_list.append(self.to_var(fixed_c, volatile=True))

        # multi-attribute transfer (H+G, H+A, G+A, H+G+A)
        for i in range(4):
            fixed_c = real_c.clone()
            for c in fixed_c:
                if i in [0, 1, 3]:  # Hair color to brown
                    c[:3] = y[2]
                if i in [0, 2, 3]:  # Gender
                    c[3] = 0 if c[3] == 1 else 1
                if i in [1, 2, 3]:  # Aged
                    c[4] = 0 if c[4] == 1 else 1
            fixed_c_list.append(self.to_var(fixed_c, volatile=True))
        return fixed_c_list

    def train(self):
        # fixed images for samples
        fixed_x = []
        real_c = []
        for i, (images, labels) in enumerate(self.data_loader):
            fixed_x.append(images)
            real_c.append(labels)
            if i == 3:
                break

        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)
        real_c = torch.cat(real_c, dim=0)

        fixed_c_list = self.make_celeb_labels(real_c)

        for epoch in range(self.niter):
            for i, (real_x, real_label) in enumerate(self.data_loader):

                rand_idx = torch.randperm(real_label.size(0))
                fake_label = real_label[rand_idx]
                real_c = real_label.clone()
                fake_c = fake_label.clone()

                real_x = self.to_var(real_x)
                real_label = self.to_var(real_label)
                fake_label = self.to_var(fake_label)
                real_c = self.to_var(real_c) # for netG
                fake_c = self.to_var(fake_c) # for netG

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                for p in self.netD.parameters(): # reset requires_grad
                    p.requires_grad = True # they are set to False below in netG update

                # train with real
                out_src, out_cls = self.netD(real_x)
                errD_real = - torch.mean(out_src)

                errD_cls = F.binary_cross_entropy_with_logits(
                    out_cls, real_label, size_average=False) / real_x.size(0)

                # train with fake
                fake_x = self.netG(real_x, fake_c)
                fake_x = Variable(fake_x.data)
                out_src, out_cls = self.netD(fake_x)
                errD_fake = torch.mean(out_src)

                # optimize
                errD = errD_real + errD_fake + 1 * errD_cls
                self.netD.zero_grad()
                errD.backward()
                self.optimizerD.step()

                # gradient penalty
                alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
                interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
                out, out_cls = self.netD(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                errD_gp = torch.mean((grad_l2norm - 1) ** 2)

                # optimize
                errD = 10 * errD_gp
                self.netD.zero_grad()
                errD.backward()
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                for p in self.netD.parameters():
                    p.requires_grad = False # to avoid computation

                if (i + 1) % 5 == 0:
                    fake_x = self.netG(real_x, fake_c)
                    rec_x = self.netG(fake_x, real_c)

                    out_src, out_cls = self.netD(fake_x)
                    errG_fake = - torch.mean(out_src)
                    errG_rec = torch.mean(torch.abs(real_x - rec_x))
                    errG_cls = F.binary_cross_entropy_with_logits(
                        out_cls, fake_label, size_average=False) / fake_x.size(0)

                    errG = errG_fake + 10 * errG_rec + 1 * errG_cls
                    self.netG.zero_grad()
                    errG.backward()
                    self.optimizerG.step()

                    print('[%d/%d][%d/%d] Loss_D_real/fake: %.4f Loss_D_cls: %.4f Loss_D_gp: %.4f'
                          'Loss_G_fake: %.4f Loss_G_rec: %.4f Loss_G_cls: %.4f'
                          % (epoch + 1, self.niter, i + 1, len(self.data_loader),
                             errD_real.data[0]+errD_fake.data[0], errD_cls.data[0], errD_gp.data[0],
                             errG_fake.data[0], errG_rec.data[0], errG_cls.data[0]))

                if (i + 1) % self.sample_step == 0:
                    fake_image_list = [fixed_x]
                    for fixed_c in fixed_c_list:
                        fake_image_list.append(self.netG(fixed_x, fixed_c))
                    fake_images = torch.cat(fake_image_list, dim=3)
                    vutils.save_image(fake_images.data,
                                      '%s/fake_samples_epoch_%03d_step_%03d.png'
                                      % (self.outf, epoch + 1, i + 1),
                                      normalize=True, nrow=1, padding=0)

                if (i + 1) % self.checkpoint_step == 0:
                    torch.save(self.netG.state_dict(), '%s/netG_epoch_%03d_step_%03d.pth' % (self.outf, epoch + 1, i + 1))
                    torch.save(self.netD.state_dict(), '%s/netD_epoch_%03d_step_%03d.pth' % (self.outf, epoch + 1, i + 1))

            if (epoch + 1) > (self.niter - self.niter_decay):
                self.lr -= (self.lr / float(self.niter_decay))
                for param_group in self.optimizerG.param_groups:
                    param_group['lr'] = self.lr
                for param_group in self.optimizerD.param_groups:
                    param_group['lr'] = self.lr