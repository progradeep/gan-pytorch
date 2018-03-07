import os, time, glob
from itertools import chain

import numpy as np

import itertools, time, os
import torch
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.optim as optim

from models import c3d


class Trainer(object):
    def __init__(self, config, image_loader, video_loader):
        self.config = config
        self.video_loader = video_loader

        self.image_size = int(config.image_size)
        self.n_channels = int(config.n_channels)
        self.video_length = int(config.video_length)

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay

        self.video_batch_size = self.video_loader.batch_size

        self.log_interval = int(config.log_interval)
        self.checkpoint_step = int(config.checkpoint_step)
        self.train_batches = int(config.batches)

        self.use_cuda = config.cuda

        self.outf = config.outf

        self.video_discriminator = c3d.C3D()
        pretrained_dict = torch.load('c3d.pickle')
        pretrained_dict['fc.weight'] = torch.FloatTensor(8192, 101).normal_(0, 1)
        pretrained_dict['fc.bias'] = torch.FloatTensor(101).normal_(0, 1)
        #pretrained_dict['fc8.weight'] = torch.FloatTensor(4096, 101).normal_(0, 1)
        #pretrained_dict['fc8.bias'] = torch.FloatTensor(101).normal_(0, 1)
        #self.video_discriminator.load_state_dict(pretrained_dict)

    def train(self):
        self.category_criterion = nn.CrossEntropyLoss()
        if self.use_cuda:
            self.video_discriminator.cuda()
        # create optimizers
        opt_video_discriminator = optim.Adam(self.video_discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        loader = iter(self.video_loader)
        valid_x = loader.next()
        valid_x_categ = valid_x["categories"]
        valid_x = valid_x["images"]

        valid_x_img = valid_x.permute(0,2,1,3,4)
        valid_x_img = self._get_variable(valid_x_img).resize(self.video_batch_size * self.video_length, self.n_channels, self.image_size, self.image_size)
        vutils.save_image(valid_x_img.data, '{}/valid_gif.png'.format(self.outf), nrow=self.video_length, normalize=True)

        valid_x = Variable(valid_x.cuda(), requires_grad=False)

        start_time = time.time()
        for epoch in range(self.train_batches):

            for step in range(len(self.video_loader)):
                try:
                    realGif = loader.next()
                    realGifCateg = realGif["categories"]
                    realGif = realGif["images"]

                except StopIteration:
                    loader = iter(self.video_loader)
                    realGif = loader.next()
                    realGifCateg = realGif["categories"]
                    realGif = realGif["images"]
                realGif = Variable(realGif.cuda(), requires_grad=False)

                image_batch_size = realGif.size(0)

                #### train D_V ####
                self.video_discriminator.zero_grad()

                real_categ = self.video_discriminator(realGif)
                categories_gt = Variable(torch.squeeze(realGifCateg.long()), requires_grad=False).cuda()
                loss = self.category_criterion(real_categ, categories_gt)

                loss.backward()
                opt_video_discriminator.step()

                step_end_time = time.time()

                print('[%d/%d][%d/%d] - time: %.2f, loss_D_V: %.3f'
                      % (epoch, self.train_batches, step, len(self.video_loader), step_end_time - start_time,
                         loss))
                valid_cate = self.video_discriminator(valid_x)
                print((torch.max(valid_cate, 1)[1].data == valid_x_categ.cuda()).sum()/image_batch_size)
                if step% self.checkpoint_step == 0 and step != 0:
                   torch.save(self.video_discriminator.state_dict(), '%s/netD_V_epoch-%d_step-%s.pth' % (self.outf, epoch, step))

                   print("Saved checkpoint")

    def _get_variable(self, inputs):
        out = Variable(inputs.cuda())
        return out
