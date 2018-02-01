import os
from itertools import chain

import numpy as np

import torch
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.optim as optim

from models import mocogan

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

class Trainer(object):
    def __init__(self, config, image_loader, video_loader):
        self.config = config
        self.image_loader = image_loader
        self.video_loader = video_loader
        self.image_sampler = image_loader
        self.video_sampler = video_loader

        self.n_channels = int(config.n_channels)
        self.dim_z_content = int(config.dim_z_content)
        self.dim_z_category = int(config.dim_z_category)
        self.dim_z_motion = int(config.dim_z_motion)
        self.video_length = int(config.video_length)


        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay

        self.image_discriminator = config.image_discriminator
        self.video_discriminator = config.video_discriminator

        self.use_noise = config.use_noise
        self.noise_sigma = float(config.noise_sigma)

        self.use_infogan = config.use_infogan
        self.use_categories = config.use_categories

        self.video_batch_size = self.video_sampler.batch_size
        self.image_batch_size = self.image_sampler.batch_size

        self.log_interval = int(config.print_every)
        self.train_batches = int(config.batches)

        self.use_cuda = config.cuda

        self.image_enumerator = None
        self.video_enumerator = None


        self.log_folder = config.outf

        self.build_model()

        if self.use_cuda:
            self.generator.cuda()
            self.image_reconstructor.cuda()
            self.video_reconstructor.cuda()
            self.image_discriminator.cuda()
            self.video_discriminator.cuda()

    def build_model(self):
        self.generator = mocogan.VideoGenerator(self.n_channels, self.dim_z_content, self.dim_z_category, self.dim_z_motion, self.video_length)

        self.image_reconstructor = mocogan.ImageReconstructor(self.n_channels, self.dim_z_content + self.dim_z_category + self.dim_z_motion)
        self.video_reconstructor = mocogan.VideoReconstructor(self.n_channels, self.video_length, self.dim_z_content + self.dim_z_category, self.dim_z_motion)

        self.image_discriminator = self.build_discriminator(self.image_discriminator, n_channels=self.n_channels,
                                                        use_noise=self.use_noise, noise_sigma=self.noise_sigma)
        
        self.video_discriminator = self.build_discriminator(self.video_discriminator, dim_categorical=self.dim_z_category,
                                                        n_channels=self.n_channels, use_noise=self.use_noise,
                                                        noise_sigma=self.noise_sigma)

    def build_discriminator(self, type, **kwargs):
        discriminator_type = getattr(mocogan, type)

        if 'Categorical' not in type and 'dim_categorical' in kwargs:
            kwargs.pop('dim_categorical')

        return discriminator_type(**kwargs)


    @staticmethod
    def ones_like(tensor, val=1.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)

    @staticmethod
    def zeros_like(tensor, val=0.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)

    def compute_gan_loss(self, discriminator, sample_true, sample_fake, is_video):
        real_batch = sample_true()
        
        batch_size = real_batch['images'].size(0)
        fake_batch, generated_categories = sample_fake(batch_size)

        real_labels, real_categorical = discriminator(Variable(real_batch['images']))
        fake_labels, fake_categorical = discriminator(fake_batch)

        fake_gt, real_gt = self.get_gt_for_discriminator(batch_size, real=0.)

        l_discriminator = self.gan_criterion(real_labels, real_gt) + \
                          self.gan_criterion(fake_labels, fake_gt)

        #  sample again and compute for generator

        fake_gt = self.get_gt_for_generator(batch_size)
        # to real_gt
        l_generator = self.gan_criterion(fake_labels, fake_gt)

        if is_video:

            # Ask the video discriminator to learn categories from training videos
            categories_gt = Variable(torch.squeeze(real_batch['categories'].long()))
            l_discriminator += self.category_criterion(real_categorical, categories_gt)

            if self.use_infogan:
                # Ask the generator to generate categories recognizable by the discriminator
                l_generator += self.category_criterion(fake_categorical, generated_categories)

        return l_generator, l_discriminator

    def sample_real_image_batch(self):
        if self.image_enumerator is None:
            self.image_enumerator = enumerate(self.image_sampler)

        batch_idx, batch = next(self.image_enumerator)
        b = batch
        if self.use_cuda:
            for k, v in batch.iteritems():
                b[k] = v.cuda()

        if batch_idx == len(self.image_sampler) - 1:
            self.image_enumerator = enumerate(self.image_sampler)

        return b

    def sample_real_video_batch(self):
        if self.video_enumerator is None:
            self.video_enumerator = enumerate(self.video_sampler)

        batch_idx, batch = next(self.video_enumerator)
        b = batch
        if self.use_cuda:
            for k, v in batch.iteritems():
                b[k] = v.cuda()

        if batch_idx == len(self.video_sampler) - 1:
            self.video_enumerator = enumerate(self.video_sampler)

        return b

    def train_discriminator(self, discriminator, sample_input, sample_true, sample_fake, opt, batch_size, use_categories):
        opt.zero_grad()

        real_batch = sample_true()
        batch = Variable(real_batch['images'], requires_grad=False)

        input_batch = sample_input()
        input_batch = Variable(input_batch['images'], requires_grad=False)
        
        fake_batch, generated_categories = sample_fake(input_batch, batch_size)

        real_labels, real_categorical = discriminator(batch)

        fake_labels, fake_categorical = discriminator(fake_batch.detach())

        ones = self.ones_like(real_labels)
        zeros = self.zeros_like(fake_labels)

        l_discriminator = self.gan_criterion(real_labels, ones) + \
                          self.gan_criterion(fake_labels, zeros)

        if use_categories:
            # Ask the video discriminator to learn categories from training videos
            categories_gt = Variable(torch.squeeze(real_batch['categories'].long()), requires_grad=False)
            l_discriminator += self.category_criterion(real_categorical.squeeze(), categories_gt)

        l_discriminator.backward()
        opt.step()

        return l_discriminator

    def train_generator(self,
                        image_discriminator, video_discriminator, image_reconstructor, video_reconstructor,
                        sample_true, sample_fake_images, sample_fake_videos,
                        opt):

        real_batch = sample_true()

        batch = Variable(real_batch['images'], requires_grad=False)

        opt.zero_grad()

        # train on images

        fake_batch, generated_categories = sample_fake_images(batch, self.image_batch_size)
        fake_labels, fake_categorical = image_discriminator(fake_batch)
        all_ones = self.ones_like(fake_labels)

        l_generator = self.gan_criterion(fake_labels, all_ones)

        recon_batch = image_reconstructor(fake_batch)
        l_generator += torch.mean(torch.abs(recon_batch - batch))

        # train on videos

        fake_batch, generated_categories = sample_fake_videos(batch, self.video_batch_size)
        fake_labels, fake_categorical = video_discriminator(fake_batch)
        all_ones = self.ones_like(fake_labels)

        l_generator += self.gan_criterion(fake_labels, all_ones)

        recon_batch = video_reconstructor(fake_batch)
        l_generator += torch.mean(torch.abs(recon_batch - batch))

        if self.use_infogan:
            # Ask the generator to generate categories recognizable by the discriminator
            l_generator += self.category_criterion(fake_categorical.squeeze(), generated_categories)

        l_generator.backward()
        opt.step()

        return l_generator

    def train(self):
        self.gan_criterion = nn.BCEWithLogitsLoss()
        self.category_criterion = nn.CrossEntropyLoss()

        if self.use_cuda:
            self.generator.cuda()
            self.image_discriminator.cuda()
            self.video_discriminator.cuda()

        # create optimizers
        opt_generator = optim.Adam(chain(self.generator.parameters(), self.image_reconstructor.parameters(), self.video_reconstructor.parameters()), lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        opt_image_discriminator = optim.Adam(self.image_discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2),
                                             weight_decay=self.weight_decay)
        opt_video_discriminator = optim.Adam(self.video_discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2),
                                             weight_decay=self.weight_decay)

        # training loop

        def sample_fake_image_batch(batch, batch_size):
            return self.generator.sample_images(batch, batch_size)

        def sample_fake_video_batch(batch, batch_size):
            return self.generator.sample_videos(batch, batch_size)

        batch_num = 0

        while True:
            self.generator.train()
            self.image_discriminator.train()
            self.video_discriminator.train()

            opt_generator.zero_grad()
            opt_image_discriminator.zero_grad()
            opt_video_discriminator.zero_grad()

            # train image discriminator
            l_image_dis = self.train_discriminator(self.image_discriminator, self.sample_real_image_batch, self.sample_real_image_batch,
                                                   sample_fake_image_batch, opt_image_discriminator,
                                                   self.image_batch_size, use_categories=False)
            # train video discriminator
            l_video_dis = self.train_discriminator(self.video_discriminator, self.sample_real_image_batch, self.sample_real_video_batch,
                                                   sample_fake_video_batch, opt_video_discriminator,
                                                   self.video_batch_size, use_categories=self.use_categories)
            # train generator
            l_gen = self.train_generator(self.image_discriminator, self.video_discriminator,
                                         self.image_reconstructor, self.video_reconstructor,
                                         self.sample_real_image_batch, sample_fake_image_batch, sample_fake_video_batch,
                                         opt_generator)

            batch_num += 1
            print('step: ' + str(batch_num))
            if batch_num % self.log_interval == 0:

                self.generator.eval()

                input_batch = self.sample_real_image_batch()
                input_batch = Variable(input_batch['images'], requires_grad=False)

                images, _ = sample_fake_image_batch(input_batch, self.image_batch_size)
                videos, _ = sample_fake_video_batch(input_batch, self.video_batch_size)

                vutils.save_image(denorm(images.data),
                        '%s/image_samples_epoch_%05d.png' % (self.log_folder, batch_num), nrow=10)
                videos.data.resize_(self.video_batch_size * 10, 3, 64, 64)
                vutils.save_image(denorm(videos.data),
                        '%s/video_samples_epoch_%05d.png' % (self.log_folder, batch_num), nrow=10)

                torch.save(self.generator, os.path.join(self.log_folder, 'generator_%05d.pytorch' % batch_num))

            if batch_num >= self.train_batches:
                torch.save(self.generator, os.path.join(self.log_folder, 'generator_%05d.pytorch' % batch_num))
                break
