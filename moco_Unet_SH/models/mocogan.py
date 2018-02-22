"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

import numpy as np

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * Variable(T.FloatTensor(x.size()).normal_(), requires_grad=False)
        return x


class ImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, use_noise=False, noise_sigma=None):
        super(ImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()
        return h, None


class PatchImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, use_noise=False, noise_sigma=None):
        super(PatchImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()
        return h, None


class PatchVideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(PatchVideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 4, 1, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()

        return h, None


class VideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(VideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, (3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, (3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, (3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 4, ndf * 8, (3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 8, n_output_neurons, (2, 4, 4), 1, 0, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()

        return h, None


class CategoricalVideoDiscriminator(VideoDiscriminator):
    def __init__(self, n_channels, dim_categorical, n_output_neurons=1, use_noise=False, noise_sigma=None):
        super(CategoricalVideoDiscriminator, self).__init__(n_channels=n_channels,
                                                            n_output_neurons=n_output_neurons + dim_categorical,
                                                            use_noise=use_noise,
                                                            noise_sigma=noise_sigma)

        self.dim_categorical = dim_categorical

    def split(self, input):
        return input[:, :input.size(1) - self.dim_categorical], input[:, input.size(1) - self.dim_categorical:]

    def forward(self, input):
        h, _ = super(CategoricalVideoDiscriminator, self).forward(input)
        labels, categ = self.split(h)
        return labels, categ


class SequenceDiscriminator(nn.Module):
    def __init__(self, n_channels, video_len, dim_z_motion, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(SequenceDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma
        self.video_len = video_len
        self.dim_z_motion = dim_z_motion

        self.encoder = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, self.dim_z_motion, 4, 1, 0, bias=False),
        )

        self.rnn = nn.GRUCell(self.dim_z_motion, self.dim_z_motion)
        self.discriminator = nn.Sequential(
            nn.Linear(self.dim_z_motion, 1),
            nn.Sigmoid()
        )

    def z_dynamics(self, gif):
        video_len = gif.size(0)
        batch_size = gif.size(1)
        # input: len, bs, nc, 64, 64

        h_0 = Variable(T.FloatTensor(batch_size, self.dim_z_motion).normal_())
        h_t = []

        for i in range(video_len):
            e_t = self.encoder(gif[i]).squeeze()
            # e_t: bs, dim_z_motion
            ht = self.rnn(h_0, e_t)
            h_t.append(ht)

        # h_t: len, bs, dim z motion

        z_d_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_d = torch.cat(z_d_t, dim=1)
        # z_d: bs, len, dim z motion
        z_d = z_d.contiguous().view(-1, self.dim_z_motion)
        # z_d: bs * len, dim z motion

        return z_d

    def forward(self, gif):
        # input: bs, 3, len, 64, 64
        gif = gif.permute(0, 2, 1, 3, 4)
        # gif: bs, len, 3, 64, 64

        f1 = gif[:, 0, :, :, :]
        # f1: bs, 3, 64, 64
        im_gif = f1.contiguous().view(gif.size(0), 1, self.n_channels, gif.size(3), gif.size(4))
        im_gif = torch.cat([im_gif] * gif.size(1), dim=1)  # bs, len, nc, 64, 64
        im_gif = im_gif.permute(1, 0, 2, 3, 4)

        video = gif.permute(1, 0, 2, 3, 4)

        dynamics = self.z_dynamics(video)
        spacial = self.z_dynamics(im_gif)
        # size: bs, dim_z_m

        z_d = dynamics - spacial
        # z_d: bs, len, dim z motion

        output = self.discriminator(z_d)

        return output, None



class VideoGenerator(nn.Module):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ngf=64):
        super(VideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_category = dim_z_category
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length

        self.ngf = ngf

        dim_z = dim_z_motion + dim_z_category + dim_z_content

        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)

        self.main1 = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        self.main2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.main3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.main4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.main5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.encoder_motion = nn.Sequential(
            nn.Conv2d(self.n_channels, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, dim_z_motion, 4, 1, 0, bias=False)
        )

        self.encoder_content1 = nn.Sequential(
            nn.Conv2d(self.n_channels, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_content2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_content3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_content4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8), 
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_content5 = nn.Sequential(
            nn.Conv2d(ngf * 8, dim_z_content, 4, 1, 0, bias=False)
        )

    def sample_z_m(self, image_batches, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        #h_t = [self.get_gru_initial_state(num_samples)]
        
        motion = self.encoder_motion(image_batches).view(num_samples, self.dim_z_motion)
        h_t = [motion]

        for frame_num in range(video_len):
            e_t = self.get_iteration_noise(num_samples)
            h_t.append(self.recurrent(e_t, h_t[-1]))

        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)

        return z_m

    def sample_z_categ(self, num_samples, video_len, categories):
        video_len = video_len if video_len is not None else self.video_length

        #classes_to_generate = np.random.randint(self.dim_z_category, size=num_samples)
        classes_to_generate = categories
        one_hot = np.zeros((num_samples, self.dim_z_category), dtype=np.float32)
        one_hot[np.arange(num_samples), classes_to_generate] = 1
        one_hot_video = np.repeat(one_hot, video_len, axis=0)

        one_hot_video = torch.from_numpy(one_hot_video)

        if torch.cuda.is_available():
            one_hot_video = one_hot_video.cuda()

        return Variable(one_hot_video), classes_to_generate

    def sample_z_content(self, image_batches, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        #content = np.random.normal(0, 1, (num_samples, self.dim_z_content)).astype(np.float32)
        content1 = self.encoder_content1(image_batches)
        content2 = self.encoder_content2(content1)
        content3 = self.encoder_content3(content2)
        content4 = self.encoder_content4(content3)
        content = self.encoder_content5(content4)
        
        #content = content.data.view(num_samples, self.dim_z_content)
        #content = torch.cat([content] * 10)
        content = content.data.view(num_samples, 1, self.dim_z_content)
        content = torch.cat([content] * video_len, dim=1)
        content = content.view(num_samples * video_len, content.size(2))

        content4 = content4.data.view(num_samples, 1, self.ngf*8, content4.size(2), content4.size(3))
        content4 = torch.cat([content4] * video_len, dim=1)
        content4 = content4.view(num_samples * video_len, content4.size(2), content4.size(3), content4.size(4))

        content3 = content3.data.view(num_samples, 1, self.ngf*4, content3.size(2), content3.size(3))
        content3 = torch.cat([content3] * video_len, dim=1)
        content3 = content3.view(num_samples * video_len, content3.size(2), content3.size(3), content3.size(4))

        content2 = content2.data.view(num_samples, 1, self.ngf*2, content2.size(2), content2.size(3))
        content2 = torch.cat([content2] * video_len, dim=1)
        content2 = content2.view(num_samples * video_len, content2.size(2), content2.size(3), content2.size(4))

        content1 = content1.data.view(num_samples, 1, self.ngf, content1.size(2), content1.size(3))
        content1 = torch.cat([content1] * video_len, dim=1)
        content1 = content1.view(num_samples * video_len, content1.size(2), content1.size(3), content1.size(4))

        #content = torch.from_numpy(content)
        #if torch.cuda.is_available():
        #    content = content.cuda()
        return Variable(content), Variable(content1), Variable(content2), Variable(content3), Variable(content4)

    def sample_z_video(self, image_batches, categories, num_samples, video_len=None):
        z_content, z_content1, z_content2, z_content3, z_content4 = self.sample_z_content(image_batches, num_samples, video_len)
        
        z_category, z_category_labels = self.sample_z_categ(num_samples, video_len, categories)
        
        z_motion = self.sample_z_m(image_batches, num_samples, video_len)
        
        z = torch.cat([z_content, z_category, z_motion], dim=1)

        return z, z_category_labels, z_content1, z_content2, z_content3, z_content4

    def sample_videos(self, image_batches, categories, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        z, z_category_labels, z_content1, z_content2, z_content3, z_content4 = self.sample_z_video(image_batches, categories, num_samples, video_len)

        h = self.main1(z.view(z.size(0), z.size(1), 1, 1))
        h = self.main2(torch.cat([h, z_content4], dim=1))
        h = self.main3(torch.cat([h, z_content3], dim=1))
        h = self.main4(torch.cat([h, z_content2], dim=1))
        h = self.main5(torch.cat([h, z_content1], dim=1))
        h = h.view(int(h.size(0) / video_len), video_len, self.n_channels, h.size(3), h.size(3))
        h = h.permute(0, 2, 1, 3, 4)
        #z_category_labels = torch.from_numpy(z_category_labels).type("torch.LongTensor")
        z_category_labels = z_category_labels.type("torch.LongTensor")

        if torch.cuda.is_available():
            z_category_labels = z_category_labels.cuda()

        return h, Variable(z_category_labels, requires_grad=False)

    def sample_images(self, image_batches, categories, num_samples):
        #z, z_category_labels = self.sample_z_video(image_batches, num_samples * self.video_length * 2)
        z, z_category_labels, z_content1, z_content2, z_content3, z_content4 = self.sample_z_video(image_batches, categories, num_samples)

        j = np.sort(np.random.choice(z.size(0), num_samples, replace=False)).astype(np.int64)
        z = z[j, ::]
        z = z.view(z.size(0), z.size(1), 1, 1)
        z_content4 = z_content4[j, ::]
        z_content3 = z_content3[j, ::]
        z_content2 = z_content2[j, ::]
        z_content1 = z_content1[j, ::]
        h = self.main1(z)
        h = self.main2(torch.cat([h, z_content4], dim=1))
        h = self.main3(torch.cat([h, z_content3], dim=1))
        h = self.main4(torch.cat([h, z_content2], dim=1))
        h = self.main5(torch.cat([h, z_content1], dim=1))

        return h, None

    def get_gru_initial_state(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())

    def get_iteration_noise(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())

class ImageReconstructor(nn.Module):
    def __init__(self, n_channels, dim_z, ngf=64):
        super(ImageReconstructor, self).__init__()

        self.n_channels = n_channels
        self.dim_z = dim_z

        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_channels, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, self.dim_z, 4, 1, 0, bias=False)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, input):
        h = self.encoder(input)
        return self.decoder(h)

class VideoReconstructor(nn.Module):
    def __init__(self, n_channels, video_len, dim_z, ngf=64):
        super(VideoReconstructor, self).__init__()

        self.n_channels = n_channels
        self.video_len = video_len
        self.dim_z = dim_z

        self.encoder = nn.Sequential(
            nn.Conv3d(self.n_channels, ngf, (3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ngf, ngf * 2, (3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ngf * 2, ngf * 4, (3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ngf * 4, ngf * 8, (3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ngf * 8, self.dim_z, (2, 4, 4), 1, 0, bias=False),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, input):
        h = self.encoder(input)
        h = h.view(-1, self.dim_z, 1, 1)
        return self.decoder(h)


