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


class VideoGenerator(nn.Module):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ngf=64):
        super(VideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_category = dim_z_category
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length

        dim_z = dim_z_motion + dim_z_category + dim_z_content

        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.encoder_motion = nn.Sequential(
            # 3, 64, 64
            nn.Conv2d(self.n_channels, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # ngf, 32, 32
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # ngf * 2, 16, 16
            nn.Conv2d(ngf * 2, dim_z_motion, 4, 2, 1, bias=False)

            # nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            # # ngf * 4, 8, 8
            # nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # ngf * 8, 4, 4
            # nn.Conv2d(ngf * 8, dim_z_motion, 4, 1, 0, bias=False)
            # # dim z motion, 1, 1
        )

        self.encoder_content = nn.Sequential(
            nn.Conv2d(self.n_channels, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 32
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16
            nn.Conv2d(ngf * 2, dim_z_content, 4, 2, 1, bias=False)
            # 8
        )

    def sample_z_m(self, image_batches, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        #h_t = [self.get_gru_initial_state(num_samples)]
        
        # motion = self.encoder_motion(image_batches).view(num_samples, self.dim_z_motion)
        motion = self.encoder_motion(image_batches)

        # motion: bs, dim z motion, 8, 8
        motion = motion.permute(0, 2, 3, 1)
        # motion: bs, 8, 8, dim z motion

        motion = motion.contiguous().view(-1, self.dim_z_motion)

        msize = motion.size(0)

        h_t = [motion]

        for frame_num in range(video_len):
            e_t = self.get_iteration_noise(msize)
            h_t.append(self.recurrent(e_t, h_t[-1]))

        # h_t: 10, 640, 10
        # h_k: 640, 10
        # -> 10, 8, 8, 10 -> 10, 10, 8, 8


        z_m_t = [h_k.view(num_samples, 8, 8, self.dim_z_motion).permute(0, 3, 1, 2) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion, 8, 8)


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
        content = self.encoder_content(image_batches)
        
        #content = content.data.view(num_samples, self.dim_z_content)
        #content = torch.cat([content] * 10)
        content = content.data.view(num_samples, 1, self.dim_z_content, content.size(2), content.size(3))
        content = torch.cat([content] * video_len, dim=1)
        content = content.view(num_samples * video_len, self.dim_z_content, content.size(3), content.size(4))
        
        #content = torch.from_numpy(content)
        #if torch.cuda.is_available():
        #    content = content.cuda()
        return Variable(content)

    def sample_z_video(self, image_batches, categories, num_samples, video_len=None):
        z_content = self.sample_z_content(image_batches, num_samples, video_len)

        z_category, z_category_labels = self.sample_z_categ(num_samples, video_len, categories)
        
        z_motion = self.sample_z_m(image_batches, num_samples, video_len)
       
        z_category = z_category.unsqueeze(2).unsqueeze(3)
        z_category = z_category.expand(z_category.size(0), z_category.size(1), z_content.size(2), z_content.size(3))
        # z_motion = z_motion.unsqueeze(2).unsqueeze(3)
        # z_motion = z_motion.expand(z_motion.size(0), z_motion.size(1), z_content.size(2), z_content.size(3))

        z = torch.cat([z_content, z_category, z_motion], dim=1)

        return z, z_category_labels

    def sample_videos(self, image_batches, categories, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        z, z_category_labels = self.sample_z_video(image_batches, categories, num_samples, video_len)

        h = self.main(z)
        h = h.view(int(h.size(0) / video_len), video_len, self.n_channels, h.size(3), h.size(3))
        h = h.permute(0, 2, 1, 3, 4)
        #z_category_labels = torch.from_numpy(z_category_labels).type("torch.LongTensor")
        z_category_labels = z_category_labels.type("torch.LongTensor")

        if torch.cuda.is_available():
            z_category_labels = z_category_labels.cuda()

        return h, Variable(z_category_labels, requires_grad=False)

    def sample_images(self, image_batches, categories, num_samples):
        #z, z_category_labels = self.sample_z_video(image_batches, num_samples * self.video_length * 2)
        z, z_category_labels = self.sample_z_video(image_batches, categories, num_samples)

        j = np.sort(np.random.choice(z.size(0), num_samples, replace=False)).astype(np.int64)
        z = z[j, ::]
        h = self.main(z)

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


