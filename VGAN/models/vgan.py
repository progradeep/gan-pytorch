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


class VideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1):
        super(VideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons

        self.main = nn.Sequential(
            # Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, 128, (3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(128, 256, (3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(256, 512, (3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(512, 1024, (3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(1024, 2, (2, 4, 4), 1, 0, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()

        return h


class VideoGenerator(nn.Module):
    def __init__(self, n_channels, video_length):
        super(VideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.video_length = video_length

        self.background = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, 4, 1, 0, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=True),
            nn.Tanh()
            )

        self.video = nn.Sequential(
            nn.ConvTranspose3d(1024, 1024, kernel_size=(2,4,4), stride=(1,2,2), padding=0),
            nn.BatchNorm3d(1024),
            nn.ReLU(True),

            nn.ConvTranspose3d(1024, 512, 4, stride=(1,2,2), padding=(0,1,1), bias=True),
            nn.BatchNorm3d(512),
            nn.ReLU(True),

            nn.ConvTranspose3d(512, 256, 4, stride=(1,2,2), padding=(0,1,1), bias=True),
            nn.BatchNorm3d(256),
            nn.ReLU(True),

            nn.ConvTranspose3d(256, 128, kernel_size=(2, 4, 4), stride=(1,2,2), padding=(0,1,1), bias=True),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
        )

        self.gen_net = nn.Sequential(nn.ConvTranspose3d(128, 3, kernel_size=(2, 4, 4), stride=(1,2,2), padding=(0,1,1)),
                                     nn.Tanh())
        self.mask_net = nn.Sequential(nn.ConvTranspose3d(128, 1, kernel_size=(2,4,4), stride=(1,2,2), padding=(0,1,1)),
                                      nn.Sigmoid())

        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_channels, 128, 4, 2, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 4, 1, 0, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )


    def forward(self, input):
        # print("generator input:", input.shape)
        x = self.encoder(input) # bs, 1024, 1, 1
        x_v = x.unsqueeze(2) # bs, 1024, 1, 1, 1
        # print(x_v.shape)
        video = self.video(x_v)

        foregound = self.gen_net(video)
        mask = self.mask_net(video)
        # print("fg",foregound.shape, "mask",mask.shape)

        background = self.background(x)
        # print("bg", background.shape)
        background_frames = background.unsqueeze(2).repeat(1,1,self.video_length,1,1)
        # print("fg",foregound.shape,"mask",mask.shape,"bg",background.shape)

        output = mask * foregound + (1-mask) * background_frames
        # print("output",output.shape)
        return output
