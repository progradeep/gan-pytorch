import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.padding = padding
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel, stride, 0),
            nn.InstanceNorm2d(channel),
            nn.ReLU(True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel, stride, 0),
            nn.InstanceNorm2d(channel))

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv1(x)
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2(x)

        return input + x

class _netG(nn.Module):
    def __init__(self, ngpu, nc, ngf):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            # input is (nc) * 256 * 256
            nn.Conv2d(nc, ngf, 7, 1, 0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 256 x 256
            nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 128 x 128
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True))
            # state size. (ngf*4) x 64 x 64

        self.resnet_blocks = []
        for i in range(9):
            self.resnet_blocks.append(resnet_block(ngf * 4, 3, 1, 1))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
            # state size. (ngf*4) x 64 x 64

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 128 x 128
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True))
            # state size. (ngf) x 256 x 256
        self.decoder2 = nn.Sequential(
            nn.Conv2d(ngf, nc, 7, 1, 0),
            nn.Tanh())
            # state size. (nc) x 256 x 256

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            x = F.pad(x, (3, 3, 3, 3), 'reflect')
            x = nn.parallel.data_parallel(self.encoder, x, range(self.ngpu))
            x = nn.parallel.data_parallel(self.resnet_blocks, x, range(self.ngpu))
            x = nn.parallel.data_parallel(self.decoder, x, range(self.ngpu))
            x = F.pad(x, (3, 3, 3, 3), 'reflect')
            output = nn.parallel.data_parallel(self.decoder2, x, range(self.ngpu))
        else:
            x = F.pad(x, (3, 3, 3, 3), 'reflect')
            x = self.encoder(x)
            x = self.resnet_blocks(x)
            x = self.decoder(x)
            x = F.pad(x, (3, 3, 3, 3), 'reflect')
            output = self.decoder2(x)

        return output

class _netD(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),
            # state size. (ndf*8) x 32 x 32
            nn.Conv2d(ndf * 8, 1, 4, 1, 1))
            # state size. 1 x 32 x 32

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)

        return output