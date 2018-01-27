import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

class _netG(nn.Module):
    def __init__(self, ngpu, ngf, input_nc, output_nc, nb=9):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.nb = nb

        self.conv = nn.Sequential(
            # input. (nc) * 256 * 256
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 256 x 256

            nn.Conv2d(ngf, ngf * 2, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf *2 ),
            nn.ReLU(True),
            # state size. (ngf*2) x 128 x 128

            nn.Conv2d(ngf*2, ngf*4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(True),
            # state size. (ngf*4) x 64 x 64
        )

        self.resnet = []
        for i in range(nb):
            self.resnet.append(resnet_block(ngf*4, 3, 1, 1))
        self.resnet = nn.Sequential(*self.resnet)
        # state size. (ngf*4) x 64 x 64

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 128 x 128

            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 256 x 256

            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7, 1, 0),
            nn.Tanh()
            # state size. (nc) x 256 x 256)
        )

    def main(self, input):
        x = self.conv(input)
        x = self.resnet(x)
        x = self.deconv(x)

        return x

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self, ngpu, ndf, input_nc):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input. (nc) x 256 x 256
            nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16

            nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 15 x 15

            nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False),
            # state size. 1 x 14 x 14
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output



# resnet block with reflect padding
class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.padding = padding
        self.main = nn.Sequential(
            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(channel, channel, kernel, stride, 0),
            nn.InstanceNorm2d(channel),
            nn.ReLU(True),

            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(channel, channel, kernel, stride, 0),
            nn.InstanceNorm2d(channel))

    def forward(self, input):
        x = self.main(input)

        return input + x