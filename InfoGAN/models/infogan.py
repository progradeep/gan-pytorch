import torch
import torch.nn as nn
import torch.nn.parallel


class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution (MNIST)
            nn.ConvTranspose2d(nz + 12, ngf * 16, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf x 16) x 1 x 1
            nn.ConvTranspose2d(ngf * 16, ngf * 2, 7, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf x 2) * 7 * 7
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 14 x 14
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 28 x 28
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class _netShareDQ(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netShareDQ, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf x 2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 16, 7, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf x 16) x 1 x 1
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

class _netD(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # state size. (ndf x 16) x 1 x 1
            nn.Conv2d(ndf * 16, 1, 1),
            nn.Sigmoid()

        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class _netQ(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netQ, self).__init__()
        self.ngpu = ngpu

        self.conv = nn.Conv2d(ndf * 16, ndf * 2, 1, bias=False)
        self.conv_disc = nn.Conv2d(ndf * 2, 10, 1)
        self.conv_mu = nn.Conv2d(ndf * 2, 2, 1)
        self.conv_var = nn.Conv2d(ndf * 2, 2, 1)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            y = nn.parallel.data_parallel(self.conv, input, range(self.ngpu))
            disc_logits = nn.parallel.data_parallel(self.conv_disc, y, range(self.ngpu))
            mu = nn.parallel.data_parallel(self.conv_mu, y, range(self.ngpu))
            var = nn.parallel.data_parallel(self.conv_var, y, range(self.ngpu))
        else:
            y = self.conv(input)
            disc_logits = self.conv_disc(y).squeeze()
            mu = self.conv_mu(y).squeeze()
            var = self.conv_var(y).squeeze().exp()

        return disc_logits, mu, var


