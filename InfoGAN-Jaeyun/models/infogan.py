import torch
import torch.nn as nn
import torch.nn.parallel

class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + 12, 1024, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. 
            nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 
            nn.ConvTranspose2d(64, nc, 4, 2, 1, bias=False),
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
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. 
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # state size.
            nn.Conv2d(ndf * 2, ndf * 16, 7, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. 
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
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

class _netQ(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netQ, self).__init__()
        self.ngpu = ngpu
        self.conv = nn.Conv2d(ndf * 16, 128, 1, bias=False)
        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, input):
        input = self.conv(input)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            digit = nn.parallel.data_parallel(self.conv_disc, input, range(self.ngpu))
            mu = nn.parallel.data_parallel(self.conv_mu, input, range(self.ngpu))
            var = nn.parallel.data_parallel(self.conv_var, input, range(self.ngpu))
        else:
            digit = self.conv_disc(input).squeeze()
            mu = self.conv_mu(input).squeeze()
            var = self.conv_var(input).squeeze().exp()

        return digit, mu, var


