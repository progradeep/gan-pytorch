import torch
import torch.nn as nn
import torch.nn.parallel

class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class _netD(nn.Module):
    def __init__(self, ngpu, nc, ndf, f_div):
        super(_netD, self).__init__()
        self.f_div = f_div
        self.ngpu = ngpu
        self.layer = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

            ## delete sigmoid layer in f-gan
        )

    def activation_func(self, x):
        if self.f_div == "KL" or "Pearson":
            return x
        elif self.f_div == "RKL":
            return -x.exp()
        elif self.f_div == "Neyman" or "Squared_Hellinger":
            return 1-x.exp()
        elif self.f_div == "JS":
            return (2*x.sigmoid()).log()
        else:       # GAN
            return x.sigmoid().log()

    def f_star(self, x):
        if self.f_div == "KL":
            return (x-1).exp()
        elif self.f_div == "RKL":
            return 1-(-x).exp()
        elif self.f_div == "Pearson":
            return 0.25*x*x+x
        elif self.f_div == "Neyman":
            return 2-2*(1-x).sqrt()
        elif self.f_div == "Squared_Hellinger":
            return x/(1-x)
        elif self.f_div == "JS":
            return -(2-x.exp()).log()
        else:        # GAN
            return -(1-x.exp()).log()

    def main(self, input):
        output = self.layer(input)
        return self.activation_func(output)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main(input), input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


