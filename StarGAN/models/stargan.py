import torch
import torch.nn as nn
import torch.nn.parallel


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class _netG(nn.Module):
    def __init__(self, ngpu, na, ngf):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            # input is (3+nl) * 128 * 128
            nn.Conv2d(3 + na, ngf, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 2, affine=True),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 4, affine=True),
            nn.ReLU(True))
            # state size. (ngf*4) x 32 x 32

        self.resnet_blocks = []
        for i in range(6):
            self.resnet_blocks.append(ResidualBlock(ngf * 4, ngf * 4))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
            # state size. (ngf*4) x 32 x 32

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 2, affine=True),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128

            nn.Conv2d(ngf, 3, 7, 1, 3, bias=False),
            nn.Tanh())
            # state size. 3 x 128 x 128

    def forward(self, x, c):
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            x = nn.parallel.data_parallel(self.encoder, x, range(self.ngpu))
            x = nn.parallel.data_parallel(self.resnet_blocks, x, range(self.ngpu))
            output = nn.parallel.data_parallel(self.decoder, x, range(self.ngpu))
        else:
            x = self.encoder(x)
            x = self.resnet_blocks(x)
            output = self.decoder(x)
        return output

class _netD(nn.Module):
    def __init__(self, ngpu, na, ndf):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is 3 x 128 x 128
            nn.Conv2d(3, ndf, 4, 2, 1),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ndf) x 64 x 64

            # 5 Hidden layers=================
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1),
            nn.LeakyReLU(0.01, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1),
            nn.LeakyReLU(0.01, inplace=True)
            # state size. (ndf*32) x 2 x 2
            # ================================
        )

        self.conv1 = nn.Conv2d(ndf * 32, 1, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf * 32, na, 2, bias=False)

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
            out_dis = nn.parallel.data_parallel(self.conv1, output, range(self.ngpu))
            out_aux = nn.parallel.data_parallel(self.conv2, output, range(self.ngpu))
        else:
            output = self.main(x)
            out_dis = self.conv1(output)
            out_aux = self.conv2(output)
        return out_dis.squeeze(), out_aux.squeeze()