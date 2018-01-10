import torch
import torch.nn as nn
import torch.nn.parallel

class _netG(nn.Module):
    def __init__(self, ngpu, nz, nl, ngf, nc):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.layer1 = nn.Sequential(
            # input is Z
            nn.Linear(nz + nl, ngf * 16),
            nn.ReLU(True),
        )
        self.layer2 = nn.Sequential(
            # state size. (ngf*16) x 1 x 1
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def main(self,input):
        x = self.layer1(input)
        x = x.view(x.size()[0], -1, 1, 1)
        return self.layer2(x)


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main(input), input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class _netD(nn.Module):
    def __init__(self, ngpu, nl, ndf, nc):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
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
            # state size. (ndf*8) x 4 x 4)
            nn.Conv2d(ndf * 8, ndf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True))
            # state size. (ndf*16) x 1 x 1)
        self.dis = nn.Sequential(
            nn.Linear(ndf * 16, 1),
            nn.Sigmoid())
        self.aux = nn.Sequential(
            nn.Linear(ndf * 16, nl),
            nn.Sigmoid())

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            output_dis = nn.paralle.data_parallel(self.dis, output, range(self.ngpu))
            output_aux = nn.paralle.data_parallel(self.aux, output, range(self.ngpu))
        else:
            output = self.main(input)
            output = output.squeeze()
            output_dis = self.dis(output)
            output_aux = self.aux(output)
        return output_dis.view(-1, 1).squeeze(), output_aux.view(-1, 10).squeeze()