import torch
import torch.nn as nn
import torch.nn.parallel

class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        self.nz = nz

        self.fc = nn.Linear(nz, ngf * 8)
        self.tconv = nn.Sequential(
            # state size. (ngf*16)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (ngf) x 32 x 32
            )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = input.view(-1, self.nz)
            output = nn.parallel.data_parallel(self.fc, input, range(self.ngpu))
            output = output.view(-1, self.ngf * 8, 1, 1)
            output = nn.parallel.data_parallel(self.tconv, output, range(self.ngpu))
        else:
            input = input.view(-1, self.nz)
            output = self.fc(input)
            output = output.view(-1, self.ngf * 8, 1, 1)
            output = self.tconv(output)
        return output


class _netD(nn.Module):
    def __init__(self, ngpu, nl, ndf, nc):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf

        self.conv = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, ndf * 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
            # state size. (ndf*32) x 4 x 4
            )
        self.dis = nn.Sequential(
            nn.Linear(ndf * 32 * 4 * 4, 1),
            nn.Sigmoid()
            )
        self.aux = nn.Sequential(
            nn.Linear(ndf * 32 * 4 * 4, nl),
            nn.Softmax()
            )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.conv, input, range(self.ngpu))
            output = output.view(-1, self.ndf * 32 * 4 * 4)
            dis_output = nn.parallel.data_parallel(self.dis, output, range(self.ngpu)).view(-1, 1).squeeze(1)
            aux_output = nn.parallel.data_parallel(self.aux, output, range(self.ngpu))
        else:
            output = self.conv(input)
            output = output.view(-1, self.ndf * 32 * 4 * 4)
            dis_output = self.dis(output).view(-1, 1).squeeze(1)
            aux_output = self.aux(output)
        return dis_output, aux_output