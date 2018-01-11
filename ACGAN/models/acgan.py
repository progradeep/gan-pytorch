import torch
import torch.nn as nn
import torch.nn.parallel


class _netG(nn.Module):
    def __init__(self, ngpu, nz, nl, ngf, nc):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        self.fc = nn.Sequential(
            # input is Z
            nn.Linear(nz + nl, ngf * 8 * 4 * 4),
            nn.ReLU(True))
        self.deconv = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh())
            # state size. (nc) x 64 x 64


    def forward(self, input, label):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = torch.cat((input, label), 1)
            output = nn.parallel.data_parallel(self.fc(output), output, range(self.ngpu))
            output = output.view(-1, self.ngf * 8, 4, 4)
            output = nn.parallel.data_parallel(self.deconv(output), output, range(self.ngpu))
        else:
            output = torch.cat((input, label), 1)
            output = self.fc(output)
            output = output.view(-1, self.ngf * 8, 4, 4)
            output = self.deconv(output)
        return output


class _netD(nn.Module):
    def __init__(self, ngpu, nl, ndf, nc):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.nl = nl
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
            nn.LeakyReLU(0.2, inplace=True))
            # state size. (ndf*8) x 4 x 4)

        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, ndf * 16),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf*16) x 1 x 1)
        self.dis = nn.Sequential(
            nn.Linear(ndf * 16, 1),
            nn.Sigmoid())
        self.aux = nn.Sequential(
            nn.Linear(ndf * 16, nl),
            nn.Softmax()
            )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            output = output.view(-1, self.ndf * 16)
            output_dis = nn.paralle.data_parallel(self.dis, output, range(self.ngpu))
            output_aux = nn.paralle.data_parallel(self.aux, output, range(self.ngpu))
        else:
            output = self.main(input)
            output = output.view(-1, self.ndf * 8 * 4 * 4)
            output = self.fc(output)
            output_dis = self.dis(output)
            output_aux = self.aux(output)
        return output_dis.squeeze(), output_aux.squeeze()