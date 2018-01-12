import torch
import torch.nn as nn
import torch.nn.parallel


class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + 100, ngf * 8, 4, 1, 0, bias=False),
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

class _netShareDQ(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netShareDQ, self).__init__()
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
            # state size. (ndf*8) x 4 x 4
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
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
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

        self.conv_1 = nn.Conv2d(ndf * 8, 10, 4, 1, 0, bias=False)
        self.conv_2 = nn.Conv2d(ndf * 8, 10, 4, 1, 0, bias=False)
        self.conv_3 = nn.Conv2d(ndf * 8, 10, 4, 1, 0, bias=False)
        self.conv_4 = nn.Conv2d(ndf * 8, 10, 4, 1, 0, bias=False)
        self.conv_5 = nn.Conv2d(ndf * 8, 10, 4, 1, 0, bias=False)
        self.conv_6 = nn.Conv2d(ndf * 8, 10, 4, 1, 0, bias=False)
        self.conv_7 = nn.Conv2d(ndf * 8, 10, 4, 1, 0, bias=False)
        self.conv_8 = nn.Conv2d(ndf * 8, 10, 4, 1, 0, bias=False)
        self.conv_9 = nn.Conv2d(ndf * 8, 10, 4, 1, 0, bias=False)
        self.conv_10 = nn.Conv2d(ndf * 8, 10, 4, 1, 0, bias=False)

    def forward(self, input):
        q_list = []
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            q_list.append(nn.parallel.data_parallel(self.conv_1, input, range(self.ngpu)))
            q_list.append(nn.parallel.data_parallel(self.conv_2, input, range(self.ngpu)))
            q_list.append(nn.parallel.data_parallel(self.conv_3, input, range(self.ngpu)))
            q_list.append(nn.parallel.data_parallel(self.conv_4, input, range(self.ngpu)))
            q_list.append(nn.parallel.data_parallel(self.conv_5, input, range(self.ngpu)))
            q_list.append(nn.parallel.data_parallel(self.conv_6, input, range(self.ngpu)))
            q_list.append(nn.parallel.data_parallel(self.conv_7, input, range(self.ngpu)))
            q_list.append(nn.parallel.data_parallel(self.conv_8, input, range(self.ngpu)))
            q_list.append(nn.parallel.data_parallel(self.conv_9, input, range(self.ngpu)))
            q_list.append(nn.parallel.data_parallel(self.conv_10, input, range(self.ngpu)))
        else:
            q_list.append(self.conv_1(input).squeeze())
            q_list.append(self.conv_2(input).squeeze())
            q_list.append(self.conv_3(input).squeeze())
            q_list.append(self.conv_4(input).squeeze())
            q_list.append(self.conv_5(input).squeeze())
            q_list.append(self.conv_6(input).squeeze())
            q_list.append(self.conv_7(input).squeeze())
            q_list.append(self.conv_8(input).squeeze())
            q_list.append(self.conv_9(input).squeeze())
            q_list.append(self.conv_10(input).squeeze())

        return q_list


