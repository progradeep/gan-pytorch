import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable

class _netG(nn.Module):
    def __init__(self, ngpu, ngf, input_nc, nz, nsize):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        self.input_nc = input_nc
        self.nz = nz
        self.nsize = nsize
        self.ntimestep = 9
        self.lstm = nn.LSTMCell(self.nz, self.nz)

        self.encoder = nn.Sequential(
            # input. (nc) * 128 * 128
            nn.Conv2d(self.input_nc, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64

            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32

            nn.Conv2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16

            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8

            nn.Conv2d(ngf * 4, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4

            nn.Conv2d(ngf * 4, ngf *8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 2 x 2

            nn.Conv2d(ngf * 8, self.nz, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. (nz) x 1 x 1
        )


        self.decoder = nn.Sequential(
            # state size. nz x 1 x 1
            nn.ConvTranspose2d(self.nz, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 2 x 2

            nn.ConvTranspose2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 4 x 4

            nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf * 2) x 8 x 8

            nn.ConvTranspose2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf * 4) x 16 x 16

            nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf * 4) x 32 x 32

            nn.ConvTranspose2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf * 8) x 64 x 64

            nn.ConvTranspose2d(ngf * 8, self.input_nc, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Tanh()
            # state size. (nc * 9) x 128 x 128
        )

    def main(self, input):
        x = input.view((-1, self.input_nc, self.nsize, self.nsize))
        x = self.encoder(x)
        # print(x.shape) # should be (bs, nz, 1, 1)
        x = x.view((1, -1, self.nz))
        # print(x.shape) # should be (1, bs, nz)
        bs = x.size(1)

        hx = Variable(torch.zeros(bs, self.nz).cuda())
        cx = Variable(torch.zeros(bs, self.nz).cuda())

        im_list = []

        for i in range(self.ntimestep):
            input = x.contiguous().view(1, bs, self.nz)
            hx, cx = self.lstm(input, (hx, cx))
            hx_view = hx.contiguous().view(bs, self.nz, 1, 1)
            im = self.decoder(hx_view)
            # print(im.shape) # should be (bs, nc, 128, 128)
            im_list.append(im)

        output = torch.cat(im_list)
        # print(output.shape) # (bs, nc * 9, 128, 128)
        return output

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self, ngpu, ndf, input_nc, nsize, bs):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.ntimestep = 9
        self.input_nc = input_nc
        self.nsize = nsize
        self.nz = 100
        self.batch_size = bs
        self.lstm = nn.LSTMCell(self.nz, self.nz)
        self.encoder = nn.Sequential(
            # input. (nc) x 128 x 128
            nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 7 x 7

            nn.Conv2d(ndf * 8, self.nz, 7, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. nz x 1 x 1
        )

        self.decoder = nn.Sequential(
            # input size. nz x 1 x 1
            nn.Conv2d(self.nz, ndf, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 1 x 1
            nn.Conv2d(ndf, ndf * 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 1 x 1
            nn.Conv2d(ndf * 2, ndf * 4, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 1 x 1
            nn.Conv2d(ndf * 4, ndf * 8, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 1 x 1
            nn.Conv2d(ndf * 8, 1, 1, bias=False),
            nn.Sigmoid()

        )

    def main(self, input):
        # input. (bs, 9, 3, 128, 128)
        input = input.contiguous().view(-1, 9, self.input_nc, self.nsize, self.nsize)
        bs = input.size(0)
        hx = Variable(torch.zeros(bs, self.nz).cuda())
        cx = Variable(torch.zeros(bs, self.nz).cuda())

        for i in range(input.size(1)):
            x = input[:,i,:,:,:]
            x = x.contiguous().view(bs, self.input_nc, self.nsize, self.nsize)
            x = self.encoder(x)
            # print("x.",x.shape) # bs, nz, 1, 1
            x = x.contiguous().view(1, -1, self.nz) # 1, bs, nz
            hx, cx = self.lstm(x, (hx, cx))

        # print("hx", hx.shape) # bs, nz
        x = hx.view(bs, self.nz, 1, 1)
        output = self.decoder(x)
        # print("output",output.shape)

        return output



    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

