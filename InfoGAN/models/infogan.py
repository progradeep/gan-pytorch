import torch
import torch.nn as nn
import torch.nn.parallel

class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, code_size):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z + code
            #nn.Linear(nz + code_size, 1024, bias=False),
            nn.ConvTranspose2d(nz + code_size, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. 1024
            nn.ConvTranspose2d(1024, ngf * 2, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf * 2) x 7 x 7
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 14 x 14
            nn.ConvTranspose2d(ngf, nc, 4, 2, 0, bias=False),
            nn.Tanh()
            # state size. (nc) x 28 x 28
        )
        '''
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + code_size, ngf * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 1 x 1
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 7 x 7
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 14 x 14
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 28 x 28
        )
        '''

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class _netD(nn.Module):
    def __init__(self, ngpu, nc, ndf, code_size):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.share = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
	)
        self.fc = nn.Sequential(
            # state size. (ndf*2*7*7)
            nn.Linear((ndf*2)*7*7, 1024, bias=False),
            nn.BatchNorm2d(1024)
            # state size. 1024
	)
        '''
        self.share = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 7 x 7
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 3 x 3

	)
        '''
        self.main_d = nn.Sequential(
            # state size. 1024
            nn.Linear(1024, 1, bias=False),
	    nn.Sigmoid()
	)
        self.main_q = nn.Sequential(
            # state size. 1024
            nn.Linear(1024, 128, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128
            nn.Linear(128, code_size, bias=False),
	)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            share = nn.parallel.data_parallel(self.share, input, range(self.ngpu))
        else:
            share = self.share(input)

        share = share.view(-1, self.ndf * 2 * 7 * 7)
        share = self.fc(share)

        out_d = self.main_d(share)
        #out_q = share.view(-1, self.ndf * 8 * 7 * 7)
        out_q = self.main_q(share)	

        return out_d.squeeze(), out_q.squeeze()


