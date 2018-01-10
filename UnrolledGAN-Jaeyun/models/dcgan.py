import torch
import torch.nn as nn
import torch.nn.parallel

class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a FC
            nn.Linear(nz, ngf, bias=False),
            nn.ReLU(),
            # state size. ngf
            nn.Linear(ngf, ngf, bias=False),
            nn.ReLU(),
            # state size. ngf
            nn.Linear(ngf, nc, bias=False)
            # state size. nc
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
            # input is nc
            nn.Linear(nc, ndf, bias=False),
            nn.ReLU(),
            # state size. ndf
            nn.Linear(ndf, 1, bias=False),
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


