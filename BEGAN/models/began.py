import torch
import torch.nn as nn
import torch.nn.parallel

class _netG(nn.Module):
    def __init__(self, ngpu, nz, n_hidden, nc):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.n_hidden = n_hidden
        self.fc = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(nz, 8 * 8 * n_hidden)
            # state size. (n_hidden) x 8 x 8
        )
        self.conv = nn.Sequential(
            # state size. (n_hidden) x 8 x 8
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Upsample(scale_factor=2),
            # state size. (n_hidden) x 16 x 16
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Upsample(scale_factor=2),
            # state size. (n_hidden) x 32 x 32
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Upsample(scale_factor=2),
            # state size. (n_hidden) x 64 x 64
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(n_hidden, 3, 3, 1, 1),
            # state size. (nc) x 64 x 64
        )

    def main(self, input):
        h = self.fc(input)
        h = h.view(-1, self.n_hidden, 8, 8)
        return self.conv(h)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class _netD(nn.Module):
    def __init__(self, ngpu, nz, n_hidden, nc):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.n_hidden = n_hidden
        self.conv_encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            # state size. (n_hidden) x 64 x 64
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(n_hidden, n_hidden * 2, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2, 2),
            # state size. (n_hidden*2) x 32 x 32
            nn.Conv2d(n_hidden * 2, n_hidden * 2, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(n_hidden * 2, n_hidden * 3, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2, 2),
            # state size. (n_hidden*3) x 16 x 16
            nn.Conv2d(n_hidden * 3, n_hidden * 3, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(n_hidden * 3, n_hidden * 3, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.MaxPool2d(2, 2),
            # state size. (n_hidden*3) x 8 x 8
            nn.Conv2d(n_hidden * 3, n_hidden * 3, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(n_hidden * 3, n_hidden * 3, 3, 1, 1),
            nn.ELU(inplace=True)
        )

        self.fc_encoder = nn.Sequential(
            # state size. (n_hidden*8) x 8 x 8
            nn.Linear(8 * 8 * 3 * n_hidden, nz)
            # input is Z, going into a convolution
        )

        self.fc_decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(nz, 8 * 8 * n_hidden)
            # state size. (n_hidden) x 8 x 8
        )

        self.conv_decoder = nn.Sequential(
            # state size. (n_hidden) x 8 x 8
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Upsample(scale_factor=2),
            # state size. (n_hidden) x 16 x 16
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Upsample(scale_factor=2),
            # state size. (n_hidden) x 32 x 32
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Upsample(scale_factor=2),
            # state size. (n_hidden) x 64 x 64
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(n_hidden, n_hidden, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(n_hidden, 3, 3, 1, 1)
            # state size. (nc) x 64 x 64
        )

    def main(self, input):
        h = self.conv_encoder(input)
        h = h.view(-1, 8 * 8 * 3 * self.n_hidden)
        h = self.fc_encoder(h)
        h = self.fc_decoder(h)
        h = h.view(-1, self.n_hidden, 8, 8)
        return self.conv_decoder(h)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


