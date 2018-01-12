import torch
import torch.nn as nn
import torch.nn.parallel


class _netG(nn.Module):
    def __init__(self, ngpu, nz, dataset):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.dataset = dataset
        if self.dataset == "mnist":
            self.layer = nn.Sequential(
                nn.Linear(self.nz, 1200),
                nn.BatchNorm1d(1200),
                nn.ReLU(),
                nn.Linear(1200, 1200),
                nn.BatchNorm1d(1200),
                nn.ReLU(),
                nn.Linear(1200, 784),
                nn.Tanh()
            )
        if self.dataset == "lsun":
            self.layer1 = nn.Sequential(
                nn.Linear(self.nz, 6 * 6 * 512),
                nn.BatchNorm1d(6 * 6 * 512),
                nn.ReLU(),
            )
            self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, 4, 2, 1),
                nn.Tanh()
            )

    def main(self, input):
        if self.dataset == "mnist" :
            output = self.layer(input)
            return output.view(-1, 1, 28, 28)
        if self.dataset == "lsun" :
            output = self.layer1(input)
            output = output.view(-1, 512, 6, 6)
            return self.layer2(output)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self, ngpu, f_div, dataset):
        super(_netD, self).__init__()
        self.f_div = f_div
        print("Using " + f_div + " divergence")
        self.ngpu = ngpu
        self.dataset = dataset
        if dataset == "mnist":
            self.layer = nn.Sequential(
                nn.Linear(784, 240),
                nn.ReLU(),
                nn.Linear(240, 240),
                nn.ReLU(),
                nn.Linear(240, 1)
            )
        if dataset == "lsun":
            self.layer1 = nn.Sequential(
                nn.Conv2d(3,64,4,2,1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,128,4,2,1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,256,4,2,1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256,512,4,2,1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            self.layer2 = nn.Linear(512*6*6, 1)


    def activation_func(self, x):
        if self.f_div == "KL" or self.f_div == "Pearson":
            return x
        elif self.f_div == "RKL":
            return -x.exp()
        elif self.f_div == "Neyman" or self.f_div == "Squared_Hellinger":
            return 1 - x.exp()
        elif self.f_div == "JS":
            return (2 * x.sigmoid()).log()
        else:  # GAN
            return x.sigmoid().log()

    def f_star(self, x):
        if self.f_div == "KL":
            return (x - 1).exp()
        elif self.f_div == "RKL":
            return -1 - (- x).log()
        elif self.f_div == "Pearson":
            return 0.25 * x * x + x
        elif self.f_div == "Neyman":
            return 2 - 2 * (1 - x).sqrt()
        elif self.f_div == "Squared_Hellinger":
            return x / (1 - x)
        elif self.f_div == "JS":
            return -(2 - x.exp()).log()
        else:  # GAN
            return -(1 + 0.00001 - x.exp()).log()

    def main(self, input):
        if self.dataset == "mnist":
            input = input.view(input.size(0), -1)
            output = self.layer(input)
            output = self.activation_func(output)
            return output
        if self.dataset == "lsun":
            output = self.layer1(input)
            output = output.view(-1, 512*6*6)
            return self.layer2(output)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1)
