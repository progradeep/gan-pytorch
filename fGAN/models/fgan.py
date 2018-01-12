import torch
import torch.nn as nn
import torch.nn.parallel


class _netG (nn.Module):
    def __init__ (self, ngpu, nz):
        super (_netG, self).__init__ ()
        self.ngpu = ngpu
        self.nz = nz
        self.layer = nn.Sequential (
            nn.Linear(self.nz, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(),
            nn.Linear(1200,1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(),
            nn.Linear(1200, 784),
            nn.Tanh()
        )

    def main(self, input):
        output = self.layer(input)
        return output.view(-1, 1, 28, 28)

    def forward (self, input):
        if isinstance (input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel (self.main, input, range (self.ngpu))
        else:
            output = self.main (input)
        return output


class _netD (nn.Module):
    def __init__ (self, ngpu, f_div):
        super (_netD, self).__init__ ()
        self.f_div = f_div
        print ("Using " + f_div + " divergence")
        self.ngpu = ngpu
        self.layer = nn.Sequential (
            nn.Linear(784, 240),
            nn.ReLU(),
            nn.Linear(240, 240),
            nn.ReLU(),
            nn.Linear(240,1)
        )

    def activation_func (self, x):
        if self.f_div == "KL" or self.f_div == "Pearson":
            return x
        elif self.f_div == "RKL":
            return -x.exp ()
        elif self.f_div == "Neyman" or self.f_div == "Squared_Hellinger":
            return 1 - x.exp ()
        elif self.f_div == "JS":
            return (2 * x.sigmoid ()).log ()
        else:  # GAN
            return x.sigmoid ().log ()

    def f_star (self, x):
        if self.f_div == "KL":
            return (x - 1).exp ()
        elif self.f_div == "RKL":
            return -1 - (- x).log ()
        elif self.f_div == "Pearson":
            return 0.25 * x * x + x
        elif self.f_div == "Neyman":
            return 2 - 2 * (1 - x).sqrt ()
        elif self.f_div == "Squared_Hellinger":
            return x / (1 - x)
        elif self.f_div == "JS":
            return -(2 - x.exp ()).log ()
        else:  # GAN
            return -(1 - x.exp ()).log ()

    def main (self, input):
        input = input.view(input.size(0), -1)
        output = self.layer (input)
        output = self.activation_func (output)
        return output

    def forward (self, input):
        if isinstance (input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel (self.main, input, range (self.ngpu))
        else:
            output = self.main (input)

        return output.view (-1)
