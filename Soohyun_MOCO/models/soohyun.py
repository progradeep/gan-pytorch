import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class _netG(nn.Module):
    def __init__(self, ngpu, ngf, input_nc, dim_z_content, dim_z_category, dim_z_motion, ntimestep):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        self.input_nc = input_nc
        self.dim_z_content = dim_z_content
        self.dim_z_category = dim_z_category
        self.dim_z_motion = dim_z_motion
        # self.nsize = nsize
        self.ntimestep = ntimestep

        dim_z = dim_z_motion + dim_z_category + dim_z_content
        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)

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

            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16

            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8

            # nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4

        )

        self.encoder_motion = nn.Sequential(
            nn.Conv2d(ngf * 8, self.dim_z_motion, 4, 1, 0, bias=False),
            # state size. (nz) x 1 x 1
        )

        self.encoder_content = nn.Sequential(
            nn.Conv2d(ngf * 8, self.dim_z_content, 4, 1, 0, bias=False),
            # state size. (nz) x 1 x 1
        )


        self.decoder = nn.Sequential(
            # state size. nz x 1 x 1
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, self.input_nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def sample_z_m(self, num_samples, input):

        motion = self.encoder(input)
        motion = self.encoder_motion(motion).view(num_samples, self.dim_z_motion)
        # print("motion", motion.shape)
        # motion size. (bs, 10)
        h_t = [motion]

        for framenum in range(self.ntimestep):
            e_t = Variable(torch.FloatTensor(num_samples, self.dim_z_motion).normal_().cuda())
            h_t.append(self.recurrent(e_t, h_t[-1]))

        z_m_t = [h_k.view(-1,1,self.dim_z_motion) for h_k in h_t]
        # z_m_t size. (self.ntimestep * bs, 1, self.dim_z_motion)

        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)
        # print("zm",z_m.shape) # bs * ntimestep, self.dim_z_motion (64, 10)
        return z_m


    def sample_z_categ(self, bs):
        classes_to_generate = np.random.randint(self.dim_z_category, size=bs)
        # state size. (bs)

        one_hot = np.zeros((bs, self.dim_z_category), dtype=np.float32)
        one_hot[np.arange(bs), classes_to_generate] = 1
        one_hot_video = np.repeat(one_hot, self.ntimestep, axis=0)
        # state size. (self.ntimestep* bs, self.dim_z_category (6))

        one_hot_video = torch.from_numpy(one_hot_video)

        if torch.cuda.is_available():
            one_hot_video = one_hot_video.cuda()

        return Variable(one_hot_video), classes_to_generate

    def sample_z_content(self, bs, input):
        content = self.encoder(input)
        content = self.encoder_content(content)
        # state size. (bs, self.dim_z_content (50))
        content = content.data.view(bs, self.dim_z_content)
        content = torch.cat([content] * self.ntimestep)

        # content size. (bs * 10, (80, 50)
        # print("contetn",content.shape)
        return Variable(content)


    def z_gif(self, bs, input):
        z_content = self.sample_z_content(bs, input)
        z_category, z_category_labels = self.sample_z_categ(bs)
        z_motion = self.sample_z_m(bs, input)
        # torch.Size([80, 50]) torch.Size([80, 6]) torch.Size([80, 10])

        z = torch.cat([z_content, z_category, z_motion], dim = 1)

        return z, z_category_labels

    def sample_gif(self, bs, input):
        z, z_category = self.z_gif(bs, input)
        # z size. 80, 66
        # print("z",z.shape)

        h = self.decoder(z.view(z.size(0), z.size(1), 1, 1))
        # print("h", h.shape)
        # h size. 80, 3, 128, 128
        h = h.view(int(h.size(0) / self.ntimestep), self.ntimestep, self.input_nc, h.size(3), h.size(3))

        z_category_labels = torch.from_numpy(z_category)

        if torch.cuda.is_available():
            z_category_labels = z_category_labels.cuda()

        h = h.permute(0, 2, 1, 3, 4)
        return h, Variable(z_category_labels, requires_grad=False)

    def sample_im(self, bs, input):
        z, z_category_labels = self.z_gif(bs, input)

        j = np.sort(np.random.choice(z.size(0), bs, replace=False)).astype(np.int64)
        z = z[j, ::]
        z = z.view(z.size(0), z.size(1), 1, 1)
        h = self.decoder(z)
        # print("imsize",h.shape)

        return h, None

    def main(self, input):
        x = input.view((-1, self.input_nc, 128, 128))

        # print(x.shape) # should be (bs, nz, 1, 1)
        # x = x.view((1, -1, self.dim_z_motion))
        # print(x.shape) # should be (1, bs, nz)
        bs = x.size(0)

        return self.sample_gif(bs, x), self.sample_im(x.size(0), x)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD_V(nn.Module):
    def __init__(self, ngpu, n_channels, n_output_neurons=1, ndf=64):
        super(_netD_V, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons


        self.main = nn.Sequential(
            # Input. bs, nc, depth, 128, 128
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf, d/2, 64, 64

            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf * 2, d/4, 32, 32

            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf * 4, d/8, 16, 16

            nn.Conv3d(ndf * 4, 1, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            # 1, d/16, 8, 8

        )

    def forward(self, input):
        h = self.main(input)
        h = h.squeeze()

        return h

class _netD_I(nn.Module):
    def __init__(self, n_channels, ndf=64):
        super(_netD_I, self).__init__()


        self.main = nn.Sequential(
            # 3, 128, 128
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf, 64, 64

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf * 2, 32, 32

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf * 4, 16, 16

            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            # 1, 8, 8
        )

    def forward(self, input):
        h = self.main(input).squeeze()
        return h


class ImageReconstructor(nn.Module):
    def __init__(self, n_channels, dim_z, ngf=64):
        super(ImageReconstructor, self).__init__()

        self.n_channels = n_channels
        self.dim_z = dim_z

        self.encoder = nn.Sequential(
            # 3, 128, 128
            nn.Conv2d(self.n_channels, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # ngf, 64, 64

            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # ngf * 2, 32, 32

            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # ngf * 4, 16, 16

            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # ngf * 8, 8, 8

            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # ngf * 8, 4, 4

            nn.Conv2d(ngf * 8, self.dim_z, 4, 1, 0, bias=False),
            # z, 1, 1

        )

        self.decoder = nn.Sequential(
            # state size. nz x 1 x 1
            nn.ConvTranspose2d(dim_z, ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 4 x 4

            nn.ConvTranspose2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf * 2) x 8 x 8

            nn.ConvTranspose2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf * 4) x 16 x 16

            nn.ConvTranspose2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf * 8) x 32 x 32

            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf * 8) x 64 x 64

            nn.ConvTranspose2d(ngf * 8, self.n_channels, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        h = self.encoder(input)
        # print("h",h.shape)
        # h size. (bs, dim z, 1, 1)
        h = h.view(-1, self.dim_z, 1, 1)
        output = self.decoder(h)
        # print("output",output.shape)
        # output size. bs, nc, 128, 128
        return output


class VideoReconstructor(nn.Module):
    def __init__(self, n_channels, video_len, dim_z, ngf=64):
        super(VideoReconstructor, self).__init__()

        self.n_channels = n_channels
        self.video_len = video_len
        self.dim_z = dim_z

        self.encoder = nn.Sequential(
            # Input. bs, nc, depth, 128, 128
            nn.Conv3d(n_channels, ngf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf, d/2, 64, 64

            nn.Conv3d(ngf, ngf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf * 2, d/4, 32, 32

            nn.Conv3d(ngf * 2, ngf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf * 4, d/8, 16, 16

            nn.Conv3d(ngf * 4, ngf * 8, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf * 8, d/8, 8, 8

            nn.Conv3d(ngf * 8, ngf * 8, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf * 8, d/8, 4, 4

            nn.Conv3d(ngf * 8, self.dim_z, (1, 4, 4), 1, 0, bias=False),
            # dimz, d/8, 1, 1
        )

        self.decoder = nn.Sequential(
            # state size. nz x 1 x 1
            nn.ConvTranspose2d(dim_z, ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 4 x 4

            nn.ConvTranspose2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf * 2) x 8 x 8

            nn.ConvTranspose2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf * 4) x 16 x 16

            nn.ConvTranspose2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf * 8) x 32 x 32

            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf * 8) x 64 x 64

            nn.ConvTranspose2d(ngf * 8, self.n_channels, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        h = self.encoder(input)
        # print(h.shape)
        # h size. (bs, dim z, 1, 1, 1)
        h = h.view(-1, self.dim_z, 1, 1)
        output = self.decoder(h)
        # print("output",output.shape)
        # output size. bs, nc, 128, 128
        return output