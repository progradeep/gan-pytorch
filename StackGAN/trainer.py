from __future__ import print_function
from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time

import numpy as np
import torchfile

from utils import mkdir_p
from utils import save_img_results, save_model
from utils import KL_loss
from utils import compute_discriminator_loss, compute_generator_loss

import models.stageI as stageI
import models.stageII as stageII

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class Trainer(object):
    def __init__(self, config, data_loader, datapath):
        self.config = config
        self.data_loader = data_loader
        self.datapath = datapath
        
        self.text_dim = int(config.text_dim)

        self.ngpu = int(config.ngpu)
        self.nz = int(config.nz)
        self.nef = int(config.nef)
        self.ngf = int(config.ngf)
        self.ndf = int(config.ndf)
        self.r_num = int(config.r_num)

        self.cuda = config.cuda
        
        self.training = config.training
        self.stage = config.stage

        self.batch_size = config.batch_size
        self.image_size = config.image_size
        
        self.lrG = config.lrG
        self.lrD = config.lrD
        self.lr_decay_step = config.lr_decay_step
        self.beta1 = config.beta1

        self.niter = config.niter

        self.vis_count = config.vis_count
        self.snapshot_interval = config.snapshot_interval

        self.outf = config.outf

        self.coeff_KL = config.coeff_KL

        if self.training:
            self.model_dir = os.path.join(self.outf, 'models')
            self.image_dir = os.path.join(self.outf, 'images')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

    # ############# For training stageI GAN #############
    def load_network_stageI(self):
        netG = stageI._netG(self.text_dim, self.nz, self.nef, self.ngf, self.cuda)
        netG.apply(weights_init)
        netD = stageI._netD(self.ndf, self.nef)
        netD.apply(weights_init)

        if self.config.netG != '':
            state_dict = \
                torch.load(self.config.netG,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
        if self.config.netD != '':
            state_dict = \
                torch.load(self.config.netD,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
        
        if self.cuda:
            netG.cuda()
            netD.cuda()
        return netG, netD

    # ############# For training stageII GAN  #############
    def load_network_stageII(self):
        stageI_G = stageI._netG(self.text_dim, self.nz, self.nef, self.ngf, self.cuda)
        netG = stageII._netG(stageI_G, self.text_dim, self.nz, self.nef, self.ngf, self.r_num, self.cuda)
        netG.apply(weights_init)
        
        if self.config.netG != '':
            state_dict = \
                torch.load(self.config.netG,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
        elif self.config.stageI_G != '':
            state_dict = \
                torch.load(self.config.stageI_G,
                           map_location=lambda storage, loc: storage)
            netG.stageI_G.load_state_dict(state_dict)
        else:
            print("Please give the StageI_G path")
            return

        netD = stageII._netD(self.ndf, self.nef)
        netD.apply(weights_init)
        if self.config.netD != '':
            state_dict = \
                torch.load(self.config.netD,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)

        if self.cuda:
            netG.cuda()
            netD.cuda()
        return netG, netD

    def train(self):
        if self.stage == 1:
            netG, netD = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()

        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(self.batch_size, self.nz))
        fixed_noise = \
            Variable(torch.FloatTensor(self.batch_size, self.nz).normal_(0, 1),
                     volatile=True)
        real_labels = Variable(torch.FloatTensor(self.batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(self.batch_size).fill_(0))
        if self.cuda:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()

        optimizerD = \
            optim.Adam(netD.parameters(),
                       lr=self.lrD, betas=(self.beta1, 0.999))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para,
                                lr=self.lrG,
                                betas=(self.beta1, 0.999))
        count = 0
        for epoch in range(self.niter):
            if epoch % self.lr_decay_step == 0 and epoch > 0:
                self.lrG *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = self.lrG
                self.lrD *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = self.lrD

            for i, data in enumerate(self.data_loader, 0):
                ######################################################
                # (1) Prepare training data
                ######################################################
                real_img_cpu, txt_embedding = data
                real_imgs = Variable(real_img_cpu)
                txt_embedding = Variable(txt_embedding)
                if self.cuda:
                    real_imgs = real_imgs.cuda()
                    txt_embedding = txt_embedding.cuda()

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (txt_embedding, noise)
                _, fake_imgs, mu, logvar = \
                    nn.parallel.data_parallel(netG, inputs, range(self.ngpu))

                ############################
                # (3) Update D network
                ###########################
                netD.zero_grad()
                errD, errD_real, errD_wrong, errD_fake = \
                    compute_discriminator_loss(netD, real_imgs, fake_imgs,
                                               real_labels, fake_labels,
                                               mu, range(self.ngpu))
                errD.backward()
                optimizerD.step()
                ############################
                # (2) Update G network
                ###########################
                netG.zero_grad()
                errG = compute_generator_loss(netD, fake_imgs,
                                              real_labels, mu, range(self.ngpu))
                kl_loss = KL_loss(mu, logvar)
                errG_total = errG + kl_loss * self.coeff_KL
                errG_total.backward()
                optimizerG.step()

                count = count + 1
                if i % 100 == 0:
                    # save the image result for each epoch
                    inputs = (txt_embedding, fixed_noise)
                    lr_fake, fake, _, _ = \
                        nn.parallel.data_parallel(netG, inputs, range(self.ngpu))
                    save_img_results(real_img_cpu, fake, epoch, self.image_dir, self.vis_count)
                    if lr_fake is not None:
                        save_img_results(None, lr_fake, epoch, self.image_dir, self.vis_count)
                    print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                        Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                        '''
                        % (epoch, self.niter, i, len(data_loader),
                            errD.data[0], errG.data[0], kl_loss.data[0],
                            errD_real, errD_wrong, errD_fake))
            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, epoch, self.model_dir)
        #
        save_model(netG, netD, self.n_iters, self.model_dir)

    def sample(self):
        if self.stage == 1:
            netG, _ = self.load_network_stageI()
        else:
            netG, _ = self.load_network_stageII()
        netG.eval()

        # Load text embeddings generated from the encoder
        t_file = torchfile.load(self.datapath)
        captions_list = t_file.raw_txt
        embeddings = np.concatenate(t_file.fea_txt, axis=0)
        num_embeddings = len(captions_list)
        print('Successfully load sentences from: ', self.datapath)
        print('Total number of sentences:', num_embeddings)
        print('num_embeddings:', num_embeddings, embeddings.shape)
        # path to save generated samples
        save_dir = self.config.netG[:self.config.netG.find('.pth')]
        mkdir_p(save_dir)

        batch_size = np.minimum(num_embeddings, self.batch_size)
        noise = Variable(torch.FloatTensor(self.batch_size, self.nz))
        if self.cuda:
            noise = noise.cuda()
        count = 0
        while count < num_embeddings:
            if count > 3000:
                break
            iend = count + self.batch_size
            if iend > num_embeddings:
                iend = num_embeddings
                count = num_embeddings - batch_size
            embeddings_batch = embeddings[count:iend]
            # captions_batch = captions_list[count:iend]
            txt_embedding = Variable(torch.FloatTensor(embeddings_batch))
            if self.cuda:
                txt_embedding = txt_embedding.cuda()

            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            inputs = (txt_embedding, noise)
            _, fake_imgs, mu, logvar = \
                nn.parallel.data_parallel(netG, inputs, range(self.ngpu))
            for i in range(self.batch_size):
                save_name = '%s/%d.png' % (save_dir, count + i)
                im = fake_imgs[i].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                # print('im', im.shape)
                im = np.transpose(im, (1, 2, 0))
                # print('im', im.shape)
                im = Image.fromarray(im)
                im.save(save_name)
            count += self.batch_size

