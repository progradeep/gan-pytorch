import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import itertools, time, os
from glob import glob
from utils import save_imgs
import models.cyclegan as cyclegan


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Trainer(object):
    def __init__(self, config, dataloader):
        self.config = config
        self.train_loader_A = dataloader[0]
        self.train_loader_B = dataloader[1]
        self.test_loader_A = dataloader[2]
        self.test_loader_B = dataloader[3]
        self.ngpu = int(config.ngpu)
        self.input_nc = int(config.input_nc)
        self.output_nc = int(config.output_nc)

        self.ngf = int(config.ngf)
        self.ndf = int(config.ndf)
        self.cuda = config.cuda

        self.batch_size = config.batch_size
        self.image_size = config.image_size

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.niter = config.niter
        self.decay_epoch = config.decay_epoch
        self.cycle_lambda = config.cycle_lambda

        self.outf = config.outf
        self.sample_step = config.sample_step
        self.checkpoint_step = config.checkpoint_step

        self.build_model()

        if self.cuda:
            self.netG_AB.cuda()
            self.netD_A.cuda()
            self.netG_BA.cuda()
            self.netD_B.cuda()

    def load_model(self):
        print("[*] Load models from {}...".format(self.outf))

        paths = glob(os.path.join(self.outf, 'net*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.outf))
            return

        epochs = [int(os.path.basename(path.split('.')[0].split('_')[-2].split('-')[-1])) for path in paths]
        self.start_epoch = str(max(epochs))
        steps = [int(os.path.basename(path.split('.')[0].split('_')[-1].split('-')[-1])) for path in paths]
        self.start_step = str(max(steps))



        G_AB_filename = '{}/netG_A_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)
        G_BA_filename = '{}/netG_B_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)
        D_A_filename = '{}/netD_A_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)
        D_B_filename = '{}/netD_B_epoch-{}_step-{}.pth'.format(self.outf, self.start_epoch, self.start_step)


        self.netG_AB.load_state_dict(torch.load(G_AB_filename))
        self.netG_BA.load_state_dict(torch.load(G_BA_filename))
        self.netD_A.load_state_dict(torch.load(D_A_filename))
        self.netD_B.load_state_dict(torch.load(D_B_filename))


        print("[*] Model loaded: {}".format(G_AB_filename))


    def build_model(self):
        self.netG_AB = cyclegan._netG(self.ngpu, self.ngf,
                                      self.input_nc, self.output_nc)
        self.netG_AB.apply(weights_init)

        self.netG_BA = cyclegan._netG(self.ngpu, self.ngf,
                                      self.input_nc, self.output_nc)
        self.netG_BA.apply(weights_init)


        self.netD_A = cyclegan._netD(self.ngpu, self.ndf, self.input_nc)
        self.netD_A.apply(weights_init)

        self.netD_B = cyclegan._netD(self.ngpu, self.ndf, self.input_nc)
        self.netD_B.apply(weights_init)

        if self.config.model_path != '':
            self.load_model()


    def train(self):
        MSELoss = nn.MSELoss()
        L1loss = nn.L1Loss()

        if self.cuda:
            MSELoss.cuda()
            L1loss.cuda()

        # setup optimizer
        optimizerD_A = optim.Adam(self.netD_A.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        optimizerD_B = optim.Adam(self.netD_B.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        optimizerG = optim.Adam(itertools.chain(self.netG_AB.parameters(), self.netG_BA.parameters()), lr=self.lr, betas=(self.beta1, self.beta2))

        test_loader_A, test_loader_B = iter(self.test_loader_A), iter(self.test_loader_B)

        for epoch in range(self.niter):
            if (epoch+1) > self.decay_epoch:
                optimizerD_A.param_groups[0]['lr'] -= self.lr / (self.niter - self.decay_epoch)
                optimizerD_B.param_groups[0]['lr'] -= self.lr / (self.niter - self.decay_epoch)
                optimizerG.param_groups[0]['lr'] -= self.lr / (self.niter - self.decay_epoch)

            start_time = time.time()
            for step, (realA, realB) in enumerate(itertools.izip(self.train_loader_A, self.train_loader_B)):

                realA, realB = Variable(realA.cuda()), Variable(realB.cuda())
                ############################
                # (1) Update G network: minimize Lgan(MSE) + Lcycle(L1)
                ###########################
                for p in self.netD_A.parameters():
                    p.requires_grad = False
                for p in self.netD_B.parameters():
                    p.requires_grad = False

                self.netG_AB.zero_grad()
                self.netG_BA.zero_grad()

                # GAN loss: D_A(G_A(A))
                fakeB = self.netG_AB(realA)
                output = self.netD_A(fakeB)
                loss_G_A = MSELoss(output, Variable(torch.ones(output.size()).cuda()))

                # GAN loss: D_B(G_B(B))
                fakeA = self.netG_BA(realB)
                output = self.netD_B(fakeA)
                loss_G_B = MSELoss(output, Variable(torch.ones(output.size()).cuda()))

                # Forward cycle loss: A <-> G_B(G_A(A))
                cycleA = self.netG_BA(fakeB)
                loss_cycle_A = L1loss(cycleA, realA) * self.cycle_lambda

                # Backward cycle loss: B <-> G_A(G_B(B))
                cycleB = self.netG_AB(fakeA)
                loss_cycle_B = L1loss(cycleB, realB) * self.cycle_lambda

                # Combined Generator loss
                loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B
                loss_G.backward()
                optimizerG.step()


                ############################
                # (2) Update D network: minimize LSGAN loss
                ###########################
                for p in self.netD_A.parameters():
                    p.requires_grad = True
                for p in self.netD_B.parameters():
                    p.requires_grad = True

                ## train D_A ##
                # train with real
                self.netD_A.zero_grad()

                D_A_real = self.netD_A(realB)
                loss_D_A_real = MSELoss(D_A_real, Variable(torch.ones(D_A_real.size()).cuda()))

                # train with fake
                D_A_fake = self.netD_A(fakeB.detach())
                loss_D_A_fake = MSELoss(D_A_fake, Variable(torch.zeros(D_A_fake.size()).cuda()))

                loss_D_A = loss_D_A_real + loss_D_A_fake
                loss_D_A.backward()
                optimizerD_A.step()

                ## train D_B ##
                # train with real
                self.netD_B.zero_grad()

                D_B_real = self.netD_B(realB)
                loss_D_B_real = MSELoss(D_B_real, Variable(torch.ones(D_B_real.size()).cuda()))

                # train with fake
                D_B_fake = self.netD_B(fakeB.detach())
                loss_D_B_fake = MSELoss(D_B_fake, Variable(torch.zeros(D_B_fake.size()).cuda()))

                loss_D_B = loss_D_B_real + loss_D_B_fake
                loss_D_B.backward()
                optimizerD_B.step()


                step_end_time = time.time()

                print('[%d/%d][%d/%d] - time_passed: %.2f, loss_D_A: %.3f, loss_D_B: %.3f, '
                      'loss_G_A: %.3f, loss_G_B: %.3f, loss_A_cycle: %.3f, loss_B_cycle: %.3f'
                      % (epoch, self.niter, step, len(self.train_loader_A), step_end_time - start_time,
                         loss_D_A, loss_D_B, loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B))


                if step % self.sample_step == 0:
                    realA = test_loader_A.next()
                    realB = test_loader_B.next()

                    realA = Variable(realA.cuda(), volatile=True)
                    fakeB = self.netG_AB(realA)
                    cycleA = self.netG_BA(fakeB)

                    realB = Variable(realB.cuda(), volatile=True)
                    fakeA = self.netG_BA(realB)
                    cycleB = self.netG_AB(fakeA)

                    save_imgs(realA, fakeB, cycleA, realB, fakeA, cycleB, self.outf, epoch, step)

                if step% self.checkpoint_step == 0 and step != 0:
                    torch.save(self.netG_AB.state_dict(), '%s/netG_A_epoch-%d_step-%s.pth' % (self.outf, epoch, step))
                    torch.save(self.netD_A.state_dict(), '%s/netD_A_epoch-%d_step-%s.pth' % (self.outf, epoch, step))
                    torch.save(self.netG_BA.state_dict(), '%s/netG_B_epoch-%d_step-%s.pth' % (self.outf, epoch, step))
                    torch.save(self.netD_B.state_dict(), '%s/netD_B_epoch-%d_step-%s.pth' % (self.outf, epoch, step))
                    print("Saved checkpoint")

