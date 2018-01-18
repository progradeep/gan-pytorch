import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default=os.path.split(os.getcwd())[0] + '/datasets', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG_AB', default='', help="path to netG_AB (to continue training)")
parser.add_argument('--netG_BA', default='', help="path to netG_BA (to continue training)")
parser.add_argument('--netD_A', default='', help="path to netD_A (to continue training)")
parser.add_argument('--netD_B', default='', help="path to netD_B (to continue training)")
parser.add_argument('--outf', default=None, help='folder to output images and model checkpoints')
parser.add_argument('--sample_step', type=int, default=100, help='sample steps')
parser.add_argument('--checkpoint_step', type=int, default=400, help='checkpoint steps')

def get_config():
    return parser.parse_args()
