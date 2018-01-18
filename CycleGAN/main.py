from __future__ import print_function
import random
import torch
import torch.backends.cudnn as cudnn
import os

from data_loader import train_loader
from config import get_config
from trainer import Trainer

def main(config):
    if config.outf is None:
        config.outf = 'samples'
    os.system('mkdir {0}'.format(config.outf))

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    if config.cuda:
        torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not config.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    train_loader_A, train_loader_B = train_loader(config.dataroot, config.batch_size, config.workers)

    trainer = Trainer(config, train_loader_A, train_loader_B)
    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)
