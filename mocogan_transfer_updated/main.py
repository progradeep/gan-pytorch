"""
Usage: main.py [options] --dataroot <dataroot> --cuda
"""

import os

import random
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from trainer import Trainer

from data_loader import get_loader

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

    image_loader, video_loader = get_loader(config.dataroot, config.image_batch, config.image_size, 
                                            num_workers=2)

    trainer = Trainer(config, image_loader, video_loader)

    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)
