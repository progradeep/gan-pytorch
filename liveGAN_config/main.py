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

    # dataroot, cache, image_size, n_channels, image_batch, video_batch, video_length):
    image_loader, video_loader = get_loader(dataroot=config.dataroot, image_size=int(config.image_size),
                                            n_channels=int(config.n_channels), image_batch=int(config.image_batch),
                                            video_batch=int(config.video_batch), video_length=int(config.video_length))

    trainer = Trainer(config, image_loader, video_loader)

    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)
