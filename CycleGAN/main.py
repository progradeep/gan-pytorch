from __future__ import print_function
import random
import torch
import torch.backends.cudnn as cudnn
import os

from data_loader import get_loader
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

    data_loader = get_loader(_dataset=config.dataset, dataroot=config.dataroot, batch_size=config.batch_size,
                                num_workers=int(config.workers), image_size=config.image_size)

    trainer = Trainer(config, data_loader)
    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)
