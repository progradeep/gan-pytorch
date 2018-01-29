from __future__ import print_function
import random
import torch
import torch.backends.cudnn as cudnn
import os
from data_loader import get_loader
from config import get_config
from trainer import Trainer

def main(config):
    if not os.path.exists('./samples'):
        os.makedirs('./samples')
    if config.outf is None:
        config.outf = 'samples/%s'%(config.dataset)
    if not os.path.exists(config.outf):
        os.makedirs(config.outf)
        print("Directory",config.outf,"has created")
    else:
        print("Directory",config.outf,"already exsits")

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    if config.cuda:
        torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not config.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    data_lodaer = get_loader(config.dataroot, config.batch_size, config.image_size, 
                            num_workers=int(config.workers))

    trainer = Trainer(config, data_lodaer)
    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)
