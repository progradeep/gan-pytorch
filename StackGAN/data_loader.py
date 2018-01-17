import os
import torch.utils.data
import torchvision.transforms as transforms

from datasets import TextDataset

def get_loader(_dataset, dataroot, batch_size, num_workers, image_size, shuffle=True):
    if _dataset in ['birds', 'flowers', 'coco']:
        dataset = TextDataset(dataroot, 'train', imsize=image_size,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(image_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             drop_last=True, shuffle=True, num_workers=num_workers)

    return dataloader
