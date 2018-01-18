import os
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
from glob import glob


class Cityscapes(torch.utils.data.Dataset):
    def __init__(self, root, data_type):
        self.root = root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.paths = glob(os.path.join(self.root, '{}/*'.format(data_type)))
        if len(self.paths) == 0:
            raise Exception("No images are found in {}".format(self.root))

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return len(self.paths)


def train_loader(root, batch_size, num_workers):
    train_set_A = Cityscapes(root, 'trainA')
    train_set_B = Cityscapes(root, 'trainB')

    train_loader_A = torch.utils.data.DataLoader(train_set_A, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    train_loader_B = torch.utils.data.DataLoader(train_set_B, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

    return train_loader_A, train_loader_B

