import torch
import os
import random
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


class CelebA(torch.utils.data.Dataset):
    def __init__(self, image_path, attribute_path, mode, transform):
        self.image_path = image_path
        self.mode = mode
        self.transform = transform
        self.lines = open(attribute_path, 'r').readlines()
        self.num_data = int(self.lines[0])
        self.idx2attr = {}

        self.preprocess()

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

    def preprocess(self):
        attrs = self.lines[1].split()
        for i, attr in enumerate(attrs):
            self.idx2attr[i] = attr

        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []

        lines = self.lines[2:]
        random.shuffle(lines)   # random shuffling
        for i, line in enumerate(lines):

            splits = line.split()
            filename = splits[0]
            values = splits[1:]

            label = []
            for idx, value in enumerate(values):
                attr = self.idx2attr[idx]

                if attr in self.selected_attrs:
                    if value == '1':
                        label.append(1)
                    else:
                        label.append(0)

            if (i+1) < 2000:
                self.test_filenames.append(filename)
                self.test_labels.append(label)
            else:
                self.train_filenames.append(filename)
                self.train_labels.append(label)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(os.path.join(self.image_path, self.train_filenames[index]))
            label = self.train_labels[index]
        elif self.mode in ['test']:
            image = Image.open(os.path.join(self.image_path, self.test_filenames[index]))
            label = self.test_labels[index]

        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_data

def get_loader(image_path, attribute_path, batch_size, num_workers, crop_size, image_size, mode):
    if mode == 'train':
        dataset = CelebA(image_path, attribute_path, mode,
                         transform = transforms.Compose([
                             transforms.CenterCrop(crop_size),
                             transforms.Resize(image_size, interpolation=Image.ANTIALIAS),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        shuffle = True

    elif mode == 'test':
        dataset = CelebA(image_path, attribute_path, mode,
                         transform = transforms.Compose([
                             transforms.CenterCrop(crop_size),
                             transforms.Scale(image_size, interpolation=Image.ANTIALIAS),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        shuffle = False

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)

    return data_loader