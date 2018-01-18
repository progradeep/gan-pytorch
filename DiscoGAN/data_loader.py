import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import pandas as pd

import torch
from torchvision import transforms

PIX2PIX_DATASETS = [
    'facades', 'cityscapes', 'maps', 'edges2shoes', 'edges2handbags']

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def pix2pix_split_images(root):
    paths = glob(os.path.join(root, "train/*"))

    a_path = os.path.join(root, "A")
    b_path = os.path.join(root, "B")

    makedirs(a_path)
    makedirs(b_path)

    for path in tqdm(paths, desc="pix2pix processing"):
        filename = os.path.basename(path)

        a_image_path = os.path.join(a_path, filename)
        b_image_path = os.path.join(b_path, filename)

        if os.path.exists(a_image_path) and os.path.exists(b_image_path):
            continue

        image = Image.open(os.path.join(path)).convert('RGB')
        data = np.array(image)

        height, width, channel = data.shape

        a_image = Image.fromarray(data[:,:width/2].astype(np.uint8))
        b_image = Image.fromarray(data[:,width/2:].astype(np.uint8))

        a_image.save(a_image_path)
        b_image.save(b_image_path)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, scale_size, data_type, skip_pix2pix_processing=False):
        self.root = root
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))

        self.name = os.path.basename(root)
        if self.name in PIX2PIX_DATASETS and not skip_pix2pix_processing:
            pix2pix_split_images(self.root)

        self.paths = glob(os.path.join(self.root, '{}/*'.format(data_type)))
        if len(self.paths) == 0:
            raise Exception("No images are found in {}".format(self.root))
        self.shape = list(Image.open(self.paths[0]).size) + [3]

        self.transform = transforms.Compose([
            transforms.Resize(scale_size), 
            transforms.CenterCrop(scale_size), 
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return len(self.paths)

def get_loader(root, batch_size, scale_size, num_workers=2,
               skip_pix2pix_processing=False, shuffle=True):
    a_data_set, b_data_set = \
        Dataset(root, scale_size, "A", skip_pix2pix_processing), \
        Dataset(root, scale_size, "B", skip_pix2pix_processing)
    
    a_data_loader = torch.utils.data.DataLoader(dataset=a_data_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)
    b_data_loader = torch.utils.data.DataLoader(dataset=b_data_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)
    a_data_loader.shape = a_data_set.shape
    b_data_loader.shape = b_data_set.shape

    return a_data_loader, b_data_loader

class CelebA_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, scale_size, style, style_type, constraint, constraint_type):
        self.root = root
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))

        self.name = os.path.basename(root)
       
        attr_file = os.path.join(self.root, 'list_attr_celeba.txt')
        image_dir = os.path.join(self.root, 'img_align_celeba')
        image_data = read_attr_file(attr_file, image_dir)

        if constraint != '':
            image_data = image_data[image_data[constraint] == constraint_type]

        self.paths = image_data[image_data[style] == str(style_type)]['image_path'].values
        if len(self.paths) == 0:
            raise Exception("No images are found in {}".format(self.root))
        self.shape = list(Image.open(self.paths[0]).size) + [3]

        self.transform = transforms.Compose([
            transforms.Resize(scale_size), 
            transforms.CenterCrop(scale_size), 
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return len(self.paths)

def get_celebA_loader(root, batch_size, scale_size, style_A, style_B, 
                constraint, constraint_type, num_workers=2, shuffle=True):

    if style_B != '':
        a_data_set, b_data_set = \
            CelebA_Dataset(root, scale_size, style_A, 1, constraint, constraint_type), \
            CelebA_Dataset(root, scale_size, style_B, 1, constraint, constraint_type)
    else:
        a_data_set, b_data_set = \
            CelebA_Dataset(root, scale_size, style_A, 1, constraint, constraint_type), \
            CelebA_Dataset(root, scale_size, style_A, -1, constraint, constraint_type)

    a_data_loader = torch.utils.data.DataLoader(dataset=a_data_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)
    b_data_loader = torch.utils.data.DataLoader(dataset=b_data_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)
    a_data_loader.shape = a_data_set.shape
    b_data_loader.shape = b_data_set.shape

    return a_data_loader, b_data_loader

def read_attr_file(attr_path, image_dir):
    f = open(attr_path)
    lines = f.readlines()
    lines = map(lambda line: line.strip(), lines)
    columns = ['image_path'] + lines[1].split()
    lines = lines[2:]

    items = map(lambda line: line.split(), lines)
    df = pd.DataFrame( items, columns=columns )
    df['image_path'] = df['image_path'].map( lambda x: os.path.join( image_dir, x ) )

    return df
