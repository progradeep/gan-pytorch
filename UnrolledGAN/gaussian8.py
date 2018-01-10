import torch
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

def circle(num_data=1000):
    if num_data % 8 != 0:
        raise ValueError('num_data should be multiple of 8. num_data = {}'.format(num_data))

    center1 = 10
    center2 = math.sqrt(50)
    sigma = 1

    # init data
    d1x = torch.FloatTensor(int(num_data/8), 1)
    d1y = torch.FloatTensor(int(num_data/8), 1)
    d1x.normal_(0,       sigma * 1)
    d1y.normal_(center1, sigma * 1)

    d2x = torch.FloatTensor(int(num_data/8), 1)
    d2y = torch.FloatTensor(int(num_data/8), 1)
    d2x.normal_(center2, sigma * 1)
    d2y.normal_(center2, sigma * 1)

    d3x = torch.FloatTensor(int(num_data/8), 1)
    d3y = torch.FloatTensor(int(num_data/8), 1)
    d3x.normal_(center1, sigma * 1)
    d3y.normal_(0,       sigma * 1)

    d4x = torch.FloatTensor(int(num_data/8), 1)
    d4y = torch.FloatTensor(int(num_data/8), 1)
    d4x.normal_(center2, sigma * 1)
    d4y.normal_(-center2, sigma * 1)

    d5x = torch.FloatTensor(int(num_data/8), 1)
    d5y = torch.FloatTensor(int(num_data/8), 1)
    d5x.normal_(0,        sigma * 1)
    d5y.normal_(-center1, sigma * 1)

    d6x = torch.FloatTensor(int(num_data/8), 1)
    d6y = torch.FloatTensor(int(num_data/8), 1)
    d6x.normal_(-center2, sigma * 1)
    d6y.normal_(-center2, sigma * 1)

    d7x = torch.FloatTensor(int(num_data/8), 1)
    d7y = torch.FloatTensor(int(num_data/8), 1)
    d7x.normal_(-center1, sigma * 1)
    d7y.normal_(0,        sigma * 1)

    d8x = torch.FloatTensor(int(num_data/8), 1)
    d8y = torch.FloatTensor(int(num_data/8), 1)
    d8x.normal_(-center2, sigma * 1)
    d8y.normal_(center2, sigma * 1)

    d1 = torch.cat((d1x, d1y), 1)
    d2 = torch.cat((d2x, d2y), 1)
    d3 = torch.cat((d3x, d3y), 1)
    d4 = torch.cat((d4x, d4y), 1)
    d5 = torch.cat((d5x, d5y), 1)
    d6 = torch.cat((d6x, d6y), 1)
    d7 = torch.cat((d7x, d7y), 1)
    d8 = torch.cat((d8x, d8y), 1)

    d = torch.cat((d1, d2, d3, d4, d5, d6, d7, d8), 0)

    label = torch.IntTensor(num_data).zero_()
    for i in range(8):
        label[i * (int(num_data/8)):(i + 1) * (int(num_data/8))] = i

    return d, label


num_data = 2000
data, label = circle(num_data)
data = data.numpy()
label = label.numpy()

print(data.shape)
print(label.shape)

colors = ['red','orange','yellow','green','blue','purple','brown','black']
fig, ax = plt.subplots()
plt.scatter(data[:,0], data[:,1], c=label, alpha=0.1, cmap=matplotlib.colors.ListedColormap(colors))
plt.axis('equal')
plt.minorticks_on()
plt.grid(True)
plt.xlabel('x', fontsize=14, color='black')
plt.ylabel('y', fontsize=14, color='black')
plt.title('Toy dataset')
plt.savefig('toy.png')
plt.show()