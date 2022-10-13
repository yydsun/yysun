import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose(
    #函数的集合
    [transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))]
)

trainset = torchvision.datasets.FashionMNIST('./data',
    download = True,
    train = True,
    transform = transform)
testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

trainloader = torch.utils.data.Dataloader(trainset, batch_size = 4, shuffle = True, num_workers = 2)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = True, num_workers = 2)
#dataloader可按批取出样本，dataset一次只取一个

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

def matplotlib_imshow(img, one_channel = False):
    if one_channel:
        img = img.mean(dim = 0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap = 'Greys')
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)))
