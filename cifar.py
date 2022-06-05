import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, Subset
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim


#creating a dinstinct transform class for the train, validation and test dataset
transform = transforms.Compose([transforms.Resize((227,227)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])])


BATCHRATIO = 2**11
TRAINSIZE = 2**15
VALSIZE = int(TRAINSIZE/3)
BATCHSIZE = int(TRAINSIZE/BATCHRATIO)

trainset = CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
                                        
testset = CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainset, valset = random_split(trainset, [TRAINSIZE, VALSIZE]) #Extracting the 10,000 validation images from the train set

trainloader = DataLoader(trainset, batch_size=BATCHSIZE,
                                          shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=BATCHSIZE,
                                          shuffle=False, num_workers=4)
testloader = DataLoader(testset, batch_size=BATCHSIZE,
                                         shuffle=False, num_workers=4)


#passing the train, val and test datasets to the dataloader
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)