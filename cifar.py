import torch
import torchvision
from models import AlexNet
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, Subset
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim
import torch.nn as nn

BATCHRATIO = 2**11
TRAINSIZE = 2**15
VALSIZE = int(TRAINSIZE/3)
BATCHSIZE = int(TRAINSIZE/BATCHRATIO)

def main():
    #creating a dinstinct transform class for the train, validation and test dataset
    transform = transforms.Compose([transforms.Resize((227,227)), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])])

    trainset = CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
                                            
    testset = CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

    trainset, valset, _ = random_split(trainset, [TRAINSIZE, VALSIZE, len(trainset)-TRAINSIZE-VALSIZE]) #Extracting the 10,000 validation images from the train set

    trainloader = DataLoader(trainset, batch_size=BATCHSIZE,
                                            shuffle=True, num_workers=4) 
    valloader = DataLoader(valset, batch_size=64,
                                            shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=64,
                                            shuffle=False, num_workers=4)

    print(f"Train set f{len(trainloader)}, validation set f{len(valloader)}, testloader f{len(testloader)}")


    device = torch.device('cuda')

    model = AlexNet() #to compile the model
    model = model.to(device=device) #to send the model for training on either cuda or cpu

    ## Loss and optimizer
    learning_rate = 1e-4 #I picked this because it seems to be the most used by experts
    load_model = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= learning_rate) #Adam seems to be the most popular for deep learning

    loss_values = []
    acc_values = []

    for epoch in range(50):
        loss_ep = 0
        
        for _, (data, targets) in enumerate(trainloader):
            data = data.to(device=device)
            targets = targets.to(device=device)
            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores,targets)
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
        running_loss = loss_ep/len(trainloader)
        print(f"Loss in epoch {epoch} :::: {running_loss}")
        loss_values.append(running_loss)

        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for _, (data,targets) in enumerate(valloader):
                data = data.to(device=device)
                targets = targets.to(device=device)
                scores = model(data)
                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)
            acc = float(num_correct) / float(num_samples) * 100
            print(
                f"Got {num_correct} / {num_samples} with accuracy {acc:.2f}"
            )
        acc_values.append(acc)


    torch.save(model.state_dict(), f"saved/alexnet_cifar_{BATCHSIZE}:{BATCHRATIO}.pt") #SAVES THE TRAINED MODEL

    num_correct = 0
    num_samples = 0
    for batch_idx, (data,targets) in enumerate(testloader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        scores = model(data)
        _, predictions = scores.max(1)
        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)
    print(
        f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
    )

if __name__ == '__main__':
    main()