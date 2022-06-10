from logging import raiseExceptions
import torch
import torchvision
from models import AlexNet, ResNet18
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, Subset
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import csv

import os
import argparse


parser = argparse.ArgumentParser(description='CS439 Experiment, by Seungil Lee')

parser.add_argument('-epoch', '-e', default = 200, type = int,
                    help='number of epochs to be trained')
parser.add_argument('-trainsize', '-t', default = 2**15, type = int,
                    help='size of trainset')
parser.add_argument('-batchratio', '-b', default = 2**11, type = int,
                    help='ratio of trainsize to batchsize')
parser.add_argument('-model', '-m', required = True,
                    help='Choose between ResNet and AlexNet')
args = parser.parse_args()

MODEL = args.model
BATCHRATIO = args.batchratio
TRAINSIZE = args.trainsize
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

    print(f"Train set {len(trainloader)}, validation set {len(valloader)}, testloader {len(testloader)}")


    ## Loss and optimizer
    learning_rate = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr= learning_rate) #Adam seems to be the most popular for deep learning

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise NotImplementedError

    if MODEL == 'AlexNet':
        model = AlexNet()
    elif MODEL == 'ResNet':
        model = ResNet18()
    model = model.to(device=device)

    epoch_values = []
    trainloss_values = []
    valloss_values = []
    acc_values = []
    test_value = []

    def save_model(model):
        if not os.path.isdir('saved'):
            os.mkdir('saved')
        torch.save(model.state_dict(), f"saved/{MODEL}_cifar_{BATCHSIZE}:{BATCHRATIO}.pt")
        print(f"saved/{MODEL}_cifar_{BATCHSIZE}:{BATCHRATIO}.pt saved!")

    def save_csv():
        if not os.path.isdir('log'):
            os.mkdir('log')
        with open(f"./log/saved/{MODEL}_cifar_{BATCHSIZE}:{BATCHRATIO}.csv", 'w') as f:
            write = csv.writer(f)
            write.writerow(["Epoch","Train Loss","Validation Loss","Accuracy","Result"])
            write.writerow([epoch_values, trainloss_values, valloss_values, acc_values, test_value])


    def train(epoch):
        best_acc = 0.
        print(f'\nEpoch {epoch}')
        model.train()
        for batch_idx, (data, targets) in enumerate(trainloader):
            data = data.to(device=device)
            targets = targets.to(device=device)
            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores,targets)
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
            
        train_loss = loss_ep/len(trainloader)
            
        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for batch_idx, (data,targets) in enumerate(valloader):
                data = data.to(device=device)
                targets = targets.to(device=device)
                scores = model(data)
                val_loss = criterion(scores,targets)

                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)
                acc = 100.*(num_correct/num_samples)

                print(
                    f"Epoch {epoch}:: Train Loss {train_loss:.4f}, Validation Loss {val_loss:.4f}, Accuracy {acc:.2f} ({num_correct} / {num_samples})"
                )
                
                epoch_values.append(epoch)
                trainloss_values.append(train_loss)
                valloss_values.append(val_loss)
                acc_values.append(acc)
        
        if acc > best_acc:
            save_model(model)
            best_acc = acc

    def test():
        if MODEL == 'AlexNet':
            model = AlexNet()
        elif MODEL == 'ResNet':
            model = ResNet18()

        model.load_state_dict(torch.load("saved/{MODEL}_cifar_{BATCHSIZE}:{BATCHRATIO}.pt")) #loads the trained model
        model = model.to(device=device) #to send the model for training on either cuda or cpu
        model.eval()

        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for batch_idx, (data,targets) in enumerate(testloader):
                data = data.to(device=device)
                targets = targets.to(device=device)
                scores = model(data)

                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)
                acc = float(num_correct) / float(num_samples) * 100

                print("Testing...")
                print(
                    f"Got {num_correct} / {num_samples} with accuracy {acc:.2f}"
                )
                test_value.append(acc)

    for epoch in range(args.epoch):
        train(epoch)
    save_csv()
    test()

if __name__ == '__main__':
    main()