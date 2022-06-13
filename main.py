from logging import raiseExceptions
import torch
from models import AlexNet, ResNet18
from torchvision.datasets import CIFAR10, MNIST

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import csv
  
import os
import argparse
import time


parser = argparse.ArgumentParser(description='CS439 Experiment, by Seungil Lee')

parser.add_argument('-epoch', '-e', default = 100, type = int,
                    help='number of epochs to be trained')
parser.add_argument('-trainsize', '-t', default = 15, type = int,
                    help='size of trainset')
parser.add_argument('-batchratio', '-b', default = 11, type = int,
                    help='ratio of trainsize to batchsize')
parser.add_argument('-data', '-d', required = True,
                    help='Choose between cifar and mnist')
parser.add_argument('-model', '-m', required = True,
                    help='Choose between ResNet and AlexNet')
parser.add_argument('-gpu', '-g', required = True, default = 2, type = int,
                    help='Choose between ResNet and AlexNet')
parser.add_argument('-lr', '-l', default = 1e-2, type = float,
                    help='Learning Rate')
args = parser.parse_args()

MODEL = args.model
BATCHRATIO = 2**args.batchratio
TRAINSIZE = 2**args.trainsize
VALSIZE = int(TRAINSIZE/4)
BATCHSIZE = int(TRAINSIZE/BATCHRATIO)
EPOCH = int(2**15/TRAINSIZE)*args.epoch

if torch.cuda.is_available():
    DEVICE = torch.device(f'cuda:{args.gpu}')
else:
    raise NotImplementedError

def init_model(modelname = None):
    if MODEL == 'AlexNet':
        model = AlexNet()
    elif MODEL == 'ResNet':
        model = ResNet18()
    if modelname is not None:
        model.load_state_dict(torch.load(modelname))
    model = model.to(device=DEVICE)
    return model

def save_model(model):
    if not os.path.isdir('saved'):
        os.mkdir('saved')
    torch.save(model.state_dict(), f"saved/{MODEL}_cifar_{BATCHSIZE}_{TRAINSIZE}.pt")
    print(f"{MODEL}_cifar_{BATCHSIZE}_{TRAINSIZE}.pt saved!")

def save_csv(epoch, trainloss, valloss, acc, test, runningtime):
    if not os.path.isdir('log'):
        os.mkdir('log')
    with open(f"./log/{MODEL}_cifar_{BATCHSIZE}_{TRAINSIZE}.csv", 'w') as f:
        write = csv.writer(f)
        write.writerow(["Epoch","Train Loss","Validation Loss","Validation Accuracy","Test Accuracy","Running Time"])
        write.writerow([epoch[0],trainloss[0],valloss[0],acc[0],test[0],runningtime[0]])
        for i in range(1,len(epoch)):
            write.writerow([epoch[i], trainloss[i], valloss[i], acc[i]])

def load_dataset():
    # 다름
    transform = transforms.Compose([transforms.Resize((227,227)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])])
    if args.data =="cifar":
        trainset = CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
                                            
        testset = CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    if args.data =="mnist":
        trainset = MNIST(root='./data', train=True,
                                            download=True, transform=transform)
                                            
        testset = MNIST(root='./data', train=False,
                                        download=True, transform=transform)  
    trainset, valset, _ = random_split(trainset, [TRAINSIZE, VALSIZE, len(trainset)-TRAINSIZE-VALSIZE])

    return trainset, valset, testset            


def main():
    trainset, valset, testset = load_dataset()

    print(f"Train {len(trainset)}, Validation {len(valset)}, Test {len(testset)}")

    trainloader = DataLoader(trainset, batch_size=BATCHSIZE,
                                            shuffle=True, num_workers=4) 
    valloader = DataLoader(valset, batch_size=2048,
                                            shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=2048,
                                            shuffle=False, num_workers=4)


    model = init_model()
    ## Loss and optimizer
    #다름
    learning_rate = args.lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr= learning_rate)

    epoch_values = []
    trainloss_values = []
    valloss_values = []
    acc_values = []
    test_value = []
    running_time_value = []

    def train(epoch):
        for epoch in range(EPOCH):
            best_acc = 0.
            best_val_loss = 10
            epochs_no_improve = 0
            patience = 2
            train_loss_ep = 0
            print(f'\n[Epoch {epoch}]')
            model.train()
            print("Training...")
            for _, (data, targets) in enumerate(trainloader):
                data = data.to(device=DEVICE)
                targets = targets.to(device=DEVICE)
                optimizer.zero_grad()
                scores = model(data)
                loss = criterion(scores,targets)
                loss.backward()
                optimizer.step()
                train_loss_ep += loss.item()
                
            train_loss = train_loss_ep/len(trainloader)
            
            model.eval()
            print("Validating...")

            with torch.no_grad():
                num_correct = 0
                num_samples = 0
                val_loss_ep = 0
                for _, (data,targets) in enumerate(valloader):
                    data = data.to(device=DEVICE)
                    targets = targets.to(device=DEVICE)
                    scores = model(data)
                    loss = criterion(scores,targets)
                    
                    val_loss_ep += loss.item()

                    _, predictions = scores.max(1)
                    num_correct += (predictions == targets).sum()
                    num_samples += predictions.size(0)

                acc = 100.*(num_correct/num_samples).item()
                val_loss = val_loss_ep/len(valloader)

                print(
                    f"[Epoch {epoch}] Train Loss {train_loss:.4f}, Validation Loss {val_loss:.4f}, Accuracy {acc:.2f} ({num_correct} / {num_samples})"
                )
                    
                epoch_values.append(epoch)
                trainloss_values.append(train_loss)
                valloss_values.append(val_loss)
                acc_values.append(acc)

            #Early Stopping
            if val_loss > best_val_loss:
                epochs_no_improve += 1
                if epochs_no_improve > patience:
                    print(f"Early stopped at epoch {epoch}")
                    break
            else: 
                epochs_no_improve = 0
                best_val_loss = val_loss

            if acc > best_acc:
                print(f"New top accuracy achieved {acc:.2f}! Saving the model...")
                save_model(model)
                best_acc = acc

    def test():
        model = init_model(f"saved/{MODEL}_cifar_{BATCHSIZE}_{TRAINSIZE}.pt")
        model.eval()
        
        print("Testing...")

        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for _, (data,targets) in enumerate(testloader):
                data = data.to(device=DEVICE)
                targets = targets.to(device=DEVICE)
                scores = model(data)

                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)
            acc = 100.*(num_correct/num_samples).item()

            print(
                f"Got {num_correct} / {num_samples} with accuracy {acc:.2f}"
            )
            test_value.append(acc)

    # Main Loop
    start_time = time.time()
    train(EPOCH)
    running_time = time.time() - start_time
    running_time_value.append(running_time)
    test()
    save_csv(epoch_values, trainloss_values, valloss_values, acc_values, test_value, running_time_value)


if __name__ == '__main__':
    main()