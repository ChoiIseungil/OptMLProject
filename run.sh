#!/bin/bash

python main.py -g 2 -d cifar -m ResNet -t 12 -b 3
python main.py -g 2 -d cifar -m ResNet -t 12 -b 4
python main.py -g 2 -d cifar -m ResNet -t 12 -b 5
python main.py -g 2 -d cifar -m ResNet -t 12 -b 6
python main.py -g 2 -d cifar -m ResNet -t 12 -b 7
python main.py -g 2 -d cifar -m ResNet -t 12 -b 8

python main.py -g 2 -d cifar -m ResNet -t 11 -b 3
python main.py -g 2 -d cifar -m ResNet -t 11 -b 4
python main.py -g 2 -d cifar -m ResNet -t 11 -b 5
python main.py -g 2 -d cifar -m ResNet -t 11 -b 6
python main.py -g 2 -d cifar -m ResNet -t 11 -b 7
python main.py -g 2 -d cifar -m ResNet -t 11 -b 8

python main.py -g 2 -d cifar -m ResNet -t 10 -b 3
python main.py -g 2 -d cifar -m ResNet -t 10 -b 4
python main.py -g 2 -d cifar -m ResNet -t 10 -b 5
python main.py -g 2 -d cifar -m ResNet -t 10 -b 6
python main.py -g 2 -d cifar -m ResNet -t 10 -b 7
python main.py -g 2 -d cifar -m ResNet -t 10 -b 8