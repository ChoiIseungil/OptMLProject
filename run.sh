#!/bin/bash

python main.py -g 2 -d cifar -m ResNet -t 15 -b 3
python main.py -g 2 -d cifar -m ResNet -t 15 -b 4
python main.py -g 2 -d cifar -m ResNet -t 15 -b 5
python main.py -g 2 -d cifar -m ResNet -t 15 -b 6
python main.py -g 2 -d cifar -m ResNet -t 15 -b 7
python main.py -g 2 -d cifar -m ResNet -t 15 -b 8

python main.py -g 2 -d cifar -m ResNet -t 14 -b 3
python main.py -g 2 -d cifar -m ResNet -t 14 -b 4
python main.py -g 2 -d cifar -m ResNet -t 14 -b 5
python main.py -g 2 -d cifar -m ResNet -t 14 -b 6
python main.py -g 2 -d cifar -m ResNet -t 14 -b 7
python main.py -g 2 -d cifar -m ResNet -t 14 -b 8

python main.py -g 2 -d cifar -m ResNet -t 13 -b 3
python main.py -g 2 -d cifar -m ResNet -t 13 -b 4
python main.py -g 2 -d cifar -m ResNet -t 13 -b 5
python main.py -g 2 -d cifar -m ResNet -t 13 -b 6
python main.py -g 2 -d cifar -m ResNet -t 13 -b 7
python main.py -g 2 -d cifar -m ResNet -t 13 -b 8