#!/bin/bash

python main.py -g 2 -m AlexNet -t 2**15 -b 2**3
python main.py -g 2 -m AlexNet -t 2**15 -b 2**4
python main.py -g 2 -m AlexNet -t 2**15 -b 2**5
python main.py -g 2 -m AlexNet -t 2**15 -b 2**6
python main.py -g 2 -m AlexNet -t 2**15 -b 2**7
python main.py -g 2 -m AlexNet -t 2**15 -b 2**8

python main.py -g 2 -m AlexNet -t 2**14 -b 2**3
python main.py -g 2 -m AlexNet -t 2**14 -b 2**4
python main.py -g 2 -m AlexNet -t 2**14 -b 2**5
python main.py -g 2 -m AlexNet -t 2**14 -b 2**6
python main.py -g 2 -m AlexNet -t 2**14 -b 2**7
python main.py -g 2 -m AlexNet -t 2**14 -b 2**8

python main.py -g 2 -m AlexNet -t 2**13 -b 2**3
python main.py -g 2 -m AlexNet -t 2**13 -b 2**4
python main.py -g 2 -m AlexNet -t 2**13 -b 2**5
python main.py -g 2 -m AlexNet -t 2**13 -b 2**6
python main.py -g 2 -m AlexNet -t 2**13 -b 2**7
python main.py -g 2 -m AlexNet -t 2**13 -b 2**8