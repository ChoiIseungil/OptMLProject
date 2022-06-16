# OptMLProject

## Effect of Batch Size Varying by Train Set Size in Mini-Batch Gradient Descent
Mini-project for CS439 Optimization for Machine Learning, EPFL

You can find [a report](./report/report.pdf).

Dependencies are specified in [requirements.txt](./requirements.txt)
All the results are reproducable by running follow command.


```
python main.py -l 0.01 -g 2 -d mnist -m FCN -t 10 -b 8
```
* ```-e``` The number of epochs to be trained
* ```-t``` The size of trainset
* ```-b``` The ratio of trainsize to batchsize
* ```-l``` Learning rate
* ```-d``` The dataset. Choose between cifar and mnist.
* ```-m``` Model to be trained. Choose among ResNet, AlexNet, VGG and FCN
* ```-g``` Number of gpu device you want to use




