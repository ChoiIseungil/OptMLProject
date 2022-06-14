import torch.nn as nn

# class FCN(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(FCN, self).__init__()
#         self.layer1 = nn.Linear(input_size*input_size, 512),
#         self.relu1 = nn.ReLU(),
#         self.dropout1 = nn.Dropout(p=0.2),
#         self.layer2 = nn.Linear(512, 256),
#         self.relu2 = nn.ReLU(),
#         self.dropout2 = nn.Dropout(p=0.2),
#         self.layer3 = nn.Linear(256, num_classes),
#         self.softmax = nn.Softmax()


#     def forward(self, x):
#         out = self.dropout1(self.relu1(self.layer1(x)))
#         out = self.dropout2(self.relu2(self.layer2(out)))
#         out = self.softmax(self.layer3(out))
#         return out

def FCN(input_size,num_classes):
    model = nn.Sequential(
        nn.Linear(input_size*input_size, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
        nn.Softmax()
    )
    return model