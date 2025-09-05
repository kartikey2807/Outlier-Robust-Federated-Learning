## conditionalGAN: comprises Generator
## and discriminator network. Both the
## networks are conditiioned on labels
## for targeted image generation

import torch
import torch.nn as nn
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1_1 = self.linear(100,256)
        self.fc1_2 = self.linear(10 ,256)
        self.fc2_1 = self.linear(512,512)
        self.fc3_1 = self.linear(512,1024)
        self.fc4_1 = self.linear(1024,784,last=True)

        self.embed = nn.Embedding(10,10)
    
    def linear(self,i,o,last=False):
        if last:
            return nn.Sequential(nn.Linear(i,o),nn.Tanh())
        else:
            return nn.Sequential(nn.Linear(i,o),
                   nn.BatchNorm1d(o),nn.ReLU())
    
    def forward(self,z,y):
        y_embed = self.embed(y)

        input = torch.cat([self.fc1_1(z),self.fc1_2(y_embed)],dim = 1)
        input = self.fc2_1(input)
        input = self.fc3_1(input)
        input = self.fc4_1(input)

        input = input.view(-1,1,28,28)
        return input

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1_1 = self.linear(784,1024)
        self.fc1_2 = self.linear(10, 1024)
        self.fc2_1 = self.linear(1024,512)
        self.fc3_1 = self.linear(512, 256)
        self.fc4_1 = nn.Sequential(nn.Linear(256,1),
                     nn.Sigmoid())
        
        self.embed = nn.Embedding(10,10)

    def linear(self,i,o):
        return nn.Sequential(nn.Linear(i,o),
               nn.LeakyReLU(0.2))

    def forward(self,x):
        # y_embed = self.embed(y)
        x = x.view(-1,784)

        # input = torch.cat([self.fc1_1(x),self.fc1_2(y_embed)],dim=1)
        input = self.fc1_1(x)
        input = self.fc2_1(input)
        input = self.fc3_1(input)
        input = self.fc4_1(input)
        
        return input

class Auxillary(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1_1 = self.linear(784,1024)
        self.fc2_1 = self.linear(1024,512)
        self.fc3_1 = self.linear(512, 256)
        self.fc4_1 = nn.Linear(256,10)
    
    def linear(self,i,o):
        return nn.Sequential(nn.Linear(i,o),
               nn.LeakyReLU(0.2))
    
    def forward(self,x):
        input = self.fc1_1(x.view(-1,784))
        input = self.fc2_1(input)
        input = self.fc3_1(input)
        input = self.fc4_1(input)

        return input

def weight_initialization(model):
    for m in model.modules():
        if isinstance(m,(nn.Linear,nn.BatchNorm1d)):## 0 μ / 0.02 std
            nn.init.normal_(m.weight.data,0.0,0.02)