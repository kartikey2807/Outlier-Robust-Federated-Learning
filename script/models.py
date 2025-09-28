## Wasserstein GAN models with DP-GAN setup.
## 1-Lipschitz constraint is satisfoed with
## weight clipping. Classifier is simple NN
## that predicts the label class, given the
## image. The generator is conditioned with
## the label to support targeted generation

import torch
import torch.nn as nn
from torchsummary import summary

## no. channels in MNIST?
IN = 1

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = self.blocks(IN,16)
        self.conv2 = self.blocks(16,32)
        self.conv3 = self.blocks(32,64)

        self.fc1_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*64,10),
        )

    def blocks(self,i,o):
        return nn.Sequential(nn.Conv2d(i,o,4,2,1),
               nn.LeakyReLU(0.2))

    def forward(self,image):

        input = self.conv1(image)
        input = self.conv2(input)
        input = self.conv3(input)
        input = self.fc1_1(input)
        return input

class Critic(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = self.blocks(IN,16)
        self.conv2 = self.blocks(16,32)
        self.conv3 = self.blocks(32,64)

        self.fc1_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*64,100),
            nn.LeakyReLU(0.2),
            nn.Linear(100,1)
        )

    def blocks(self,i,o):
        return nn.Sequential(nn.Conv2d(i,o,4,2,1),
               nn.LeakyReLU(0.2))

    def forward(self,image):

        input = self.conv1(image)
        input = self.conv2(input)
        input = self.conv3(input)
        input = self.fc1_1(input)
        return input

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(10,100)
        self.fc1_1 = nn.Sequential(
            nn.Linear(200, 4*4*64),
            nn.BatchNorm1d(4*4*64),
            nn.ReLU()
        )
        self.conv1 = self.blocks(64,32)
        self.conv2 = self.blocks(32,16)
        self.conv3 = self.blocks(16,IN,last=True)
    
    def blocks(self,i,o,last=False):
        if last:
            return nn.Sequential(
                   nn.ConvTranspose2d(i,o,4,2,1),
                   nn.Tanh())
        else:
            return nn.Sequential(
                   nn.ConvTranspose2d(i,o,4,2,1),
                   nn.BatchNorm2d(o), nn.ReLU())
    
    def forward(self,noise,label):

        input = torch.cat([noise,self.embed(label)],dim=1)
        input = self.fc1_1(input)
        
        input = input.view(-1,64,4,4)
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        return input

def weight_init(model):
    for param in model.modules():
        if isinstance(param,(nn.Linear,nn.Conv2d,nn.ConvTranspose2d)):
            nn.init.normal_(param.weight.data,0.0,0.02)