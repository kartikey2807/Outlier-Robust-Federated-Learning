## We implement CGAN model. The models are
## used for targeted image generation. The
## Generator/Discriminator are conditioned
## on the input label.

## E[log[D(x|y)]] +  E[log[1-D(G(z|y)|y)]]
## E[log(D(G(z|y)|y))]

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch.nn import BCELoss,CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from torchsummary import summary
from torchvision.utils import make_grid

from tqdm import tqdm
import matplotlib.pyplot as plt

## Hyper-parameters
EMBEDDING  = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
EPOCH = 50
NOISE = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1_1 = self.linear(100,256)
        self.fc1_2 = self.linear(10, 256)
        self.fc2_1 = self.linear(512,512)
        self.fc3_1 = self.linear(512,1024)
        self.fc4_1 = self.linear(1024,784,last=True)

        self.embed = nn.Embedding(10,EMBEDDING)

    def linear(self,i,o,last=False):
        if last:
            return nn.Sequential(nn.Linear(i,o),nn.Tanh())
        else:
            return nn.Sequential(nn.Linear(i,o),
                   nn.BatchNorm1d(o),nn.ReLU())
    
    def forward(self,z,y):
        y_embed = self.embed(y)

        input = torch.cat([self.fc1_1(z),self.fc1_2(y_embed)],dim=1)
        
        input = self.fc2_1(input)
        input = self.fc3_1(input)
        input = self.fc4_1(input) ## x784

        return input.view(-1,1,28,28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(10,EMBEDDING)

        self.fc1_1 = self.linear(784,1024)
        self.fc1_2 = self.linear(10, 1024)
        self.fc2_1 = self.linear(2048,512)
        self.fc3_1 = self.linear(512, 256)
        self.fc4_1 = nn.Sequential(nn.Linear(256,1),
                     nn.Sigmoid())

    def linear(self,i,o):
        return nn.Sequential(nn.Linear(i,o),
               nn.LeakyReLU(0.2))
    
    def forward(self,x,y):
        y_embed = self.embed(y)
        x = x.view(-1,784)

        input = torch.cat([self.fc1_1(x),self.fc1_2(y_embed)],dim=1)
        input = self.fc2_1(input)
        input = self.fc3_1(input)
        input = self.fc4_1(input)

        return input

def weight_initialization(model):
    for m in model.modules():
        if isinstance(m,(nn.Linear,nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data,0.0,0.02)

Dnet = Discriminator().to(DEVICE)
Gnet = Generator().to(DEVICE)

weight_initialization(Dnet)
weight_initialization(Gnet)

bcloss = BCELoss()

Doptim = Adam(Dnet.parameters(),lr=LEARNING_RATE)
Goptim = Adam(Gnet.parameters(),lr=LEARNING_RATE)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])])

datasets = MNIST('/MNIST/dataset',True,transform=transform,download=True)
dataload = DataLoader(datasets,BATCH_SIZE,shuffle=True)

for epoch in range(EPOCH):

    Dnet.train()
    Gnet.train()
    for image,label in tqdm(dataload):
        Doptim.zero_grad()

        image = image.to(DEVICE)
        label = label.to(DEVICE)
        noise = torch.randn(image.shape[0],NOISE)
        noise = noise.to(DEVICE)
        fakes = Gnet(noise,label)

        real_logit = Dnet(image,label)
        fake_logit = Dnet(fakes,label)

        real_loss  = bcloss(real_logit,torch.ones_like (real_logit))
        fake_loss  = bcloss(fake_logit,torch.zeros_like(fake_logit))
        disc_loss  = real_loss + fake_loss
        
        disc_loss.backward(retain_graph=True)
        Doptim.step()

        Goptim.zero_grad()

        fake_logit = Dnet(fakes,label)
        fake_loss  = bcloss(fake_logit,torch.ones_like (fake_logit))

        fake_loss.backward()
        Goptim.step()