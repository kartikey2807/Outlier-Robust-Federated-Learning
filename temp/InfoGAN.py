## We implement InfoGAN by OpenAI. Used for
## targeted sample generation and you could
## additionally specify other features like
## rotation or brightness. They are part of
## loss objective.

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

class Generator(nn.Module):
    def __init__(self,noise:int=62,label=10):
        super().__init__()
        feature = noise+label
        
        self.layer = nn.Sequential(
        nn.Linear(feature,1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024,128*7*7),
        nn.BatchNorm1d(128*7*7),
        nn.ReLU())

        self.trans = nn.Sequential(self.block(128,64),
        self.block(64,1,last=True))

    def block(self,i,o,last=False):
        if last:
            return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1),nn.Tanh())
        else:
            return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1),
                                 nn.BatchNorm2d(o), nn.ReLU())
    
    def forward(self,z,y):
        ## y here will be one_hot encoded so
        ## its shape will be BATCH_SIZE x 10
        input = torch.cat([z,y],dim=1)
        input = self.layer(input)
        input = input.view(-1,128,7,7)
        return self.trans(input)

class Discriminator(nn.Module):
    def __init__(self,inp:int):
        super().__init__()

        self.layer = nn.Sequential(
        self.block(inp,64),
        self.block(64,128),
        nn.Flatten(),
        nn.Linear(128*7*7,1024),
        nn.LeakyReLU(0.2))

        self.fc1 = nn.Linear(1024,1)
        self.sg1 = nn.Sigmoid()

    def block(self,i,o):
        return nn.Sequential(nn.Conv2d(i,o,4,2,1),nn.LeakyReLU(0.2)) ##/2
    
    def forward(self,x):
        part = self.layer(x) ## BATCH_SIZE x 1024
        return part, self.sg1(self.fc1(part))

class Auxillary(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(1024,128)
        self.active = nn.LeakyReLU(0.2)
        self.layer2 = nn.Linear(128,10)
    
    def forward(self,x):
        input = self.active(self.layer1(x))
        return self.layer2(input)

def weight_initialization(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight.data,0.0,0.02)
            nn.init.constant_(m.bias.data,0.0)

#### HYPER-PARAMETERS ####
INP = 1
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") ###

DNet = Discriminator(INP)
GNet = Generator()
ANet = Auxillary()

DNet = DNet.to(DEVICE)
GNet = GNet.to(DEVICE)
ANet = ANet.to(DEVICE)

## WEIGHT INITIALIZATION: initialize weight
## to have 0 mean / 0.02 standard deviation
weight_initialization(DNet)
weight_initialization(GNet)
weight_initialization(ANet)

Doptim = Adam(DNet.parameters(),LEARNING_RATE)
Goptim = Adam([{"params":GNet.parameters()},{"params":ANet.parameters()}],
               lr=LEARNING_RATE)

bcloss = BCELoss()
celoss = CrossEntropyLoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))])

datasets = MNIST("MNIST/dataset",True, transform=transform,download=True)
dataload = DataLoader(datasets,BATCH_SIZE,shuffle=True)

for epoch in range(EPOCHS): ## ADD TQDM on loop
    for image, label in tqdm(dataload):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        DNet.train()
        GNet.train()
        ANet.train()
        ## Discriminator training ##
        Doptim.zero_grad()
        real_loss = DNet(image)[1]
        
        noise = torch.randn(image.shape[0] , 62)
        label_indices = label
        label = one_hot(label,10)

        noise = noise.to(DEVICE)
        label = label.to(DEVICE)
        label_indices = label_indices.to(DEVICE)

        fakes = GNet(noise,label)
        fake_loss = DNet(fakes)[1]

        loss1 = bcloss(real_loss,torch.ones_like (real_loss))
        loss2 = bcloss(fake_loss,torch.zeros_like(fake_loss))
        combined_loss = loss1 + loss2
        combined_loss.backward(retain_graph=True)
        Doptim.step()## ONE GRADIENT DESCENT STEP

        ## Generator step ##
        Goptim.zero_grad()
        latent, fake_loss = DNet(fakes)

        logit = ANet(latent)
        loss1 = bcloss(fake_loss,torch.ones_like (fake_loss))
        loss2 = celoss(logit,label_indices)

        generator_loss = loss1 + loss2
        generator_loss.backward()
        Goptim.step() ## GRADIENT STEP
    
    image,label = next(iter(dataload))
    noise = torch.randn(image.shape[0],62)

    label = one_hot(label,10)
    
    noise = noise.to(DEVICE)
    image = image.to(DEVICE)
    label = label.to(DEVICE)
    
    
    fakes = GNet(noise,label)
    plt.imshow(make_grid(fakes).permute(1,2,0).cpu().detach().numpy()) ##
    plt.set_cmap("gray")
    plt.axis("off")
    plt.show()