## Wasserstein GANs with weight clipping
## to enforce the 1-Lipschitz constraint
## Tune hyper-parameters to handle model
## collapse. Used in the original DP-GAN
## implementation.

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm

from torchsummary import summary
from torchvision.transforms import transforms
from torchvision.datasets  import MNIST
from torch.optim import RMSprop,Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

## Hyper-parameters
ROOT = 'MNIST/dataset'
W_CLIP = 0.01
EPOCHS = 100
BATCH_SIZE = 64
CRITIC_ITERS = 5
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):

    def __init__(self):
        super().__init__()

        IN = 1
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
            nn.ReLU(),
        )

        self.conv1 = self.blocks(64,32)
        self.conv2 = self.blocks(32,16)
        self.conv3 = self.blocks(16,1,last=True)
    
    def blocks(self,i,o,last=False):
        if last:
            return nn.Sequential(
                   nn.ConvTranspose2d(i,o,4,2,1),
                   nn.BatchNorm2d(o),nn.Tanh())
        else:
            return nn.Sequential(
                   nn.ConvTranspose2d(i,o,4,2,1),
                   nn.BatchNorm2d(o),nn.ReLU())
    
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
            nn.init.normal_(param.weight.data,0,0.02)

Cnet = Critic().to(DEVICE)
Gnet = Generator().to(DEVICE)

weight_init(Cnet)
weight_init(Gnet)

Coptim = RMSprop(Cnet.parameters(),LEARNING_RATE)
Goptim = RMSprop(Gnet.parameters(),LEARNING_RATE)

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

datasets = MNIST(ROOT,
           train=True,
           transform=transform,
           download = True)

dataload = DataLoader(datasets,BATCH_SIZE,shuffle=True)

for epoch in range(EPOCHS): ## train for each epoch

    Cnet.train()
    Gnet.train()

    for image,label in tqdm(dataload):
        
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        for _ in range(CRITIC_ITERS):
            Coptim.zero_grad()

            noise = torch.randn(image.shape[0],100)
            noise = noise.to(DEVICE)
            
            fakes = Gnet(noise,label)
            fake_logits = Cnet(fakes)
            real_logits = Cnet(image)
            Closs =-(torch.mean(real_logits)-torch.mean(fake_logits))
            
            Closs.backward(retain_graph=True)
            Coptim.step()
        
            ## weight clipping
            for param in Cnet.parameters():
                param.data.clamp_(-W_CLIP,W_CLIP)

        Goptim.zero_grad()
        Gloss = -torch.mean(Cnet(fakes)) ## f(G(z|y))
        Gloss.backward()
        Goptim.step()