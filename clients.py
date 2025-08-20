## Client works in 2 phases: one phase is
## image classifier using a cross-entropy
## objective. And, the other phase trains
## the GAN's Discriminator model over the
## real image log[D(x|y)]

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset,DataLoader

from tqdm import tqdm

from script.models import *
from config import *

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])])

class CustomMNIST(Dataset):
    def __init__(self,index):
        
        assert index < COUNT_CLIENT ## index represent each client

        self.datasets = MNIST(ROOT,True,transform)
        start = index*SAMPLES_PER_CLIENT
        final = start+SAMPLES_PER_CLIENT

        self.datasets.data = self.datasets.data[start:final]
        self.datasets.targets = self.datasets.targets[start:final]
    
    def __len__(self):
        return len(self.datasets.data)
    
    def __getitem__(self,index):
        return self.datasets[index]

class Client():
    def __init__(self,index):
        
        ## define models
        ## define loss
        ## define optimizer
        ## initialize weights
        ## create custom dataset
        
        self.Dnet = Discriminator().to(DEVICE)
        self.Anet = Auxillary().to(DEVICE)

        weight_initialization(self.Dnet)
        weight_initialization(self.Anet)

        self.bcloss = BCELoss()
        self.celoss = CrossEntropyLoss()

        self.Doptim = Adam(self.Dnet.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))
        self.Aoptim = Adam(self.Anet.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))

        self.datasets = CustomMNIST(index).datasets
    
    def train(self,start,flag):
        
        image = []
        label = []
        for i in range(start * BATCH_SIZE,(start + 1) * BATCH_SIZE):
            image.append(self.datasets[i][0])
            label.append(self.datasets[i][1])
        
        image = torch.stack (image)
        label = torch.tensor(label)

        self.Doptim.zero_grad()
        self.Aoptim.zero_grad()

        image = image.to(DEVICE)
        label = label.to(DEVICE)

        real_logit = self.Dnet(image,label)

        ## Compute the loss over the real
        ## logits and the 10-D prediction
        Dloss = self.bcloss(real_logit,torch.ones_like(real_logit))
        Dloss.backward()

        ## shares discriminator gradients
        real_grad = [p.grad.clone()for p in self.Dnet.parameters()]

        if flag:
            preds = self.Anet(image)
            Aloss = self.celoss(preds,label)
            Aloss.backward()
            self.Aoptim.step()
        
        return real_grad,label

    def eval(self):
        self.Anet.eval()

        start = torch.randint(0,150,(1,))[0] ## for random sampling
        
        image = []
        label = []
        for i in range(start,start+BATCH_SIZE):
            image.append(self.datasets[i][0])
            label.append(self.datasets[i][1])
        
        image = torch.stack (image)
        label = torch.tensor(label)

        image = image.to(DEVICE)
        label = label.to(DEVICE)
        preds = self.Anet(image)

        accur = (torch.argmax(preds,dim=1)==label).sum()/BATCH_SIZE
        print(f"Accuracy: {accur*100:.2f}%")
    
    def weight_attack(self):
        STD = 1.2

        with torch.no_grad():
            for param in self.Dnet.parameters(): ## gaussian noises
                param.add_(torch.randn_like(param)*STD)
            for param in self.Anet.parameters():
                param.add_(torch.randn_like(param)*STD)