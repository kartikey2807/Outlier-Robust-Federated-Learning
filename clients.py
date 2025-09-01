## We work batch-by-batch. Client works is two
## phases: one computes the real component for
## Discriminator loss log[D(x|y)] and the next
## computes the cross-entropy loss -clog[h(x)]
## We share gradients and labels to the server

from script.models import *
from config import *

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam,SGD
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset,DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])])

class CustomMNIST(Dataset):

    def __init__(self,index):

        self.dataset = MNIST(ROOT,True,transform=transform,download=True)
        start = index*SAMPLES_PER_CLIENT
        final = start+SAMPLES_PER_CLIENT

        self.images = []
        self.labels = []

        for i in range(start,final):
            self.images.append(self.dataset[i][0])
            self.labels.append(self.dataset[i][1])

        self.images = torch.stack (self.images)
        self.labels = torch.tensor(self.labels)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        return self.images[index],self.labels[index]

class Client():

    def __init__(self,index):

        ## Cannot exceed total client instance
        assert index < COUNT_CLIENT

        self.Dnet = Discriminator().to(DEVICE)
        self.Anet = Auxillary().to(DEVICE)

        weight_initialization(self.Dnet)
        weight_initialization(self.Anet)

        self.bcloss = BCELoss()
        self.celoss = CrossEntropyLoss()

        self.Doptim = Adam(
                      self.Dnet.parameters(),
                      LEARNING_RATE,
                      (0.500,0.999))
        self.Aoptim = Adam(
                      self.Anet.parameters(),
                      LEARNING_RATE,
                      (0.500,0.999))

        self.datasets = CustomMNIST(index) ## transformed image and label
        self.STD = 1.2
    
    def train(self,start,flag=False):
        
        image = []
        label = []
        for i in range(start*BATCH_SIZE,(start+1)*BATCH_SIZE):
            image.append(self.datasets.images[i])
            label.append(self.datasets.labels[i])
        
        image = torch.stack (image).to(DEVICE)
        label = torch.tensor(label).to(DEVICE)
        
        self.Doptim.zero_grad()
        self.Aoptim.zero_grad()

        real_logit = self.Dnet(image,label)
        
        Dloss = self.bcloss(
                real_logit,
                torch.ones_like(real_logit)
                )
        
        Dloss.backward()

        real_grad = []
        for param in self.Dnet.parameters():
            p = param.grad.detach().clone()

            real_grad.append(p)

        if flag:
            preds = self.Anet(image)
            Aloss = self.celoss(preds,label)
            Aloss.backward()
            self.Aoptim.step()

        return real_grad,label
    
    def eval(self):
        
        self.Anet.eval()
        start = torch.randint(0,150,(1,))[0]
        
        image = []
        label = []
        for i in range(start, start + BATCH_SIZE):
            image.append(self.datasets.images[i])
            label.append(self.datasets.labels[i])
        
        image = torch.stack (image).to(DEVICE)
        label = torch.tensor(label).to(DEVICE)

        preds = self.Anet(image)
        accuracy = (torch.argmax(preds,dim=1)==label).sum() / BATCH_SIZE
        print(f"Accuracy: {accuracy*100:.2f}%")
    
    def weight_attack(self):

        with torch.no_grad():
            for param in self.Dnet.parameters(): ## std. normal gaussian
                param.add_(torch.randn_like(param)*self.STD)
            for param in self.Anet.parameters():
                param.add_(torch.randn_like(param)*self.STD)

    def gradient_attack(self):
        posionous_grads = []

        with torch.no_grad():
            for param in self.Dnet.parameters(): ## std. noraml gaussian
                param.grad = torch.randn_like(param.grad)* self.STD
                posionous_grads.append(param.grad.detach().clone())
        
        return posionous_grads,torch.tensor([])