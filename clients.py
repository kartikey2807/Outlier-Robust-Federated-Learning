## We work batch-by-batch. Client has two models
## First is Critic f(x), and other is classifier
## H(x). Critic shares the gradients of the real
## loss with the server side. Classifier trained
## with cross-entropy loss -c.log[H(x)]. DP will
## get applied to discriminator first.

from script.models import *
from config import *

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torch.optim import RMSprop,SGD

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

datasets = MNIST(
    ROOT,
    train=True,
    transform=transform,
    download=True
)

class CustomMNISTDataset(Dataset):

    def __init__(self,index):
        '''
        index: partitions global
        dataset and assigns each
        chunk to one client. All
        chunks are idependent of
        each other.
        '''

        global datasets
        self.image = []
        self.label = []

        for i in range(index*SAMPLE_LEN,(index+1)*SAMPLE_LEN):
            self.image.append(datasets[i][0])
            self.label.append(datasets[i][1])
    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):
        return self.image[index],self.label[index]

class Client():

    def __init__(self,index):

        self.Dnet = Critic()
        self.Anet = Classifier()
        self.Dnet.to(DEVICE)
        self.Anet.to(DEVICE)

        self.datasets = CustomMNISTDataset(index)

        self.Doptim = RMSprop(
            lr=LEARNING_RATE_G,
            params=self.Dnet.parameters()
        )

        self.Aoptim = SGD(
            lr=LEARNING_RATE_C,
            params=self.Anet.parameters()
        )

        self.ce_loss = CrossEntropyLoss()

    def train(self,index,flag=False):
        
        '''
        index: defines the batch
        of samples used to train
        critic & classifier nets
        '''

        image = []
        label = []

        for i in range(index*BATCH_SIZE,(index+1)*BATCH_SIZE):
            image.append(self.datasets[i][0])
            label.append(self.datasets[i][1])
        
        ## BATCH
        image = torch.stack (image)
        label = torch.tensor(label)

        self.Doptim.zero_grad()
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        Dloss = -self.Dnet(image).mean()
        Dloss.backward()

        real_grad = []
        for param in self.Dnet.parameters():
            x = param.grad.detach().clone()
            x = x + (1/BATCH_SIZE)*torch.randn_like(x)*SIGMA
            real_grad.append(x)
        
        if flag:
            
            ## taken from the paper
            ## https://arxiv.org/pdf/1607.00133
            NORM = 1.0
            D = math.sqrt(2*math.log10(1.25/DELTA))/EPSILON_2

            aggregate = dict()
            for i, param in enumerate(self.Anet.parameters()):
                aggregate[i] = 0

            for img,lab in zip(image,label):
                self.Aoptim.zero_grad()
                
                img = img.unsqueeze(0)
                lab = lab.unsqueeze(0)
                preds = self.Anet(img)

                Aloss = self.ce_loss(preds,lab)
                Aloss.backward()

                ## clip
                j = 0
                for param in self.Anet.parameters():
                    x = param.grad.detach().clone()
                    x = x/max(1.0,x.norm(p=2)/NORM)
                    aggregate[j] += x ## takes sums
                    j += 1
            
            for i, param in enumerate(self.Anet.parameters()):
                temp = aggregate[i]
                temp = temp + (torch.randn_like(temp)*D*NORM)
                temp = temp/BATCH_SIZE

                param.grad = temp.to(DEVICE)
            
            self.Aoptim.step()
        
        return real_grad

    def eval(self):

        index = torch.randint(0,10,(1,))
        index = index.item()

        image = []
        label = []

        for i in range(index*BATCH_SIZE,(index+1)*BATCH_SIZE):
            image.append(self.datasets[i][0])
            label.append(self.datasets[i][1])
        
        ## BATCH
        image = torch.stack (image)
        label = torch.tensor(label)
        
        self.Anet.eval()
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        preds = self.Anet(image)

        right_pred = (torch.argmax(preds,dim=1)==label).sum()
        return right_pred/BATCH_SIZE
    
    def weight_attack(self,scalar):

        with torch.no_grad():
            for param in self.Dnet.parameters(): ## Gaussian
                param.add_(torch.randn_like(param)*scalar)
            for param in self.Anet.parameters():
                param.add_(torch.randn_like(param)*scalar)
    
    def gradient_attack(self):
        p_grad = []

        with torch.no_grad():
            for param in self.Dnet.parameters(): ## Guassian
                param.grad = torch.randn_like(param.grad)
                p_grad.append(param.grad.detach().clone())
        
        return p_grad