## Server receives real loss gradients -del f(x)
## and samples noise and labels, passing through
## generator G(z|y*), and then critic f(G(z|y*))
## Critic is updated 5 times, for each generator
## update. Weight clipping is also applied. Test
## the global classifier accuracy.

from script.models import *
from config import *
from clients import transform

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import RMSprop

from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import MNIST
from torch.nn import CrossEntropyLoss

class Server():

    def __init__(self):

        self.celoss = CrossEntropyLoss()

        self.Dnet = Critic()
        self.Gnet = Generator()
        self.Dnet.to(DEVICE)
        self.Gnet.to(DEVICE)

        self.test = MNIST(
            ROOT,
            train=False,
            transform=transform,
            download=True
        )

        self.Doptim = RMSprop(
            lr=LEARNING_RATE,
            params=self.Dnet.parameters()
        )

        self.Goptim = RMSprop(
            lr=LEARNING_RATE,
            params=self.Gnet.parameters()
        )

        weight_init(self.Dnet)
        weight_init(self.Gnet)
    
    def train(self,classifier,real_grad,flag):
        
        classifier.to(DEVICE)
        
        noise = torch.randn(BATCH_SIZE,100)
        label = torch.randint(0,10,(BATCH_SIZE,))
        noise = noise.to(DEVICE)
        label = label.to(DEVICE)
        fakes = self.Gnet(noise,label)

        if flag == False:
            '''
            If flag is true, generator
            weights gets updated. Else
            discriminator weights gets
            updated.
            '''

            self.Doptim.zero_grad()

            for param,real_g in zip(self.Dnet.parameters(),real_grad):
                param.grad = real_g.to(DEVICE)

            Dloss = self.Dnet(fakes).mean()
            Dloss.backward()
            self.Doptim.step()

            for param in self.Dnet.parameters():
                param.data.clamp_(-CLIP,CLIP)
        else:
            
            self.Goptim.zero_grad()
            
            '''
            Add classifier loss to the
            generator loss to targeted
            sampling for MNIST samples
            '''
            preds = classifier(fakes)
            Aloss = self.celoss(preds,label)
            Gloss = -self.Dnet(fakes).mean()
            overall_loss_val = Gloss + Aloss
           
            overall_loss_val.backward()
            self.Goptim.step()
    
    def test_global_classifier(self,classifier):
        
        classifier.to(DEVICE)
        classifier.eval()

        loader = DataLoader(
            self.test,
            batch_size=10_000
        )
        image,label = next(iter(loader))
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        preds = classifier(image)
        accuracy = (torch.argmax(preds,dim=1)==label).sum() / 10_000
        return accuracy