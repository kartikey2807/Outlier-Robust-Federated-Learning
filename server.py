## Server receives real discriminator gradients
## and trained classifer. No real images/labels
## are shared. It randomly samples labels, pass
## through Generator and computes discriminator
## loss log[1-D(G(z|y*))]. Additionally G(z|y*)
## are passed to classifier, and computes cross
## entropy loss.

from script.models import *
from config import *
from clients import transform

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.nn import BCELoss, CrossEntropyLoss

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

class Server():
    def __init__(self):

        ## define models
        ## define loss
        ## define optimizers
        ## initialize weights

        self.Dnet = Discriminator().to(DEVICE)
        self.Gnet = Generator().to(DEVICE)

        weight_initialization(self.Dnet)
        weight_initialization(self.Gnet)

        self.bcloss = BCELoss()
        self.celoss = CrossEntropyLoss()

        self.Goptim = Adam(self.Gnet.parameters(),
                      LEARNING_RATE,(0.50,0.999))
        self.Doptim = Adam(self.Dnet.parameters(),
                      LEARNING_RATE,(0.50,0.999))
        
        self.datasets = MNIST("MNIST/dataset",
                             train=False,
                             transform=transform,
                             download=True)
        
        self.dataload = DataLoader(self.datasets,10000)

    def train(self,real_gradients,classifier):

        self.Doptim.zero_grad()
        self.Goptim.zero_grad()

        classifier.to(DEVICE)

        for param,real_grad in zip(self.Dnet.parameters(),real_gradients):
            param.grad = real_grad.to(DEVICE)

        noise = torch.randn(BATCH_SIZE,NOISE)
        label = torch.randint(0,LABEL, (32,))
        noise = noise.to(DEVICE)
        label = label.to(DEVICE)

        fakes = self.Gnet(noise,label)
        fake_logit = self.Dnet(fakes)

        Dloss = self.bcloss(fake_logit,torch.zeros_like(fake_logit))

        Dloss.backward(retain_graph=True)
        self.Doptim.step()

        fake_logit = self.Dnet(fakes)

        Gloss = self.bcloss(fake_logit,torch.ones_like (fake_logit)) + \
                self.celoss(classifier(fakes),label)
        
        Gloss.backward(retain_graph=True)
        self.Goptim.step()
    
    def test_global_classifier(self,classifier):
        
        classifier.eval()
        image,label = next(iter(self.dataload))

        image = image.to(DEVICE)
        label = label.to(DEVICE)
        preds = classifier(image,label)

        accuracy = (torch.argmax(preds,dim=1)==label).sum() / 10_000
        print(f"Global model accuracy: {accuracy*100:.2f}%")