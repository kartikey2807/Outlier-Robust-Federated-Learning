## Server receives real discriminator gradients
## and labels. No real image are leaked. Server
## computes fake disc loss log[1-D(G(z|y))] and
## Generator loss log[D(G(z|y))]. Model updates
## happen here

from script.models import *
from config import *

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.nn import BCELoss

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

        self.Goptim = Adam(self.Gnet.parameters(),
                      LEARNING_RATE,(0.50,0.999))
        self.Doptim = Adam(self.Dnet.parameters(),
                      LEARNING_RATE,(0.50,0.999))

    def train(self,real_grads,label):

        self.Doptim.zero_grad()

        for param, real_g in zip(self.Dnet.parameters(),real_grads):
            param.grad = real_g.to(DEVICE)
        
        noise = torch.randn(BATCH_SIZE,NOISE)
        noise = noise.to(DEVICE)
        label = label.to(DEVICE)

        fakes = self.Gnet(noise,label)
        fake_logit = self.Dnet(fakes,label)

        ## log[1-D(G(z|y))]
        Dloss = self.bcloss(fake_logit,torch.zeros_like(fake_logit))
        Dloss.backward(retain_graph=True)
        
        self.Doptim.step()

        self.Goptim.zero_grad()
        fake_logit = self.Dnet(fakes,label)

        ## log[D(G(z|y))]
        Gloss = self.bcloss(fake_logit,torch.ones_like (fake_logit))
        Gloss.backward()
        
        self.Goptim.step()