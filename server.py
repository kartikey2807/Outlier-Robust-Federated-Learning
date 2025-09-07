## Server receives real discriminator gradients
## and trained classifer. No real images/labels
## are shared. It randomly samples labels, pass
## through Generator and computes discriminator
## loss log[1-D(G(z|y*))]. Additionally G(z|y*)
## are passed to classifier, and computes cross
## entropy loss.

from script.models import *
from config import *

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.nn import BCELoss, CrossEntropyLoss

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

        Dloss = self.bcloss(
            fake_logit,
            torch.zeros_like(fake_logit)
            )

        Dloss.backward(retain_graph = True)
        self.Doptim.step()

        fake_logit = self.Dnet(fakes)

        Gloss = self.bcloss(
            fake_logit,
            torch.ones_like(fake_logit)
            ) + \
            self.celoss(classifier(fakes),label)
        Gloss.backward()

        self.Goptim.step()