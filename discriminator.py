## The model acts in 2 cases: On the client side
## it is classifier with cross-entropy loss, and
## on the server-side, it is a Critic in WGAN-GP
## And so we have two outputs: 10-dim logits for
## softmax classification and 1 value for Critic

from config import *
import torch
import torch.nn as nn
from torchsummary import summary

class Critic(nn.Module):
    def __init__(self,out:list,imsize:int,label:int,embedding:int):
        super(Critic,self).__init__()
        assert embedding == imsize*imsize
        assert len(out) == 4
        self.imsize = imsize

        self.layer = nn.Sequential( ## conv-leakyrelu block
            self._block(INC+1 ,out[0]), ## //2
            self._block(out[0],out[1]), ## //4
            self._block(out[1],out[2]), ## //8
            self._block(out[2],out[3]), ## //16
            nn.Flatten())
        self.fc_01 = nn.Linear(out[3]*(imsize//16)*(imsize//16),10)
        self.leaky = nn.LeakyReLU(0.2)
        self.fc_02 = nn.Linear(10,1)
        self.embed = nn.Embedding(label,embedding)
    def _block(self,i,o):
        return nn.Sequential(nn.Conv2d(i,o,4,2,1),nn.LeakyReLU(negative_slope=0.2))
    def forward(self,x,y,classify=False):
        image = None
        if classify:
            image = x
        else:
            image = self.embed(y).view(-1,1,self.imsize,self.imsize)
        input = torch.cat([x,image],dim=1)
        input = self.layer(input)
        input_client = self.fc_01(input)
        input_server = self.fc_02(self.leaky(input_client))
        return input_client,input_server