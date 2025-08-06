## It will be a regular WGAN generator inputs
## noise and label pass through convtranspose
## layers and then outputs a grayscale images
## pixel values between -1 and 1.

import torch
import torch.nn as nn
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self,noise:int,chn:list,label:int,embedding:int):
        super(Generator,self).__init__()

        ## Pass through dense layers, reshape
        ## and then pass through convtranspos
        ## batchnorm-relu blocks. Last layers
        ## will have torch.tanh() activations
        assert len(chn) == 4
        self.chn = chn

        self.fc_01 = nn.Linear(noise+embedding,chn[0]*4*4)
        self.batch = nn.BatchNorm1d(chn[0]*4*4)
        self.relu_ = nn.ReLU()
        self.layer = nn.Sequential( ## ConvTranspose-ReLU
            self._block(chn[0],chn[1]), ## x8
            self._block(chn[1],chn[2]), ## x16
            self._block(chn[2],chn[3]), ## x32
            self._block(chn[3],1,last=True)) ## x64
        self.embed = nn.Embedding(label,embedding)
    def _block(self,i,o,last=False):
        if last:
            return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1),
                                 nn.Tanh())
        else:
            return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1),nn.BatchNorm2d(o),
                                 nn.ReLU())
    def forward(self,z,y):
        input = torch.cat([z,self.embed(y)],dim=1)
        input = self.fc_01(input)
        input = self.batch(input)
        input = self.relu_(input)
        input = input.view(-1,self.chn[0],4,4)
        input = self.layer(input)
        return input