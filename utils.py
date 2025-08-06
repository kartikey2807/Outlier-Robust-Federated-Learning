## Need to create CLIENT_COUNT number of CLIENT
## each client has # SAMPLES_PER_CLIENT of data
## points. Assert that the product of these two
## terms is 60_000 (total number of samples)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from config import *

assert CLIENT_COUNT*SAMPLES_PER_CLIENT == 60000

transform = transforms.Compose([
    transforms.Resize((IMSIZE,IMSIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])])

class MNISTDataset(Dataset):
    def __init__(self,root:str,start:int):
        assert start < CLIENT_COUNT
        self.dataset = MNIST(root=root,train=True,download=True,transform=transform)
        start = start*SAMPLES_PER_CLIENT
        after = start+SAMPLES_PER_CLIENT
        self.dataset.data = self.dataset.data[start:after]
        self.dataset.targets = self.dataset.targets[start:after]
    def __len__(self):
        return len(self.dataset.data) ## = SAMPLES_PER_CLIENT
    def __getitem__(self, index):
        image,label = self.dataset[index]
        return image,label

def penalty(critic,real,fake,label,device):
    B,C,H,W = real.shape
    ## add the real and fake image to create an
    ## interpolated image. Compute gradient wrt
    ## input images and force the gradient norm
    ## to 1 Sample epsilon uniform distribution
    epsilon = torch.rand(B,1,1,1).repeat(1,C,H,W).to(device)
    interpolated_image = (real*epsilon) + (fake*(1-epsilon)) ## combine real / fake
    _, interpolated_score = critic(interpolated_image,label)
    gradient = torch.autograd.grad(
        outputs=interpolated_score,
        inputs =interpolated_image,
        grad_outputs=torch.ones_like(interpolated_score),
        create_graph=True,
        retain_graph=True)[0]
    gradient = gradient.view(gradient.shape[0],-1)
    gradeint_norm = gradient.norm(2,dim=1)
    return ((gradeint_norm-1)**2).mean()

def weight_initialization(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight.data,0.0,0.02)
            nn.init.constant_(m.bias.data,0.0)