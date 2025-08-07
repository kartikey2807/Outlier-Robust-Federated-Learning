#### CLIENT-SIDE ####
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from discriminator import Critic
from config import *
from utils  import *
from torchvision.datasets import MNIST

class Client():
    def __init__(self,start:int):
        self.model = Critic(OUT,IMSIZE,LABEL,C_EMBEDDING)
        self.model = self.model.to(DEVICE)
        self.optim = Adam(self.model.parameters(), lr=LR)
        self.loss_term = CrossEntropyLoss()

        ## CLIENT DATASET ##
        self.datasets = MNISTDataset(ROOT,start).dataset
        self.dataload = DataLoader(self.datasets,batch_size=BATCH_SIZE,shuffle=True)
    def _train(self):
        for epoch in range(EPOCHS):
            #### TRAINING ####
            pbar = tqdm(self.dataload)
            self.model.train()
            for image, label in pbar:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                pred, _ = self.model(image,label,classify=True)
                loss_values = self.loss_term(pred,label)
                
                self.optim.zero_grad() ## GRAD ~ 0
                loss_values.backward()
                self.optim.step()

                loss = loss_values.item()
                pbar.set_postfix({"Loss":f"{loss:.3f}"})
    def _share_dataset(self):
        agg_x = []
        agg_y = []
        total_samples = FRAC*SAMPLES_PER_CLIENT
        for i in range(int(total_samples)):
            x,y = self.datasets[i]
            agg_x.append(x)
            agg_y.append(y)
        
        tensor_x = torch.stack (agg_x)
        tensor_y = torch.tensor(agg_y)
        return tensor_x,tensor_y
    def _poison(self):
        ## WEIGHTS POISONING: replacing weights
        ## with random noise. DEVIATION is this
        ## standard deviation so the new weight
        ## is a standard Gaussian variable with
        ## 0 mean and 'DEVIATION' deviation.
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d,nn.Linear)):
                m.weight.data = torch.randn_like(m.weight.data)
    def _eval_(self):
        #### EVALUATION ####
        ## Calculate accuracy on the test MNIST
        self.model.eval()
        test_dataset = MNIST(root=ROOT,train=False,transform=transform,download=True)
        test_loaders = DataLoader(test_dataset,BATCH_SIZE)
        x,y = next(iter(test_loaders))
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        p,_ = self.model(x,y,classify=True)
        correct_samples = (torch.argmax(p,dim=1)==y).sum()
   
        accuracy = correct_samples/float(BATCH_SIZE)
        print(f"Accuracy: {(accuracy*100):.3f}%")
    def _init_(self):
        weight_initialization(self.model)
    def _share_weights(self):
        return self.model.state_dict()