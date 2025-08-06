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
        ## client model is an images classifier
        ## Takes in an image, outputs 10 LOGITS
        ## and computes CROSS-ENTROPY LOSS. THE
        ## MODEL SHARES WEIGHTS AND SOME SAMPLE
        ## TO THE SERVER-SIDE (GLOBAL MODEL).
        self.model = Critic(INC,OUT,IMSIZE,LABEL,C_EMBEDDING).to(DEVICE)
        self.optim = Adam(self.model.parameters(),lr=LEARNING_RATE)
        self.loss_term = CrossEntropyLoss()
        ## The datasets for client[i] takes the
        ## SAMPLES_PER_CLIENT data points. Also
        ## these datasets  are non-intersecting
        temp = MNISTDataset(ROOT,start=start)
        self.datasets = temp.dataset
        self.dataload = DataLoader(self.datasets, batch_size=BATCH_SIZE)
    def _eval_(self):
        #### EVALUATION ####
        ## Calculate accuracy % for  test MNIST
        ## DATASET. THE EXPECTATION IS THAT THE
        ## ACCURACY IS NOT IMPACTED BY A LOT.
        self.model.eval()
        test_dataset = MNIST(root=ROOT,train=False,transform=transform,download=True)
        test_loaders = DataLoader(test_dataset,BATCH_SIZE)
        x,y = next(iter(test_loaders))
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        p,_ = self.model(x,y,classify=True)
        correct_samples = (torch.argmax(p,dim=1)==y).sum()

        accuracy = correct_samples/float(BATCH_SIZE)
        print(f"Acc: {(accuracy*100):.3f}%")
    def _train(self):
        for epoch in range(EPOCHS):
            #### TRAINING ####
            pbar = tqdm(self.dataload)
            self.model.train()
            for image, label in pbar:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                pred, _ = self.model(image,label,classify=True)
                loss_value = self.loss_term(pred,label)

                self.optim.zero_grad() ## GRAD ~ 0
                loss_value.backward()
                self.optim.step()
                pbar.set_postfix({"Loss val":f"{loss_value.item():.3f}"})

    def _init_(self):
        weight_initialization(self.model)
    def _share_weights(self):
        return self.model.state_dict()
    def _share_dataset(self):
        ## BY ALTERING FRAC TERM, WE CAN DECIDE
        ## HOW MANY SAMPLES TO PASS!
        agg_x, agg_y = [],[]
        for i in range(int(FRAC*SAMPLES_PER_CLIENT)):
            x,y = self.datasets[i]
            agg_x.append(x)
            agg_y.append(y)
        return torch.stack(agg_x,dim=0), torch.tensor(agg_y)
    def _poison(self):
        ## WEIGHTS POISONING: replacing weights
        ## with random noise. DEVIATION is this
        ## standard deviation so the new weight
        ## is a standard Gaussian variable with
        ## 0 mean and 'DEVIATION' deviation.
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d,nn.Linear)):
                m.weight.data = DEVIATION*torch.randn_like(m.weight.data)