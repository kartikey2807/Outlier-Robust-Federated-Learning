## WGAN-GP model... Create GENERATOR and CRITIC
## Critic takes its weights from the aggregated
## model and generator gets initialized. Try to
## generate good and diverse samples, and  then
## the process continues.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from config import *
from generator import Generator
from utils import penalty, weight_initialization
from discriminator import Critic
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class CustomDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        print(f"Sample size: {x.shape[0]}")
        for k in range(10):
            count = 0
            for i in range(y.shape[0]):
                if y[i] == k:
                    count+=1
            print(f"{k} -> {count}")
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index],self.y[index]

class Server():
    def __init__(self):
        self.critic = Critic(INC,OUT,IMSIZE,LABEL,C_EMBEDDING).to(DEVICE)
        self.gen = Generator(NOISE,CHN,LABEL,G_EMBEDDING).to(DEVICE)
        ## beta (momentum) terms here are taken
        ## from the WGAN-GP paper. They seem to
        ## result in diverse samples They cause
        ## gradients to  move  smoothly without
        ## jumping around.
        self.c_optim = Adam(self.critic.parameters(),lr=LEARNING_RATE,betas=(0.0,0.9))
        self.g_optim = Adam(self.gen.parameters(),lr=LEARNING_RATE,betas=(0.0,0.9))
    def _initialize_critic(self,model_weights):
        self.critic.load_state_dict(model_weights)
    def _initialize_gen(self):
        weight_initialization(self.gen)
    def _get_critic_weights(self):
        return self.critic.state_dict()
    def _train(self,x,y):
        dataset = CustomDataset(x,y)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE)
        ## batches of 64 samples
        for epoch  in range(WGAN_EPOCHS):
            for image,label in loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                ## Train the critic 5 time more
                ## than the generator. Add this
                ## gradient penalty term to the
                ## critic loss.
                self.gen.train()
                self.critic.train()
                for _ in range(CRITIC_ITER):
                    noise = torch.randn(image.shape[0],NOISE)
                    noise = noise.to(DEVICE)
                    fakes = self.gen(noise,label)
                    _,fake_loss = self.critic(fakes,label)
                    _,real_loss = self.critic(image,label)
                    ## Gradient penalty is need
                    ## to produce diverse fakes
                    ## TRIED vanillla WGAN. Did
                    ## not work!
                    gp = penalty(self.critic,image,fakes,label,DEVICE)
                    critic_loss = -(real_loss.mean()-fake_loss.mean()) + LAMBDA_GP*gp
                    self.c_optim.zero_grad()
                    critic_loss.backward(retain_graph=True)
                    self.c_optim.step()
                
                _,fake_loss = self.critic(fakes,label)
                gen_loss = -fake_loss.mean()
                self.g_optim.zero_grad()
                gen_loss.backward()
                self.g_optim.step()
            
            ## EVALUATION: Sample label & noise
            ## and pass through generator. Take
            ## the fake image, and pass through
            ## critic and note the loss.  Print
            ## the fake images.
            if (epoch+1)%100 != 0:
                continue
            self.gen.eval()
            self.critic.eval()
            image,label = next(iter(loader))
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            noise = torch.randn(image.shape[0],NOISE)
            noise = noise.to(DEVICE)
            fakes = self.gen(noise,label)
            _,fake_loss = self.critic(fakes,label)
            _,real_loss = self.critic(image,label)
            critic_loss = -(real_loss.mean() - fake_loss.mean())
            print(f"Test Loss: {critic_loss.item():.3f}")

            if (epoch+1)%WGAN_EPOCHS != 0:
                continue
            print("Generator output .....")
            plt.imshow(make_grid(fakes).permute(1,2,0).cpu().detach().numpy()*0.5+0.5)
            plt.set_cmap("gray")
            plt.axis("off")
            plt.show()