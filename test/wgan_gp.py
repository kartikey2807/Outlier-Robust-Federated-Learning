import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import make_grid

import warnings
warnings.filterwarnings("ignore")

## Hyper-parameters
INC = 1
OUT = [16,32,64,128]
IMSIZE = 64
LABEL = 10
C_EMBEDDING = 4096
NOISE = 256
CHN = [128,64,32,16]
G_EMBEDDING = 256
LEARNING_RATE = 0.00005
BATCH_SIZE = 64
EPOCHS = 500
LAMBDA = 10
CRITIC_ITER = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self,inc:int,out:list,imsize:int,label,embedding):
        super(Critic,self).__init__()
        ## The image is concatenated with label
        ## embeddings. Passed through  bunch of
        ## conv layers and then to dense layers
        ## HERE WE WILL KEEP 2 DENSE LAYERS FOR
        ## 10 LOGITS AND FINALLY FOR 1 LOGIT. 1
        ## LOGIT IS USED TO TRAIN CRITIC IN GAN
        ## AND 10 LOGITS ARE  USED TO TRAIN THE
        ## CLIENT-SIDE neural nets
        assert embedding == imsize*imsize
        assert len(out) == 4
        self.imsize = imsize

        self.layer = nn.Sequential( ## Conv-LeakyReLU block
            self._block(inc+1, out[0]), ## //2
            self._block(out[0],out[1]), ## //4
            self._block(out[1],out[2]), ## //8
            self._block(out[2],out[3]), ## //16
            nn.Flatten())
        self.fc_01 = nn.Linear(out[3]*(imsize//16)*(imsize//16),10)
        self.leaky = nn.LeakyReLU(0.2)
        self.fc_02 = nn.Linear(10,1)
        self.embed = nn.Embedding(label,embedding)
    
    def _block(self,i,o):
        return nn.Sequential(nn.Conv2d(i,o,4,2,1),
                             nn.LeakyReLU(0.2))
    
    def forward(self,x,y):
        input = torch.cat([x,self.embed(y).view(-1,1,self.imsize,self.imsize)],dim=1)
        input = self.layer(input)
        input_client = self.fc_01(input)
        input_server = self.fc_02(self.leaky(input_client))
        return input_client,input_server

class Generator(nn.Module):
    def __init__(self,noise:int,chn:list,label,embedding):
        super(Generator,self).__init__()
        ## takes in noise and concat with label
        ## embedding. reshape, and pass through
        ## conv-transpose layers The last layer
        ## should have tanh() output to put the
        ## pixels between -1 and 1.
        assert len(chn) == 4
        self.chn = chn

        self.fc_01 = nn.Sequential(
            nn.Linear(noise+embedding,chn[0]*4*4),
            nn.BatchNorm1d(chn[0]*4*4),nn.ReLU())
        self.layer = nn.Sequential( ## convtranspose-batchnorm-relu
            self._block(chn[0],chn[1]), ## x8
            self._block(chn[1],chn[2]), ## x16
            self._block(chn[2],chn[3]), ## x32
            self._block(chn[3],1,last=True)) ## x64
        self.embed = nn.Embedding(label,embedding)
    
    def _block(self,i,o,last=False):
        if last:
            return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1),
                                 nn.Tanh()) ## output between -1 / 1 for image tensors
        else:
            return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1),
                                 nn.BatchNorm2d(o), nn.ReLU())
    def forward(self,z,y):
        input = self.fc_01(torch.cat([z,self.embed(y)],dim=1))
        return self.layer(input.view(-1,self.chn[0],4,4))

def weight_initialization(model):
    for m in model.modules():
        if isinstance(m,[nn.Conv2d,nn.ConvTranspose2d]):
            nn.init.normal_(m.weight.data,0.0,0.02)
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight.data,0.0,0.02)
            nn.init.constant_(m.bias.data,0.0)

def gradient_penalty(critic,real,fake,label,device):
    B,C,H,W = real.shape
    ## take epsilon between 0 and 1. combine to
    ## create an interpolated image. Pass image
    ## through critic and force the norm of the
    ## gradeint to be close to 1.
    epsilon = torch.rand(B,1,1,1).repeat(1,C,H,W).to(device)
    interpolated_image = (real*epsilon) + (fake*(1-epsilon))
    _, interpolated_score = critic(interpolated_image,label) ## server-side o/p scores
    gradient = torch.autograd.grad(
        outputs=interpolated_score,
        inputs =interpolated_image,
        grad_outputs=torch.ones_like(interpolated_score),
        create_graph=True,
        retain_graph=True)[0]
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim=1)
    return ((gradient_norm-1)**2).mean()

transform = transforms.Compose([
    transforms.Resize((IMSIZE,IMSIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])])

train_dataset = MNIST(root='MNIST/dataset/',transform=transform,download=True) ## load
loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

critic = Critic(INC,OUT,IMSIZE,LABEL,C_EMBEDDING).to(DEVICE)
gen = Generator(NOISE,CHN,LABEL,G_EMBEDDING).to(DEVICE)
optim_critic = Adam(critic.parameters(),lr=LEARNING_RATE,betas=(0.0,0.9))
optim_gen = Adam(gen.parameters(),lr=LEARNING_RATE,betas=(0.0,0.9))

for epoch in range(EPOCHS):
    pbar = tqdm(loader,desc=f"Epoch:{epoch+1}/{EPOCHS}")
    ## Sample noise, pass through generator and
    ## get fake samples. Pass the fake and real
    ## samples to Critic and calculate the loss
    for i, (x,y) in enumerate(pbar):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        ## train
        gen.train()
        critic.train()
        for _ in range(CRITIC_ITER):
            z = torch.randn(x.shape[0],NOISE)
            z = z.to(DEVICE)
            fake = gen(z,y)
            _,fake_loss = critic(fake,y)
            _,real_loss = critic(x,y)
            gp = gradient_penalty(critic,x,fake,y,DEVICE)
            critic_loss = -(real_loss.mean() - fake_loss.mean()) + LAMBDA*gp## penalty
            optim_critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            optim_critic.step()
        
        _,fake_loss = critic(fake,y)
        gen_loss = -fake_loss.mean()
        optim_gen.zero_grad()
        gen_loss.backward()
        optim_gen.step()
    
    ## eval
    gen.eval()
    critic.eval()
    x,y = next(iter(loader))
    z = torch.randn(x.shape[0] , NOISE)
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    z = z.to(DEVICE)
    sample_fake = gen(z,y)
    _,real_loss = critic(x,y)
    _,fake_loss = critic(sample_fake,y)
    t_loss = -(real_loss.mean() - fake_loss.mean())
    print(f"Test Loss: {t_loss:.4f}")
    plt.imshow(make_grid(sample_fake).permute(1,2,0).cpu().detach().numpy()*0.5+0.5, cmap='gray')
    plt.axis('off')
    plt.show()