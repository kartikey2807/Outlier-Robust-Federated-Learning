## Outlier-robust Federated Learning setup: For
## each batch sample client grads and implement
## KRUM. You will have a trusted gradient, with
## high probability of being benign. GAN forces
## similar distribution. Classifier forces them
## to be correct labels.

from config import *

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision.utils import make_grid
from clients import Client
from server  import Server

def aggregate(weights):
    num = len(weights)
    return {k: sum(w[k] for w in weights)/num for k in weights[0].keys()}

def krum(gradient):
    ## Apply KRUM
    dist_mat = torch.zeros(COUNT_CLIENT,COUNT_CLIENT)
    for c1 in range(COUNT_CLIENT):

        for c2 in range(COUNT_CLIENT):
            
            total = 0
            for x,y in zip(gradient[c1],gradient[c2]):
                total += (x-y).norm(p=2)
            
            dist_mat[c1,c2] = total
            dist_mat[c2,c1] = total
    
    score = []
    for c3 in range(COUNT_CLIENT):
        dist = torch.sort(dist_mat[c3,:])[0]
        score.append(torch.sum(dist[:COUNT_CLIENT-MAX_BYZANTINE]))
    
    return torch.tensor(score)

server  = Server()
clients = []

for i in  range(COUNT_CLIENT):
    clients.append(Client(i))

stable = clients[0]
stable.Anet.train()
for i in range(75):
    stable.train(i,True)

for epoch in tqdm(range(EPOCH)):
    server.Dnet.train()
    server.Gnet.train()
    stable.Dnet.train()
    for i in range(75):

        for _ in range(CRITIC_ITER):

            stable.Dnet.load_state_dict(server.Dnet.state_dict())
            real_grad = stable.train(i)
            
            server.train(stable.Anet,
                real_grad,flag=False
            )
        
        server.train(stable.Anet,[],flag=True)

for _ in range(ROUNDS):

    weights = []

    noise = torch.randn(64,100)
    label = torch.randint(0,10,(64,))
    
    noise = noise.to(DEVICE)
    label = label.to(DEVICE)
    fakes = server.Gnet(noise,label)

    byzantines = torch.randperm(COUNT_CLIENT)[:MAX_BYZANTINE + 0]

    for j,client in enumerate(clients):

        if j in byzantines:

            print(f"Client @ {j}")
            print("POISONED")
            client.train(0,True)
            client.weight_attack(scalar=1.5)

        else:

            print(f"Client @ {j}")
            print("TRAINING")

            client.Anet.train()
            for i in range(75):
                client.train(i,True)
        
            accuracy = client.eval()
            print(f"Accuracy: {accuracy*100 :.2f}%")
    
    benign = None
    malicious = []
    for j,client in enumerate(clients):

        preds = client.Anet(fakes)
        preds = preds.argmax(dim=1)
        accuracy = (preds==label).sum() / BATCH_SIZE

        if accuracy < THRESHOLD:
            malicious.append(j)

        else:
            benign = j
            weights.append(
                client.Anet.state_dict()
            )
    
    print("MALICIOUS CLIENTS")
    print(malicious)
    
    average_weights = aggregate(weights)

    for client in clients:
        client.Anet.load_state_dict(average_weights)
    
    for epoch in tqdm(range(EPOCH)):
        server.Dnet.train()
        server.Gnet.train()

        for i in range(75):

            ## For each batch, get the
            ## gradients from from all
            ## clients, and apply KRUM
            ## to get the most 'likely'
            ## gradient.

            for _ in range(CRITIC_ITER):
                trust = None
                gradient = []

                for j,client in enumerate(clients):
                    client.Anet.train()
                    client.Dnet.train()

                    if j in malicious:
                        gradient.append(client.gradient_attack())
                    
                    else:

                        client.Dnet.load_state_dict(
                            server.Dnet.state_dict()
                        )
                        gradient.append(client.train(i))

                score = krum(gradient)
                trust = torch.argmin(score)

                server.train(clients[benign].Anet,gradient[trust],
                             flag=False)
            
            server.train(
                clients[benign].Anet,
                gradient[trust],True
            )