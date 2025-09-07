## Outlier-robust Federated Learning setup: For
## each batch sample client grads and implement
## KRUM. You will have a trusted gradient, with
## high probability of being benign. Assumption
## initially we have 1 (or a subset of) trusted
## clients. The GAN forces Generator to produce
## similar distribution. Classifier forces them
## to be correct labels.

from config import *

import numpy as np
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

clients = []
for i  in range(COUNT_CLIENT):
    clients.append(Client(i))

server = Server()

## Assume that client '0' is the trusted client
stable = clients[0]
stable.Anet.train()
for i in range(300):

    stable.train(i,True)

print("INITIAL TRAINING")
for epoch in tqdm(range(EPOCH)):

    server.Dnet.train()
    server.Gnet.train()
    stable.Dnet.train()
    stable.Anet.train()

    for i in range(300):

        stable.Dnet.load_state_dict(server.Dnet.state_dict()) ## EQUAL θs
        real_gradients = stable.train(i)
        server.train(real_gradients,stable.Anet)

TEMP = Client(0)
for client in clients:
    client.Anet.load_state_dict(TEMP.Anet.state_dict())

for _ in range(ROUNDS):

    print("GENERATOR OUTPUT")
    sample = 64

    noise = torch.randn(sample,NOISE)
    label = torch.randint(0,10,(64,))

    noise = noise.to(DEVICE)
    label = label.to(DEVICE)
    fakes = server.Gnet(noise, label)

    num = torch.randint(1,MAX_BYZANTINE+1,(1,)).item()
    byzantine = torch.randperm(COUNT_CLIENT)[:num]

    for j,client in enumerate(clients):

        if j in byzantine:

            print(f"Client @ {j}")
            print("POISONED")
            client.train(0)
            client.weight_attack()
        
        else:

            print(f"Client @ {j}")
            print("TRAINING")

            client.Anet.train()
            for i in range(300):
                client.train(i,True)

        client.eval()

    malicious = []
    for j,client in enumerate(clients):

        client.Anet.eval()
        preds = \
        client.Anet(fakes)

        accuracy = (torch.argmax(preds,dim=1)==label).sum()/float(sample)
        if accuracy < THRESHOLD:
            malicious.append(j)
    
    print("MALICIOUS CLIENTS")
    print(malicious)
    print("FEDERATED AVERAGE")

    weights = []
    for j,client in enumerate(clients):

        if j not in malicious:
            weights.append(client.Anet.state_dict())
    avg_weights = aggregate(weights)

    ## FEDERATED-AVERAGING - aggregates  weights
    ## and loads back to the clients' classifier
    ## This is applied to : malicious and benign
    for client in clients:
        client.Anet.load_state_dict(avg_weights)

    for epoch in tqdm(range(EPOCH)):

        for i in range(300):
            batch_grad = []

            for j,client in enumerate(clients):

                if j in byzantine:

                    batch_grad.append(client.gradient_attack())

                else:
                    client.Dnet.train()
                    client.Anet.train()
                    server.Dnet.train()
                    server.Gnet.train()

                    client.Dnet.load_state_dict(server.Dnet.state_dict())
                    batch_grad.append(client.train(i))
            
            ## KRUM: compute euclidean distance
            ## between each pair Sum to closest
            ## N-F distances and select the one
            ## with smallest value The selected
            ## gradient is the most "similar".

            dist = torch.zeros(COUNT_CLIENT,COUNT_CLIENT)

            for c1 in range(COUNT_CLIENT):

                for c2 in range(COUNT_CLIENT):

                    add = 0
                    for dist1,dist2 in zip(batch_grad[c1],batch_grad[c2]):
                        add += (dist1-dist2).norm(p=2)
                    
                    dist[c1,c2] = add
                    dist[c2,c1] = add
            
            score = []
            for c3 in range(COUNT_CLIENT):

                sorted_dist = torch.sort(dist[c3,:])[0]
                score.append(
                    torch.sum(sorted_dist[:COUNT_CLIENT - MAX_BYZANTINE])
                )

            score = torch.tensor(score)
            trust = torch.argmin(score)
            
            ## all classifiers have same weight
            server.train(batch_grad[trust],
                         clients[0].Anet)