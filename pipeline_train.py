## Outlier-robust Federated Learning setup: For
## each batch sample client grads and implement
## KRUM. You will have a trusted gradient, with
## high probability of being benign. Train with
## the server side and update the GAN models.

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

clients = []
for i  in range(COUNT_CLIENT):
    clients.append(Client(i))

server = Server()

for _ in range(ROUNDS):

    num = torch.randint(1,MAX_BYZANTINE+1,(1,)).item()
    byzantine = torch.randperm(COUNT_CLIENT)[:num]

    print("CLIENT CLASSIFIER TRAINED")
    for j,client in enumerate(clients):

        if j in byzantine:
            client.weight_attack()
            print(f"Client @ {j}")
            print("POISONED")
        
        else:
            
            print(f"Client @ {j}")
            print("TRAINING")

            client.Anet.train()
            client.Dnet.train()
            for i in range(300):

                client.train(i,flag=True)
            
            client.eval()

    print("DISCRIMINATOR TRAINING")
    for epoch in tqdm(range(EPOCH)):

        for i in range(300):
            
            batch_grads = []
            batch_label = []

            for j,client in enumerate(clients):
                
                real_grad = None
                label = None

                if j not in byzantine:

                    server.Dnet.train()
                    server.Gnet.train()
                    client.Dnet.train()
                    client.Anet.train()

                    client.Dnet.load_state_dict(server.Dnet.state_dict())
                    real_grad,label = client.train(i)
                else:

                    client.train(i)
                    real_grad,label = client.gradient_attack()
                
                batch_grads.append(real_grad)
                batch_label.append(label)         
            
            ## IMPLEMENT KRUM aggregation part
            ## where the goal is to select the
            ## most similar client of majority
            ## Expected to filter "Byzanitine"
            ## clients.

            dist = torch.zeros(COUNT_CLIENT,COUNT_CLIENT)
            dist = dist.to(DEVICE)

            for c1 in range(COUNT_CLIENT):

                for c2 in range(COUNT_CLIENT):

                    if c1 != c2:
                        for d1,d2 in zip(batch_grads[c1],batch_grads[c2]):
                            ## add norms
                            dist[c1,c2] += (d1-d2).norm(p=2)
                            dist[c2,c1] += (d1-d2).norm(p=2)
            score = []
            for cx in range(COUNT_CLIENT):

                sort_dist = torch.sort(dist[cx,:])
                score.append(torch.sum(sort_dist[0][: closest_neighbor]))
            
            score = torch.tensor(score)
            trust = torch.argmin(score)

            server.train(
                batch_grads[trust],
                batch_label[trust]
            )
    
    print("GENERATED MNIST OUTPUT")
    
    ## Sample gaussian noise and generate fake
    ## image. Plot the mnist images and use it
    ## to filter out all the malicious clients
    ## Aggregate the benign weights and shares
    ## with all the clients.

    sample = 64
    noise = torch.randn(sample,NOISE)
    label = torch.randint(0,10,(64,))

    noise = noise.to(DEVICE)
    label = label.to(DEVICE)
    fakes = server.Gnet(noise, label)
    
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
    average = aggregate(weights)

    ## Load back the aggregated weights back
    ## to each client. (malicious or benign)
    for client in clients:
        client.Anet.load_state_dict(average)