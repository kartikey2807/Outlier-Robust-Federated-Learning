## Federated learning pipeline: train client
## server models The Discriminator models on
## both side will have same weight and train
## Auxilllary model once in each epoch.

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

from config import *
from server import *
from clients import *

def aggregate(weights):
    agg_weights = {}

    for key in weights[0].keys():
        agg_weights[key] = torch.zeros_like(weights[0][key])

    for weight in weights:
        for key in agg_weights:
            agg_weights[key] = agg_weights[key]+weight[key]
    
    for key in agg_weights:
        agg_weights[key] /= float(len(weights)) ## averaged
    
    return agg_weights

clients = []
for i  in range(COUNT_CLIENT):
    clients.append(Client(i))

server = Server()

## Assume that initially we have one trusted
## client. We train with that client for the
## first round, get the generator and figure
## out malicious clients.

print(f"Client: {0}")
print("TRAINING")
stable = clients[0]

for epoch in tqdm(range(EPOCH)):
    server.Dnet.train()
    server.Gnet.train()
    stable.Dnet.train()
    stable.Anet.train()

    ## HOW MANY GRADIENT ARE SUFFIEICIENT TO
    ## LEAK??

    for i in range(300):
        stable.Dnet.load_state_dict(server.Dnet.state_dict()) ## same Θs
        gradients_real,label = stable.train(i,False)
        server.train(gradients_real,label) ## server

stable.eval()
for _ in range(ROUNDS):

    ## Generate fake samples
    
    sample = 64
    print("------- Generator O/P --------")
    noise = torch.randn(sample,NOISE)
    label = torch.randint(0,10,(64,))

    noise = noise.to(DEVICE)
    label = label.to(DEVICE)
    fakes = server.Gnet(noise,label)

    print(label)

    plt.imshow(make_grid(fakes).permute(1,2,0).detach().numpy()*0.5+0.5)
    plt.set_cmap("gray")
    plt.axis("off")
    plt.show()

    for index, client in enumerate(clients):

        if torch.bernoulli(PROBABILITY)== 1:

            print(f"Client: {index}")
            print("TRAINING")

            ## Dnet <- attack, Anet <- ok <= ignore
            ## Dnet <- ok, Anet <- attack
            ## Both Dnet & Anet <- attack
            
            client.Dnet.train()
            client.Anet.train()
            for i in tqdm(range(300)):
                client.train(i,True)
                
        else:
            print(f"Client: {index}")
            print("POISONED")
            client.weight_attack()
    
        client.eval()
    
    malicious = []
    for index, client in enumerate(clients):
        client.Anet.eval()

        preds = client.Anet(fakes)
        accuracy = (torch.argmax(preds,dim=1)==label).sum()/float(sample)

        if accuracy < THRESHOLD:
            print(f"Poisoned client index {index}")
            print(f"Accuracy: {accuracy*100:.2f}%")
            malicious.append(index)

    client_weights = []
    print("-------- RE-TRAINING ---------")
    for index, client in enumerate(clients):

        if index not in malicious:

            print(f"Client: {index}")
            print("TRAINING")

            for epoch in tqdm(range(EPOCH)):
                server.Dnet.train()
                server.Gnet.train()
                client.Dnet.train()
                client.Anet.train()

                for i in range(300):
                    client.Dnet.load_state_dict(server.Dnet.state_dict())
                    gradients_real,label = client.train(i,False)
                    server.train(gradients_real,label) ## server

            client.eval()
            client_weights.append(client.Anet.state_dict())
    
    agg_weights = aggregate(client_weights)
    print(len(client_weights))
    print("Load aggregate weights")
    for client in clients:
        client.Anet.load_state_dict(agg_weights)