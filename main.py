from client import *
from utils  import *
from server import *
from config import *
import torch
import torch.nn as nn

print("--- CLIENT SIDE ---")
clients = [None]*CLIENT_COUNT
poisoned_clients = []

## Assumption: Clients that are corrupted remain that
## way. And vice versa. POISONED_PROBABIBILITY let us
## control how many models could be compromised. Also
## we only distort model weights, and don't alter the
## the dataset (data poisoning).

## TO_DO:-
## Track client's (non-poisoned ones) accuracy
## Verify that the image generated are diverse
## Verify that malicious clients are getting filtered

for i in range(CLIENT_COUNT):
    clients[i] = Client(i)
    clients[i]._init_()
    print(f"client: {i}")
    if i == 0 or torch.bernoulli(torch.tensor([POISONED_PROBABILITY])) == 1:
        clients[i]._train()
        clients[i]._eval_()
    else:
        print("model POISONED")
        clients[i]._poison()
        poisoned_clients.append(i) ## client's index
        clients[i]._eval_()

server = Server()
server._initialize_gen()

## LOOP
for round in range(10):
    agg_x = []
    agg_y = []
    print("extracting datasets")
    for i in range(CLIENT_COUNT):
        x,y = clients[i]._share_dataset()
        agg_x.append(x)
        agg_y.append(y)
    DATASET_X = torch.cat(agg_x,dim=0)
    DATASET_Y = torch.cat(agg_y,dim=0)

    ## FILTER-BASED Federated Learning:: In the first
    ## round train the take the weight of only single
    ## client (let's call this a TRUSTED CLIENT) This
    ## is another assumption: this client will not be
    ## compromised. In our case it is the 0th client.
    ## From 2nd round onward, filter out based on the
    ## accuracy for each model. Set a threshold.
    if round == 0:
        print("first round: take 0th client weight")
        server._initialize_critic(clients[0]._share_weights())
    else:
        noise = torch.randn(BATCH_SIZE,NOISE)
        label = torch.randint(0,LABEL,(BATCH_SIZE,))
        noise = noise.to(DEVICE)
        label = label.to(DEVICE)
        fakes = server.gen(noise,label)

        legit_clients = [] ## stores index
        for i in range(CLIENT_COUNT):
            ## compute accuracy
            pred, _ = clients[i].model(fakes,label,classify=True)
            pred = torch.argmax(pred,dim=1)
            correct = (pred == label).sum()
            accuracy= correct/float(BATCH_SIZE)

            if accuracy < THRESHOLD:
                print(f"BAD: {i}")
            else:
                legit_clients.append(i)
        
        print("weights aggregation")
        weights = []
        for i in range(CLIENT_COUNT):
            if i in legit_clients:
                weights.append(clients[i]._share_weights())

        agg_w = {}
        for key in weights[0].keys():
            agg_w[key]  = torch.zeros_like(weights[0][key])
        for weight in weights:
            for key in agg_w:
                agg_w[key] += weight[key]
        for key in agg_w:
            agg_w[key] = agg_w[key]/len(legit_clients) ## aggregate weights
        
        server._initialize_critic(agg_w)

    print("--- SERVER SIDE ---")
    server._train(x=DATASET_X,y=DATASET_Y)
    new_weights = server._get_critic_weights()
    ## Put the weights back into NON-POISONED clients
    ## and re-train them. Poisoned ones are untouched
    ## Expect that the client model's accuracy is not
    ## impacted by a LOT. Track the client's accyracy
    print("--- CLIENT SIDE ---")
    for i in range(CLIENT_COUNT):
        print(f"client: {i}")
        if i not in poisoned_clients:
            clients[i].model.load_state_dict(new_weights)
            clients[i]._train()
            clients[i]._eval_()
        else:
            clients[i]._eval_()