## Byzantine-Robust Federated Learning with WCGAN

<div align="justify">Federated learning is a way to jointly train multiple ML models without exposing the underlying data. It comprises multiple clients and a single server. The general flow looks like this: the server shares a global model with a client subset (based on activity or network connection), clients train the model on their local data and send back the updated weights, and then these updated weights are aggregated to output a new global model. Federated learning (FL) is preferred when data privacy and network latency are a concern. <i>But what if one or many of the clients are malicious and send poisoned weights?</i> This would compromise the aggregation process and the global model, and all clients would suffer as a result. To prevent this, robust aggregation methods have been investigated. These include trimmed mean, median, and KRUM (select the weights that are most similar to the majority). However, these methods either discard useful information or are susceptible to heterogeneous data distributions. <a href="https://arxiv.org/pdf/2503.20884">Usama et al.</a> suggested using a GAN model on the server with a modified discriminator. The discriminator is the global classifier itself, and we expect the trained GAN to generate samples that are representative of the underlying client dataset. During FL aggregation, clients are evaluated against this synthetic dataset and filtered out if their performance deviates significantly from the rest. But IN PRACTICE, the synthetic data generated did not resemble the underlying client data when we trained it on the MNIST and CIFAR-10 datasets. This research problem aims to fix this by changing the underlying model.</div>
<br>

***Pros***
- The malicious client identification performance is sufficiently good.

***Cons***
- Generated images are non-interpretable (low model explainability).
