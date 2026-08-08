## Byzantine-Robust Federated Learning with WCGAN

Federated Learning
- allows us to train multiple edge ML models collaboratively.
- data remains localized and protected.
- weights are shared with the server for aggregation.
- *what if one or many of the clients are malicious and send poisoned weights?*
- this would compromise the aggregation scheme and the global model, and all the clients will suffer as a result.

---

Byzantine-robust FL
- [Usama et. al](https://arxiv.org/pdf/2503.20884) suggests using a modified GAN on the server.
- GAN is trained in sync with the **FL training** process.
- trained GAN generator outputs a synthetic dataset.
- dataset should ideally be representative of the underlying client data.
- during an aggregation round, the clients are evaluated against the said dataset.
- if a client's performance deviates from majority, it is likely poisoned.
- during experiments on MNIST and CIFAR-10 data the synthetic dataset did not resemble the client distribution.
- this work tries to resolve this problem.

*Given Methodology*
- we use a distributed conditional Wasserstein GAN model.
- one discriminator is deployed on each client.
- there is a single generator on the server side.
- Figure 1 describes the loss functions and how it flows.
- on client side, discriminator computes gradients for first WGAN loss term $\nabla f(x)$.
- on client side, classifier $H$ (typical FL) is also trained.
- they are robustly aggregated using KRUM to select gradients closest to the *majority* $\nabla \hat{f}(x)$
- instead of discarding the useful information from the trained client models we discard discriminator gradients.
- we then generate a sample from the conditional generator $G(z|y)$.
- we compute the second WGAN loss term and aggregated classifier output.

<img src="./images/setup.png" width="600px">

*Figure 1. Flow of gradients during training and KRUM*

<img src="./images/pipeline.png" width="600px">

*Figure 2. The overall FL training process*

---

We note that the generated samples from the learned WGAN generator are interpretable and achieve up to 80.91% accuracy on MNIST data. It should be noted that we make the process secure by trimming the gradients and adding Gaussian noise.
