## Robust Federated Learning with WGAN-GP
References: [Usama et al](https://doi.org/10.48550/arXiv.2503.20884)   
**Federated Learning**   
* collaborative training
* between decentralized devices
* no need to share raw data samples
* updated weights are aggregated at a central server
* averaged weights are sent back to the clients
---

**Poison attacks**   
* models are prone to poison attacks
* model's weights can get altered
* propagates to the global model
* *un-targeted* attacks: degrades overall performance
* *targeted* attacks: only targets certain classes
---

**Filter-based methods**   
* filters out the malicious clients
* use *Generator* from GANs
* produce dummy data samples that mimic real data
* measure client's accuracy on the dataset
---

**Improvements**
* GANs suffer from mode collapse
* do not produce diverse samples
* In case of MNIST, they stuck with generating *blobs*
* WGAN-GP generates better samples
---

**Note:** 0.5% of the samples are shared with the server model to compute gradient penalty and train the generator. It can be argued that this is negligible and can be ignored.
