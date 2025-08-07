## Robust Federated Learning with WGAN-GP
References: [Usama et al](https://doi.org/10.48550/arXiv.2503.20884)   
**Federated Learning**   
* for collaborative training
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
* measure client's accuracy on dummy dataset
---

**Improvements**
* GANs suffer from mode collapse
* do not produce diverse samples
* In case of MNIST, they stuck with generating *blobs*
* WGAN-GP generates better samples
---

**Observations**   
In each communication round:-
* *poisonous* clients get filtered out
* *non-poisonous* clients' accuracy is same
* Diverse set of images are obtained from Generator

![](https://www.kaggleusercontent.com/kf/254650135/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..0Kh9hJrYZUC-4SSb5r7nVQ.u5Ey60-VPhDRjqwhrMFvDy08SNl2nizJGG8-5BqVH-AdFamkPeGXS1_caoPauoispP-WXo1oolR5gbxbMTnd_JxHUjEm3f53Uh3J_0VKHUu3Ybbphz4gMPRB1cqB-b80fBq9aJ5QdB2HalR1kLKfLRQ2cMf4xR2QNgCUvCY48M0Ey3OoMjpcSeF2hV2t_UJ3KWlw82pPI7uJZIfVMbwkCn099SoKAdwBtOtu99_ZgaPk46lfaEkoeFdG5n0ygYiwfVHOLbqPYNdtFVfXrkMY2L1KYvTZ2k2g2M6adJYZ3hJMtE1L_T5LqtGpebaIX-54XIuUG8-RDbw0fb5Fx8AQfFxnqSmteYvw99D84QsSvL3VdprHhHF-lo4lC7NbNH1SenkMdThwz77neNdpkMb-PS0QNPrIug7wuDEwWfqNhLIdMQ6_HEi150VbG4MqG2ILjPuW2sMRXaPV6SXk_axIjNv5Ilw135aa-oLLEicgtTXP_1ZilsNkH5MFc93lH4Puh6_3Hly4OmJpJs-DPayoRrYEm8HRzKGxrj0QJdTw0u_F-KJLkF-x48wT5CprLdsrE3-WPqt-oKu3yH_23JrFxP2ZEOfQfPFbEQ8CzRYAucXZGb276SVQCWsAFXKTTpnjrAaG7D7mjUr45dYgkbuy0y8XQeS_fhDAO9HDJghJfno.56AtGQHJMH8VuWtrHqsggQ/__results___files/__results___6_73.png)
---
**Note:** 0.5% of the samples are shared with the server model to compute gradient penalty and train the generator. It can be argued that this is negligible and can be ignored.
