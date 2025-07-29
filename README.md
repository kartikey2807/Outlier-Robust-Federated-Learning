## Plan
* What was suggested:-
  * we train a bunch of client models
  * aggregate the model weights
  * that becomes discriminator
  * Retrain the discriminator
    * does it work with stale loss?
    * what data is needed?
* Can we work without gradient penalty
* The best case plan is:-
  * client models can have WGAN + softmax for probability (SEPARATE) \[train on client side as well\]
  * we still get *critic(x|y)*
  * we collect a bunch of them (per class per sample)
  * aggregate them on server-side
  * so *real_loss.mean()* is approximated
  * pass the generated samples from *G*
  * compute *fake_loss.mean()*
  * train the critic model and generator model (plain wgan with clipping)
