## Plan
* What was suggested:-
  * we train a bunch of client models
  * aggregate the model weights
  * that becomes discriminator
  * Retrain the discriminator
    * does it work with stale loss?
    * what data is needed?
* The best case plan is:-
  * client models can have WGAN + softmax for probability
  * we still get *critic(x|y)*
  * we collect a bunch of them (per class per sample)
  * aggregate them on server-side
  * so *real_loss.mean()* is approximated
  * pass the generated samples from *G*
  * compute *fake_loss.mean()*
  * train the critic model and generator model (plain wgan with clipping) <-- fail :X
* UPDATE 2025-07-31
  * cannot work without GP
  * intense mode collapse
  * no way around it
  * let's keep a subset of real samples from some trusted clients
  * try WGAN-GP
  * If critic can use both Wasserstein objective and Cross-entropy loss, then this is great

`CLIENT x` --> 10%/5%/2%/1% images (x) and corresponding labels (y)    
`SERVER` collects around 6000 of them (let's say)    
Can make critic(fake,y) and critic(s,y) and  GRADIENT PENALTY TERM   
Use 2 objectives `Cross entropy` on server level and `WGAN loss` on server level
