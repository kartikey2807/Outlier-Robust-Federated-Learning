## Byzantine-Robust Federated Learning with WCGAN

<div align="justify">Federated learning is a way to jointly train multiple ML models without exposing the underlying data. It comprises multiple clients and a single server. The flow looks as follows: the server shares a global model with a client subset (based on activity or network connection), clients train the model on their local data and send back updated weights, and then the server aggregates these updates to produce a new global model.
</div>
<br>

***Pros***
- The malicious client identification performance is sufficiently good.

***Cons***
- Generated images are non-interpretable (low model explainability).
