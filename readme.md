# Two-Layer Toy Model

The file `model3.py` implements a two-layer two model that can be trained on feature decoding tasks. This is primarily meant for interpretability experiments.

Calling `run(...)` trains a model and returns the loss curve, the final model, checkpointed models (uniformly spaced in log-trainings-steps), and a dictionary containing details of the model and task setup.

The parameters `N`, `m`, and `k` specify the number of features, the embedding dimension, and the nonlinear dimension of the model. The parameter `eps` specifies the mean feature frequency (feature sparsity `S=1-eps`). The parameter `sample_kind` specifies whether to use uniform ("equal") or power-law ("power_law") feature frequencies. `init_bias` is the initial mean bias of the nonlinear units. `nonlinearity` specifies which nonlinearity to use, from "ReLU", "GeLU", and "SoLU". `task` specifies the task, either "decoder" (for the feature decoder) or "abs" for the absolute-value feature decoder. `decay` specifies the weight decay rate on the biases. This gets multiplied by the learning rate to determine the per-step weight decay.
