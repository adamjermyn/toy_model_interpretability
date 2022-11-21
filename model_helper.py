import torch
import torch.nn as nn
import numpy as np

r_threshold = 0.999

# Reconstruct the model
def model_builder(N, m, k, nonlinearity):
    if nonlinearity == 'ReLU':
        activation = nn.ReLU()
    elif nonlinearity == 'GeLU':
        activation = nn.GELU()
    elif nonlinearity == 'SoLU':
        activation = lambda x: x*torch.exp(x)    
    
    model = torch.nn.Sequential(
                nn.Linear(m, k, bias=True),
                activation,
                nn.Linear(k, N, bias=False)
            )
    return model

# Helper methods for extracting properties of the model

def single_dimension_output(d, data): # Computes the output for each feature on its own
    setup = data['setup']
    k = d['0.weight'].shape[0]
    m = d['0.weight'].shape[1]
    N = d['2.weight'].shape[0]
    nonlinearity = data['nonlinearity']
    embedder = setup['fixed_embedder']

    model = model_builder(N, m, k, nonlinearity)
    model.load_state_dict(d)
    inputs = torch.eye(m)
    outputs = model.forward(inputs)
    return outputs.detach().numpy()

def single_feature_output(d, data): # Computes the output for each feature on its own
    setup = data['setup']
    k = d['0.weight'].shape[0]
    m = d['0.weight'].shape[1]
    N = d['2.weight'].shape[0]
    nonlinearity = data['nonlinearity']
    embedder = setup['fixed_embedder']

    model = model_builder(N, m, k, nonlinearity)
    model.load_state_dict(d)
    inputs = torch.eye(N)
    outputs = model.forward(torch.einsum('ji,ik->jk',embedder,inputs).T)
    return outputs.detach().numpy()

def many_feature_output(d, data, inds): # Computes the output for each feature on its own
    setup = data['setup']
    k = d['0.weight'].shape[0]
    m = d['0.weight'].shape[1]
    N = d['2.weight'].shape[0]
    nonlinearity = data['nonlinearity']
    embedder = setup['fixed_embedder']

    model = model_builder(N, m, k, nonlinearity)
    model.load_state_dict(d)
    inputs = torch.eye(N)
    for i in inds:
        inputs[:,i] = 1
    outputs = model.forward(torch.einsum('ji,ik->jk',embedder,inputs).T)
    return outputs.detach().numpy()

def single_feature_activations(d, data, setup): # Computes the activations for each feature on its own
    m = d['0.weight'].shape[1]
    k = d['0.weight'].shape[0]
    N = setup['N']
    nonlinearity = data['nonlinearity']

    vectors = torch.eye(N)
    embedder = setup['fixed_embedder']
    inputs = torch.matmul(vectors,embedder.T)

    model = model_builder(d['2.weight'].shape[0], m, k, nonlinearity)
    model.load_state_dict(d)
    outputs = model[:2].forward(inputs).T
    return outputs.detach().numpy()

def many_feature_activations(d, data, setup, inds): # Computes the activations for each feature on its own plus feature i
    m = d['0.weight'].shape[1]
    N = d['2.weight'].shape[0]
    k = d['0.weight'].shape[0]
    nonlinearity = data['nonlinearity']

    vectors = torch.eye(N)
    for i in inds:
        vectors[:,i] = 1
    embedder = setup['fixed_embedder']
    inputs = torch.matmul(vectors,embedder.T)

    model = model_builder(N, m, k, nonlinearity)
    model.load_state_dict(d)
    outputs = model[:2].forward(inputs).T

    return outputs.detach().numpy()

def get_net_embedder(d, setup): # Computes the net effect of the fixed embedder and the learned one.
    m = d['0.weight'].shape[1]
    N = d['2.weight'].shape[0]

    vectors = torch.eye(N)
    embedder = setup['fixed_embedder']
    inputs = torch.matmul(vectors,embedder.T)
    
    e = torch.matmul(d['0.weight'], inputs.T)

    return e.detach().numpy()

def get_linear_model(d, setup):
    m = d['0.weight'].shape[1]
    N = d['2.weight'].shape[0]

    vectors = torch.eye(N)
    embedder = setup['fixed_embedder']
    inputs = torch.matmul(vectors,embedder.T)
    
    f = torch.matmul(d['2.weight'],torch.matmul(d['0.weight'], inputs.T))

    return f.detach().numpy()

    