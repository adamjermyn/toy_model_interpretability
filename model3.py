import torch
import numpy as np
import torch_optimizer as optim
from copy import deepcopy

ReLU = torch.nn.ReLU()

print('GPU:',torch.cuda.get_device_name(0))

@torch.jit.script
def sample_vectors_power_law(N, eps, batch_size, embedder):
	'''
	Generates random uniform vectors in a tensor of shape (N,batch_size)
	with sparsity 1-eps. These are returned as v.
	
	Applies embedding matrix to v to produce a low-dimensional embedding,
	returned as x.    
	'''
	v = torch.rand((int(batch_size), int(N)), device='cuda')

	compare = 1. / torch.arange(1,int(N)+1,device='cuda')**1.1
	compare *= N * eps / torch.sum(compare)
	compare[compare >= 1] = 1

	sparsity = torch.bernoulli(compare.repeat(int(batch_size),1))
				
	v *= sparsity
	x = torch.matmul(v,embedder.T) # Embeds features in a low-dimensional space

	return v, x

@torch.jit.script
def sample_vectors_equal(N, eps, batch_size, embedder):
	'''
	Generates random uniform vectors in a tensor of shape (N,batch_size)
	with sparsity 1-eps. These are returned as v.
	
	Applies embedding matrix to v to produce a low-dimensional embedding,
	returned as x.    
	'''
	v = torch.rand((int(batch_size), int(N)), device='cuda')
	
	compare = eps * torch.ones((int(batch_size), int(N)), device='cuda')
	sparsity = torch.bernoulli(compare)
			
	v *= sparsity
	x = torch.matmul(v,embedder.T) # Embeds features in a low-dimensional space

	return v, x


@torch.jit.script
def loss_func(batch_size, outputs, vectors):
	loss = torch.sum((outputs - vectors)**2) / batch_size
	return loss

@torch.jit.script
def abs_loss_func(batch_size, outputs, vectors):
	loss = torch.sum((outputs - torch.abs(vectors))**2) / batch_size
	return loss

def train(setup, model, training_steps):
	N = setup['N']
	eps = setup['eps']
	learning_rate = setup['learning_rate']
	batch_size = setup['batch_size']
	fixed_embedder = setup['fixed_embedder']
	task = setup['task']
	decay = setup['decay']

	if task == 'decoder':
		l_func = loss_func
		sample_vectors = setup['sampler']
	elif task == 'abs':
		l_func = abs_loss_func
		# I need to cut eps in half to make this equivalent density.
		# Different samples have different sparse choices so doubles the density.
		def sample_vectors(N, eps, batch_size, fixed_embedder):
			v1,i1 = setup['sampler'](N, eps / 2, batch_size, fixed_embedder)
			v2,i2 = setup['sampler'](N, eps / 2, batch_size, fixed_embedder)
			return v1 - v2, i1 - i2
	else:
		print('Task not recognized. Exiting.')
		exit()

	optimizer = optim.Lamb(model.parameters(), lr=setup['learning_rate'])
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2**9, eta_min=0)

	losses = []
	models = []

	# Training loop
	for i in range(training_steps):
		optimizer.zero_grad(set_to_none=True)

		vectors, inputs = sample_vectors(N, eps, batch_size, fixed_embedder)
		outputs = model.forward(inputs)
		loss = l_func(batch_size, outputs, vectors)
		loss.backward()

		optimizer.step()
		scheduler.step()

		if i < training_steps / 2:
			state = model.state_dict()
			state['0.bias'] *= (1 - decay * learning_rate)
			model.load_state_dict(state)

		if i%2**4 == 0: # Avoids wasting time on copying the scalar over
			losses.append(float(loss))

		if (i & (i+1) == 0) and (i+1) != 0: # Checks if i is a power of 2
			models.append(deepcopy(model))

	return losses, model, models

import torch.nn as nn

def make_random_embedder(N,m):
	matrix = np.random.randn(N,m) # Make a random matrix that's (N,m)
	u,s,v = np.linalg.svd(matrix, full_matrices=False)
	# Now u is a matrix (N,m) with orthogonal columns and nearly-orthogonal rows
	# Normalize the rows of u
	u /= (np.sum(u**2,axis=1)**0.5)[:,np.newaxis]
	t = torch.tensor(u.T, requires_grad=False, device='cuda', dtype=torch.float)
	return t

class SoLU(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, input):
		return input*torch.exp(input)
	
def run(N,m,k,eps,batch_size,learning_rate,training_steps,sample_kind,init_bias,nonlinearity,task,decay):
	if sample_kind == 'equal':
		sampler = sample_vectors_equal
	elif sample_kind == 'power_law':
		sampler = sample_vectors_power_law
	else:
		print('Sample kind not recognized. Exiting.')
		exit()

	setup = {
		'N':N,
		'm':m,
		'k':k,
		'batch_size':batch_size,
		'learning_rate':learning_rate,
		'eps':eps,
		'fixed_embedder':make_random_embedder(N,m),
		'sampler':sampler,
		'task': task,
		'decay': decay
	}
	
	if nonlinearity == 'ReLU':
		activation = nn.ReLU()
	elif nonlinearity == 'GeLU':
		activation = nn.GELU()
	elif nonlinearity == 'SoLU':
		activation = SoLU()
	else:
		print('No valid activation specified. Quitting.')
		exit()

	model = torch.jit.script(
				torch.nn.Sequential(
					nn.Linear(m, k, bias=True),
					activation,
					nn.Linear(k, N, bias=False)
				)
		).to('cuda')

	state = model.state_dict()
	state['0.bias'] += init_bias
	model.load_state_dict(state)
				
	losses, model, models = train(setup, model, training_steps)
	return losses, model, models, setup
