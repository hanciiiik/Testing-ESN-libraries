import torch
import numpy as np

from util import *


class ESN:
    def __init__(self, dim_in, dim_res, dim_out, spectral_radius, sparsity = 0.8, weight_scale = 0.01):
        # initializing dimesnions
        self.dim_in = dim_in
        self.dim_res = dim_res
        self.dim_out = dim_out

        # initializing wights
        self.W_in = torch.randn(self.dim_res, self.dim_in + 1) * weight_scale
        self.W_res = torch.randn(self.dim_res, self.dim_res)
        self.W_out = torch.rand(self.dim_out, self.dim_res + 1)

        # handling sparsity
        prob_matrix = np.random.choice([0, 1], size=(self.dim_res, self.dim_res), p=[sparsity, 1-sparsity])
        self.W_res = np.multiply(self.W_res, prob_matrix)

        # spectral normlaization
        # idk if OK
        eigenvalues, eigenvectors = np.linalg.eig(self.W_res)
        radius = torch.max(torch.abs(torch.linalg.eigvals(self.W_res)))
        self.W_res *= spectral_radius / radius

        # define reservoir
        self.reservoir = self.f_res(torch.zeros(self.dim_res))

    def f_res(self, input):
        return torch.tanh(input)

    def predict(self, input):
        # single input, one time-stamp
        pre_activation = torch.mm(self.W_in, add_bias(input)) + torch.mm(self.W_res, self.reservoir)
        reservoir = self.f_res(pre_activation)
        output = torch.mm(self.dim_out, reservoir)

        # remember previous state of the reservoir
        self.reservoir = reservoir

        return output


    def fit(self, inputs, targets):
        # fit each input and train the whole network

        # collect reservoir states for each input
        states = []
        for input in inputs:
            self.predict(input)
            states.append(torch.cat([self.reservoir, input]).detach())

        # convert list of tensors to a tensor
        states = torch.stack(states)

        # train output weights with ridge regression
        regularization = alpha * torch.eye(states.shape[1])
        self.W_out = torch.linalg.pinv(states.T @ states + regularization) @ states.T @ targets

        #train output via pseudoinverse
        R_pinv = torch.linalg.pinv(R.T)
        self.W_out = torch.mm(R_pinv, targets.T)



#!/usr/bin/python3


#from ESN import ESN
from util import *


# # Generate sequence data
length = 800
train_test_ratio = 0.25
X = np.linspace(0, 2*np.pi, num=length) / train_test_ratio

# # Choose from following datasets:
# data = 0*X
# data = 0*X + 4
# data = 0.5*X
# data = 50.0*X
# data = X**2
# data = (2*X-4)**2

data = np.sin(X*3)
# data = np.sin(X*3) + 5
# data = np.sin(X) * np.sin(0.5*X)
# data = np.sin(X*8) + 0.4 * np.cos(X*2)
# data = np.sin(3*X)**3
# data = np.sin(X) * np.sin(2*X) # last that should definitely work
# data = np.sin(X) * np.sin(3*X)
# data = np.sin(X*3) * X
# data = np.sin(X**2*2)
# data = np.sin(X**2*9) + np.cos(X*3)
# data = np.tan(X*3)


# # Prepare inputs+targets
step = 1

split = int(train_test_ratio * length)
train_data = data[:split]

train_inputs = train_data[:-step]
train_targets = train_data[step:]

full_inputs = data[:-step]
full_targets = data[step:]

# plot_sequence(full_inputs, full_targets, split=split)


# # Train model
# FIXME: Tune parameters to make the network perform better. Do not increase number of reservoir neurons.
model = ESN(dim_in=1, dim_res=20, dim_out=1, spectral_radius=0.5, sparsity=0.5)
model.fit(train_inputs, train_targets)


# # Test model
# a) one-step prediction of next input
outputs, R = model.one_step_predict_seq(full_inputs)
plot_cells(R, split=split)
plot_sequence(full_targets, outputs=outputs.flatten(), split=split)

# b) repeated one-step generation
outputs, R = model.generate_seq(inputs=train_inputs, count=length-split)
plot_cells(R, split=split)
plot_sequence(full_targets, outputs=outputs.flatten(), split=split, title='Sequence generation')





