#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 30 June 2020
"""

import math
import numpy  as np
from copy import copy
from functools import reduce
from operator import mul

from from_scratch.activations import sigmoid, relu, softmax,  activation_function

class Layer():
    def __init__(self, *args, **kwargs):
       self.input_shape = None
       self.output_shape = None
       self.parameters = None
       self.activation = None
       self.initialized = False
       self.compiled = False

    def layer_name(self):
        return self.__class__.__name__

    def initialize(self, *args, **kwargs):
        self.initialized = True

    def compile(self, *args, **kwargs):
        self.compiled = True

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, grad):
        raise NotImplementedError()

    def _summary_table(self):
        total_input_shape = (None,)+self.input_shape if self.input_shape else None
        total_output_shape = (None,)+self.output_shape if self.output_shape else None
        return [self.layer_name(), total_input_shape, total_output_shape, self.parameters]

    def __str__(self, out_format='{:>12}'*4+'\n'):
        return out_format.format(self.layer_name(), self.input_shape, self.output_shape, self.parameters)

class Dense(Layer):
    def __init__(self, n_neurons, input_shape=None, activation=None):
        super().__init__()
        self.input_neurons = None
        self.n_neurons = n_neurons
        self.activation = activation
        self.initialized = False
        self.compiled = False
        self.inputs = None
        self.weights = None
        self.biases = None
        self.weights_optim = None
        self.biases_optim = None
        if input_shape is not None:
            self.initialize(input_shape)

    def initialize(self, input_shape):
        if not isinstance(input_shape, tuple): input_shape = tuple(input_shape)
        assert len(input_shape) == 1
        self.input_shape = input_shape
        self.output_shape = (self.n_neurons,)
        self.input_neurons = input_shape[0]

        limit = 1 / math.sqrt(self.input_neurons)
        self.weights = np.random.uniform(-limit, limit, (self.input_neurons, self.n_neurons))
        self.biases = np.zeros((1, self.n_neurons))

        self.parameters = reduce(mul, self.weights.shape) + reduce(mul, self.biases.shape)
        self.initialized = True

    def compile(self, optimizer):
        self.weights_optim = copy(optimizer)
        self.biases_optim = copy(optimizer)
        self.compiled = True

    def forward(self, inputs):
        assert self.compiled and self.initialized

        if not self.initialized:
            self.initialize(inputs.shape[1])
        else:
            assert self.input_shape == (inputs.shape[1],)

        self.inputs = inputs
        self.output = np.matmul(inputs, self.weights) + self.biases
        return self.output

    def backward(self, grad):
        assert self.compiled and self.initialized

        grad_inputs = np.matmul(grad, self.weights.T)
        grad_weights = np.matmul(self.inputs.T, grad)
        grad_biases = np.mean(grad, axis=0, keepdims=True)

        self.weights = self.weights_optim.update(self.weights, grad_weights)
        self.biases = self.biases_optim.update(self.biases, grad_biases)

        return grad_inputs

class Flatten(Layer):
    def __init__(self, input_shape=None):
        super().__init__()
        self.compiled = True
        self.batch_size = None
        if input_shape is not None:
            self.initialize(input_shape)

    def initialize(self, input_shape):
        if not isinstance(input_shape, tuple) : input_shape = tuple(input_shape)
        self.input_shape = input_shape
        self.output_shape = (reduce(mul, input_shape),)
        self.initialized = True

    def forward(self, x):
        self.batch_size = x.shape[0]
        if not self.initialized:
            self.initialize(x.shape[1:])

        return x.reshape((self.batch_size, -1))

    def backward(self, grad):
        return grad.reshape((self.batch_size,) + self.input_shape)


activation_functions_dict = {
    'relu':relu,
    'sigmoid':sigmoid,
    'softmax':softmax
}

class Activation(Layer):
    def __init__(self, activation, input_shape=None):
        super().__init__()
        self.compiled = True
        self.inputs = None

        if input_shape is not None:
            self.initialize(input_shape)

        if isinstance(activation, activation_function):
            self.activation = activation
        elif activation in activation_functions_dict:
            self.activation = activation_functions_dict[activation]()
        else:
            raise NotImplemented('Activation function %s not existing, choose within '%activation, activation_functions_dict.keys())


    def initialize(self, input_shape):
        if not isinstance(input_shape, tuple):
            input_shape = tuple(input_shape)
        self.input_shape = self.output_shape = input_shape

        self.parameters = 0
        self.initialized = True


    def forward(self, x):
        if not self.initialized : self.initialize(tuple(x[1:]))
        self.inputs = x
        return self.activation(x)

    def backward(self, grad):
        return self.activation.gradient(self.inputs) * grad

    def layer_name(self):
        return f'Activation ({self.activation.__class__.__name__})'