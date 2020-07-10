#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux 
@date : 09 July 2020 
"""
import time
import numpy as np
from tabulate import tabulate

from from_scratch.utils import batch_iterator
from from_scratch.layers import Activation

class Model():
    pass

class Sequential(Model):
    def __init__(self, layers):
        self.initialized = False
        self.compiled = False
        self.input_shape = None
        self.input = None
        self.loss_function = None
        self.layers = layers
        self.input_layer = layers[0]
        if layers[0].input_shape is not None:
            self.initialize()

    def compile(self, loss, optimizer, metrics=None):
        self.loss_function = loss
        for layer in self.layers:
            layer.compile(optimizer)
        self.metrics = metrics
        self.compiled = True

    def initialize(self):
        self.input_shape = self.input_layer.input_shape
        prev_input_shape = self.input_shape
        final_layers = []
        for layer in self.layers:
            final_layers.append(layer)
            layer.initialize(prev_input_shape)
            if layer.activation is not None:
                final_layers.append(Activation(layer.activation))
            prev_input_shape = layer.output_shape
        self.layers = final_layers
        self.initialized = True

    def forward(self, x):
        if x is not isinstance(x, np.ndarray) : x = np.array(x)
        self.input = x
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def batch_train_pass(self, X, y):
        y_hat = self.forward(X)
        loss = np.mean(self.loss_function.loss(y, y_hat))
        grad = self.loss_function.gradient(y, y_hat)
        self.backward(grad)
        return loss

    def fit(self, X, y, batch_size=64, epochs=1):
        nb_batches = 'Unknown'
        losses = []
        for epoch in range(epochs):
            batch_loss = []
            mean_batch_loss = 0.0
            idx_batch = 0
            for X_batch, y_batch in batch_iterator(X, y, batch_size):
                loss = self.batch_train_pass(X_batch, y_batch)
                batch_loss.append(loss)
                mean_batch_loss = np.mean(batch_loss)
                idx_batch += 1
                print(f'Epoch {epoch+1}/{epochs} : batch {idx_batch}/{nb_batches} : loss {mean_batch_loss:.3E}', end='\r')
            losses.append(mean_batch_loss)
            nb_batches = idx_batch
            print()
        return {'loss':np.array(losses), 'epochs':np.arange(epochs)}

    def _old_summary(self):
        out_format = '{:<18}' * 4 + '\n'
        out = out_format.format('Name', 'Input shape', 'Output shape', 'Nb parameters')
        for layer in self.layers:
            out += str(layer, out_format)
        return out

    def summary(self):
        total_parameters = 0
        out_table = []
        for layer in self.layers:
            out_table.append(layer._summary_table())
            total_parameters += layer.parameters or 0
        out = tabulate(out_table, headers=['Name', 'Input Shape', 'Output Shape', 'Nb Parameters'])
        out += f'\n\nTotal parameters : {total_parameters}\n'
        return out

    def __iter__(self):
        yield self.layers