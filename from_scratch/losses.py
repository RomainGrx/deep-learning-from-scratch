#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux 
@date : 09 July 2020 
"""

import numpy as np

class loss():
    def __init__(self):
        pass

    def loss(self, y, y_hat):
        raise NotImplementedError()

    def gradient(self, y, y_hat):
        raise NotImplementedError()

class mse(loss):
    def loss(self, y, y_hat):
        return 0.5 * np.power(y - y_hat, 2)

    def gradient(self, y, y_hat):
        return y_hat - y

class crossentropy(loss):
    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)
    def gradient(self, y, p):
        p += 1e-15
        return - (y/p) + (1 - y)/(1 - p)

class sparse_crossentropy(loss):
    def loss(self, y, p):
        y_one_hot = np.zeros_like(p)
        y_one_hot[np.arange(len(y)), y] = 1
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y_one_hot * np.log(p) - (1 - y_one_hot) * np.log(1 - p)
    def gradient(self, y, p):
        y_one_hot = np.zeros_like(p)
        y_one_hot[np.arange(len(y)), y] = 1
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y_one_hot / p) + (1 - y_one_hot) / (1 - p)