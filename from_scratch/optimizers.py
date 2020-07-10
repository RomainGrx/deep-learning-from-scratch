#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux 
@date : 09 July 2020 
"""

import numpy as np

class sgd():
    def __init__(self, lr, momentum=0):
        self.lr = lr
        self.momentum = momentum
        self.cur_w = None

    def update(self, w, grad_w):
        if self.cur_w is None:
            self.cur_w = np.zeros_like(w)

        self.cur_w = self.momentum * self.cur_w + (1 - self.momentum) * grad_w
        return w - self.lr * self.cur_w