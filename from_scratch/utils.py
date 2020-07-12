#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux 
@date : 09 July 2020 
"""
import numpy as np
from functools import wraps

def batch_iterator(X, y=None, batch_size=64):
    n = X.shape[0]
    for start_idx in np.arange(0, n, batch_size):
        end_idx = min(n, start_idx+batch_size)
        if y is not None:
            yield X[start_idx:end_idx], y[start_idx:end_idx]
        else:
            yield X[start_idx:end_idx]

def input_to_numpy(fun):
    @wraps(fun)
    def inner(*args, **kwargs):
        for param in args:
            if isinstance(param, list):
                param = np.array(param)
        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = np.array(value)
        return fun(*args, **kwargs)
    return inner

