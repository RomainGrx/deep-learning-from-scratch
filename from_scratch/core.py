#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 30 June 2020
"""

import cupy as np
from . import layers


X = np.eye(3,4)

dense = layers.Dense(1, (1,))
