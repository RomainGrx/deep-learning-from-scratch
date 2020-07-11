#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux 
@date : 09 July 2020 
"""

import numpy as np
import matplotlib.pyplot as plt

from from_scratch import layers, optimizers, models, losses

X = np.linspace(0, 1, 10000).reshape(-1, 1)
y = (X*2)

y = y + np.random.rand(*y.shape)/10


model = models.Sequential([
    layers.Dense(1, input_shape=(1,))
])


model.compile(
    loss = losses.mse(),
    optimizer = optimizers.sgd(lr=1e-4, momentum=0.9)
)

history = model.fit(X, y, batch_size=512, epochs=2000)

out = model.forward([[0.5]])
print(out)

plt.figure()
plt.plot(X, y, alpha=.5)
plt.figure()
plt.semilogx(history['epochs'], history['loss'])
plt.show()



