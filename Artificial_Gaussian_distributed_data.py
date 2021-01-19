# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:08:38 2020

@author: mathi
"""

import numpy as np
import matplotlib.pyplot as plt

params = [[[ 0,1],  [ 0,1]], 
          [[ 5,1],  [ 5,1]], 
          [[-2,5],  [ 2,5]],
          [[ 2,1],  [ 2,1]],
          [[-5,1],  [-5,1]]]

n = 300
dims = len(params[0])

data = []
y = []
for ix, i in enumerate(params):
    inst = np.random.randn(n, dims)
    for dim in range(dims):
        inst[:,dim] = params[ix][dim][0]+params[ix][dim][1]*inst[:,dim]
        label = ix + np.zeros(n)

    if len(data) == 0: data = inst
    else: data = np.append( data, inst, axis= 0)
    if len(y) == 0: y = label
    else: y = np.append(y, label)

num_clusters = len(params)

print(y.shape)
print(data.shape)
plt.scatter(data[:,0], data[:,1])
plt.show()