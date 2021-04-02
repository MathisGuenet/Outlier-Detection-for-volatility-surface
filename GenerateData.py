# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:08:44 2020

@author: mathi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

def CreateGaussian(distribution):
    if distribution == "gaussian" :
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
    return data,num_clusters, y        


def CreateDispatchGaussian(distribution):
    if distribution == "gaussian" :
        params = [[[ 0,1],  [ 0,1]], 
          [[ 5,1],  [ -2.5,1]], 
          [[-10,1],  [ -10,1]],
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
    return data, y 

def createNoisyCircle():
    n = 1500
    noisy_circle = datasets.make_circles(n_samples=n, factor=.5,noise=.05)
    return noisy_circle

def createNoisyMoons():
    n = 1500
    noisy_moons = datasets.make_moons(n_samples=n, noise=0.10)
    return noisy_moons

def createBlobs():
    n = 1500
    varied = datasets.make_blobs(n_samples=n,cluster_std=[1.0, 2.5, 0.5], random_state=170)
    return varied

def createUniforme():
    n = 1500
    no_structure = np.random.rand(n, 2), None
    return no_structure

def createAnisotropiclyDistribution():
    n= 1500
    random_state = 170      
    X, y = datasets.make_blobs(n_samples=n, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    return X_aniso, y


def plotData(data):
      plt.scatter(data[:,0], data[:,1])
      plt.show()   