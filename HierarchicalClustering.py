# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:42:23 2020

@author: mathi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering 


def CreateGaussian(distribution):
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
    return data,num_clusters, y        


def hierarchical_clustering_printData(data_pd, k):
    hc = AgglomerativeClustering(affinity = 'euclidean', linkage ='ward', n_clusters = k)
    y_hc=hc.fit_predict(data_pd)
    plt.scatter(data[:,0], data[:,1], c = y_hc)
    plt.xlabel('x_absis')
    plt.ylabel('y_absis')
    plt.show()
    return y_hc

def performance(y_predict, y_initial):
    score = 0
    for i in range(len(y_predict)):
        if y_predict[i] == y_initial[i]:
            score = score + 1 #count the true positive
    return score
     
def hierarchical_clustering_printDendrogram(data_pd):
    dendrogram = sch.dendrogram(sch.linkage(data_pd, method  = "ward"))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()

data, k, y = CreateGaussian("gaussian")
data_pd = pd.DataFrame(data)
plt.scatter(data[:,0], data[:,1])
plt.show()

#hierarchical_clustering_printDendrogram(data_pd)
y_predict = hierarchical_clustering_printData(data_pd, 4) 
score = performance(y_predict, y)
print(score)

