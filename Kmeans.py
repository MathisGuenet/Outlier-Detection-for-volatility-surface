# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:46:06 2020

@author: mathi
"""

import numpy as np
import matplotlib.pyplot as plt
from toolsData import *

class Kmeans(object) :
     def __init__(self, k=1):
        self.k = k #k is the number of cluster
     def train(self, data, y,verbose = 1):
        shape = data.shape  #"Tuple of array dimensions. here(1500,2)
        ranges = np.zeros((shape[1], 2)) #is an array of 0 and dimensions(shape[1],2)
        centroids = np.zeros((shape[1], 2))#is an array of 0 and dimensions(shape[1],2)
         
        for dim in range(shape[1]):
            ranges[dim, 0] = np.min(data[:,dim]) #get the minimums
            ranges[dim, 1] = np.max(data[:,dim]) #get the maximums
            
        if verbose == 1:
            print('Ranges: ')
            print(ranges)
            
        centroids = np.zeros((self.k, shape[1])) #we need k centroids since there is k cluster
        for i in range(self.k): #for i in number of cluster
            for dim in range(shape[1]): #for each dimension
                centroids[i, dim] = np.random.uniform(ranges[dim, 0], ranges[dim, 1], 1)
                #give random uniform coordinate for each centroids
        
        if verbose == 1:
            print('Centroids: ')
            print(centroids)

            plt.scatter(data[:,0], data[:,1])
            plt.scatter(centroids[:,0], centroids[:,1], c = 'r')
            plt.show()
        
        count = 0
        while count < 100:
            count += 1
            if verbose == 1:
                print('-----------------------------------------------')
                print('Iteration: ', count)
            distances = np.zeros((shape[0],self.k)) #array of dimensions (number of point, number of cluster)
            #for each point we're gonna calculate the euclidienne distance with each centroids
            for ix, cData in enumerate(data):
                for ic, cCentroid in enumerate(centroids):
                    distances[ix, ic] = np.sqrt(np.sum((cData-cCentroid)**2))
            
            labels = np.argmin(distances, axis = 1) #return indices of the minimum elements of distances with axis = 1

            #we want to calcul the new centroid for each clusters now
            new_centroids = np.zeros((self.k, shape[1])) #we need k new centroids since there is k cluster
            for centroid in range(self.k): #for each cluster
                temp = data[labels == centroid] #temp is array with the coordinate when labels == centroid(cluster) if data belongs to a specific cluster
                if len(temp) == 0:
                    return 0
                for dim in range(shape[1]): #calculate the new 2 coordinnates of the centroid
                    new_centroids[centroid, dim] = np.mean(temp[:,dim])
            
            if verbose == 1:
                plt.scatter(data[:,0], data[:,1], c = labels)
                plt.scatter(new_centroids[:,0], new_centroids[:,1], c = 'r')
                plt.show()
             
            #if the new and old centroids are almost equal means that we finally found our clusters
            #so we can break
            if np.linalg.norm(new_centroids - centroids) < np.finfo(float).eps: 
                print("DONE!")
                break

            centroids = new_centroids
        self.centroids = centroids
        self.labels = labels
        if verbose == 1:
            print(labels)
            print(centroids)
        
        return 1



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
    return data,num_clusters, y 

if __name__ == "__main__":
    data, k, y = CreateGaussian("gaussian")
    k1 = 1
    k2 = 2
    k3 = 3
    k4 = 4
    k5 = 5
    kmeans = Kmeans(k3)
    score = kmeans.train(data, y)
