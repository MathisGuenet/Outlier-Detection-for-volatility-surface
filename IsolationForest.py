# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:31:10 2020

@author: mathi
"""
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

class LeafNode:
    def __init__(self, size, data):
        self.size = size
        self.data = data


class DecisionNode:
    def __init__(self, left, right, splitFeature, splitValue):
        self.left = left
        self.right = right
        self.splitFeature = splitFeature
        self.splitValue = splitValue


class IsolationTree:
    def __init__(self, height, maxDepth):
        self.height = height
        self.maxDepth = maxDepth

    def fit(self, X):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.
        """
        if self.height >= self.maxDepth or X.shape[0] <= 2: #X.shapes[0] number of points
            self.root = LeafNode(X.shape[0], X)
            return self.root

        # Choose Random Split features and Value
        num_features = X.shape[1] #X.shapes[1] number of features
        splitFeature = np.random.randint(0, num_features) #take radomly a feature
        splitValue = np.random.uniform(min(X[:, splitFeature]), max(X[:, splitFeature])) #take randomly a value

        X_left = X[X[:, splitFeature] < splitValue]
        X_right = X[X[:, splitFeature] >= splitValue]

        leftTree = IsolationTree(self.height + 1, self.maxDepth)
        rightTree = IsolationTree(self.height + 1, self.maxDepth)
        leftTree.fit(X_left)
        rightTree.fit(X_right)
        self.root = DecisionNode(leftTree.root, rightTree.root, splitFeature, splitValue)
        self.n_nodes = self.count_nodes(self.root)
        return self.root

    def count_nodes(self, root):
        count = 0
        stack = [root]
        while stack:
            node = stack.pop()
            count += 1
            if isinstance(node, DecisionNode):
                stack.append(node.right)
                stack.append(node.left)
        return count
    
class IsolationForest:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees

    def fit(self, X): #X must be ndarray
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        self.trees = [] #array of n treess
        if isinstance(X, pd.DataFrame):
            X = X.values
        n_rows = X.shape[0]
        height_limit = np.ceil(np.log2(self.sample_size))
        for i in range(self.n_trees):
            # data_index = np.random.choice(range(n_rows), size=self.sample_size, replace=False)
            #choose randomly the sample (size = sample_size) from the dataSet wich we are going to apply isolation forest 
            data_index = np.random.randint(0, n_rows, self.sample_size)
            X_sub = X[data_index]
            tree = IsolationTree(0, height_limit)
            tree.fit(X_sub)
            self.trees.append(tree)
        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        paths = []
        for row in X:
            path = []
            for tree in self.trees:
                node = tree.root
                length = 0
                while isinstance(node, DecisionNode):
                    if row[node.splitFeature] < node.splitValue:
                        node = node.left
                    else:
                        node = node.right
                    length += 1
                leaf_size = node.size
                pathLength = length + c(leaf_size)
                path.append(pathLength)
            paths.append(path)
        paths = np.array(paths)
        return np.mean(paths, axis=1)

    def anomaly_score(self, X:pd.DataFrame) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        avg_length = self.path_length(X)
        scores = np.array([np.power(2, -l/c(self.sample_size))for l in avg_length])
        return scores

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return np.array([1 if s >= threshold else 0 for s in scores])

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        scores = self.anomaly_score(X)
        prediction = self.predict_from_anomaly_scores(scores, threshold)
        for i in prediction:
            if prediction[i] == 0:
                print(prediction[i])

        return prediction    
    
def c(size):
    if size > 2:
        return 2 * (np.log(size-1)+0.5772156649) - 2*(size-1)/size
    if size == 2:
        return 1
    return 0

def createAnisotropiclyDistribution():
    n= 1500
    random_state = 170      
    X, y = datasets.make_blobs(n_samples=n, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    return X_aniso, y

def CreateDispatchGaussian():
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
    return data, y 

def pltData(data, label):
    plt.scatter(data[:,0], data[:,1],c=label, alpha=1, marker='.')
    plt.show()



data, y = createAnisotropiclyDistribution()
point = [[5,-10]]
data = np.concatenate((data, point))
forest = IsolationForest(data.shape[0], 10)
forest.fit(data)
forest.path_length(data)
pltData(data,forest.predict(data, 0.70))
    

