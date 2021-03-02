# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:17:55 2020

@author: alexh
"""
import numpy as np
import matplotlib.pyplot as plt
import collections
import queue
from GenerateData import *
from sklearn.neighbors import NearestNeighbors


class DBSCAN:
    
    def __init__(self, data, radius, MinPt):
        self.data = data
        self.radius = radius
        self.MinPt = MinPt
        self.noise = 0
        self.unassigned = 0
        self.core=-1
        self.edge=-2
        
    #function to find all neigbor points in radius
    def neighbor_points(self, pointId):
        points = []
        for i in range(len(self.data)):
            #Euclidian distance using L2 Norm
            if sum((self.data[i] - self.data[pointId])**2) <= self.radius**2:
                points.append(i)
        return points
   
    
    def dbscan(self):
        #initilize all pointlabel to unassign
        pointlabel  = [self.unassigned] * len(self.data)
        pointcount = []
        #initialize list for core/noncore point
        corepoint=[]
        noncore=[]    
        
        #Find all neigbor for all point
        for i in range(len(self.data)):
            pointcount.append(DBSCAN.neighbor_points(self, i))
    
        #Find all core point, edgepoint and noise
        for i in range(len(pointcount)):
            if (len(pointcount[i])>=self.MinPt):
                pointlabel[i]=self.core
                corepoint.append(i)
            else:
                noncore.append(i)    
                
        for i in noncore:
            for j in pointcount[i]:
                if j in corepoint:
                    pointlabel[i]=self.edge
                    break
                
        #start assigning point to luster
        cl = 1
        #Using a Queue to put all neigbor core point in queue and find neigboir's neigbor
        for i in range(len(pointlabel)):
            q = queue.Queue()
            if (pointlabel[i] == self.core):
                pointlabel[i] = cl
                for x in pointcount[i]:
                    if(pointlabel[x]==self.core):
                        q.put(x)
                        pointlabel[x]=cl
                    elif(pointlabel[x]==self.edge):
                        pointlabel[x]=cl
                #Stop when all point in Queue has been checked   
                while not q.empty():
                    neighbors = pointcount[q.get()]
                    for y in neighbors:
                        if (pointlabel[y]==self.core):
                            pointlabel[y]=cl
                            q.put(y)
                        if (pointlabel[y]==self.edge):
                            pointlabel[y]=cl            
                cl=cl+1 #move to next cluster
                
        return pointlabel,cl
    
    #Function to plot final result with different clusters and anomalies
    def plotRes(self, clusterRes, clusterNum):
        nPoints = len(self.data)
        scatterColors = ['black', 'green', 'brown', 'red', 'purple', 'orange', 'yellow']
        for i in range(clusterNum):
            if (i==0):
                #Plot all noise point as blue
                color='blue'
            else:
                color = scatterColors[i % len(scatterColors)]
            abscissa = []
            ordinate = []
            for j in range(nPoints):
                if clusterRes[j] == i:
                    abscissa.append(self.data[j, 0])
                    ordinate.append(self.data[j, 1])
            plt.scatter(abscissa, ordinate, c=color, alpha=1, marker='.')

def days_outliers(data):
        res = []
        for i in range(len(data)):
            if data[i]==0:
                res.append(i)
        return res

def Distances(data):
    distance=[]
    for i in range(len(data)):
        distance1=0
        for j in range(len(data)):
            if i!= j:
                distance1 += np.linalg.norm(data[i] - data[j])
        distance.append(distance)
    return distance

if __name__=='__main__':
    RandomPts = createNoisyCircle()
    
    k = 10
    NN = NearestNeighbors(n_neighbors = k).fit(RandomPts)
    distances, indices = NN.kneighbors(RandomPts)
    distanceSorted = sorted(distances[:,k-1], reverse = True)
    
    distances2 = Distances(RandomPts)
    #plt.plot(indices[:,0], distanceSorted)
    
    radiusArray = np.percentile(distanceSorted,99)
    
    clustering = DBSCAN(RandomPts, radiusArray, 20)
    
    print('Set radius = ' +str(clustering.radius)+ ', Minpoints = '+str(clustering.MinPt))
    pointlabel, cluster = clustering.dbscan()
    clustering.plotRes(pointlabel, cluster)
    plt.show()
    print('Number of clusters found: ' + str(cluster - 1))
    counter=collections.Counter(pointlabel)
    print(counter)
    outliers  = pointlabel.count(0)
    print('Numbrer of outliers found: '+str(outliers) +'\n')

    
    