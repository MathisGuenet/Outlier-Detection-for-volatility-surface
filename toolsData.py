# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:32:00 2021

@author: mathi
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import pickle
import numpy as np

def plotVolatility3DPoint(tab):
    moyeness = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]

    #filling the arrays for the plot's axis (1 dimension)
    X_maturity = []
    Y_moyeness = []
    Z_volatility = []
    for list_ in tab:
        for i in range(4, len(list_)):
            X_maturity.append(list_[1])
            Y_moyeness.append(moyeness[i - 4])
            Z_volatility.append(list_[i])

    
    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_maturity, Y_moyeness, Z_volatility, c = Z_volatility)
    plt.xlabel("expiry")
    plt.ylabel("strike")
    plt.show()

def plotVolatilitySurface(tab):
    moyeness = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]

    #filling the arrays for the plot's axis (1 dimension)
    X_maturity = []
    Y_moyeness = []
    Z_volatility = []
    for list_ in tab:
        for i in range(4, len(list_)):
            X_maturity.append(list_[1])
            Y_moyeness.append(moyeness[i - 4])
            Z_volatility.append(list_[i])
    
    #plot
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(X_maturity, Y_moyeness, Z_volatility, cmap = cm.coolwarm)
    plt.xlabel("expiry")
    plt.ylabel("strike")
    plt.show()

def loadVolatility(path, day):
    """
    Load all the data in a numpy array
    """
    #loading data (pickle file)
    unpickled_df = pd.read_pickle(path)
    df = []
    df = unpickled_df[day]
    df = df.dropna()

    #converting data frame to numpy
    tab=df.to_numpy()
    return tab

def load_volatility_strike(path, day, maturity = 0):
    """
    Return a 2d numpy array, volatility & moyeness for a given maturity
    """
    tab = loadVolatility(path, day)
    tab = tab[maturity]
    tab = tab[3:len(tab) - 1]
    moyeness = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
    data = np.empty((len(tab),2))
    for i in range(len(tab)):
        data[i] = np.array([moyeness[i], tab[i]])
    return data

def load_volatility_strike_maturity(path, day):
    tab = loadVolatility(path, day)
    data = np.empty((tab.shape[0]*(tab.shape[1] - 2),3))
    i = 0
    for list_ in tab : #for each maturity
            moyeness = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
            for j in range(len(moyeness)): #for each volatility   
                data[i] = np.array([moyeness[j], list_[1], list_[3+j]])
                i = i + 1
    return data



def plotVolatilityPoint_strike(tab):
    moyeness = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
    Z_volatility = []
    Y_moyeness = []
    colors = []
    j = 2
    for list_ in tab:
        for i in range(4, len(list_)):
            Y_moyeness.append(moyeness[i - 4])
            Z_volatility.append(list_[i])
            colors.append(2*j)   
        j = j*j
        plt.scatter(Y_moyeness, Z_volatility)
    plt.xlabel("moyeness")
    plt.ylabel("volatility")
    plt.show()
    

def plotVolatilityPoint_maturity(tab):
    moyeness = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]

if __name__ == "__main__":
    #tab = load_volatility_strike("NKY_clean.pkl", 80)
    #plotVolatilitySurface(tab)
    #plotVolatility3DPoint(tab)
    #plotVolatilityPoint_strike(tab)
    data = load_volatility_strike_maturity("NKY_clean.pkl", 80)
    print("done")


