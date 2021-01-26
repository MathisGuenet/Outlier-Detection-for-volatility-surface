# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:32:00 2021

@author: mathi
"""

import pandas as pd
from pandas import DataFrame
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
    #loading data (pickle file)
    unpickled_df = pd.read_pickle(path)
    df = []
    df = unpickled_df[day]
    df = df.dropna()

    #converting data frame to numpy
    tab=df.to_numpy()
    return tab


tab = loadVolatility("NKY_clean.pkl", 1)
plotVolatility3DPoint(tab)
plotVolatilitySurface(tab)






