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
    moneyness = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]

    #filling the arrays for the plot's axis (1 dimension)
    X_maturity = []
    Y_moneyness = []
    Z_volatility = []
    for list_ in tab:
        for i in range(4, len(list_)):
            X_maturity.append(list_[1])
            Y_moneyness.append(moneyness[i - 4])
            Z_volatility.append(list_[i])

    
    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_maturity, Y_moneyness, Z_volatility, c = Z_volatility)
    plt.xlabel("maturity")
    plt.ylabel("moneyness")
    plt.show()

def plotVolatilitySurface(tab):
    moneyness = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]

    #filling the arrays for the plot's axis (1 dimension)
    X_maturity = []
    Y_moneyness = []
    Z_volatility = []
    for list_ in tab:
        for i in range(4, len(list_)):
            X_maturity.append(list_[1])
            Y_moneyness.append(moneyness[i - 4])
            Z_volatility.append(list_[i])
    
    #plot
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_trisurf(X_maturity, Y_moneyness, Z_volatility, cmap = cm.coolwarm)
    plt.xlabel("maturity")
    plt.ylabel("moneyness")
    plt.show()

def loadVolatility(path, day):
    """
    Load all the volatility for one day in a numpy array
    Use for plot3D
    """
    #loading data (pickle file)
    unpickled_df = pd.read_pickle(path)
    df = []
    df = unpickled_df[day]
    print(df)
    df = df.dropna()

    #converting data frame to numpy
    tab =df.to_numpy()
    return tab

def loadAllData(path):
    """
    Load all the volatility for all day in a numpy array
    Use for the PCA
    """
    #loading data (pickle file)
    unpickled_df = pd.read_pickle(path)
    Ndays = len(unpickled_df)
    data = np.empty((Ndays,441))
    j = 0
    maturities = [5,13,36,58,80,110,140,200,300,400,500,800,900,1150,1400,1550,2000,2450,2700,2900,3200]
    myrow = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    for df in unpickled_df:
        df.set_index(['nBizDays'], inplace = True)
        #Add rows in df for maturity that we want
        for i in maturities:
            exitingIndex = i in df.index
            if exitingIndex == False :
                
                df.loc[i] = myrow
        df.sort_index(inplace = True)
        
        #interpolation
        for col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.interpolate(method = "values", limit_direction = "both", inplace = True)
        #Fill allData in 1 np.array
        df = df.loc[maturities,:]
        df.drop(columns=['Forwards', 'nCalDays', 'diff Days'], inplace = True)
        Nrows = len(df)
        Ncolumns = len(df.columns)
        data[j] = np.array(df.iloc[:,:], dtype = np.float).reshape(-1)
        j = j + 1
    return data
        
def createLinearOutlier(data):
    day = 0
    k = 3 #K is our coefficient
    prepData = data[day]
    outlier = np.empty((len(prepData),1))
    for i in range(len(prepData)):
        outlier[i] = k*prepData[i]
    data = np.concatenate((data, prepData))
    return data

def createInverseOutlier(data):
        return data

        

if __name__ == "__main__":
    tab = loadVolatility("NKY_clean.pkl", 1250)
    plotVolatilitySurface(tab)
    plotVolatility3DPoint(tab)
    #plotVolatilityPoint_strike(tab)
    #data = loadAllData("NKY_clean.pkl")
    #tab = loadAllData("NKY_clean.pkl")
    print("done")

