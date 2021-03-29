'''
must use python 3.7
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolsData import *

from keras.layers import Input, Dense
from keras.models import Model

from sklearn import preprocessing

def autoencoder_outlierDetection(data, verbose = 0, threshold = 1.1):
    # No test data needed, trainData is all our dataset
    n_train = int(len(data)*100/100)
    trainData = data[:n_train]
    #Preprocessing : we need our input data between 0 and 1 
    # Since we'll be using sigmoid for the output layer
    mms = preprocessing.MinMaxScaler()
    trainDataScaled = mms.fit_transform(trainData)

    # this is the size of our encoded representations
    encoding_dim = 40  # 40 floats -> compression of factor 0.1, assuming the input is 441 floats
    # this is our input placeholder
    input = Input(shape=(441,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(441, activation='sigmoid')(encoded)
    # Since we create an autoencoder this model must map an input to its reconstruction
    autoencoder = Model(inputs=input, outputs=decoded)
    # create the encoder model that maps an input to its encoded representation
    encoder = Model(inputs=input, outputs=encoded)
    # create a placeholder for an encoded (40-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model that maps the encoded inputs to its reconstruction
    decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

    # We train over 200 epochs, with a batch size of 100
    # Finally, we get the prediction of the network for our data
    autoencoder.compile(optimizer='Nadam', loss='binary_crossentropy')
    model_autoencoder = autoencoder.fit(trainDataScaled, 
                                        trainDataScaled,
                                        epochs=200,
                                        batch_size=100,
                                        shuffle=True,
                                        verbose=verbose)
    if verbose == 1 :
        plt.plot(model_autoencoder.history["loss"], color = "r")
        autoencoder.summary()
    encoded = encoder.predict(trainDataScaled)
    decoded = decoder.predict(encoded)
    if verbose == 1 :
        print("TrainDataScaled[10, 10:30] \n" + trainDataScaled[10, 10:30])
        print("decoded[10,10:30] \n" + decoded[10,10:30])

    # We compute the euclidean distance from each point to its reconstruction. 
    # We use it as an outlier score:
    dist = np.zeros(len(trainDataScaled))
    for i, x in enumerate(trainDataScaled):
        dist[i] = np.linalg.norm(x-decoded[i])
    if verbose == 1 :
        plt.figure(figsize=(30,10))
        plt.plot(dist)
        plt.xlim((0,1296))
        plt.ylim((0,2))
        plt.xlabel('Index')
        plt.ylabel('Outlier Score')
        plt.title("Outlier Score for each observations")
    
    outliers = [i for (i, x) in enumerate(dist) if x > threshold]
    return outliers

    
def autoencoder_dimensionReduction(data, verbose):
    # No test data needed, trainData is all our dataset
    n_train = int(len(data)*100/100)
    trainData = data[:n_train]
    #Preprocessing : we need our input data between 0 and 1 
    # Since we'll be using sigmoid for the output layer
    mms = preprocessing.MinMaxScaler()
    trainDataScaled = mms.fit_transform(trainData)

    # this is the size of our encoded representations
    encoding_dim = 8 
    # this is our input placeholder
    input = Input(shape=(441,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='sigmoid')(input)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(441, activation='sigmoid')(encoded)
    # Since we create an autoencoder this model must map an input to its reconstruction
    autoencoder = Model(inputs=input, outputs=decoded)
    # create the encoder model that maps an input to its encoded representation
    encoder = Model(inputs=input, outputs=encoded)
    # create a placeholder for an encoded (40-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model that maps the encoded inputs to its reconstruction
    decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

    # We train over 200 epochs, with a batch size of 100
    # Finally, we get the prediction of the network for our data
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    model_autoencoder = autoencoder.fit(trainDataScaled, 
                                        trainDataScaled,
                                        epochs=500,
                                        batch_size=100,
                                        shuffle=True,
                                        verbose=verbose)
    if verbose == 1 :
        plt.plot(model_autoencoder.history["loss"], color = "r")
        autoencoder.summary()
    encoded = encoder.predict(trainDataScaled)
    decoded = decoder.predict(encoded)
    if verbose == 1 :
        print("TrainDataScaled[10, 10:30] \n" + trainDataScaled[10, 10:30])
        print("decoded[10,10:30] \n" + decoded[10,10:30])

        print("encoded[10] \n" + encoded[10])
    return encoded


if __name__=='__main__':
    data = loadAllData("NKY_clean.pkl")
    outliers = autoencoder_outlierDetection(data, 1, 1.1)
    print(outliers)