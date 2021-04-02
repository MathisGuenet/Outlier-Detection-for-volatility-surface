from toolsData import *
from GenerateData import *
from IsolationForest import *
from Kmeans import *
from HierarchicalClustering import *
from DBSCAN_Clustering import *
from PCA import *
from autoencoder import *

def predict_IsolationForest(data):
    nbrTree = 100
    forest = IsolationForest(data.shape[0], 100) #Create the isolation forest
    forest.fit(data) #fit data
    forest.path_length(data) #compute mean length in the trees for each observation
    prediction = forest.predict(data, 0.55) #array of prediction's label with respect of the treshold
    pltData(data, prediction)
    print(outliersDays(prediction))

def predict_DBSCAN(data):
    k = 10
    NN = NearestNeighbors(n_neighbors = k).fit(data)
    distances, indices = NN.kneighbors(data)
    distanceSorted = sorted(distances[:,k-1], reverse = True)
    radiusArray = np.percentile(distanceSorted,99)
    clustering = DBSCAN(data, radiusArray, 10)
    print('Set radius = ' +str(clustering.radius)+ ', Minpoints = '+str(clustering.MinPt))
    pointlabel, cluster = clustering.dbscan()
    clustering.plotRes3D(cluster)
    plt.show()
    print('Number of clusters found: ' + str(cluster - 1))
    counter=collections.Counter(pointlabel)
    print(counter)
    outliers  = pointlabel.count(0)
    print('Numbrer of outliers found: '+str(outliers) +'\n')

def predict_Kmeans():
    data = load_volatility_strike("NKY_clean.pkl", 94)
    kmeans = Kmeans(2)
    score = kmeans.train(data, y)



if __name__ == "__main__":
    data = loadAllData("NKY_clean.pkl")
    data_AE = autoencoder_dimensionReduction(data, 0)
    predict_DBSCAN(data_AE)
    print("done")
    
    