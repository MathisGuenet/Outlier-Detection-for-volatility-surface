import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def PCA(X:np.ndarray):
    #Standardize data  
    X_cent = X - np.mean(X , axis = 0)

    # calculating the covariance matrix of the mean-centered data.
    cov_mat = np.cov(X_cent , rowvar = False)

    #Calculating Eigenvalues and Eigenvectors of the covariance matrix
    #Eigen Values is 1D array and Eigen Vectors is ndarray
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    
    #sort the eigen values and eigen vectors in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    
    # Calculating the explained variance on each of components
    variance_explained = []
    for i in sorted_eigenvalue:
        variance_explained.append((i/sum(sorted_eigenvalue))*100)
    # Cumulative explained variance
    cumulative_variance_explained = np.cumsum(variance_explained)
    # Visualizing the eigenvalues
    sns.lineplot(x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21], y=cumulative_variance_explained)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Explained variance vs Number of components")

    #Projection Matrix
    n_components = 17
    projection_matrix = (eigen_vectors.T[:][:n_components]).T
    X_pca = np.dot(X, projection_matrix)
    
    return X_pca

if __name__ == "__main__":
    X = np.random.randint(10,50,400).reshape(20,20) 
    components = PCA(X)
    print(components)