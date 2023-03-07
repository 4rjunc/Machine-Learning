import numpy as np

def PCA(X, num_components):
    #Step1
    X_meaned = X - np.mean(X,axis=0)
    
    #Step2
    cov_mat = np.cov(X_meaned, rowvar=False)
    
    #Step3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    
    #Step4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    
    #Step5
    eigenvectors_subset = sorted_eigenvectors[:,0:num_components]
    
    #Step6
    X_reduced = np.dot(eigenvectors_subset.transpose(), X_meaned.transpose() ).transpose()
    
    return X_reduced
    
import pandas as mypd
mydata = mypd.read_csv("./Iris_data.csv")
print(mydata)