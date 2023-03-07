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

#Prepare the data
x = mydata.iloc[:,0:4]

#prepare the target

target = mydata.iloc[:,4]

#Applying it to PCA Function
mat_reduced = PCA(x, 2)

#Creating a Pandas DataFrame of reduced Dataset
principal_df = mypd.DataFrame(mat_reduced, columns=['PC1','PC2'])

#Concat it with target variable to create a complete Dataset
principal_df = mypd.concat([principal_df , mypd.DataFrame(target)], axis = 1)
print(principal_df)
