import pandas as mypd
import matplotlib.pyplot as myplot
import seaborn as mysb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,accuracy_score
from sklearn import metrics
import numpy as np

mydata = mypd.read_csv("./Iris_data.csv")
print(mydata)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = mypd.DataFrame(scaler.fit_transform(mydata))
print(scaled_data)
mysb.heatmap(scaled_data.corr())

#Outputs in .ipynb file
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(mydata)
data_pca = pca.transform(mydata)
data_pca = mypd.DataFrame(data_pca,columns=['PC1','PC2'])
print(data_pca)