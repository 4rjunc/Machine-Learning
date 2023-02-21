import numpy as np
import matplotlib.pyplot as pyplot
import pandas as mypd
from sklearn.metrics import confusion_matrix,accuracy_score

mydata = mypd.read_csv("./Iris_data.csv")

X = mydata.iloc[:,:4].values
Y = mydata['Species'].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print(Y_pred)
mymatrix = confusion_matrix(Y_test, Y_pred)
print(mymatrix)

acc = accuracy_score(Y_pred, Y_test)
print(round(acc*100,2))

