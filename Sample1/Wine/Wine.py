import pandas as mypd
import matplotlib.pyplot as myplot
import seaborn as mysb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np

mydata = mypd.read_csv("Wine.csv")
print(mydata)

'''
x = mydata.iloc[:,5:7]
print(x)

y = mydata.iloc[:,10:11]
print(y)

mymodel = LogisticRegression()
mymodel = mymodel.fit(x,y)
print(mymodel)
'''

mydata.isnull().sum()

for col in mydata.columns:
    if mydata[col].isnull().sum() > 0:
        mydata[col] = mydata[col].fillna(mydata[col].mean())
    
mydata.isnull().sum().sum()

mydata.hist(bins=20, figsize=(10,10))
myplot.show()

myplot.figure(figsize=(12, 12))
mysb.heatmap(mydata.corr(), annot=True)
myplot.show()


mydata.replace({'white':1,'red':0},inplace=True)
mydata['best quality'] = mydata.quality.apply(lambda x:1 if x > 5 else 0)
print(mydata['best quality'].value_counts())


from sklearn.model_selection import train_test_split
features = mydata.drop(['quality','best quality'],axis=1)
target = mydata['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40, shuffle=True)

print(xtrain.shape)
print(xtrain.shape)

model = LogisticRegression()
model.fit(xtrain,ytrain)

ypred = model.predict(xtest)

from sklearn.metrics import accuracy_score
model_acc = accuracy_score(ypred,ytest)

round(model_acc*100,2)

mymatrix = confusion_matrix(ypred,ytest)
print(mymatrix)
