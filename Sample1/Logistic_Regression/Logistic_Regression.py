import pandas as mypd
import matplotlib.pyplot as myplot
import seaborn as mysb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np

mydata = mypd.read_csv("Iris_data.csv")
print(mydata)

print(mydata.describe())

print(mysb.pairplot(mydata,hue = 'Species'))
print(myplot.show())

x = mydata.iloc[:,0:3]
y = mydata.Species

mymodel = LogisticRegression(C = 1e08)
print(mymodel)

mymodel = mymodel.fit(x,y)
print(mymodel)

x = mydata.iloc[:,0:3]
print(x)

y = mydata.Species
print(y)


#Model Intercept
print(mymodel.intercept_)

#Predicted probabilities
preprob = mymodel.predict_proba(x)
print(preprob)

predclass = mymodel.predict(x)
print(predclass)

mytable = mypd.crosstab(y,predclass)
print(mytable)


predprob = mypd.DataFrame(preprob, columns=["Predicted 0","Predicted = 1","Predicted = 2"])
print(predprob)

predclass = mypd.DataFrame(predclass,columns=["Predicate Class"])
print(predclass)

myresult = mydata.join(predclass)
print(myresult)

predclass = mymodel.predict(x)
print(predclass)

mytable = mypd.crosstab(y, predclass)
print(myresult)

myresult = myresult.join(predprob)
print(round(myresult.head(15),4))

myscore = cross_val_score(mymodel,x ,y,scoring='accuracy',cv = 5)
print(myscore)

cv_accuracy = myscore.mean()
round(cv_accuracy*100,2)

mymatrix = confusion_matrix(y, predclass)
print(mymatrix)

myreport = classification_report(y, predclass)
print(myreport)

myscore = cross_val_score(mymodel,x,y,scoring='accuracy',cv = 5)
print(myscore)
