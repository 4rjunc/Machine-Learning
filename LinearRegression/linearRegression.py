import pandas as mypd
import matplotlib.pyplot as myplot
import seaborn as mysb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

mydata = mypd.read_csv("Mult_Reg_Yield.csv")
print(mydata)

print(mydata.describe())


#Separating X's and Y's
x = mydata.iloc[:,0:2]
print(x)


#Separating x's and y's
y = mydata.Yield
print(y)

#Correlation Analysis
#Scatter plot

print(mysb.pairplot(mydata))
print(myplot.show())

#Regression Modeling
#Fitting the model
mymodel = LinearRegression()
print(mymodel)

mymodel = mymodel.fit(x,y)
print(mymodel)

pred = mymodel.predict(x)
print(pred)

print(mymodel.coef_)

print(mymodel.intercept_)

#model accuracy-R Square value
rsq = mymodel.score(x,y)
print(rsq)

round(rsq*100,2)

mse = mean_squared_error(y,pred)
print(mse)

import math as mymath
rmse = mymath.sqrt(mse)
print(rmse)

#Residual Analysis

res = y- pred
print(res)