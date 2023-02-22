import pandas as pd
import matplotlib.pyplot as myplot
import seaborn as mysb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

import numpy as np

pima = pd.read_csv("./diabetes.csv")
print(pima.head())
myplot.figure(figsize=(12, 12))
mysb.heatmap(pima.corr(), annot=True)
print(myplot.show())
feature_cols = ['Pregnancies','Insulin','BMI','Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
x = pima[feature_cols]
y = pima.Outcome

X_train, X_test, Y_train, Y_test = train_test_split(x , y, test_size=0.3, random_state=1)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
confusion_matrix(Y_test,y_pred)
print(confusion_matrix(Y_test, y_pred))

print("Accuracy : ", metrics.accuracy_score(Y_test,y_pred))  

import six
import sys
sys.modules['sklearn.externals.six'] = six
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data  = StringIO()
export_graphviz(classifier, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())