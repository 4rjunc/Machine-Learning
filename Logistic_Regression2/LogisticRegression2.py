import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

np.random.seed(1)
#create Dataframe
df = pd.DataFrame({'points':np.random.randint(30,size = 1000),
                   'assets':np.random.randint(12, size = 1000),
                   'drafted':np.random.randint(2, size=1000)
    })
print(df.head(500))

x = df[['points','assets']]
print(x)

y = df['drafted']
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train,y_train)
y_pred = logistic_regression.predict(x_test)
print(y_pred)

print(classification_report(y_test,y_pred))
