import pandas as pd
import numpy  as np 
from sklearn.preprocessing import Binarizer, KBinsDiscretizer, FunctionTransformer

df = pd.DataFrame({
    'A':[1,2,3,4,5],
    'B':[6,2,7,9,0],
    'C':[11,55,67]
})
print(df)

print("Values")
print(df.values)