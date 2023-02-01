import pandas  as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

vg_df = pd.read_csv('vgsales.csv',encoding='utf-8')
print(vg_df[['Name','Platform','Year','Genre','Publisher']].iloc[1:7])
genres = np.unique(vg_df['Genre'])
print(genres)
gle = LabelEncoder()
genre_labels = gle.fit_transform(vg_df['Genre'])
genre_mappings = {index:label for index,label in enumerate(gle.classes_)}
print(genre_mappings)
vg_df['GenreLabel'] = genre_labels
print(vg_df[['Name','Platform','Year','Genre','GenreLabel']].iloc[1:16000])