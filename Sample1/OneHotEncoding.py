import pandas  as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

poke_df = pd.read_csv("pokemon.csv",encoding='utf-8')

# transform and map pokemon generations

gen_le = LabelEncoder()
gen_labels = gen_le.fit_transform(poke_df['Generations'])
poke_df['Gen_Label'] = gen_labels

poke_df_sub = poke_df['Name','Generations','Gen_Label','Legendary']
poke_df_sub.iloc[4:10]