import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('Dataset470K_with_smiles.csv')
df.columns = df.columns.str.strip()
DRC_cutoff = 0.05
Seed = 42 
df['DRC_class'] = np.where(df['DRC'] >= DRC_cutoff, 1, 0)
df_shuffled = df.sample(frac=1, random_state=Seed).reset_index(drop=True)

train_spind = int(len(df_shuffled) * 0.8)
val_spind = int(len(df_shuffled) * 0.9)

train_df = df_shuffled.iloc[:train_spind]
val_df = df_shuffled.iloc[train_spind:val_spind]
test_df = df_shuffled.iloc[val_spind:]
#testcv_df = df_shuffled[train_spind:]
#train_df.to_csv("Training_data.csv")
#testcv_df.to_csv("Test_data.csv")
#print(train_df)
#print(val_df)
#print(test_df)


