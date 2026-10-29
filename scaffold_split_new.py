import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

# 1. Sample Data (Replace with your actual dataset)
df = pd.read_csv("Dataset470K_with_smiles.csv")
DRC_cutoff = 0.05
df.columns = df.columns.str.strip()
Seed = 42 
df['DRC_class'] = np.where(df['DRC'] >= DRC_cutoff, 1, 0)
print(df)
# 2. Extract Bemis-Murcko Scaffolds using RDKit
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else "Invalid"

df["scaffold"] = df["Smiles"].apply(get_scaffold)
print(df)
# ==========================================
# 3. Two-Step Scaffold Split (80:10:10)
# ==========================================

# Step 1: Separate 10% for the Test set
# Remaining data = 90%
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=Seed)
rem_idx, test_idx = next(gss_test.split(df, groups=df["scaffold"]))

df_remaining = df.iloc[rem_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# Step 2: Split the remaining 90% into Train and Validation.
# To get 10% of the total from the remaining 90%, the validation fraction must be:
# 10 / 90 ≈ 0.1111 (or 1/9)
val_fraction = 0.10 / (1.0 - 0.10)

gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
train_idx, val_idx = next(gss_val.split(df_remaining, groups=df_remaining["scaffold"]))

train_df = df_remaining.iloc[train_idx].reset_index(drop=True)
val_df = df_remaining.iloc[val_idx].reset_index(drop=True)

# ==========================================
# 4. Verify the Splits
# ==========================================
print(f"Total rows: {len(df)}")
print(f"Train size:  {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val size:    {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test size:   {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)\n")

# Verify that scaffolds do not overlap between sets
train_scaffolds = set(train_df["scaffold"])
val_scaffolds = set(val_df["scaffold"])
test_scaffolds = set(test_df["scaffold"])

print("Overlap Train & Val:", train_scaffolds.intersection(val_scaffolds))
print("Overlap Train & Test:", train_scaffolds.intersection(test_scaffolds))
print("Overlap Val & Test:", val_scaffolds.intersection(test_scaffolds))
#print(df_train)
