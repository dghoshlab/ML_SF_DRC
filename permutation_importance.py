import copy
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (precision_score, f1_score, accuracy_score, recall_score,
                             jaccard_score, roc_auc_score, confusion_matrix, matthews_corrcoef, average_precision_score)
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset470K_with_smiles.csv")
#print(df)
df.columns = df.columns.str.strip()
DRC_cutoff = 0.05
Seed = 42
df['DRC_class'] = np.where(df['DRC'] >= DRC_cutoff, 1, 0)

X = df[['S1(CAS2-1) [eV]','DS2(CAS2-1) [eV]','INDOS-OscStr(CAS2-1)','CI-Coef**2(CAS2-1)','T1(CAS2-1) [eV]','ALogP','ALogp2',
            'AMR','apol','naAromAtom','nAromBond','nAtom','nHeavyAtom','nC','nN','nO','nBonds','nBonds2','nBondsS','nBondsD',
            'nBondsD2','C1SP2','C2SP2','C3SP2','C1SP3','C2SP3','Sv','Sse','Spe','Mv','Mse','Mp','Mi','ECCEN','hmax','hmin','nRing',
            'n5Ring','n6Ring','nTRing','n5HeteroRing','n6HeteroRing','nT5HeteroRing','nT6HeteroRing','topoRadius','topoShape',
            'RotBtFrac','topoDiameter']
]

y = df['DRC_class']
#print(X)
scaler = StandardScaler()
feature_names = X.columns.tolist()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

import xgboost as xgb
xgb_model=xgb.XGBClassifier(
    objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.01,
            max_depth=50,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss'
)
print("Training started")
xgb_model.fit(X_train,y_train)
print("Training Complete")
y_pred=xgb_model.predict(X_test)
y_probs_xgb = xgb_model.predict_proba(X_test)[:, 1] # Probability estimates for AUC
tn_xgb, fp_xgb, fn_xgb, tp_xgb = confusion_matrix(y_test, y_pred).ravel()

ca_xgb = accuracy_score(y_test, y_pred)
se_xgb = recall_score(y_test, y_pred)
pr_xgb = precision_score(y_test,y_pred)
sp_xgb = tn_xgb / (tn_xgb + fp_xgb)
fm_xgb = f1_score(y_test, y_pred)
j_xgb  = jaccard_score(y_test, y_pred)
auc_xgb = roc_auc_score(y_test, y_probs_xgb)
pr_auc = average_precision_score(y_test, y_probs_xgb)
print(f"Precision-Recall AUC (Average Precision): {pr_auc:.3f}")
print(f"--- XGB Results ---")
print(f"Accuracy: {ca_xgb:.4f}")
print(f"Sensitivity: {se_xgb:.4f}")
print(f"Specificity: {sp_xgb:.4f}")
print(f"AUC Score: {auc_xgb:.4f}")
print(f"FM Score: {fm_xgb:.4f}")
print(f"Jaccard Index: {j_xgb:.4f}")
print(f"Precision: {pr_xgb:.4f}")

result = permutation_importance(
    xgb_model,
    X_test,
    y_test,
    n_repeats=30,        # shuffle each feature 30 times
    random_state=42,
    scoring='roc_auc'    # or 'f1', 'accuracy', 'average_precision'
)

# Organize results
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False)

print(importance_df)

# Bar plot with error bars
plt.figure(figsize=(20, 20))
plt.barh(
    importance_df['feature'],
    importance_df['importance_mean'],
    xerr=importance_df['importance_std'],
    align='center',
    color='steelblue',
    ecolor='black'
)
plt.xlabel('Mean decrease in ROC-AUC')
plt.title('Permutation Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('permutation_importance.png', dpi=300)
plt.show()












