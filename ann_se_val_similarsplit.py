import copy
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, f1_score, accuracy_score, recall_score,
                             jaccard_score, roc_auc_score, confusion_matrix, matthews_corrcoef, average_precision_score)
from torch.utils.data import DataLoader, TensorDataset
from split_dataset import train_df, val_df, test_df

SEED= 42
torch.manual_seed(SEED)
np.random.seed(SEED)

features = ['S1(CAS2-1) [eV]','DS2(CAS2-1) [eV]','INDOS-OscStr(CAS2-1)','CI-Coef**2(CAS2-1)','T1(CAS2-1) [eV]',
            'AMR','apol','nAromBond','nHeavyAtom','nC','nN','nO','nBonds','nBonds2','nBondsS','nBondsD',
            'nBondsD2','C1SP2','C2SP2','C3SP2','C1SP3','C2SP3','Sv','Spe','Sp','Si','Mv','Mp','Mi','ECCEN','hmax','hmin','nRing',
            'n5Ring','n6Ring','nTRing','nFRing','nT5HeteroRing','nT6HeteroRing','nT5Ring','nT6Ring','RotBtFrac','topoDiameter']

X_train_raw = train_df[features]
X_val_raw = val_df[features]
X_test_raw = test_df[features]

y_train = train_df['DRC_class']
y_val = val_df['DRC_class']
y_test = test_df['DRC_class']

print(f"Train: {len(X_train_raw)} | Val: {len(X_val_raw)} | Test: {len(X_test_raw)}")


# Fit scaler on TRAIN only; transform val and test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

# Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=2048, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=2048, shuffle=False)

# Model Architecture 
class WeightedBinaryNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layer_stack(x)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def ece_score(y_true, prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)[1:-1]     
    bin_id = np.digitize(prob, edges)               

    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_id == b                        
        if not in_bin.any():                        
            continue
        weight = in_bin.mean()                      
        observed = y_true[in_bin].mean()            
        predicted = prob[in_bin].mean()             
        ece += weight * abs(observed - predicted)   
    return float(ece)

# Model Training Setup
model = WeightedBinaryNN(input_dim=len(features))
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2]))
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
print("Model Summary", model)

# Training Loop 
epochs = 1000
best_val_loss = float('inf')
best_state = None
patience_counter = 0
early_stop_patience = 7

print(f"Training started at {datetime.datetime.now()}")
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * X_batch.size(0)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for Xv, yv in val_loader:
            v_loss = criterion(model(Xv), yv)
            total_val_loss += v_loss.item() * Xv.size(0)

    avg_train_loss = total_train_loss / len(X_train)
    avg_val_loss = total_val_loss / len(X_val)

    scheduler.step(avg_val_loss)
    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

# Restore the best weights (by validation loss)
model.load_state_dict(best_state)
model.eval()

# Probabilities
with torch.no_grad():
    train_probs = torch.sigmoid(model(X_train_tensor)).numpy().flatten()
    val_probs = torch.sigmoid(model(X_val_tensor)).numpy().flatten()
    test_probs = torch.sigmoid(model(X_test_tensor)).numpy().flatten()

# Threshold search on the VALIDATION set only 
thresholds = np.linspace(0.1, 0.9, 81)
best_f1, best_threshold = 0, 0.5
for t in thresholds:
    f1 = f1_score(y_val, (val_probs > t).astype(int))
    if f1 > best_f1:
        best_f1, best_threshold = f1, t

print(f"\nOptimal Threshold (chosen on validation): {best_threshold:.2f} | Val F1: {best_f1:.4f}")

# Final Evaluation 
y_pred_train = (train_probs > best_threshold).astype(int)
y_pred_val = (val_probs > best_threshold).astype(int)
y_pred_test = (test_probs > best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test, labels=[0, 1]).ravel()
print("\n             Classified as 0    Classified as 1")
print(f"0            {tn:<18d} {fp}")
print(f"1            {fn:<18d} {tp}")


def print_metrics(y_true, y_pred, y_prob, name):
    print(f"\n--- {name} Results ---")
    print(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision:   {precision_score(y_true, y_pred):.4f}")
    print(f"F-measure:   {f1_score(y_true, y_pred):.4f}")
    print(f"Recall:      {recall_score(y_true, y_pred):.4f}")
    print(f"Specificity: {specificity_score(y_true, y_pred):.4f}")
    print(f"MCC:      {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"PR_AUC:         {average_precision_score(y_true, y_prob):.4f}")
    print(f"Jaccard:     {jaccard_score(y_true, y_pred):.4f}")
    print(f"ROC_AUC:         {roc_auc_score(y_true, y_prob):.4f}")  
    print(f"ECE:         {ece_score(y_true, y_prob):.4f}")

print_metrics(y_train, y_pred_train, train_probs, "Train")
print_metrics(y_val, y_pred_val, val_probs, "Validation")
print_metrics(y_test, y_pred_test, test_probs, "Test")

# Saving test predictions ---
#pred_df = pd.DataFrame({'Predicted DRC_class': y_pred_test, 'Predicted_Probability': test_probs})
#result = pd.concat([test_df, pred_df], axis=1)
test_df = test_df.reset_index(drop=True)
test_df['Predicted DRC_class'] = y_pred_test
test_df['Predicted_Probability'] = test_probs
test_df.to_csv('ann_se_pred1.2_same_split42.csv', index=False)
print("\nSaved test predictions to ann_se_pred.csv")
