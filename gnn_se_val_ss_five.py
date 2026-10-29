import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import datetime
import copy
from split_dataset import train_df, val_df, test_df

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXTRA_FEATURES = ['S1(CAS2-1) [eV]', 'DS2(CAS2-1) [eV]', 'INDOS-OscStr(CAS2-1)', 'CI-Coef**2(CAS2-1)', 'T1(CAS2-1) [eV]']      
#EXTRA_MODE = 'node'        
EXTRA_MODE ='global'                           
assert EXTRA_MODE in ('node', 'global')
from rdkit.Chem.rdchem import BondType
BOND_TYPES = [
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC
]

def one_hot_bond_type(bond_type):
    idx = BOND_TYPES.index(bond_type) if bond_type in BOND_TYPES else len(BOND_TYPES)
    return F.one_hot(torch.tensor(idx), num_classes=len(BOND_TYPES)).float()

def bond_features(bond):
    features = torch.cat([
        one_hot_bond_type(bond.GetBondType()),  
        torch.tensor([
            bond.GetIsConjugated(),
            bond.IsInRing()
        ], dtype=torch.float)
    ])
    return features

def smiles_to_graph(smiles_str, label,extra=None):
    #s1 = df['S1_scaled'].loc[idx]
    #s2 = df['S2_scaled'].loc[idx]
    #os = df['Oscstr_scaled'].loc[idx]
    #cf = df['coeff_scaled'].loc[idx]
    #t1 = df['T1_scaled'].loc[idx]
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None  

    # Extract Atom Features (Node Features)
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            int(atom.GetIsAromatic()),
        ]

        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)
    if extra is not None and EXTRA_MODE == 'node':
        x = torch.cat([x, extra.view(1, -1).expand(x.size(0), -1)], dim=1)
    # Extract Bond Connections (Edge Index)
    edge_index = [[],[]]
    edge_attr = []
    for bond in mol.GetBonds():
        #print('bond',bond)
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = float(bond.GetBondTypeAsDouble())
        
        edge_index[0] +=[start, end]
        edge_index[1] +=[end, start]
        bond_feat = bond_features(bond)
        #print('Bond_feature',bond_feat)
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.stack(edge_attr) if edge_attr else torch.empty((0, 7), dtype=torch.float)
    # Create target label tensor
    y = torch.tensor([label], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.long))
    #data.smiles = smiles_str
    if extra is not None and EXTRA_MODE == 'global':
        data.g_feat = extra.view(1, -1) 
    return data

# Generate the graph dataset

#print('*********',len(df))
def generate_graphs(df):
    dataset, index = [], []
    extras = scaled_extra(df)
    for pos, (idx, row) in enumerate(tqdm(df.iterrows(),total=len(df))):
        graph = smiles_to_graph(row["Smiles"], row["DRC_class"],None if extras is None else extras[pos])
        if graph is not None:
            dataset.append(graph)
            index.append(pos)
    print(f'Total molecules converted to graphs:{len(dataset)}')
    return dataset, df.iloc[index].reset_index(drop=True)
    
extra_scaler = StandardScaler().fit(train_df[EXTRA_FEATURES].values) if EXTRA_FEATURES else None


def scaled_extra(df):
    if not EXTRA_FEATURES:
        return None
    assert not df[EXTRA_FEATURES].isna().any().any(), 'missing values in EXTRA_FEATURES'
    return torch.tensor(extra_scaler.transform(df[EXTRA_FEATURES].values), dtype=torch.float)

train_graphs, train_data = generate_graphs(train_df)
val_graphs, val_data = generate_graphs(val_df)
test_graphs, test_data = generate_graphs(test_df)

#generator = torch.Generator().manual_seed(42)
#Split data into train and test
#from torch.utils.data import random_split
#train_ds,test_ds = random_split(dataset,[0.8,0.2],generator=generator)

train_loader = DataLoader(train_graphs, batch_size=2048, shuffle=True)
train_eval_loader = DataLoader(train_graphs, batch_size=2048, shuffle=False)
val_loader = DataLoader(val_graphs, batch_size=2048, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=2048, shuffle=False)
#print("Test",test_ds)
node_feature_size=train_graphs[0].x.shape[1]
edge_feature_size=train_graphs[0].edge_attr.shape[1]
print(node_feature_size)
print(edge_feature_size)

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,edge_dim,extra_dim=0):
        super(GATNet, self).__init__()
        self.extra_dim = extra_dim
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=0.1, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=0.1, edge_dim=edge_dim)
        self.conv3 = GATConv(hidden_channels * 4, hidden_channels, heads=4, edge_dim=edge_dim)
        self.conv4 = GATConv(hidden_channels * 4, hidden_channels, heads=4, edge_dim=edge_dim)
        self.conv5 = GATConv(hidden_channels * 4, hidden_channels, heads=4, edge_dim=edge_dim)
        self.lin = torch.nn.Linear(hidden_channels * 4 + extra_dim, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = F.relu(self.conv4(x, edge_index, edge_attr))
        x = F.relu(self.conv5(x, edge_index, edge_attr))
        h = global_mean_pool(x, batch)
        if self.extra_dim:
            h = torch.cat([h, data.g_feat], dim=1)       # 'global' mode only
        return self.lin(h)
        

# Instantiate model, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
extra_dim = len(EXTRA_FEATURES) if EXTRA_MODE == 'global' else 0
model = GATNet(in_channels=node_feature_size, hidden_channels=64, out_channels=1,edge_dim= edge_feature_size, extra_dim=extra_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.2))


def train():
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in train_loader:
        batch = batch.to(device)
        #print("--- Data Shape Sanity Check ---")
        #print("edge_index shape:", batch.edge_index.shape) # Expected: [2, Num_Edges]
        #print("edge_attr shape:", batch.edge_attr.shape)   # Expected: [Num_Edges, Edge_Features]
        optimizer.zero_grad()
        #pred = model(batch)
        out = model(batch)
        target = batch.y.view(batch.num_graphs, -1)  # Reshape to [batch_size, 2]
        #print("Output shape:", out.shape)
        #print("Target shape:", target.shape)
        #print("Target dtype:", target.dtype)
        loss = criterion(out,target.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs 
    avg_loss = total_loss / total_graphs
    return avg_loss

def evaluate(loader):
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            #pred = model(batch)
            target = batch.y.view(batch.num_graphs, -1)  
            
            out = model(batch)
            loss = criterion(out, target.float())
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


def prob_pred(loader):
    model.eval()
    probs, ys =[],[]
    total_loss = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            probs.append(torch.sigmoid(pred).view(-1).cpu())
            ys.append(batch.y.view(-1).cpu())  

    return torch.cat(probs).numpy(), torch.cat(ys).numpy().astype(int)
#print("--- Data Shape Sanity Check ---")
#print("edge_index shape:", batch.edge_index.shape) # Expected: [2, Num_Edges]
#print("edge_attr shape:", batch.edge_attr.shape)   # Expected: [Num_Edges, Edge_Features]

EPOCHS = 1000
train_losses = []
val_losses = []
patience = 10 
patience_counter = 0  
best_val_loss = float('inf')
print("Training started at "+str(datetime.datetime.now()))
for epoch in range(1, EPOCHS + 1):
    train_loss = train()
    val_loss = evaluate(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f'Epoch: {epoch:03d} | Train loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  
        # Save the best model weights
        torch.save(model.state_dict(), 'best_gnn_seg_model.pt')
    else:
        patience_counter += 1  
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! No improvement in validation loss for {patience} consecutive epochs.")
            print(f"Loading the best model weights from epoch {epoch - patience}...")
            # Load the best weights back into the model before final evaluation
            model.load_state_dict(torch.load('best_gnn_seg_model.pt'))
            break

print("Training finished at "+str(datetime.datetime.now()))

# Final Evaluation 
from sklearn.metrics import (jaccard_score, precision_score, f1_score, recall_score, accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix)

def best_threshold(y_true, prob):
    ths = np.linspace(0.05, 0.95, 91)
    f1_scores=[]
    for t in ths:
        pr=(prob>t).astype(int)
        f1_scores.append(f1_score(y_true, pr))
    best_ind = np.argmax(f1_scores)
    return float(ths[best_ind])

def ece_score(y_true, prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)[1:-1]     # 0.1, 0.2, ..., 0.9
    bin_id = np.digitize(prob, edges)               # which bin (0-9) each molecule falls in

    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_id == b                        # molecules in this bin
        if not in_bin.any():                        # skip empty bins
            continue
        weight = in_bin.mean()                      # fraction of all molecules in this bin
        observed = y_true[in_bin].mean()            # fraction that are truly positive
        predicted = prob[in_bin].mean()             # average probability the model gave
        ece += weight * abs(observed - predicted)   # gap, weighted by bin size
    return float(ece)

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def metrics(name, y, p, thr):
    pred = (p>thr).astype(int)
    r = {'Accuracy': accuracy_score(y, pred),'Precision': precision_score(y, pred),'Recall': recall_score(y, pred),'F1': f1_score(y, pred),'MCC': matthews_corrcoef(y, pred),'ROC_AUC': roc_auc_score(y, p),'PR_AUC': average_precision_score(y, p),'ECE': ece_score(y, p),'Specificity':specificity_score(y,pred),'Jaccard':jaccard_score(y,pred)}
    print(f'\n---{name}---')
    for k, v in r.items():
        print(f'{k:10s} {v:.4f}')
    return r

p_val, y_val = prob_pred(val_loader)
thr = best_threshold(y_val, p_val)
print(f'\nThreshold chosen on validation: {thr:.2f}')
 
p_tr, y_tr = prob_pred(train_eval_loader)
p_te, y_te = prob_pred(test_loader)                  # test set scored once
metrics('Train', y_tr, p_tr, thr)
metrics('Validation', y_val, p_val, thr)
metrics('Test', y_te, p_te, thr)
 
pred_te = (p_te > thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred_te, labels=[0, 1]).ravel()
print('\n             Classified as 0    Classified as 1')
print(f'0            {tn:<18d} {fp}')
print(f'1            {fn:<18d} {tp}')
 
# Save test predictions
test_out = test_data.copy()
test_out['y_true'], test_out['prob'], test_out['pred'] = y_te, p_te, pred_te
test_out.to_csv(f'gnn_se_seed42n_test_predictions_g.csv', index=False)   # compare PubChemID with the ANN
cols = [c for c in ['PubChemID', 'Smiles'] if c in test_out.columns]
for tag, mask in {'tp': (y_te == 1) & (pred_te == 1), 'tn': (y_te == 0) & (pred_te == 0),
                  'fp': (y_te == 0) & (pred_te == 1), 'fn': (y_te == 1) & (pred_te == 0)}.items():
    test_out.loc[mask, cols].to_csv(f'gnn_se_seed42n_{tag}_g.csv', index=False)
print(f'\n{fp} false positives, {fn} false negatives saved to gnn_se_fp.csv / _fn.csv')

