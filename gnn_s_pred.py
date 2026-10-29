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

##Extracting edge features

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
        one_hot_bond_type(bond.GetBondType()),  # one-hot bond type
        torch.tensor([
            bond.GetIsConjugated(),
            bond.IsInRing()
        ], dtype=torch.float)
    ])
    return features
'''
def smiles_to_graph(smiles_str, label):
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
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.smiles = smiles_str
    return data

# Generate the graph dataset

#print('*********',len(df))
def generate_graphs(df):
    dataset, index = [], []
    for pos, (idx, row) in enumerate(tqdm(df.iterrows(),total=len(df))):
        graph = smiles_to_graph(row["Smiles"], row["DRC_class"])
        if graph is not None:
            dataset.append(graph)
            index.append(pos)
    print(f'Total molecules converted to graphs:{len(dataset)}')
    return dataset, df.iloc[index].reset_index(drop=True)
    
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
'''
class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,edge_dim):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=0.1, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, dropout=0.1, edge_dim=edge_dim)
        self.conv3 = GATConv(hidden_channels * 4, hidden_channels, heads=4, edge_dim=edge_dim)
        self.conv4 = GATConv(hidden_channels * 4, hidden_channels, heads=4, edge_dim=edge_dim)
        self.conv5 = GATConv(hidden_channels * 4, hidden_channels, heads=4, edge_dim=edge_dim)
        self.lin = torch.nn.Linear(hidden_channels * 4, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = F.relu(self.conv4(x, edge_index, edge_attr))
        x = F.relu(self.conv5(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        return self.lin(x)

# Instantiate model, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = GATNet(in_channels=node_feature_size, hidden_channels=64, out_channels=1,edge_dim=edge_feature_size).to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
#criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.2))
loaded_model = GATNet(in_channels=5, hidden_channels=64, out_channels=1,edge_dim=6).to(device)
state_dictionary = torch.load('best_gat_model.pt')
loaded_model.load_state_dict(state_dictionary)
loaded_model.eval()
'''
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
            target = batch.y.view(batch.num_graphs, -1)  # Reshape to [batchsize, 2]
            
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
            ys.append(batch.y.view(-1).cpu())  # Reshape to [batchsize, 2]

    return torch.cat(probs).numpy(), torch.cat(ys).numpy().astype(int)
#print("--- Data Shape Sanity Check ---")
#print("edge_index shape:", batch.edge_index.shape) # Expected: [2, Num_Edges]
#print("edge_attr shape:", batch.edge_attr.shape)   # Expected: [Num_Edges, Edge_Features]

EPOCHS = 1000
train_losses = []
val_losses = []
patience = 10  # Number of epochs to wait before stopping
patience_counter = 0  # Counter for stagnation
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
        patience_counter = 0  # Reset counter since we found a better model
        # Save the best model weights
        torch.save(model.state_dict(), 'best_gat_model.pt')
    else:
        patience_counter += 1  # No improvement, increment counter
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! No improvement in validation loss for {patience} consecutive epochs.")
            print(f"Loading the best model weights from epoch {epoch - patience}...")
            # Load the best weights back into the model before final evaluation
            model.load_state_dict(torch.load('best_gat_model.pt'))
            break

print("Training finished at "+str(datetime.datetime.now()))


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train loss')
plt.plot(val_losses, label='Val loss')
#plt.plot(train_logP_losses, label='Train logP')
#plt.plot(val_logP_losses, label='Val logP')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
 
# ------------------------------------------------------------------ save test predictions
test_out = test_data.copy()
test_out['y_true'], test_out['prob'], test_out['pred'] = y_te, p_te, pred_te
test_out.to_csv(f'gnn_s_seed42n_test_predictions.csv', index=False)   # compare PubChemID with the ANN
cols = [c for c in ['PubChemID', 'Smiles'] if c in test_out.columns]
for tag, mask in {'tp': (y_te == 1) & (pred_te == 1), 'tn': (y_te == 0) & (pred_te == 0),
                  'fp': (y_te == 0) & (pred_te == 1), 'fn': (y_te == 1) & (pred_te == 0)}.items():
    test_out.loc[mask, cols].to_csv(f'gnn_s_seed42n_{tag}.csv', index=False)
print(f'\n{fp} false positives, {fn} false negatives saved to gnn_s_fp.csv / _fn.csv')


y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for batch_data in test_loader:
        out = model(batch_data.x, batch_data.edge_index, batch_data.batch)
        pred = out.argmax(dim=-1)
        y_true.extend(batch_data.y.tolist())
        y_pred.extend(pred.tolist())

print(f"Precision:   {precision_score(y_true, y_pred):.4f}")
print(f"F-measure:   {f1_score(y_true, y_pred):.4f}")
print(f"Recall:      {recall_score(y_true, y_pred):.4f}")


# evaluate the model by predicting on test set
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        all_preds.append(pred.cpu())
        targets = batch.y.view(batch.num_graphs, -1).cpu()
        all_targets.append(targets)
#torch.save(model.state_dict(),"GNN_struc_feat2.pt")

# concatenate scaled predictions and targets
# 2. Convert both to flat numpy arrays so scikit-learn can read them smoothly
all_preds = torch.cat(all_preds, dim=0)
all_targets = torch.cat(all_targets, dim=0)
all_preds = (all_preds >= 0.0).int()
y_pred = all_preds.cpu().numpy().flatten()
y_true = all_targets.cpu().numpy().flatten()

num_true_neg=0
num_true_pos=0
num_false_neg=0
num_false_pos = 0
for k in range(y_pred.shape[0]):
    if y_true[k] == 0 and y_pred[k] == 0:
        num_true_neg += 1
    elif y_true[k] == 1 and y_pred[k] == 1:
        num_true_pos += 1
    elif y_true[k] == 1 and y_pred[k] == 0:
        num_false_neg += 1
    elif y_true[k] == 0 and y_pred[k] == 1:
        num_false_pos += 1
print("             Classified as 0             Classified as 1")
print("0                ",num_true_neg,"                ",num_false_pos)
print("1                ",num_false_neg,"               ",num_true_pos)
print("Test Results")
print("Predicted",y_pred)
print("Actual",y_true)
print(f"Accuracy:   {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision:   {precision_score(y_true, y_pred):.4f}")
print(f"F-measure:   {f1_score(y_true, y_pred):.4f}")
print(f"Recall:      {recall_score(y_true, y_pred):.4f}")

model.eval()
all_train_preds = []
all_train_targets = []
with torch.no_grad():
    for batch in train_loader:
        batch = batch.to(device)
        pred = model(batch)
        all_train_preds.append(pred.cpu())
        targets = batch.y.view(batch.num_graphs, -1).cpu()
        all_train_targets.append(targets)

# concatenate scaled predictions and targets
# 2. Convert both to flat numpy arrays so scikit-learn can read them smoothly
all_train_preds = torch.cat(all_train_preds, dim=0)
all_train_targets = torch.cat(all_train_targets, dim=0)
all_train_preds = (all_train_preds >= 0.0).int()
y_train_pred = all_train_preds.cpu().numpy().flatten()
y_train_true = all_train_targets.cpu().numpy().flatten()
print("Train Results")
print("Predicted",y_train_pred)
print("Actual",y_train_true)
print(f"Accuracy:   {accuracy_score(y_train_true, y_train_pred):.4f}")
print(f"Precision:   {precision_score(y_train_true, y_train_pred):.4f}")
print(f"F-measure:   {f1_score(y_train_true, y_train_pred):.4f}")
print(f"Recall:      {recall_score(y_train_true, y_train_pred):.4f}")


false_positives_smiles = []
false_negatives_smiles = []
true_positives_smiles = []
true_negatives_smiles = []


# Since test_loader shuffle=False, indices align perfectly with test_ds
for k in range(len(y_true)):
    actual = y_true[k]
    predicted = y_pred[k]

    # Retrieve the original graph data object from the dataset split
    original_graph_data = test_ds[k]
    smiles_string = original_graph_data.smiles

    if actual == 0 and predicted == 1:
        false_positives_smiles.append(smiles_string)
    elif actual == 1 and predicted == 0:
        false_negatives_smiles.append(smiles_string)
    elif actual == 0 and predicted == 0:
        true_negatives_smiles.append(smiles_string)
    elif actual == 1 and predicted == 1:
        true_positives_smiles.append(smiles_string)
pd.DataFrame({'Smiles': false_positives_smiles}).to_csv('gnn_s_fp.csv', index=False)
pd.DataFrame({'Smiles': false_negatives_smiles}).to_csv('gnn_s_fn.csv', index=False)
pd.DataFrame({'Smiles': true_positives_smiles}).to_csv('gnn_s_tp.csv', index=False)
pd.DataFrame({'Smiles': true_negatives_smiles}).to_csv('gnn_s_tn.csv', index=False)


# --- Print Summary Reports ---
print(f"\nFound {len(false_positives_smiles)} False Positives.")
print("Top 10 False Positive SMILES (Predicted 1, Actual 0):")
for sm in false_positives_smiles[:10]:
    print(sm)

print(f"\nFound {len(false_negatives_smiles)} False Negatives.")
print("Top 10 False Negatives SMILES (Predicted 0, Actual 1):")
for sm in false_negatives_smiles[:10]:
    print(sm)

#arr_df = pd.DataFrame(y_pred, columns=['Predicted DRC_class'])
#print(arr_df)
# 2. Concatenate along columns (axis=1)
#result = pd.concat([new_df_subset, arr_df], axis=1)
#print(pred_train)
#print(pred_train.dtype)
#result.to_csv('ann_struc_pred.csv', index=False)
'''

def predict_unknown_molecule(smiles_str, model, device):
    """
    Predicts the class of an unknown molecule.
    
    Args:
        smiles_str (str): The SMILES string of the molecule.
        raw_electronic_feats (dict): Dictionary of unscaled electronic features:
                                     {'S1': val, 'S2': val, 'Oscstr': val, 'coeff': val, 'T1': val, 'Mi': val}
        model: Your trained GATNet model.
        device: torch device ('cuda' or 'cpu').
    """
       
    # Build the PyG Data object manually (replicating smiles_to_graph logic)
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")
        
    # Extract Atom Features
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

    # Extract Bond Connections
    edge_index = [[], []]
    edge_attr = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0] += [start, end]
        edge_index[1] += [end, start]
        bond_feat = bond_features(bond)  
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.stack(edge_attr) if edge_attr else torch.empty((0, 7), dtype=torch.float)
    
    # Bundle features into Data structure
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    
    # Simulate a batch size of 1 so global_mean_pool knows what to do
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    
    # Perform Inference
    model.eval()
    with torch.no_grad():
        data = data.to(device)

        # Forward pass
        logits = model(data)
        #print('logit', logits)
        # Apply sigmoid because you trained with BCEWithLogitsLoss
        prob = torch.sigmoid(logits)
        #print('prob',prob)
        probability = torch.sigmoid(logits).item()
        
        predicted_class = 1 if probability >= 0.5 else 0

    return predicted_class, probability

# --- EXAMPLE USAGE ---
print("N,N′-Dimethyl-perylene diimide (C1-PDI)")
new_smiles = "CN1C(=O)C2=C3C(=CC=C4C3=C(C=C2)C5=C6C4=CC=C7C6=C(C=C5)C(=O)N(C7=O)C)C1=O"
pred_class, pred_prob = predict_unknown_molecule(new_smiles, loaded_model, device)
print(f"Predicted Class: {pred_class}")
print(f"Confidence Probability: {pred_prob:.4f}")

print("terylene")
new_smiles = "C1=CC2=C3C(=C1)C4=C5C(=CC=C6C5=C(C=C4)C7=CC=CC8=C7C6=CC=C8)C3=CC=C2"
pred_class, pred_prob = predict_unknown_molecule(new_smiles, loaded_model, device)
print(f"Predicted Class: {pred_class}")
print(f"Confidence Probability: {pred_prob:.4f}")

print("1,3 diphenylisobenzofuran")
new_smiles = "C1=CC=C(C=C1)C2=C3C=CC=CC3=C(O2)C4=CC=CC=C4"
pred_class, pred_prob = predict_unknown_molecule(new_smiles, loaded_model, device)
print(f"Predicted Class: {pred_class}")
print(f"Confidence Probability: {pred_prob:.4f}")

print("Aza-cibalackrot")
new_smiles = "O=c5c(c1ccccc1)c7c2ncccc2n8c(=O)c(c3ccccc3)c6c4ncccc4n5c6c78"
pred_class, pred_prob = predict_unknown_molecule(new_smiles,loaded_model, device)
print(f"Predicted Class: {pred_class}")
print(f"Confidence Probability: {pred_prob:.4f}")

print("zeaxanthin")
new_smiles = "CC1=C(C(C[C@@H](C1)O)(C)C)/C=C/C(=C/C=C/C(=C/C=C/C=C(/C=C/C=C(/C=C/C2=C(C[C@H](CC2(C)C)O)C)\C)\C)/C)/C"
pred_class, pred_prob = predict_unknown_molecule(new_smiles, loaded_model, device)
print(f"Predicted Class: {pred_class}")
print(f"Confidence Probability: {pred_prob:.4f}")

print("5,12-diphenyltetracene")
new_smiles = "C1=CC=C(C=C1)C2=C3C=CC=CC3=C(C4=CC5=CC=CC=C5C=C42)C6=CC=CC=C6"
pred_class, pred_prob = predict_unknown_molecule(new_smiles, loaded_model, device)
print(f"Predicted Class: {pred_class}")
print(f"Confidence Probability: {pred_prob:.4f}")

print("TIPS-pentacene")
new_smiles = "CC(C)[Si](C#CC1=C2C=C3C=CC=CC3=CC2=C(C4=CC5=CC=CC=C5C=C41)C#C[Si](C(C)C)(C(C)C)C(C)C)(C(C)C)C(C)C"
pred_class, pred_prob = predict_unknown_molecule(new_smiles, loaded_model, device)
print(f"Predicted Class: {pred_class}")
print(f"Confidence Probability: {pred_prob:.4f}")

print("9,10-diphenylanthracene")
new_smiles = "C1=CC=C(C=C1)C2=C3C=CC=CC3=C(C4=CC=CC=C42)C5=CC=CC=C5"
pred_class, pred_prob = predict_unknown_molecule(new_smiles, loaded_model, device)
print(f"Predicted Class: {pred_class}")
print(f"Confidence Probability: {pred_prob:.4f}")

print("9,10-bis(phenylethynyl)anthracene")
new_smiles = "C1=CC=C(C=C1)C#CC2=C3C=CC=CC3=C(C4=CC=CC=C42)C#CC5=CC=CC=C5"
pred_class, pred_prob = predict_unknown_molecule(new_smiles, loaded_model, device)
print(f"Predicted Class: {pred_class}")
print(f"Confidence Probability: {pred_prob:.4f}")

print("rubrene")
new_smiles = "C1=CC=C(C=C1)C2=C3C=CC=CC3=C(C4=C(C5=CC=CC=C5C(=C24)C6=CC=CC=C6)C7=CC=CC=C7)C8=CC=CC=C8"
pred_class, pred_prob = predict_unknown_molecule(new_smiles, loaded_model, device)
print(f"Predicted Class: {pred_class}")
print(f"Confidence Probability: {pred_prob:.4f}")

print("meso-Tetraphenylporphyrin")
new_smiles = "C1=CC=C(C=C1)C2=C3C=CC(=C(C4=NC(=C(C5=CC=C(N5)C(=C6C=CC2=N6)C7=CC=CC=C7)C8=CC=CC=C8)C=C4)C9=CC=CC=C9)N3"
pred_class, pred_prob = predict_unknown_molecule(new_smiles, loaded_model, device)
print(f"Predicted Class: {pred_class}")
print(f"Confidence Probability: {pred_prob:.4f}")

print("Octa-ethylporphyrin")
new_smiles = "CCC1=C(C2=CC3=NC(=CC4=NC(=CC5=C(C(=C(N5)C=C1N2)CC)CC)C(=C4CC)CC)C(=C3CC)CC)CC"
pred_class, pred_prob = predict_unknown_molecule(new_smiles, loaded_model, device)
print(f"Predicted Class: {pred_class}")
print(f"Confidence Probability: {pred_prob:.4f}")

print("Fullerene C60")
new_smiles = "C12=C3C4=C5C6=C1C7=C8C9=C1C%10=C%11C(=C29)C3=C2C3=C4C4=C5C5=C9C6=C7C6=C7C8=C1C1=C8C%10=C%10C%11=C2C2=C3C3=C4C4=C5C5=C%11C%12=C(C6=C95)C7=C1C1=C%12C5=C%11C4=C3C3=C5C(=C81)C%10=C23"
pred_class, pred_prob = predict_unknown_molecule(new_smiles, loaded_model, device)
print(f"Predicted Class: {pred_class}")
print(f"Confidence Probability: {pred_prob:.4f}")
