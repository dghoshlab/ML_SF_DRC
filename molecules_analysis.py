import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn3
from upsetplot import from_memberships, plot
import matplotlib.patches as patches
from upsetplot import from_contents, plot as upset_plot
df1= pd.read_csv('ann_struc_pred1.2_same_split_seed42.csv')
df2= pd.read_csv('ann_se_pred1.2_same_split42.csv')
df3= pd.read_csv('gnn_s_seed42n_test_predictions.csv')
df4= pd.read_csv('gnn_se_seed42n_test_predictions.csv')

test_ids = df1['PubChemID']
y_test = df1['DRC_class']
pred_ann_s = df1['Predicted DRC_class']
pred_ann_se = df2['Predicted DRC_class']
pred_gnn_s = df3['pred']
pred_gnn_se = df4['pred']


# After model prediction
results = pd.DataFrame({
    'molecule_id': test_ids,        
    'true_label': y_test,           
    'pred_ANN_S': pred_ann_s,       
    'pred_ANN_SE': pred_ann_se,
    'pred_GNN_S': pred_gnn_s,
    'pred_GNN_SE': pred_gnn_se,
})

results.to_csv('test_predictions.csv', index=False)

# Find misclassified molecules for each model
results['wrong_ANN_S']  = results['pred_ANN_S']  != results['true_label']
results['wrong_ANN_SE'] = results['pred_ANN_SE'] != results['true_label']
results['wrong_GNN_S']  = results['pred_GNN_S']  != results['true_label']
results['wrong_GNN_SE'] = results['pred_GNN_SE'] != results['true_label']

# Get sets of misclassified molecule IDs
miss_ann_s  = set(results[results['wrong_ANN_S']]['molecule_id'])
miss_ann_se = set(results[results['wrong_ANN_SE']]['molecule_id'])
miss_gnn_s  = set(results[results['wrong_GNN_S']]['molecule_id'])
miss_gnn_se = set(results[results['wrong_GNN_SE']]['molecule_id'])

models = {
    'ANN_S': miss_ann_s,
    'ANN_SE': miss_ann_se,
    'GNN_S': miss_gnn_s,
    'GNN_SE': miss_gnn_se
}

# Pairwise overlap
print("Pairwise misclassification overlap:\n")
model_names = list(models.keys())
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        a, b = model_names[i], model_names[j]
        overlap = models[a] & models[b]
        union   = models[a] | models[b]
        jaccard = len(overlap) / len(union)
        print(f"{a} vs {b}:")
        print(f"  Shared misclassifications: {len(overlap)}")
        print(f"  Jaccard similarity: {jaccard:.3f}\n")

# Shared across ALL models
all_wrong = miss_ann_s & miss_ann_se & miss_gnn_s & miss_gnn_se
print(f"Misclassified by ALL models: {len(all_wrong)}")

# Unique to each model
for name, miss in models.items():
    others = set().union(*[v for k,v in models.items() if k != name])
    unique = miss - others
    print(f"Unique to {name}: {len(unique)}")

# Venn diagram (up to 3 sets)
venn3([miss_ann_s, miss_gnn_s, miss_gnn_se],
      set_labels=('ANN_S', 'GNN_S', 'GNN_SE'))
plt.title('Misclassification Overlap')
plt.savefig('overlap_venn.png')

# UpSet plot
memberships = from_memberships(
    [list(models.keys()) for mol in set().union(*models.values())
     if True],  # build membership list
    data=None
)


contents = {
    'ANN_S': miss_ann_s,
    'ANN_SE': miss_ann_se,
    'GNN_S': miss_gnn_s,
    'GNN_SE': miss_gnn_se
}

upset_data = from_contents(contents)
upset_plot(upset_data,
           sort_by='cardinality',
           show_counts=True)
plt.suptitle('Misclassification Overlap Across Models',
             fontsize=14, fontweight='bold')
#plt.savefig('misclassification_overlap.png', dpi=300, bbox_inches='tight')
plt.show()



models = ['ANN_S', 'ANN_SE', 'GNN_S', 'GNN_SE']
miss = {
    'ANN_S': miss_ann_s,
    'ANN_SE': miss_ann_se,
    'GNN_S': miss_gnn_s,
    'GNN_SE': miss_gnn_se
}


n = len(models)
matrix = np.zeros((n, n))
for i, a in enumerate(models):
    for j, b in enumerate(models):
        if i == j:
            matrix[i][j] = 1.0
        else:
            overlap = len(miss[a] & miss[b])
            union   = len(miss[a] | miss[b])
            matrix[i][j] = overlap / union


fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, label='Jaccard Similarity')

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(models, fontsize=12)
ax.set_yticklabels(models, fontsize=12)


for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{matrix[i][j]:.3f}',
                ha='center', va='center',
                color='black', fontsize=11)

ax.set_title('Pairwise Misclassification Jaccard Similarity',
             fontsize=13, fontweight='bold')
plt.tight_layout()
#plt.savefig('misclassification_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(7, 5))

models_list = ['ANN_S', 'ANN_SE', 'GNN_S', 'GNN_SE']
unique_errors = [2734, 435, 598, 301]
shared_errors = [359, 359, 359, 359]

x = np.arange(len(models_list))
width = 0.35

bars1 = ax.bar(x - width/2, unique_errors, width,
               label='Unique misclassifications', color='steelblue')
bars2 = ax.bar(x + width/2, shared_errors, width,
               label='Shared by all models', color='coral')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Number of molecules', fontsize=12)
ax.set_title('Unique vs Shared Misclassifications',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_list, fontsize=11)
ax.legend(fontsize=11)


for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            str(int(bar.get_height())), ha='center', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            str(int(bar.get_height())), ha='center', fontsize=10)

plt.tight_layout()
#plt.savefig('unique_vs_shared_errors.png', dpi=300, bbox_inches='tight')
plt.show()
