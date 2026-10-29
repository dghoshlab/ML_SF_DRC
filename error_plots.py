import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, average_precision_score
)

ann_s_df  = pd.read_csv('ann_struc_pred1.2_same_split_seed42.csv')
ann_se_df = pd.read_csv('ann_se_pred1.2_same_split42.csv')
gnn_s_df  = pd.read_csv('gnn_s_seed42n_test_predictions.csv')
gnn_se_df = pd.read_csv('gnn_se_seed42n_test_predictions.csv')

true_col  = 'DRC_class'
pred_col  = 'Predicted DRC_class'
prob_col  = 'Predicted_Probability'
drc_col   = 'DRC'          
threshold = 0.05           

# Extract arrays
y_test         = ann_s_df[true_col].values
ann_s_preds    = ann_s_df[pred_col].values
ann_se_preds   = ann_se_df[pred_col].values
gnn_s_preds    = gnn_s_df['pred'].values
gnn_se_preds   = gnn_se_df['pred'].values

ann_s_probs    = ann_s_df[prob_col].values
ann_se_probs   = ann_se_df[prob_col].values
gnn_s_probs    = gnn_s_df['prob'].values
gnn_se_probs   = gnn_se_df['prob'].values

# DRC values if available
has_drc = drc_col in ann_s_df.columns
if has_drc:
    test_drc = ann_s_df[drc_col].values

models_probs = {
    'ANN_S':  ann_s_probs,
    'ANN_SE': ann_se_probs,
    'GNN_S':  gnn_s_probs,
    'GNN_SE': gnn_se_probs
}

models_preds = {
    'ANN_S':  ann_s_preds,
    'ANN_SE': ann_se_preds,
    'GNN_S':  gnn_s_preds,
    'GNN_SE': gnn_se_preds
}

colors = ['steelblue', 'coral', 'green', 'purple']
DRC_cutoff = 0.05  

# PLOT 1: CALIBRATION / RELIABILITY DIAGRAM

plt.figure(figsize=(8, 6))
for (name, probs), color in zip(models_probs.items(), colors):
    fraction_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
    plt.plot(mean_pred, fraction_pos, marker='o', color=color, label=name)

plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.xlabel('Mean Predicted Probability', fontsize=12)
plt.ylabel('Fraction of Positives', fontsize=12)
plt.title('Calibration Curves (Reliability Diagram)', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plot1_calibration_curve.pdf', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: plot1_calibration_curve.pdf")

# PLOT 2: CONFIDENCE DISTRIBUTION

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, probs), color in zip(axes.flat, models_probs.items(), colors):
    correct_mask = (probs > threshold).astype(int) == y_test
    
    ax.hist(probs[correct_mask],  bins=50, alpha=0.6,
            color='steelblue', label='Correct')
    ax.hist(probs[~correct_mask], bins=50, alpha=0.6,
            color='coral',     label='Incorrect')
    ax.axvline(threshold, color='black', linestyle='--',
               linewidth=1.5,  label=f'Threshold={threshold}')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=10)

plt.suptitle('Prediction Confidence Distribution', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_confidence_distribution.pdf', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: plot2_confidence_distribution.pdf")

# PLOT 3: ERROR DISTRIBUTION BY DRC VALUE
# Shows whether errors cluster near the DRC threshold

if has_drc:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (name, preds), color in zip(axes.flat, models_preds.items(), colors):
        errors = preds != y_test

        ax.hist(test_drc[~errors], bins=50, alpha=0.6,
                color='steelblue', label='Correct')
        ax.hist(test_drc[errors],  bins=50, alpha=0.6,
                color='coral',     label='Incorrect')
        ax.axvline(DRC_cutoff, color='black', linestyle='--',
                   linewidth=1.5, label=f'DRC cutoff={DRC_cutoff}')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('DRC Value', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.legend(fontsize=10)

    plt.suptitle('Error Distribution Along DRC Values',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot3_error_by_drc.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: plot3_error_by_drc.pdf")
else:
    print("Skipped plot3 — DRC column not found in CSV")


# PLOT 4: CONFUSION MATRICES

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax, (name, preds) in zip(axes, models_preds.items()):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'],
                ax=ax, cbar=False, annot_kws={"size": 12})
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)

plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot4_confusion_matrices.pdf', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: plot4_confusion_matrices.pdf")


# PLOT 5: ROC AND PR CURVES TOGETHER

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for (name, probs), color in zip(models_probs.items(), colors):
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    ax1.plot(fpr, tpr, color=color, linewidth=2,
             label=f'{name} (AUC={auc_val:.3f})')

    # PR curve
    prec, rec, _ = precision_recall_curve(y_test, probs)
    pr_auc_val = average_precision_score(y_test, probs)
    ax2.plot(rec, prec, color=color, linewidth=2,
             label=f'{name} (AUC={pr_auc_val:.3f})')

# ROC
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curves', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])

# PR
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('plot5_roc_pr_curves.pdf', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: plot5_roc_pr_curves.pdf")


# PLOT 6: ERROR DISTANCE FROM DECISION THRESHOLD
# Quantifies how close errors are to decision boundary


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, probs) in zip(axes.flat, models_probs.items()):
    errors = (probs > threshold).astype(int) != y_test

    error_distances   = np.abs(probs[errors]  - threshold)
    correct_distances = np.abs(probs[~errors] - threshold)

    bp = ax.boxplot(
        [error_distances, correct_distances],
        labels=['Errors', 'Correct'],
        patch_artist=True,
        boxprops=dict(facecolor='coral'),
        medianprops=dict(color='black', linewidth=2)
    )
    bp['boxes'][1].set_facecolor('steelblue')

    ax.set_ylabel('Distance from Decision Threshold', fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')

    # Annotate medians
    med_err = np.median(error_distances)
    med_cor = np.median(correct_distances)
    ax.text(1, med_err + 0.01, f'{med_err:.3f}', ha='center', fontsize=9)
    ax.text(2, med_cor + 0.01, f'{med_cor:.3f}', ha='center', fontsize=9)

plt.suptitle('Distance of Predictions from Decision Threshold',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot6_error_distance_threshold.pdf', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: plot6_error_distance_threshold.pdf")


# PLOT 7: MISCLASSIFICATION OVERLAP HEATMAP
# Shows architecture-specific vs data-specific errors


model_names = list(models_preds.keys())
miss = {
    name: set(np.where(preds != y_test)[0])
    for name, preds in models_preds.items()
}

n = len(model_names)
matrix = np.zeros((n, n))
for i, a in enumerate(model_names):
    for j, b in enumerate(model_names):
        if i == j:
            matrix[i][j] = 1.0
        else:
            overlap = len(miss[a] & miss[b])
            union   = len(miss[a] | miss[b])
            matrix[i][j] = overlap / union if union > 0 else 0

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, label='Jaccard Similarity')

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(model_names, fontsize=12)
ax.set_yticklabels(model_names, fontsize=12)

for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{matrix[i][j]:.3f}',
                ha='center', va='center',
                color='black', fontsize=11)

ax.set_title('Pairwise Misclassification Jaccard Similarity',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot7_misclassification_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: plot7_misclassification_heatmap.pdf")


# PLOT 8: UNIQUE VS SHARED ERRORS BAR CHART

all_wrong = miss['ANN_S'] & miss['ANN_SE'] & miss['GNN_S'] & miss['GNN_SE']
unique_errors = []
for name, m in miss.items():
    others = set().union(*[v for k, v in miss.items() if k != name])
    unique_errors.append(len(m - others))

shared_count = len(all_wrong)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, unique_errors, width,
               label='Unique misclassifications', color='coral')
bars2 = ax.bar(x + width/2, [shared_count]*4, width,
               label='Shared by all models', color='steelblue')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Number of Molecules', fontsize=12)
ax.set_title('Unique vs Shared Misclassifications',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.legend(fontsize=11)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 20,
            str(int(bar.get_height())),
            ha='center', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 20,
            str(int(bar.get_height())),
            ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('plot8_unique_vs_shared_errors.pdf', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: plot8_unique_vs_shared_errors.pdf")

print("\nAll plots saved successfully.")

