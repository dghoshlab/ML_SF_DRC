# ML_SF_DRC

Machine-learning models to classify singlet-fission-efficient molecules based on their diradical character value

Research group: Debashree Ghosh Lab, Indian Association for the Cultivation of Science (IACS), Kolkata

## Overview

- **Task:** classification
- **Models:** ANN, GNN

## Repository structure
```
ML_SF_DRC/
├── Dataset470K_with_smiles.csv   # full dataset (Git LFS, ~260 MB)
├── permutation_importance.py, split_dataset.py, scaffold_split_new.py   # [Permutation importance plot;Random splitting and scaffold splitting of data]
├── ann_modified_val_similarsplit.py, ann_se_val_similarsplit.py, gnn_s_val_ss.py, gnn_se_val_ss.py                    # [ANN_S, ANN_SE, GNN_S, GNN_SE models]
├── ann_struc_pred1.2_same_split_seed42.csv, ann_se_pred1.2_same_split42.csv, gnn_s_seed42n_test_predictions.csv, gnn_se_seed42n_test_predictions.csv, test_predictions.csv                        # [.csv files containing model predictions on test data]
Error_plots.py, molecules_analysis.py #[Error analysis] 
└── README.md
```

## Dataset
`Dataset470K_with_smiles.csv` contains ~470,000 molecules.
## Installation
```bash
git clone https://github.com/dghoshlab/ML_SF_DRC.git
cd ML_SF_DRC
pip install -r requirements.txt     # or: conda env create -f environment.yml
```

**Note: the dataset is stored with Git LFS.** Install it before cloning:
```bash
git lfs install
git clone https://github.com/dghoshlab/ML_SF_DRC.git
```
If you already cloned without LFS, run `git lfs pull`.

Python version: [3.x]. Main libraries: rdkit, scikit-learn, pandas, numpy

## Usage
```bash
# Train
python [train_script.py] --data Dataset470K_with_smiles.csv



## Results
[One or two key numbers, e.g., test MAE / R², and a figure:]
`![Parity plot](results/parity.png)`



## Contact
Debashree Ghosh Lab, IACS Kolkata# ML_SF_DRC
email: pcdg@iacs.res.in
