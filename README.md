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
├── [notebooks/ or scripts/]      # [what they do]
├── [models/]                     # [trained models]
├── [results/]                    # [figures, tables]
└── README.md
```

## Dataset
`Dataset470K_with_smiles.csv` contains ~470,000 molecules.

| Column | Description |
|---|---|
| `smiles` | [SMILES string] |
| [column] | [meaning, units] |

- Source / how it was generated: [method, level of theory, database]
- Units: [eV / kcal/mol]

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
