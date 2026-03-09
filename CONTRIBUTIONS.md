# Project Contributions

This document describes the responsibilities and contributions made during the development of the **Molecular Properties Prediction Project**.

---

# 1. Data Preprocessing

Responsibilities:

* Cleaning raw molecular datasets
* Handling missing values
* Standardizing SMILES format
* Removing invalid molecules
* Merging dataset splits (train, test, validation)

Output:

```
data/processed/
```

Processed datasets ready for feature extraction.

---

# 2. Feature Extraction

Responsibilities:

* Converting SMILES strings into numerical features
* Generating molecular descriptors using RDKit
* Computing Morgan fingerprints
* Preparing machine-learning-ready feature vectors

Extracted Features:

* Molecular Weight
* LogP
* TPSA
* Hydrogen Bond Donors
* Hydrogen Bond Acceptors
* Rotatable Bonds
* 1024-bit molecular fingerprints

Output:

```
data/features/
```

---

# 3. Model Training

Responsibilities:

Training machine learning models for:

### Toxicity Prediction (Tox21)

* Multi-task classification
* Random Forest with MultiOutputClassifier

### Ames Mutagenicity

* Binary classification
* Random Forest Classifier

### ESOL Solubility

* Regression
* Random Forest Regressor

Trained models saved in:

```
experiments/models/
```

---

# 4. Model Evaluation

Responsibilities:

Evaluating model performance using appropriate metrics.

### Tox21

* ROC Curve
* ROC-AUC score

### Ames

* Confusion Matrix
* Classification accuracy

### ESOL

* Root Mean Square Error (RMSE)
* Scatter plot analysis

Evaluation scripts:

```
src/Evaluation/
```

---

# 5. Visualization

Responsibilities:

Generating visual insights from model predictions.

Visualizations include:

* ROC curves for toxicity endpoints
* Ames mutagenicity confusion matrix
* Solubility prediction scatter plots

Output saved in:

```
visuals/
```

---

# 6. Final Prediction Pipeline

Responsibilities:

Combining predictions from all trained models into a single output table.

Final output includes:

* Molecule structure (SMILES)
* Toxicity probabilities
* Mutagenicity prediction
* Solubility prediction
* Molecular chemical properties

Output file:

```
results/final_molecular_predictions.csv
```

---

# Summary

The project integrates **data preprocessing, feature engineering, machine learning, evaluation, and visualization** into a complete pipeline for predicting molecular toxicity and properties.

---
