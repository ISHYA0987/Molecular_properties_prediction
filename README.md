# Molecular Properties Prediction using Machine Learning

## Project Overview

This project builds a **machine learning pipeline to predict molecular properties and toxicity using chemical structure information (SMILES)**.

The system uses **RDKit molecular descriptors and fingerprints** to train machine learning models that predict:

* **Toxicity endpoints (Tox21 dataset)**
* **Mutagenicity (Ames dataset)**
* **Solubility (ESOL dataset)**

The goal is to create a **computational screening system that helps identify potentially toxic or harmful molecules early**, reducing the need for expensive laboratory testing.

---

# Objectives

* Predict **molecular toxicity** using machine learning
* Analyze **chemical properties influencing toxicity**
* Predict **molecular solubility**
* Build a **complete molecular ML pipeline**

---

# Datasets Used

### 1. Tox21 Dataset

Used for **toxicity classification** across multiple biological targets.

Contains **12 toxicity endpoints**, including:

* NR-AR
* NR-AR-LBD
* NR-AhR
* NR-Aromatase
* NR-ER
* NR-ER-LBD
* NR-PPAR-gamma
* SR-ARE
* SR-ATAD5
* SR-HSE
* SR-MMP
* SR-p53

### 2. Ames Mutagenicity Dataset

Used for **mutagenicity classification**.

Predicts whether a molecule is:

* Mutagenic (toxic)
* Non-mutagenic

### 3. ESOL Dataset

Used for **solubility prediction**.

Predicts:

* **logS (aqueous solubility)**

---

# Technologies Used

| Tool         | Purpose                     |
| ------------ | --------------------------- |
| Python       | Programming language        |
| RDKit        | Chemical feature extraction |
| Pandas       | Data processing             |
| NumPy        | Numerical computation       |
| Scikit-learn | Machine learning models     |
| Matplotlib   | Visualization               |
| Seaborn      | Statistical visualization   |

---

# Machine Learning Models

| Task                        | Model                                 |
| --------------------------- | ------------------------------------- |
| Toxicity Prediction (Tox21) | Random Forest (MultiOutputClassifier) |
| Ames Mutagenicity           | Random Forest Classifier              |
| ESOL Solubility             | Random Forest Regressor               |

---

# Feature Engineering

Chemical features extracted using **RDKit** include:

### Molecular Descriptors

* Molecular Weight (MolWt)
* LogP
* Topological Polar Surface Area (TPSA)
* Hydrogen Bond Donors
* Hydrogen Bond Acceptors
* Rotatable Bonds

### Molecular Fingerprints

* Morgan Fingerprints
* 1024-bit molecular fingerprint representation

These features convert **SMILES chemical structures into machine learning-ready numerical vectors**.

---

# Project Pipeline

```
SMILES Molecular Structure
        ↓
Data Preprocessing
        ↓
RDKit Feature Engineering
        ↓
Machine Learning Model Training
        ↓
Evaluation
        ↓
Visualization
        ↓
Final Molecular Prediction Table
```

---

# Project Structure

```
Molecular_properties_prediction/

data/
 ├── raw/
 ├── processed/
 └── features/

experiments/
 └── models/

src/
 ├── preprocessing/
 ├── feature_engineering/
 ├── toxicity_classification/
 ├── toxicity_ranking/
 ├── property_prediction/
 ├── Evaluation/
 ├── visualization/
 └── pipeline/

visuals/

results/
```

---

# Results

### Toxicity Prediction

ROC curves generated for all **Tox21 toxicity endpoints**.

Example:

```
visuals/tox21_roc_NR-AR.png
```

### Ames Mutagenicity

Confusion matrix visualization:

```
visuals/ames_confusion_matrix.png
```

### ESOL Solubility

Scatter plot comparing:

* Actual solubility
* Predicted solubility

```
visuals/esol_prediction_plot.png
```

---

# Final Output

The project generates a **combined molecular prediction table**:

| SMILES             | Toxicity Type     | Ames Mutagenicity         | Predicted Solubility | Chemical Properties   |
| ------------------ | ----------------- | ------------------------- | -------------------- | --------------------- |
| Molecule structure | Toxic probability | Mutagenic / Non-Mutagenic | logS                 | Molecular descriptors |

Saved as:

```
results/final_molecular_predictions.csv
```

---

# How to Run the Project

### 1️⃣ Install dependencies

```
pip install pandas numpy scikit-learn matplotlib seaborn rdkit joblib
```

### 2️⃣ Run preprocessing

```
python src/preprocessing/preprocess_data.py
```

### 3️⃣ Generate molecular features

```
python src/feature_engineering/generate_features.py
```

### 4️⃣ Train models

```
python src/toxicity_classification/train_tox21_model.py
python src/toxicity_ranking/train_ames_model.py
python src/property_prediction/train_esol_model.py
```

### 5️⃣ Evaluate models

```
python src/Evaluation/evaluate_tox21.py
python src/Evaluation/evaluate_ames.py
python src/Evaluation/evaluate_esol.py
```

### 6️⃣ Run full pipeline

```
python src/pipeline/run_full_pipeline.py
```

---

# Future Improvements

* Deep learning models for molecular prediction
* Graph neural networks for molecular graphs
* Web application for toxicity prediction
* Drug discovery applications

---

# Applications

* Drug discovery
* Chemical safety screening
* Environmental toxicity prediction
* Pharmaceutical research

---

# Author

Developed as a **Machine Learning Project on Molecular Property Prediction**.

---
