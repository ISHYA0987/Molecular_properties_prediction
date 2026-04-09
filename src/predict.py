import numpy as np
import joblib
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
import base64
from io import BytesIO

from src.feature_engineering.generate_features import generate_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🔥 Load models
ames_model = joblib.load(os.path.join(BASE_DIR, "experiments/models/ames_model.pkl"))
tox21_model = joblib.load(os.path.join(BASE_DIR, "experiments/models/tox21_model.pkl"))
esol_model = joblib.load(os.path.join(BASE_DIR, "experiments/models/esol_model.pkl"))

# 🔥 Load feature columns (VERY IMPORTANT)
AMES_FEATURES = joblib.load(os.path.join(BASE_DIR, "experiments/models/feature_columns.pkl"))
TOX21_FEATURES = joblib.load(os.path.join(BASE_DIR, "experiments/models/tox21_features.pkl"))
ESOL_FEATURES = joblib.load(os.path.join(BASE_DIR, "experiments/models/esol_features.pkl"))


# ✅ Validate SMILES
def validate_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None


# 🔥 Rule-based toxicity (SMARTS-based)
def rule_based_toxicity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    alerts = []

    if mol is None:
        return alerts

    # Nitro group
    nitro = Chem.MolFromSmarts("[N+](=O)[O-]")
    if mol.HasSubstructMatch(nitro):
        alerts.append("Nitro group (mutagenic)")

    # PAH detection
    if rdMolDescriptors.CalcNumAromaticRings(mol) >= 4:
        alerts.append("Polycyclic aromatic hydrocarbon (genotoxic risk)")

    return alerts


# 🔬 Molecule image
def generate_molecule_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    img = Draw.MolToImage(mol, size=(300, 300))
    buffer = BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()


# 🚀 MAIN PREDICTION
def predict_from_smiles(smiles):

    if not validate_smiles(smiles):
        return {"error": "Invalid SMILES"}

    features = generate_features(smiles)
    if features is None:
        return {"error": "Feature generation failed"}

    df = pd.DataFrame([features])

    # 🔥 Align features
    ames_input = df.reindex(columns=AMES_FEATURES, fill_value=0).fillna(0)
    tox21_input = df.reindex(columns=TOX21_FEATURES, fill_value=0).fillna(0)
    esol_input = df.reindex(columns=ESOL_FEATURES, fill_value=0).fillna(0)

    # 🔥 Predictions
    ames_pred = int(np.array(ames_model.predict(ames_input)).flatten()[0])
    tox21_pred = int(np.array(tox21_model.predict(tox21_input)).flatten()[0])

    try:
        esol_pred = float(np.array(esol_model.predict(esol_input)).flatten()[0])
    except:
        esol_pred = None

    # 🔥 Rule-based enhancement
    alerts = rule_based_toxicity(smiles)

    if any("Nitro" in a or "Polycyclic" in a for a in alerts):
        ames_pred = 1

    # 🔥 Confidence (safe)
    try:
        prob = ames_model.predict_proba(ames_input)[0]
        confidence = round(float(max(prob)), 2)
    except:
        confidence = None

    return {
        "smiles": smiles,
        "Ames": ames_pred,
        "Tox21": tox21_pred,
        "Solubility": round(esol_pred, 3) if esol_pred is not None else "N/A",
        "alerts": alerts,
        "confidence": confidence,
        "image": generate_molecule_image(smiles)
    }