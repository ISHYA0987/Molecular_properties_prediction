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

ames_model = joblib.load(os.path.join(BASE_DIR, "experiments/models/ames_model.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "experiments/models/feature_columns.pkl"))


def validate_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None


def rule_based_toxicity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    alerts = []

    if mol is None:
        return alerts

    # Nitro
    nitro = Chem.MolFromSmarts("[N+](=O)[O-]")
    if mol.HasSubstructMatch(nitro):
        alerts.append("Nitro group (mutagenic)")

    # 🔥 PAH detection
    if rdMolDescriptors.CalcNumAromaticRings(mol) >= 4:
        alerts.append("Polycyclic aromatic hydrocarbon (genotoxic risk)")

    return alerts


def generate_molecule_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    img = Draw.MolToImage(mol, size=(300, 300))
    buffer = BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()


def predict_from_smiles(smiles):

    if not validate_smiles(smiles):
        return {"error": "Invalid SMILES"}

    features = generate_features(smiles)
    if features is None:
        return {"error": "Feature generation failed"}

    df = pd.DataFrame([features])
    X = df.reindex(columns=feature_columns, fill_value=0)

    pred = int(ames_model.predict(X)[0])

    alerts = rule_based_toxicity(smiles)

    # 🔥 Override for PAH
    if any("Polycyclic" in a for a in alerts):
        pred = 1

    try:
        prob = ames_model.predict_proba(X)[0]
        confidence = round(float(max(prob)), 2)
    except:
        confidence = None

    return {
        "smiles": smiles,
        "Ames": pred,
        "alerts": alerts,
        "confidence": confidence,
        "image": generate_molecule_image(smiles)
    }