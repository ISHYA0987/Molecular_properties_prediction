import pandas as pd
from pathlib import Path
import numpy as np
from rdkit import Chem

from .rdkit_features import compute_descriptors, compute_fingerprint

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/features")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 🔥 Substructure alerts (important for toxicity)
def count_substructures(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    nitro = Chem.MolFromSmarts("[N+](=O)[O-]")
    aromatic = Chem.MolFromSmarts("c1ccccc1")

    return {
        "nitro_count": len(mol.GetSubstructMatches(nitro)),
        "aromatic_count": len(mol.GetSubstructMatches(aromatic))
    }


# 🔥 CORE FEATURE GENERATOR (used everywhere)
def build_feature_dict(smiles):
    desc = compute_descriptors(smiles)
    fp = compute_fingerprint(smiles)
    subs = count_substructures(smiles)

    if desc is None or fp is None or subs is None:
        return None

    feature_dict = {}

    # Descriptors
    feature_dict.update(desc)

    # Fingerprints
    for i, bit in enumerate(fp):
        feature_dict[f"fp_{i}"] = bit

    # Substructure alerts
    feature_dict.update(subs)

    return feature_dict


# ✅ ESOL
def process_esol():
    print("Generating ESOL features...")

    df = pd.read_csv(DATA_DIR / "esol_clean.csv")

    feature_rows = []

    for _, row in df.iterrows():
        smi = row["SMILES"]

        features = build_feature_dict(smi)
        if features is None:
            continue

        features["logS"] = row["logS"]
        features["SMILES"] = smi

        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    features_df.to_csv(OUTPUT_DIR / "esol_features.csv", index=False)

    print("ESOL features created")
    print("Shape:", features_df.shape)


# ✅ AMES (GENOTOXICITY)
def process_ames():
    print("Generating Ames features...")

    df = pd.read_csv(DATA_DIR / "ames_clean.csv")

    feature_rows = []

    for _, row in df.iterrows():
        smi = row["SMILES"]

        features = build_feature_dict(smi)
        if features is None:
            continue

        features["genotoxicity"] = row["genotoxicity"]
        features["SMILES"] = smi

        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    features_df.to_csv(OUTPUT_DIR / "ames_features.csv", index=False)

    print("Ames features created")
    print("Shape:", features_df.shape)


# ✅ TOX21
def process_tox21():
    print("Generating Tox21 features...")

    df = pd.read_csv(DATA_DIR / "tox21_clean.csv")

    smiles_col = "SMILES" if "SMILES" in df.columns else "smiles"

    feature_rows = []

    for _, row in df.iterrows():
        smi = row[smiles_col]

        if pd.isna(smi):
            continue

        features = build_feature_dict(smi)
        if features is None:
            continue

        # Add SMILES
        features["SMILES"] = smi

        # Add all labels
        for col in df.columns:
            if col != smiles_col:
                features[col] = row[col]

        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    features_df.to_csv(OUTPUT_DIR / "tox21_features.csv", index=False)

    print("Tox21 features saved")
    print("Shape:", features_df.shape)


# 🔥 Used in Flask (prediction)
def generate_features(smiles):
    features = build_feature_dict(smiles)
    return features


# MAIN
def main():
    process_esol()
    process_ames()
    process_tox21()

    print("All feature engineering completed.")


if __name__ == "__main__":
    main()