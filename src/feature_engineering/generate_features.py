import pandas as pd
from pathlib import Path
import numpy as np
from rdkit import Chem
from .rdkit_features import compute_descriptors, compute_fingerprint

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/features")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 🔥 Substructure alerts
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


# 🔥 SAFE CLEAN FUNCTION
def clean_feature_dict(feature_dict):
    for k, v in feature_dict.items():
        if v is None:
            feature_dict[k] = 0
        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            feature_dict[k] = 0
    return feature_dict


# 🔥 CORE FEATURE GENERATOR
def build_feature_dict(smiles):
    desc = compute_descriptors(smiles)
    fp = compute_fingerprint(smiles)
    subs = count_substructures(smiles)

    if desc is None or fp is None or subs is None:
        return None

    feature_dict = {}

    # Descriptors
    feature_dict.update(desc)

    # Fingerprints (ensure fixed length)
    for i, bit in enumerate(fp):
        feature_dict[f"fp_{i}"] = int(bit)

    # Substructure features
    feature_dict.update(subs)

    # 🔥 CLEAN EVERYTHING
    feature_dict = clean_feature_dict(feature_dict)

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

        features["logS"] = float(row["logS"])
        features["SMILES"] = smi

        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    # 🔥 FINAL CLEAN
    features_df = features_df.fillna(0)

    features_df.to_csv(OUTPUT_DIR / "esol_features.csv", index=False)

    print("ESOL features created")
    print("Shape:", features_df.shape)


# ✅ AMES
def process_ames():
    print("Generating Ames features...")

    df = pd.read_csv(DATA_DIR / "ames_clean.csv")

    feature_rows = []

    for _, row in df.iterrows():
        smi = row["SMILES"]

        features = build_feature_dict(smi)
        if features is None:
            continue

        features["genotoxicity"] = int(row["genotoxicity"])
        features["SMILES"] = smi

        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    features_df = features_df.fillna(0)

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

        features["SMILES"] = smi

        for col in df.columns:
            if col != smiles_col:
                features[col] = row[col]

        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    features_df = features_df.fillna(0)

    features_df.to_csv(OUTPUT_DIR / "tox21_features.csv", index=False)

    print("Tox21 features saved")
    print("Shape:", features_df.shape)


# 🔥 Used in Flask
def generate_features(smiles):
    features = build_feature_dict(smiles)

    if features is None:
        return None

    # Extra safety
    features = clean_feature_dict(features)

    return features


# MAIN
def main():
    process_esol()
    process_ames()
    process_tox21()

    print("All feature engineering completed.")


if __name__ == "__main__":
    main()