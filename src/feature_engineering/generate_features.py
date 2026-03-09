import pandas as pd
from pathlib import Path
from rdkit_features import featurize_smiles, compute_descriptors, compute_fingerprint

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/features")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def process_esol():

    print("Generating ESOL features...")

    df = pd.read_csv(DATA_DIR / "esol_clean.csv")

    smiles = df["SMILES"]

    features = featurize_smiles(smiles)

    features["logS"] = df["logS"].values
    features["SMILES"] = smiles.values

    features.to_csv(OUTPUT_DIR / "esol_features.csv", index=False)

    print("ESOL features created")
    print("Shape:", features.shape)

def process_ames():

    print("Generating Ames features...")

    df = pd.read_csv(DATA_DIR / "ames_clean.csv")

    smiles = df["SMILES"]

    features = featurize_smiles(smiles)

    features["genotoxicity"] = df["genotoxicity"].values
    features["SMILES"] = smiles.values

    features.to_csv(OUTPUT_DIR / "ames_features.csv", index=False)

    print("Ames features created")
    print("Shape:", features.shape)

def process_tox21():

    print("Generating Tox21 features...")

    df = pd.read_csv(DATA_DIR / "tox21_clean.csv")

   
    smiles_col = "SMILES" if "SMILES" in df.columns else "smiles"

    feature_rows = []

    for _, row in df.iterrows():

        smi = row[smiles_col]

        if pd.isna(smi):
            continue

        desc = compute_descriptors(smi)
        fp = compute_fingerprint(smi)

        if desc is None or fp is None:
            continue

        feature_dict = {}

        
        feature_dict.update(desc)

        
        for i, bit in enumerate(fp):
            feature_dict[f"fp_{i}"] = bit

       
        feature_dict["SMILES"] = smi

       
        for col in df.columns:
            if col != smiles_col:
                feature_dict[col] = row[col]

        feature_rows.append(feature_dict)

    features_df = pd.DataFrame(feature_rows)

    print("Feature dataset shape:", features_df.shape)

    features_df.to_csv(OUTPUT_DIR / "tox21_features.csv", index=False)

    print("Tox21 features saved")


def main():

    process_esol()
    process_ames()
    process_tox21()

    print("All feature engineering completed.")


if __name__ == "__main__":
    main()