import pandas as pd
from pathlib import Path
from rdkit_features import featurize_smiles


DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/features")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_esol():

    print("Generating ESOL features...")

    df = pd.read_csv(DATA_DIR / "esol_clean.csv")

    smiles = df["SMILES"]

    features = featurize_smiles(smiles)

    features["logS"] = df["logS"].values

    features.to_csv(OUTPUT_DIR / "esol_features.csv", index=False)

    print("ESOL features created")


def process_ames():

    print("Generating Ames features...")

    df = pd.read_csv(DATA_DIR / "ames_clean.csv")

    smiles = df["SMILES"]

    features = featurize_smiles(smiles)

    features["genotoxicity"] = df["genotoxicity"].values

    features.to_csv(OUTPUT_DIR / "ames_features.csv", index=False)

    print("Ames features created")


def process_tox21():

    print("Generating Tox21 features...")

    df = pd.read_csv(DATA_DIR / "tox21_clean.csv")

    smiles = df["SMILES"]

    features = featurize_smiles(smiles)

    labels = df.drop(columns=["SMILES"])

    features = pd.concat([features, labels], axis=1)

    features.to_csv(OUTPUT_DIR / "tox21_features.csv", index=False)

    print("Tox21 features created")


def main():

    process_esol()
    process_ames()
    process_tox21()

    print("All feature engineering completed.")


if __name__ == "__main__":
    main()