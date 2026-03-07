import pandas as pd
from pathlib import Path
from utils import is_valid_smiles


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_tox21():

    print("Loading Tox21 dataset...")

    tox21 = pd.read_csv(RAW_DIR / "tox21.csv")

    print("Original shape:", tox21.shape)

    # Select toxicity endpoints
    tox21 = tox21[[
        "smiles",
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53"
    ]]

    # Rename column
    tox21 = tox21.rename(columns={"smiles": "SMILES"})

    # Remove rows where all targets are missing
    tox21 = tox21.dropna(
        subset=[
            "NR-AR",
            "NR-AhR",
            "NR-ER",
            "SR-ARE"
        ],
        how="all"
    )

    # Validate SMILES
    tox21 = tox21[tox21["SMILES"].apply(is_valid_smiles)]

    # Remove duplicates
    tox21 = tox21.drop_duplicates(subset="SMILES")

    # Save dataset
    tox21.to_csv(PROCESSED_DIR / "tox21_clean.csv", index=False)

    print("Tox21 preprocessing completed")
    print("Final shape:", tox21.shape)


if __name__ == "__main__":
    preprocess_tox21()