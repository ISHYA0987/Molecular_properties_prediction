import pandas as pd
from pathlib import Path
from utils import is_valid_smiles


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_esol():

    print("Loading ESOL datasets...")

    train = pd.read_csv(RAW_DIR / "train_data.csv")
    valid = pd.read_csv(RAW_DIR / "valid_data.csv")
    test = pd.read_csv(RAW_DIR / "test_data.csv")

  
    esol = pd.concat([train, valid, test], ignore_index=True)

    print("Total molecules:", esol.shape)

   
    esol = esol[[
        "smiles",
        "measured log solubility in mols per litre"
    ]]

   
    esol = esol.rename(columns={
        "smiles": "SMILES",
        "measured log solubility in mols per litre": "logS"
    })

    
    esol = esol.dropna()

    esol = esol[esol["SMILES"].apply(is_valid_smiles)]

    
    esol = esol.drop_duplicates(subset="SMILES")

   
    esol.to_csv(PROCESSED_DIR / "esol_clean.csv", index=False)

    print("ESOL preprocessing completed")
    print("Final shape:", esol.shape)


if __name__ == "__main__":
    preprocess_esol()