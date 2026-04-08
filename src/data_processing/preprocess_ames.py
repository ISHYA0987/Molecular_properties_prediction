import pandas as pd
from pathlib import Path
from utils import is_valid_smiles


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_ames():

    print("Loading Ames dataset...")
 
    ames = pd.read_csv(RAW_DIR / "ames.csv")

    print("Original shape:", ames.shape)

   
    ames = ames[[
        "SMILES RDKit",
        "Overall"
    ]]

    ames = ames.rename(columns={
        "SMILES RDKit": "SMILES",
        "Overall": "genotoxicity"
    })

    
    ames = ames.dropna()

  
    ames["genotoxicity"] = ames["genotoxicity"].replace(-1, 0)

    
    ames = ames[ames["SMILES"].apply(is_valid_smiles)]

   
    ames = ames.drop_duplicates(subset="SMILES")

  
    ames.to_csv(PROCESSED_DIR / "ames_clean.csv", index=False)

    print("Ames preprocessing completed")
    print("Final shape:", ames.shape)


if __name__ == "__main__":
    preprocess_ames()