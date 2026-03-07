from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/tox21_clean.csv")

def draw_top_toxic():

    df = pd.read_csv(DATA_PATH)

    toxicity_cols = df.columns.drop("SMILES")

    df["toxicity_score"] = df[toxicity_cols].sum(axis=1)

    top = df.sort_values("toxicity_score", ascending=False).head(9)

    mols = [Chem.MolFromSmiles(s) for s in top["SMILES"]]

    img = Draw.MolsToGridImage(mols, molsPerRow=3)

    img.save("visuals/top_toxic_structures.png")


if __name__ == "__main__":
    draw_top_toxic()