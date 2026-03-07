from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path


DATA_PATH = Path("data/processed/tox21_clean.csv")
VIS_DIR = Path("visuals")

VIS_DIR.mkdir(exist_ok=True)


def extract_substructures(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return []

    fragments = []

    for bond in mol.GetBonds():
        atoms = (bond.GetBeginAtom().GetSymbol(),
                 bond.GetEndAtom().GetSymbol())
        fragments.append("-".join(atoms))

    return fragments


def analyze_substructures():

    df = pd.read_csv(DATA_PATH)

    tox_cols = df.columns.drop("SMILES")

    df["tox_score"] = df[tox_cols].sum(axis=1)

    toxic = df[df["tox_score"] > 0]

    fragments = []

    for smi in toxic["SMILES"]:
        fragments.extend(extract_substructures(smi))

    counts = Counter(fragments)

    top = dict(counts.most_common(15))

    plt.bar(top.keys(), top.values())

    plt.xticks(rotation=90)

    plt.title("Most Common Toxic Substructures")

    plt.savefig(VIS_DIR / "toxic_substructures.png")

    plt.show()


if __name__ == "__main__":
    analyze_substructures()