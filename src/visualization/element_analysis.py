import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from collections import Counter
from pathlib import Path

DATA_PATH = Path("data/processed/tox21_clean.csv")
OUTPUT_DIR = Path("visuals")

OUTPUT_DIR.mkdir(exist_ok=True)


def extract_elements(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return []

    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def analyze_elements():

    df = pd.read_csv(DATA_PATH)

    
    toxicity_columns = df.columns.drop("SMILES")

    df["toxicity_sum"] = df[toxicity_columns].sum(axis=1)

    toxic_molecules = df[df["toxicity_sum"] > 0]

    elements = []

    for smi in toxic_molecules["SMILES"]:
        elements.extend(extract_elements(smi))

    element_counts = Counter(elements)

    elements = list(element_counts.keys())
    counts = list(element_counts.values())

    plt.figure(figsize=(10,6))

    plt.bar(elements, counts)

    plt.xlabel("Chemical Element")
    plt.ylabel("Frequency in Toxic Molecules")

    plt.title("Element Presence in Toxic Molecules")

    plt.savefig(OUTPUT_DIR / "element_toxicity_analysis.png")

    plt.show()


if __name__ == "__main__":
    analyze_elements()