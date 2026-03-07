import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

DATA_PATH = Path("data/features/tox21_features.csv")
MODEL_PATH = Path("experiments/models/tox21_model.pkl")

OUTPUT_DIR = Path("visuals")
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_top_toxic_molecules():

    df = pd.read_csv(DATA_PATH)

    model = joblib.load(MODEL_PATH)
    targets=[
        "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
        "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
        "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
    ]
    X = df.drop(columns=targets+['SMILES'], errors='ignore')

    probs = model.predict_proba(X)

    toxicity_score = probs[0][:,1]

    df["toxicity_score"] = toxicity_score

    top_toxic = df.sort_values("toxicity_score", ascending=False).head(20)

    plt.figure(figsize=(12,6))
    plt.bar(range(len(top_toxic)), top_toxic["toxicity_score"])
    plt.xlabel("Molecule Index")
    plt.ylabel("Toxicity Score")
    plt.title("Top 20 Most Toxic Molecules")

    plt.savefig(OUTPUT_DIR / "top_toxic_molecules.png")
    plt.show()


if __name__ == "__main__":
    plot_top_toxic_molecules()