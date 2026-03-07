import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import joblib
from pathlib import Path

# Paths
DATA_PATH = Path("data/features/ames_features.csv")
MODEL_PATH = Path("experiments/models/ames_model.pkl")
VIS_DIR = Path("visuals")

VIS_DIR.mkdir(exist_ok=True)


def evaluate_model():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Features (remove target + SMILES)
    X = df.drop(columns=["genotoxicity", "SMILES"], errors="ignore")

    # Target
    y = df["genotoxicity"]

    print("Dataset shape:", df.shape)

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Predict
    preds = model.predict(X)

    # Confusion matrix
    cm = confusion_matrix(y, preds)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Toxic","Toxic"],
        yticklabels=["Non-Toxic","Toxic"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Ames Mutagenicity")

    plt.savefig(VIS_DIR / "ames_confusion_matrix.png")

    plt.show()


if __name__ == "__main__":
    evaluate_model()