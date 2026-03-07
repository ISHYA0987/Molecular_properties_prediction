import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import joblib
from pathlib import Path

# Paths
DATA_PATH = Path("data/features/tox21_features.csv")
MODEL_PATH = Path("experiments/models/tox21_model.pkl")
VIS_DIR = Path("visuals")

VIS_DIR.mkdir(exist_ok=True)


def evaluate_model():

    df = pd.read_csv(DATA_PATH)

    target = "NR-AR"

    label_columns = [
        "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
        "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
        "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
    ]

    # Remove rows with missing label
    df = df.dropna(subset=[target])

    print("Dataset shape after cleaning:", df.shape)

    # Features
    X = df.drop(columns=label_columns + ["SMILES"], errors="ignore")

    y = df[target]

    # Load model
    model = joblib.load(MODEL_PATH)

    # Predict probabilities
    probs = model.predict_proba(X)

    # For MultiOutputClassifier take first task
    toxicity_probs = probs[0][:, 1]

    # ROC AUC
    auc_score = roc_auc_score(y, toxicity_probs)

    print("ROC AUC:", auc_score)

    fpr, tpr, _ = roc_curve(y, toxicity_probs)

    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0,1],[0,1],"--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Tox21 Toxicity Prediction (NR-AR)")
    plt.legend()

    plt.savefig(VIS_DIR / "tox21_roc_curve.png")

    plt.show()


if __name__ == "__main__":
    evaluate_model()