import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import joblib
from pathlib import Path
import numpy as np

# Paths
DATA_PATH = Path("data/features/esol_features.csv")
MODEL_PATH = Path("experiments/models/esol_model.pkl")
VIS_DIR = Path("visuals")

VIS_DIR.mkdir(exist_ok=True)


def evaluate_model():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    # Features (remove target + SMILES)
    X = df.drop(columns=["logS", "SMILES"], errors="ignore")

    # Target
    y = df["logS"]

    # Load model
    model = joblib.load(MODEL_PATH)

    # Predict
    preds = model.predict(X)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y, preds))

    print("RMSE:", rmse)

    # Scatter plot
    plt.figure(figsize=(6,5))

    plt.scatter(y, preds, alpha=0.6)

    plt.xlabel("Actual Solubility (logS)")
    plt.ylabel("Predicted Solubility (logS)")

    plt.title("ESOL Solubility Prediction")

    # Ideal prediction line
    min_val = min(y.min(), preds.min())
    max_val = max(y.max(), preds.max())

    plt.plot([min_val, max_val], [min_val, max_val], "r--")

    plt.savefig(VIS_DIR / "esol_prediction_plot.png")

    plt.show()


if __name__ == "__main__":
    evaluate_model()