import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from pathlib import Path


DATA_PATH = Path("data/features/ames_features.csv")
MODEL_DIR = Path("experiments/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model():

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["genotoxicity", "SMILES"], errors="ignore")
    y = df["genotoxicity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    print("Accuracy:", acc)

    joblib.dump(model, MODEL_DIR / "ames_model.pkl")


if __name__ == "__main__":
    train_model()