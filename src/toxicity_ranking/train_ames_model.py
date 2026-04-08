import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from pathlib import Path
import numpy as np


DATA_PATH = Path("data/features/ames_features.csv")
MODEL_DIR = Path("experiments/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model():

    df = pd.read_csv(DATA_PATH)

    # 🔥 Drop invalid rows
    df = df.dropna()

    # Features & labels
    X = df.drop(columns=["genotoxicity", "SMILES"], errors="ignore")
    y = df["genotoxicity"]

    # 🔥 Save feature column order (CRITICAL for Flask)
    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, MODEL_DIR / "feature_columns.pkl")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 🔥 Improved model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print("Accuracy:", acc)
    print("ROC-AUC:", auc)

    # 🔥 Feature importance (debugging)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(10)

    print("\nTop Important Features:")
    print(top_features)

    # Save model
    joblib.dump(model, MODEL_DIR / "ames_model.pkl")

    print("\nModel and feature columns saved successfully.")


if __name__ == "__main__":
    train_model()