import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_PATH = Path("data/features/tox21_features.csv")
MODEL_PATH = Path("experiments/models/tox21_model.pkl")


def train_model():

    df = pd.read_csv(DATA_PATH)

    targets = [
        "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
        "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
        "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
    ]

    
    df = df.dropna(subset=targets)

    X = df.drop(columns=targets + ["SMILES"], errors="ignore")
    y = df[targets]

    print("Dataset after removing NaN:", df.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    train_model()