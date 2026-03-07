import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from pathlib import Path


DATA_PATH = Path("data/features/esol_features.csv")
MODEL_DIR = Path("experiments/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model():

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["logS", "SMILES"], errors="ignore")
    y = df["logS"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    import numpy as np
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("RMSE:", rmse)

    joblib.dump(model, MODEL_DIR / "esol_model.pkl")


if __name__ == "__main__":
    train_model()