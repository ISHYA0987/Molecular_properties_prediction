import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from pathlib import Path

DATA_PATH = Path("data/features/esol_features.csv")
MODEL_DIR = Path("experiments/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model():

    print("Loading data...")

    df = pd.read_csv(DATA_PATH)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    df = df.drop_duplicates()

    X = df.drop(columns=["logS", "SMILES"], errors="ignore")
    y = df["logS"]

    joblib.dump(list(X.columns), MODEL_DIR / "esol_features.pkl")

  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")


    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

  
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("Test RMSE:", rmse)

  
    cv_scores = cross_val_score(
        model, X, y,
        scoring="neg_root_mean_squared_error",
        cv=5
    )
    print("CV RMSE:", -cv_scores.mean())

    importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\nTop Features:\n", importances.sort_values(ascending=False).head(10))

    joblib.dump(model, MODEL_DIR / "esol_model.pkl")

    print("Model saved successfully!")


if __name__ == "__main__":
    train_model()