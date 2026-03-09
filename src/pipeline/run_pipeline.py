import pandas as pd
import joblib
from pathlib import Path

TOX21_DATA = Path("data/features/tox21_features.csv")
AMES_DATA = Path("data/features/ames_features.csv")
ESOL_DATA = Path("data/features/esol_features.csv")

TOX21_MODEL = Path("experiments/models/tox21_model.pkl")
AMES_MODEL = Path("experiments/models/ames_model.pkl")
ESOL_MODEL = Path("experiments/models/esol_model.pkl")

OUTPUT_PATH = Path("results/final_molecular_predictions.csv")
OUTPUT_PATH.parent.mkdir(exist_ok=True)


def run_pipeline():

    print("Loading datasets...")

    tox21_df = pd.read_csv(TOX21_DATA)
    ames_df = pd.read_csv(AMES_DATA)
    esol_df = pd.read_csv(ESOL_DATA)

    print("Loading models...")

    tox21_model = joblib.load(TOX21_MODEL)
    ames_model = joblib.load(AMES_MODEL)
    esol_model = joblib.load(ESOL_MODEL)

    
    tox_targets = [
        "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
        "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
        "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
    ]

    
    print("Predicting Tox21 toxicity...")

    X_tox = tox21_df.drop(columns=tox_targets + ["SMILES"], errors="ignore")

    tox_probs = tox21_model.predict_proba(X_tox)

    tox_predictions = {}

    for i, target in enumerate(tox_targets):
        tox_predictions[target] = tox_probs[i][:,1]

    tox_results = pd.DataFrame(tox_predictions)

    tox_results["SMILES"] = tox21_df["SMILES"]


   
    print("Predicting Ames mutagenicity...")

    X_ames = ames_df.drop(columns=["genotoxicity","SMILES"], errors="ignore")

    ames_preds = ames_model.predict(X_ames)

    ames_results = pd.DataFrame({
        "SMILES": ames_df["SMILES"],
        "Ames_Mutagenicity": ames_preds
    })


   
    print("Predicting solubility...")

    X_esol = esol_df.drop(columns=["logS","SMILES"], errors="ignore")

    esol_preds = esol_model.predict(X_esol)

    esol_results = pd.DataFrame({
        "SMILES": esol_df["SMILES"],
        "Predicted_Solubility_logS": esol_preds
    })


  
    print("Merging results...")

    final_df = tox_results.merge(ames_results, on="SMILES", how="left")
    final_df = final_df.merge(esol_results, on="SMILES", how="left")

   
    chem_cols = ["MolWt","LogP","TPSA","HBD","HBA"]

    chem_features = tox21_df[["SMILES"] + chem_cols]

    final_df = final_df.merge(chem_features, on="SMILES", how="left")

    cols = ["SMILES"] + [c for c in final_df.columns if c != "SMILES"]
    final_df = final_df[cols]
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("Pipeline completed.")
    print("Results saved to:", OUTPUT_PATH)

    print("\nPreview:")
    print(final_df.head())


if __name__ == "__main__":
    run_pipeline()