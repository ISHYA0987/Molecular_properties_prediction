from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
import numpy as np
import pandas as pd


def safe_value(val):
    if val is None:
        return 0.0
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return 0.0
    return float(val)


def compute_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    desc = {}

    try:
   
        desc["MolWt"] = safe_value(Descriptors.MolWt(mol))
        desc["LogP"] = safe_value(Descriptors.MolLogP(mol))
        desc["HBD"] = safe_value(Descriptors.NumHDonors(mol))
        desc["HBA"] = safe_value(Descriptors.NumHAcceptors(mol))
        desc["TPSA"] = safe_value(Descriptors.TPSA(mol))
        desc["RotatableBonds"] = safe_value(Descriptors.NumRotatableBonds(mol))

        # Structural features
        desc["NumAromaticRings"] = safe_value(rdMolDescriptors.CalcNumAromaticRings(mol))
        desc["RingCount"] = safe_value(rdMolDescriptors.CalcNumRings(mol))
        desc["FractionCSP3"] = safe_value(Descriptors.FractionCSP3(mol))

    except Exception:
        return None

    return desc


def compute_fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    try:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048
        )

        fp = generator.GetFingerprint(mol)

        # Convert to clean list (0/1)
        arr = [int(x) for x in fp]

        return arr

    except Exception:
        return None



def featurize_smiles(smiles_list):

    feature_rows = []

    for smi in smiles_list:

        desc = compute_descriptors(smi)
        fp = compute_fingerprint(smi)

        if desc is None or fp is None:
            continue

        feature_dict = {}

    
        feature_dict.update(desc)

        for i, bit in enumerate(fp):
            feature_dict[f"fp_{i}"] = bit

        feature_rows.append(feature_dict)

    features = pd.DataFrame(feature_rows)


    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)

    return features