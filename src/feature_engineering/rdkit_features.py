from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors


# 🔥 Improved descriptors (CRITICAL FIX)
def compute_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    desc = {}

    # Basic physicochemical
    desc["MolWt"] = Descriptors.MolWt(mol)
    desc["LogP"] = Descriptors.MolLogP(mol)
    desc["HBD"] = Descriptors.NumHDonors(mol)
    desc["HBA"] = Descriptors.NumHAcceptors(mol)
    desc["TPSA"] = Descriptors.TPSA(mol)
    desc["RotatableBonds"] = Descriptors.NumRotatableBonds(mol)

    # 🔥 CRITICAL (adds structural awareness)
    desc["NumAromaticRings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
    desc["RingCount"] = rdMolDescriptors.CalcNumRings(mol)
    desc["FractionCSP3"] = Descriptors.FractionCSP3(mol)

    return desc


# 🔥 Improved fingerprint (2048 bits)
def compute_fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048   # 🔥 upgraded from 1024
    )

    return list(fp)


# 🔥 Unified featurization (consistent with pipeline)
def featurize_smiles(smiles_list):

    feature_rows = []

    for smi in smiles_list:

        desc = compute_descriptors(smi)
        fp = compute_fingerprint(smi)

        if desc is None or fp is None:
            continue

        feature_dict = {}

        # Add descriptors
        feature_dict.update(desc)

        # Add fingerprint bits with names
        for i, bit in enumerate(fp):
            feature_dict[f"fp_{i}"] = bit

        feature_rows.append(feature_dict)

    features = pd.DataFrame(feature_rows)

    return features