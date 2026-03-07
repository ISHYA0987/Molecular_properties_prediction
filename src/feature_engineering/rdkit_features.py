from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd


def compute_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    descriptors = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol)
    }


def compute_fingerprint(smiles, radius=2, n_bits=1024):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits
    )

    arr = np.zeros((1,))
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)

    return arr

def featurize_smiles(smiles_list):

    descriptor_list = []
    fingerprint_list = []

    for smi in smiles_list:

        desc = compute_descriptors(smi)
        fp = compute_fingerprint(smi)

        if desc is None or fp is None:
            continue

        descriptor_list.append(desc)
        fingerprint_list.append(fp)

    desc_df = pd.DataFrame(descriptor_list)
    fp_df = pd.DataFrame(fingerprint_list)

    features = pd.concat([desc_df, fp_df], axis=1)

    return features