
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors


def compute_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    desc = {}

    desc["MolWt"] = Descriptors.MolWt(mol)
    desc["LogP"] = Descriptors.MolLogP(mol)
    desc["HBD"] = Descriptors.NumHDonors(mol)
    desc["HBA"] = Descriptors.NumHAcceptors(mol)
    desc["TPSA"] = Descriptors.TPSA(mol)
    desc["RotatableBonds"] = Descriptors.NumRotatableBonds(mol)

    return desc
def compute_fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=1024
    )

    return list(fp)
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