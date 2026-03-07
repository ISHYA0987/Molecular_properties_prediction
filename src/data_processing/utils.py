from rdkit import Chem


def is_valid_smiles(smiles):
    """
    Check whether a SMILES string represents a valid molecule.
    """
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except:
        return False