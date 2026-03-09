from rdkit import Chem
from rdkit_features import compute_descriptors, compute_fingerprint

smi = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"

mol = Chem.MolFromSmiles(smi)

print("Mol:", mol)

desc = compute_descriptors(smi)
fp = compute_fingerprint(smi)

print("Descriptors:", desc)

if fp is not None:
    print("Fingerprint length:", len(fp))
else:
    print("Fingerprint failed")