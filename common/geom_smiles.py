"""GEOM-Drugs reference SMILES for the novelty column (paper_tables.py --geom_smi).

The keys of summary_drugs.json in the public GEOM release (rdkit_folder) are the SMILES of the
304,466 GEOM-Drugs molecules. This writes them one per line.

usage: python3 common/geom_smiles.py <path>/rdkit_folder/summary_drugs.json geom_drugs.smi
"""
import json, sys

if len(sys.argv) != 3:
    sys.exit(__doc__)
keys = list(json.load(open(sys.argv[1])).keys())
with open(sys.argv[2], "w") as f:
    for s in keys:
        f.write(s + "\n")
print("wrote %d SMILES -> %s" % (len(keys), sys.argv[2]))
