#!/usr/bin/env python3
"""per_scaffold_novelty.py - reproduce the novelty columns of Appendix B (Tables 6/7).

per_scaffold_stats.py reports kekulized / unique / heavy-atom statistics but not
novelty. This script compares each scaffold's kekulized, mol_stable SMILES
against the corresponding training set (drugs_mols_v26_max30/50.pkl) and counts
the novel (not present in training) canonical, non-isomeric SMILES.

Usage:  ADT_ROOT=/path/to/ADT python3 per_scaffold_novelty.py
Output: per-scaffold uniq / novel / novel% for the 30- and 50-atom models.
"""
import os
import pickle

_HERE = os.path.dirname(os.path.abspath(__file__))
ADT_ROOT = os.environ.get("ADT_ROOT", os.path.abspath(os.path.join(_HERE, "..", "..")))
DRUGS_DATA = os.path.join(ADT_ROOT, "Drugs", "data", "freeorder_v26")

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

SCAFFOLDS = ["benzene", "pyridine", "pyrimidine", "pyrazine",
             "furan", "thiophene", "cyclohexane"]


def canon(smi):
    """Canonical, non-isomeric SMILES (the match key)."""
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def load_train_smiles(pkl_path):
    """Build the set of canonical, non-isomeric SMILES from the training pkl
    (a list of (rdkit_mol, positions, smiles) tuples)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    out = set()
    for item in data:
        mol = item[0]
        try:
            out.add(Chem.MolToSmiles(mol, isomericSmiles=False))
        except Exception:
            pass
    return out


def report(tag, scaffold_subdir, train_pkl):
    print("=== %s : Novel / Novel%% (Appendix B) ===" % tag)
    train = load_train_smiles(train_pkl)
    print("  train unique SMILES: %d  (%s)" % (len(train), os.path.basename(train_pkl)))
    print("  %-11s %6s %6s %8s" % ("scaffold", "uniq", "novel", "novel%"))
    print("  " + "-" * 36)
    for s in SCAFFOLDS:
        smi_file = os.path.join(DRUGS_DATA, scaffold_subdir, s, "mol_stable_smiles.txt")
        uniq = set()
        with open(smi_file) as f:
            for line in f:
                c = canon(line.strip())
                if c:
                    uniq.add(c)
        novel = sum(1 for c in uniq if c not in train)
        novp = 100.0 * novel / len(uniq) if uniq else 0.0
        print("  %-11s %6d %6d %7.2f%%" % (s, len(uniq), novel, novp))
    print()


if __name__ == "__main__":
    report("30-atom (v26s, Table perscaf30)",
           "v26s_scaffolds_n10k",
           os.path.join(DRUGS_DATA, "drugs_mols_v26_max30.pkl"))
    report("50-atom (v26g, Table perscaf50)",
           "v26g_scaffolds_n10k",
           os.path.join(DRUGS_DATA, "drugs_mols_v26_max50.pkl"))
