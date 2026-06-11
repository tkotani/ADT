#!/usr/bin/env python3
"""per_scaffold_novelty.py — Appendix B (Table perscaf30/50) の Novel / Novel% 列を再現する。

per_scaffold_stats.py は Kekulized / Unique / 統計量(Mean/Std/...)を出すが novelty は出さない。
本スクリプトは各 scaffold の kekulized mol_stable SMILES を、対応する訓練セット
(drugs_mols_v26_max30/50.pkl)と照合し、novel(訓練に無い canonical SMILES)を数える。

照合キー: canonical, non-isomeric SMILES(paper と同じ)。

使い方:
  ADT_ROOT=/path/to/ADT python3 per_scaffold_novelty.py
  (ADT_ROOT 省略時は scripts/ の 2 つ上 = repo root を自動推定)

出力: 30-atom / 50-atom 各 7 scaffold の uniq / novel / novel% 表。
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
    """canonical, non-isomeric SMILES（照合キー）。"""
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def load_train_smiles(pkl_path):
    """訓練 pkl から canonical non-isomeric SMILES 集合を作る。
    pkl は (rdkit_mol, positions, smiles) のリスト。"""
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
