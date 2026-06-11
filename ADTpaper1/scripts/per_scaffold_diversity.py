#!/usr/bin/env python3
"""per_scaffold_diversity.py - Appendix C (Tables divers30/divers50) を再現する。

各 scaffold の XVR-positive 集合 (kekulized かつ xTB 収束かつ topology 保存、
unique canonical SMILES) について、以下を計算する:

  - XVR-pos.        : distinct kekulized XVR-positive 分子数
  - BM unique / BM% : 異なる Bemis-Murcko core scaffold 数とその割合
  - Tanimoto med.   : Morgan FP (radius 2, 2048 bit) の pairwise Tanimoto の
                      中央値。2000 random pairs、SEED=42 で決定論的。
  - MW (Da)         : Descriptors.MolWt の mean +/- std (population std)
  - logP            : Crippen MolLogP の mean +/- std

入力: Drugs/data/freeorder_v26/v26{s,g}_scaffolds_n10k/<scaffold>/xtb_results.json
出力: 30-atom (divers30) と 50-atom (divers50) の per-scaffold 表。

使い方:
  ADT_ROOT=/path/to/ADT python3 per_scaffold_diversity.py

注: Tanimoto median は 2000 random pairs の標本中央値で +/-0.003 程度の標本揺らぎを
持つため、再現性のため SEED=42 を固定している。論文 Table 8/9 はこの SEED=42 出力。
BM scaffold の canonical 化は RDKit バージョンに僅かに依存する (cyclohexane で
+/-1 程度)。本スクリプト確認時の RDKit は 2025.09.x。
"""
import os, json, statistics, random
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

_HERE = os.path.dirname(os.path.abspath(__file__))
ADT_ROOT = os.environ.get("ADT_ROOT", os.path.abspath(os.path.join(_HERE, "..", "..")))
DRUGS_DATA = os.path.join(ADT_ROOT, "Drugs", "data", "freeorder_v26")

SCAFFOLDS = ["benzene", "pyridine", "pyrimidine", "pyrazine",
             "furan", "thiophene", "cyclohexane"]
SEED = 42
N_PAIRS = 2000


def xvr_unique_mols(xtb_results_path):
    """XVR-positive (ok & same_topo & kekulized) の unique SMILES -> RDKit Mol list。"""
    d = json.load(open(xtb_results_path))
    uniq = list(dict.fromkeys(
        r["smi"] for r in d
        if r.get("ok") and r.get("same_topo") and r.get("sdf_tag") == "kekulized"))
    return [m for m in (Chem.MolFromSmiles(s) for s in uniq) if m is not None]


def bm_scaffold_smiles(m):
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return None


def tanimoto_median(mols):
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
    rng = random.Random(SEED)
    n = len(fps)
    sims = []
    while len(sims) < N_PAIRS:
        i = rng.randrange(n); j = rng.randrange(n)
        if i != j:
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return statistics.median(sims)


def report(tag, subdir):
    print("=== %s : Appendix C diversity ===" % tag)
    print("  %-11s %7s %7s %6s %7s %14s %14s" %
          ("scaffold", "XVRpos", "BMuniq", "BM%", "Tani", "MW(Da)", "logP"))
    print("  " + "-" * 70)
    for s in SCAFFOLDS:
        mols = xvr_unique_mols(os.path.join(DRUGS_DATA, subdir, s, "xtb_results.json"))
        n = len(mols)
        bms = set(filter(None, (bm_scaffold_smiles(m) for m in mols)))
        bmu = len(bms)
        mw = [Descriptors.MolWt(m) for m in mols]
        lp = [Descriptors.MolLogP(m) for m in mols]
        tani = tanimoto_median(mols)
        print("  %-11s %7d %7d %6.1f %7.3f  %6.1f +/-%5.1f  %+5.2f +/-%4.2f" % (
            s, n, bmu, 100.0 * bmu / n, tani,
            statistics.mean(mw), statistics.pstdev(mw),
            statistics.mean(lp), statistics.pstdev(lp)))
    print()


if __name__ == "__main__":
    report("30-atom (v26s, Table divers30)", "v26s_scaffolds_n10k")
    report("50-atom (v26g, Table divers50)", "v26g_scaffolds_n10k")
