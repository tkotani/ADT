#!/usr/bin/env python3
"""per_scaffold_diversity.py - reproduce Appendix C (Tables 8/9).

For each scaffold's XVR-positive set (kekulized, xTB-converged, topology
preserved; unique canonical SMILES) it computes:
  - XVR-pos.        : number of distinct XVR-positive molecules
  - BM unique / BM% : distinct Bemis-Murcko core scaffolds and their fraction
  - Tanimoto median : median pairwise Tanimoto of Morgan fingerprints
                      (radius 2, 2048 bits) over 2000 random pairs, SEED=42
  - MW (Da)         : mean +/- std of Descriptors.MolWt
  - logP            : mean +/- std of Crippen MolLogP

Input:  Drugs/data/freeorder_v26/v26{s,g}_scaffolds_n10k/<scaffold>/xtb_results.json
Output: per-scaffold tables for the 30-atom (Table 8) and 50-atom (Table 9) models.
Usage:  ADT_ROOT=/path/to/ADT python3 per_scaffold_diversity.py

The Tanimoto median is a 2000-random-pair sample statistic (about +/-0.003
noise), so SEED=42 is fixed for reproducibility; the paper's Tables 8/9 are this
seeded output. Bemis-Murcko canonicalisation depends slightly on the RDKit
version (verified with RDKit 2025.09.x).
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
    """Unique RDKit Mols of the XVR-positive (ok & same_topo & kekulized) SMILES."""
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
