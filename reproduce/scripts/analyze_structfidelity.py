#!/usr/bin/env python3
"""analyze_structfidelity.py - reproduce the Fig. 6 / Section 4.5 numbers.

Reads the regenerated structural-fidelity coordinates under
structfidelity/{qm9,drugs30,drugs50}/ and prints, per dataset, the median
|dr| (bond-length error) and |dtheta| (bond-angle error) plus the TikZ
histogram coordinates used in Fig. 6. Reuses process_scaffold_dir from
bond_angle_errors.py (filter: ok & same; per-bond / per-angle errors computed
from the pre/post xyz).

Usage:  ADT_ROOT=/path/to/ADT python3 analyze_structfidelity.py
"""
import os, sys, statistics
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
ADT_ROOT = os.environ.get("ADT_ROOT", os.path.abspath(os.path.join(_HERE, "..", "..")))
from bond_angle_errors import process_scaffold_dir

SF = os.path.join(ADT_ROOT, "Drugs", "data", "freeorder_v26", "structfidelity")
SC = ["benzene", "pyridine", "pyrimidine", "pyrazine", "furan", "thiophene", "cyclohexane"]
DATASETS = [("qm9", ["all"]), ("drugs30", SC), ("drugs50", SC)]

BOND_BINS = np.arange(-0.110, 0.115, 0.010)   # 22 bins, 10 mA
ANGLE_BINS = np.arange(-15.5, 16.5, 1.0)      # 1 deg


def tikz(data, bins):
    counts, _ = np.histogram(data, bins=bins)
    total = counts.sum()
    out = []
    for i in range(len(bins) - 1):
        c = (bins[i] + bins[i+1]) / 2
        pct = 100.0 * counts[i] / total if total else 0.0
        out.append("(%+.3f,%.2f)" % (c, pct))
    return out


def main():
    for ds, scaffolds in DATASETS:
        base = os.path.join(SF, ds)
        if not os.path.isdir(base):
            print("== %s : (no data) ==" % ds); continue
        all_be, all_ae, nmol = [], [], 0
        for s in scaffolds:
            be, ae, n = process_scaffold_dir(base, s)
            all_be += be; all_ae += ae; nmol += n
        if not all_be:
            print("== %s : empty ==" % ds); continue
        be_abs = [abs(x) for x in all_be]; ae_abs = [abs(x) for x in all_ae]
        print("==================== %s ====================" % ds)
        print("mols=%d  bonds=%d  angles=%d" % (nmol, len(all_be), len(all_ae)))
        print("median |dr| = %.4f A (%.1f mA)  | median |dtheta| = %.2f deg"
              % (statistics.median(be_abs), statistics.median(be_abs)*1000, statistics.median(ae_abs)))
        print("BOND-LENGTH tikz (center,pct):")
        print("  " + " ".join(tikz(all_be, BOND_BINS)))
        print("BOND-ANGLE tikz (center,pct):")
        print("  " + " ".join(tikz(all_ae, ANGLE_BINS)))
        print()


if __name__ == "__main__":
    main()
