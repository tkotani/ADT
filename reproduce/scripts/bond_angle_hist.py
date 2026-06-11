"""Dump TikZ-ready histogram coordinates for bond-length and bond-angle errors."""
import os as _os_adt
_HERE_adt = _os_adt.path.dirname(_os_adt.path.abspath(__file__))
ADT_ROOT = _os_adt.environ.get("ADT_ROOT", _os_adt.path.abspath(_os_adt.path.join(_HERE_adt, "..", "..")))
DRUGS_DATA = _os_adt.path.join(ADT_ROOT, "Drugs", "data", "freeorder_v26")
QM9_DATA = _os_adt.path.join(ADT_ROOT, "QM9", "data", "freeorder")

import os, json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, "/tmp")
from bond_angle_errors import process_scaffold_dir

SCAFFOLDS = ["benzene", "pyridine", "pyrimidine", "pyrazine", "furan", "thiophene", "cyclohexane"]

def collect(base):
    all_be, all_ae = [], []
    for s in SCAFFOLDS:
        be, ae, _ = process_scaffold_dir(base, s)
        all_be.extend(be)
        all_ae.extend(ae)
    return np.array(all_be), np.array(all_ae)

def hist_signed(data, bins, label):
    """Histogram of SIGNED errors (post - pre), with bin edges."""
    counts, _ = np.histogram(data, bins=bins)
    total = len(data)
    print(f"\n=== {label} (signed, n={total}) ===")
    for i in range(len(bins) - 1):
        center = (bins[i] + bins[i+1]) / 2
        pct = 100 * counts[i] / total
        print(f"  [{bins[i]:>+6.3f}, {bins[i+1]:>+6.3f}): center={center:>+6.3f} count={counts[i]:>5} ({pct:5.2f}%)")
    print("\nTikZ coords (center, percent):")
    for i in range(len(bins) - 1):
        center = (bins[i] + bins[i+1]) / 2
        pct = 100 * counts[i] / total
        print(f"  ({center:+.4f},{pct:.4f})")

def main():
    print("Loading 30-atom data...")
    be30, ae30 = collect("" + DRUGS_DATA + "/v26s_scaffolds_n10k")
    print(f"30-atom: {len(be30)} bonds, {len(ae30)} angles")

    print("Loading 50-atom data...")
    be50, ae50 = collect("" + DRUGS_DATA + "/v26g_scaffolds_n10k")
    print(f"50-atom: {len(be50)} bonds, {len(ae50)} angles")

    # Bond-length error: bins centered on integer mÅ from -100 to +100 mÅ (in Å scale)
    bond_bins = np.arange(-0.110, 0.115, 0.010)  # 22 bins of 10 mÅ each
    hist_signed(be30, bond_bins, "30-atom bond-length error (Å)")
    hist_signed(be50, bond_bins, "50-atom bond-length error (Å)")

    # Bond-angle error: bins of 1° from -15 to +15
    angle_bins = np.arange(-15.5, 16.5, 1.0)  # 31 bins of 1°
    hist_signed(ae30, angle_bins, "30-atom bond-angle error (°)")
    hist_signed(ae50, angle_bins, "50-atom bond-angle error (°)")

    # Summary stats
    print("\n=== SUMMARY ===")
    for label, be, ae in [("30-atom", be30, ae30), ("50-atom", be50, ae50)]:
        be_abs = np.abs(be) * 1000
        ae_abs = np.abs(ae)
        print(f"\n{label}:")
        print(f"  bond |err| (mÅ): mean={np.mean(be_abs):.2f}, median={np.median(be_abs):.2f}, std={np.std(be_abs):.2f}, p50={np.percentile(be_abs,50):.2f}, p90={np.percentile(be_abs,90):.2f}, p99={np.percentile(be_abs,99):.2f}")
        print(f"  angle |err| (°): mean={np.mean(ae_abs):.2f}, median={np.median(ae_abs):.2f}, std={np.std(ae_abs):.2f}, p50={np.percentile(ae_abs,50):.2f}, p90={np.percentile(ae_abs,90):.2f}, p99={np.percentile(ae_abs,99):.2f}")

if __name__ == "__main__":
    main()
