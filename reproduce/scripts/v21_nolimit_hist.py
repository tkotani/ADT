
import os as _os_adt
_HERE_adt = _os_adt.path.dirname(_os_adt.path.abspath(__file__))
ADT_ROOT = _os_adt.environ.get("ADT_ROOT", _os_adt.path.abspath(_os_adt.path.join(_HERE_adt, "..", "..")))
DRUGS_DATA = _os_adt.path.join(ADT_ROOT, "Drugs", "data", "freeorder_v26")
QM9_DATA = _os_adt.path.join(ADT_ROOT, "QM9", "data", "freeorder")

import pickle, statistics
from collections import Counter

with open("" + DRUGS_DATA + "/drugs_mols_v21_nolimit.pkl", "rb") as f:
    data = pickle.load(f)

def count_heavy(item):
    if hasattr(item, "GetNumHeavyAtoms"):
        return item.GetNumHeavyAtoms()
    if isinstance(item, tuple) and len(item) >= 1:
        return count_heavy(item[0])
    return None

counts = [c for c in (count_heavy(x) for x in data) if c is not None]
N = len(counts)
print(f"Total: {N}, mean={statistics.mean(counts):.2f}, std={statistics.stdev(counts):.2f}, median={statistics.median(counts):.0f}, min={min(counts)}, max={max(counts)}")

hist = Counter(counts)
# Histogram from 3 to 50
print("\n=== TikZ coords (n, count_in_thousand): ===")
for n in range(3, 51):
    print(f"({n},{hist.get(n, 0)/1000:.3f})", end=" ")
    if (n - 2) % 8 == 0:
        print()
print()

# Cumulative cuts
print("\nCumulative <=, of full N=270138:")
for t in [25, 30, 35, 40, 45, 50, 60, 70, 80]:
    n_below = sum(1 for c in counts if c <= t)
    print(f"  <= {t}: {n_below}/{N} = {100*n_below/N:.3f}%")

# Above 50 details
n_above = sum(1 for c in counts if c > 50)
print(f"\n  > 50: {n_above}/{N} = {100*n_above/N:.4f}%")
