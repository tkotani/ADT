
import os as _os_adt
_HERE_adt = _os_adt.path.dirname(_os_adt.path.abspath(__file__))
ADT_ROOT = _os_adt.environ.get("ADT_ROOT", _os_adt.path.abspath(_os_adt.path.join(_HERE_adt, "..", "..")))
DRUGS_DATA = _os_adt.path.join(ADT_ROOT, "Drugs", "data", "freeorder_v26")
QM9_DATA = _os_adt.path.join(ADT_ROOT, "QM9", "data", "freeorder")

import pickle, statistics
from collections import Counter

# Use v26_max50 which is the 50-atom-truncated training data (302k claimed)
# Then derive ≤30 subset for the "full" picture w.r.t. ADT's 30-atom cut.
print("Loading v26 max50 pickle...")
with open("" + DRUGS_DATA + "/drugs_mols_v26_max50.pkl", "rb") as f:
    data = pickle.load(f)
print(f"  type={type(data).__name__}, len={len(data)}")

def count_heavy(item):
    if hasattr(item, "GetNumHeavyAtoms"):
        return item.GetNumHeavyAtoms()
    if isinstance(item, dict):
        for k in ("n_atoms","heavy","n_heavy","mol"):
            v = item.get(k)
            if v is not None:
                if hasattr(v, "GetNumHeavyAtoms"):
                    return v.GetNumHeavyAtoms()
                if isinstance(v, int):
                    return v
        return None
    if isinstance(item, tuple) and len(item) >= 1:
        return count_heavy(item[0])
    return None

counts = []
for item in data:
    c = count_heavy(item)
    if c is not None:
        counts.append(c)
print(f"Total counted: {len(counts)}")
print(f"=== v26 max50 (50-atom-truncated) ===")
print(f"Mean = {statistics.mean(counts):.2f}, std = {statistics.stdev(counts):.2f}, median = {statistics.median(counts):.0f}")
print(f"Min = {min(counts)}, Max = {max(counts)}")
print(f"Quartiles: q25={statistics.quantiles(counts, n=4)[0]:.0f}, q75={statistics.quantiles(counts, n=4)[2]:.0f}")

bins = list(range(5, 55, 5))
hist = Counter()
for c in counts:
    for j in range(len(bins) - 1):
        if bins[j] <= c < bins[j+1]:
            hist[bins[j]] += 1
            break
total = len(counts)
print("\nHistogram (5-atom bins):")
for lo in bins[:-1]:
    n = hist[lo]
    pct = 100 * n / total
    bar = "#" * int(pct)
    print(f"  [{lo:>2},{lo+5:>2}): {n:>7} ({pct:5.1f}%)  {bar}")

# Cumulative below 30
for t in (20, 25, 30, 35, 40, 50):
    n_below = sum(1 for c in counts if c <= t)
    print(f"  <= {t} atoms: {n_below}/{total} ({100*n_below/total:.1f}%)")

# also subset to ≤30 to match the 30-atom training set stats
cnt30 = [c for c in counts if c <= 30]
print(f"\n=== Subset ≤30 atoms ({len(cnt30)} mols) ===")
print(f"Mean = {statistics.mean(cnt30):.2f}, std = {statistics.stdev(cnt30):.2f}, median = {statistics.median(cnt30):.0f}")
