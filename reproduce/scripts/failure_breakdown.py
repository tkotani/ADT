"""Compute failure-mode (A1/A2/A3) breakdown per scaffold for both 30-atom and 50-atom models."""
import os as _os_adt
_HERE_adt = _os_adt.path.dirname(_os_adt.path.abspath(__file__))
ADT_ROOT = _os_adt.environ.get("ADT_ROOT", _os_adt.path.abspath(_os_adt.path.join(_HERE_adt, "..", "..")))
DRUGS_DATA = _os_adt.path.join(ADT_ROOT, "Drugs", "data", "freeorder_v26")
QM9_DATA = _os_adt.path.join(ADT_ROOT, "QM9", "data", "freeorder")

import json

SCAFFOLDS = ["benzene", "pyridine", "pyrimidine", "pyrazine", "furan", "thiophene", "cyclohexane"]

def analyze(base, label):
    print(f"\n=== {label} ===")
    print(f"{'scaffold':<12} {'kek':>5} {'A3':>5} {'A2':>5} {'A1':>5}  {'A3%':>6} {'A2%':>6} {'A1%':>6}")
    print("-" * 70)
    for s in SCAFFOLDS:
        path = f"{base}/{s}/xtb_results.json"
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            continue
        kek = [r for r in data if r.get("sdf_tag") == "kekulized" and r.get("ok")]
        n = len(kek)
        a3 = sum(1 for r in kek if r.get("same_topo"))
        a2 = sum(1 for r in kek if not r.get("same_topo") and r.get("same_inchi"))
        a1 = n - a3 - a2
        if n > 0:
            print(f"{s:<12} {n:>5} {a3:>5} {a2:>5} {a1:>5}  {100*a3/n:>5.1f}% {100*a2/n:>5.1f}% {100*a1/n:>5.1f}%")

analyze("" + DRUGS_DATA + "/v26s_scaffolds_n10k", "30-atom (v26s)")
analyze("" + DRUGS_DATA + "/v26g_scaffolds_n10k", "50-atom (v26g)")
