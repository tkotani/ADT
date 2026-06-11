"""Extract representative Type A1 failure cases (pre/post SMILES that differ)."""
import os as _os_adt
_HERE_adt = _os_adt.path.dirname(_os_adt.path.abspath(__file__))
ADT_ROOT = _os_adt.environ.get("ADT_ROOT", _os_adt.path.abspath(_os_adt.path.join(_HERE_adt, "..", "..")))
DRUGS_DATA = _os_adt.path.join(ADT_ROOT, "Drugs", "data", "freeorder_v26")
QM9_DATA = _os_adt.path.join(ADT_ROOT, "QM9", "data", "freeorder")

import json
from rdkit import Chem

SCAFFOLDS = ["benzene", "pyridine", "pyrimidine", "pyrazine", "furan", "thiophene", "cyclohexane"]

def cases(base, label, max_per_scaffold=3):
    print(f"\n=== {label} : Type A1 failure examples ===")
    for s in SCAFFOLDS:
        try:
            d = json.load(open(f"{base}/{s}/xtb_results.json"))
        except FileNotFoundError:
            continue
        a1 = []
        for r in d:
            if r.get("sdf_tag") != "kekulized": continue
            if not r.get("ok"): continue
            if r.get("same_topo"): continue
            if r.get("same_inchi"): continue  # exclude A2
            pre = r.get("smi")
            post = r.get("topo_post")
            if not pre or not post or pre == post: continue
            a1.append((pre, post))
            if len(a1) >= max_per_scaffold: break
        print(f"\n  {s} ({len(a1)} examples):")
        for pre, post in a1:
            print(f"    pre:  {pre}")
            print(f"    post: {post}")

cases("" + DRUGS_DATA + "/v26s_scaffolds_n10k", "30-atom", max_per_scaffold=2)
cases("" + DRUGS_DATA + "/v26g_scaffolds_n10k", "50-atom", max_per_scaffold=2)
