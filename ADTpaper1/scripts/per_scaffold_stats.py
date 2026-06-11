"""Per-scaffold statistics: heavy-atom distribution, uniqueness."""
import os as _os_adt
_HERE_adt = _os_adt.path.dirname(_os_adt.path.abspath(__file__))
ADT_ROOT = _os_adt.environ.get("ADT_ROOT", _os_adt.path.abspath(_os_adt.path.join(_HERE_adt, "..", "..")))
DRUGS_DATA = _os_adt.path.join(ADT_ROOT, "Drugs", "data", "freeorder_v26")
QM9_DATA = _os_adt.path.join(ADT_ROOT, "QM9", "data", "freeorder")

import json, statistics
from rdkit import Chem
from rdkit.Chem import AllChem

SCAFFOLDS = ["benzene", "pyridine", "pyrimidine", "pyrazine", "furan", "thiophene", "cyclohexane"]

def stats(base, label):
    print(f"\n=== {label} ===")
    print(f"{'scaffold':<12} {'kek':>5} {'unique':>7} {'uniq%':>7} {'mean':>5} {'std':>5} {'med':>4} {'min':>4} {'max':>4}")
    print("-" * 70)
    for s in SCAFFOLDS:
        try:
            d = json.load(open(f"{base}/{s}/xtb_results.json"))
        except FileNotFoundError:
            continue
        kek = [r for r in d if r.get("sdf_tag") == "kekulized"]
        smis = [r["smi"] for r in kek if r.get("smi")]
        nas = []
        for smi in smis:
            m = Chem.MolFromSmiles(smi)
            if m is None: continue
            nas.append(m.GetNumHeavyAtoms())
        n = len(nas)
        if n == 0: continue
        unique_smi = len(set(smis))
        print(f"{s:<12} {n:>5} {unique_smi:>7} {100*unique_smi/n:>6.1f}% "
              f"{statistics.mean(nas):>5.1f} {statistics.stdev(nas):>5.1f} "
              f"{statistics.median(nas):>4.0f} {min(nas):>4} {max(nas):>4}")

stats("" + DRUGS_DATA + "/v26s_scaffolds_n10k", "30-atom (v26s)")
stats("" + DRUGS_DATA + "/v26g_scaffolds_n10k", "50-atom (v26g)")
