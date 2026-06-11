
import os as _os_adt
_HERE_adt = _os_adt.path.dirname(_os_adt.path.abspath(__file__))
ADT_ROOT = _os_adt.environ.get("ADT_ROOT", _os_adt.path.abspath(_os_adt.path.join(_HERE_adt, "..", "..")))
DRUGS_DATA = _os_adt.path.join(ADT_ROOT, "Drugs", "data", "freeorder_v26")
QM9_DATA = _os_adt.path.join(ADT_ROOT, "QM9", "data", "freeorder")

import json, statistics
from rdkit import Chem

all_na = []
print(f"{'scaffold':<12} {'N':>6} {'mean':>5} {'std':>5} {'med':>5} {'min':>5} {'max':>5} {'%>30':>5}")
print("-" * 60)
for s in ["benzene","pyridine","pyrimidine","pyrazine","furan","thiophene","cyclohexane"]:
    d = json.load(open(f"" + DRUGS_DATA + "/v26s_scaffolds_n10k/{s}/xtb_results.json"))
    nas = []
    for r in d:
        m = Chem.MolFromSmiles(r["smi"])
        if m is None:
            continue
        nas.append(m.GetNumHeavyAtoms())
    all_na.extend(nas)
    n_over30 = sum(1 for x in nas if x > 30)
    pct_over = 100 * n_over30 / len(nas)
    print(f"{s:<12} {len(nas):>6} {statistics.mean(nas):>5.1f} {statistics.stdev(nas):>5.1f} {statistics.median(nas):>5.0f} {min(nas):>5} {max(nas):>5} {pct_over:>5.1f}")
print("-" * 60)
n_over30_all = sum(1 for x in all_na if x > 30)
pct = 100 * n_over30_all / len(all_na)
print(f"{'ALL':<12} {len(all_na):>6} {statistics.mean(all_na):>5.1f} {statistics.stdev(all_na):>5.1f} {statistics.median(all_na):>5.0f} {min(all_na):>5} {max(all_na):>5} {pct:>5.1f}")
