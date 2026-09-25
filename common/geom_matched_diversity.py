"""GEOM side of Table 3: scaffold diversity of GEOM-Drugs sub-sampled to the model's N^gen.

Scaffold-diversity counts grow with the number of molecules, so each model row is compared with
GEOM-Drugs randomly sub-sampled to the same count N^gen. For each scaffold the GEOM pool is the
set of distinct (non-isomeric canonical) GEOM molecules that contain the scaffold ring (SMARTS);
the unconditional row (bootstrap3) uses all of GEOM. The pool order comes from one random.seed(0)
and one in-place shuffle per scaffold, in the fixed order below, so the numbers are reproducible
(all eight shuffles always run, whichever rows are requested).

Output per row: N_eff^scaf = exp(Shannon entropy of Murcko-scaffold frequencies) and #distinct
Murcko scaffolds over the first N^gen pool molecules. The GEOM IntDiv_1 in Table 3 does not depend
on N and is not recomputed here.

usage: python3 common/geom_matched_diversity.py geom_drugs.smi benzene=9562 pyridine=9589 ...
       (scaffold names as in SCAFFOLDS below; N^gen from paper_tables.py)
"""
import sys, random, math
from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

SCAFFOLDS = [("benzene", "c1ccccc1"), ("pyridine", "c1ccncc1"), ("pyrimidine", "c1cncnc1"),
             ("pyrazine", "c1cnccn1"), ("furan", "c1ccoc1"), ("thiophene", "c1ccsc1"),
             ("cyclohexane", "C1CCCCC1"), ("bootstrap3", "")]

if len(sys.argv) < 3:
    sys.exit(__doc__)
want = dict((kv.split("=")[0], int(kv.split("=")[1])) for kv in sys.argv[2:])
unknown = set(want) - {n for n, _ in SCAFFOLDS}
if unknown:
    sys.exit("unknown scaffold(s): %s" % ", ".join(sorted(unknown)))

random.seed(0)
GEOM = [l.strip() for l in open(sys.argv[1]) if l.strip()]


def geom_pool(smarts, cap=60000):
    pat = Chem.MolFromSmarts(smarts) if smarts else None
    random.shuffle(GEOM)
    u = {}
    for s in GEOM:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        if pat is not None and not m.HasSubstructMatch(pat):
            continue
        try:
            u.setdefault(Chem.MolToSmiles(m, isomericSmiles=False), MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            continue
        if len(u) >= cap:
            break
    return list(u.values())


def neff(bms):
    c = Counter(bms); n = len(bms)
    return math.exp(-sum((v / n) * math.log(v / n) for v in c.values()))


print("%-12s %7s %8s %9s" % ("scaffold", "N", "N_eff", "#distinct"))
for name, smarts in SCAFFOLDS:
    pool = geom_pool(smarts)                  # always shuffle, to keep the RNG sequence fixed
    if name in want:
        bms = pool[:want[name]]
        if len(bms) < want[name]:
            print("%-12s  pool has only %d molecules" % (name, len(bms)))
        print("%-12s %7d %8.0f %9d" % (name, len(bms), neff(bms), len(set(bms))), flush=True)
