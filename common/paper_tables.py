"""Paper tables from molrecord banks: Table 2 (funnel), Table 3 (diversity/properties), strain, Fig. 6.

Reads every <bank_dir>/<scaffold>/pfree_bank_<scaffold>.pt written by run_gen_records.sh and prints,
per scaffold:
  Table 2: N_Hprerx, N_XTP, N_XTP^smiles, N^gen (rate), N^Novel, N_fullrx, heavy-atom size
           (mean/std/median/min/max), median Kabsch RMSD, median strain dE (kcal/mol)
  Table 3: N^gen, N_eff^scaf = exp(Shannon entropy of Murcko scaffolds), #distinct Murcko,
           IntDiv_1 (Morgan r=2, 2048 bits, 1500-molecule sample), mean MW / Crippen logP / QED
and, over all banks, the per-heavy-atom strain percentiles and the Fig. 6 bond/angle shift
(generated -> xTB-relaxed) medians and histograms.

N_XTP^smiles counts XTP-accepted molecules that RDKit reads back as a neutral molecule; N^gen is the
number of distinct canonical SMILES among them; N^Novel those absent from the GEOM-Drugs set
(--geom_smi, built by geom_smiles.py). funnel_stats.py gives the xTB-side columns only.

usage: python3 common/paper_tables.py <bank_dir> --geom_smi geom_drugs.smi
"""
import os, sys, glob, random, math, argparse
from collections import Counter
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Crippen, QED, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
from util_validation import validate_3D

ap = argparse.ArgumentParser()
ap.add_argument("bank_dir")
ap.add_argument("--geom_smi", required=True, help="GEOM-Drugs SMILES, one per line (geom_smiles.py)")
args = ap.parse_args()

geom = set()
for line in open(args.geom_smi):
    s = line.split()[0] if line.strip() else ""
    m = Chem.MolFromSmiles(s) if s else None
    if m:
        geom.add(Chem.MolToSmiles(m))
print("GEOM reference set: %d canonical SMILES" % len(geom), flush=True)


def bond_angle_err(r):
    """Bond-length and bond-angle change, generated (H-placed) -> xTB-relaxed, over bonds0."""
    ci = np.asarray(r["struct_init"]["coords"], float)
    cr = np.asarray(r["struct_relaxed"]["coords"], float)
    B = [(int(a), int(b)) for a, b in r["bonds0"]]
    dr = [np.linalg.norm(cr[a] - cr[b]) - np.linalg.norm(ci[a] - ci[b]) for a, b in B]
    adj = {}
    for a, b in B:
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)

    def ang(c, x, y, z):
        v1 = c[x] - c[y]; v2 = c[z] - c[y]
        cs = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        return math.degrees(math.acos(np.clip(cs, -1, 1)))
    dth = []
    for y, nb in adj.items():
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                dth.append(ang(cr, nb[i], y, nb[j]) - ang(ci, nb[i], y, nb[j]))
    return dr, dth


def smiles_of(r):
    sr = r["struct_relaxed"]
    try:
        m, _, _ = validate_3D(list(sr["anums"]), np.asarray(sr["coords"], float), charge=0)
        if m is None:
            return None
        mh = Chem.RemoveAllHs(m)
        return Chem.MolToSmiles(mh), mh
    except Exception:
        return None


def neff(mols):
    c = Counter(MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols)
    tot = sum(c.values())
    H = -sum((n / tot) * math.log(n / tot) for n in c.values())
    return math.exp(H), len(c)


def intdiv(mols):
    s = random.Random(0).sample(mols, min(1500, len(mols)))
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in s]
    t = n = 0
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]); t += sum(sims); n += len(sims)
    return 1 - t / n if n else 0


allDR, allDTH, rows, strain_all = [], [], [], []
banks = sorted(glob.glob(os.path.join(args.bank_dir, "*", "pfree_bank_*.pt")))
if not banks:
    sys.exit("no pfree_bank_*.pt under %s" % args.bank_dir)
for bank in banks:
    scaf = os.path.basename(os.path.dirname(bank))
    B = torch.load(bank, weights_only=False)["mols"]
    xtp = [r for r in B if r.get("xvr")]
    Hpre = sum(1 for r in B if r.get("hprerelax_ok"))
    full = sum(1 for r in B if r.get("full_ok"))
    sizes = [r["n_heavy"] for r in xtp]
    strain_all += [r["strain_pa"] for r in xtp if r["strain_pa"] is not None]
    smis, mols = [], []
    for r in xtp:
        pr = smiles_of(r)
        if pr:
            smis.append(pr[0]); mols.append(pr[1])
        dr, dth = bond_angle_err(r); allDR += dr; allDTH += dth
    uniq = {}
    for s, m in zip(smis, mols):
        uniq.setdefault(s, m)
    umols = list(uniq.values())
    ne, nmk = neff(umols)
    rows.append(dict(
        scaf=scaf, gen=len(B), Hpre=Hpre, full=full, xtp=len(xtp), nsmi=len(smis), nd=len(uniq),
        novel=sum(1 for s in uniq if s not in geom),
        sz=(np.mean(sizes), np.std(sizes), int(np.median(sizes)), min(sizes), max(sizes)),
        rmsd=np.median([r["rmsd_heavy"] for r in xtp if r["rmsd_heavy"] is not None]),
        dE=np.median([r["strain_dE"] for r in xtp if r["strain_dE"] is not None]),
        neff=ne, nmk=nmk, idiv=intdiv(umols),
        MW=(np.mean([Descriptors.MolWt(m) for m in umols]), np.std([Descriptors.MolWt(m) for m in umols])),
        lP=(np.mean([Crippen.MolLogP(m) for m in umols]), np.std([Crippen.MolLogP(m) for m in umols])),
        qed=(np.mean([QED.qed(m) for m in umols]), np.std([QED.qed(m) for m in umols]))))
    print("done", scaf, flush=True)

print("\n=== Table 2 (funnel) ===")
print("%-18s %6s %6s %8s %14s %6s %6s  %-24s %5s %5s" % (
    "scaffold", "Hprerx", "XTP", "XTPsmi", "Ngen(rate)", "Novel", "fullrx", "size mean/std/med/min/max", "RMSD", "dE"))
for r in rows:
    print("%-18s %6d %6d %8d %7d(%.1f%%) %6d %6d  %-24s %5.2f %5.1f" % (
        r["scaf"], r["Hpre"], r["xtp"], r["nsmi"], r["nd"], r["nd"] / r["gen"] * 100, r["novel"], r["full"],
        "%.1f/%.1f/%d/%d/%d" % r["sz"], r["rmsd"], r["dE"]))
print("\n=== Table 3 (diversity / properties, model side) ===")
print("%-18s %6s %6s %8s %7s %12s %12s %14s" % ("scaffold", "Ngen", "Neff", "#Murcko", "IntDiv", "MW", "logP", "QED"))
for r in rows:
    print("%-18s %6d %6.0f %8d %7.3f %6.1f+-%4.1f %+5.2f+-%4.2f %6.3f+-%5.3f" % (
        r["scaf"], r["nd"], r["neff"], r["nmk"], r["idiv"], *r["MW"], *r["lP"], *r["qed"]))
print("\n=== strain per heavy atom over all XTP molecules (kcal/mol): p25/50/75/90 ===")
print(np.round(np.percentile(strain_all, [25, 50, 75, 90]), 3))
print("\n=== Fig. 6: generated -> relaxed shift ===")
dr = np.array(allDR); dth = np.array(allDTH)
print("median |dr| = %.4f A, median |dtheta| = %.2f deg" % (np.median(np.abs(dr)), np.median(np.abs(dth))))
be = np.arange(-0.155, 0.156, 0.01); hb, _ = np.histogram(dr, bins=be); hb = hb / hb.sum() * 100
print("BOND_HIST(%):", " ".join(f"({be[i] + 0.005:.3f},{hb[i]:.2f})" for i in range(len(hb))))
ae = np.arange(-15.5, 16.5, 1); ha, _ = np.histogram(dth, bins=ae); ha = ha / ha.sum() * 100
print("ANGLE_HIST(%):", " ".join(f"({ae[i] + 0.5:.1f},{ha[i]:.2f})" for i in range(len(ha))))
