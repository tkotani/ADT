"""GEOM-Drugs reference values of the paper, with the same definitions as the model side (paper_tables.py).

subcommands
  props   --smi geom_drugs.smi
          full GEOM-Drugs: mean +- std of MW, Crippen logP and QED over all valid molecules (Table 3 caption),
          heavy-atom size mean +- std and heavy-atom composition (Table 4, GEOM column)
  intdiv  --smi geom_drugs.smi
          IntDiv_1 of each scaffold's GEOM pool (distinct non-isomeric molecules containing the ring; all of GEOM
          for bootstrap3), computed exactly as for the model: all pairs of a 1500-molecule sample
          (random.Random(0) on the sorted pool), Morgan radius 2, 2048 bits (Table 3, GEOM IntDiv column)
  extract --rdkit_folder <GEOM rdkit_folder> --n 5000 --seed 0 --out geom_confs.pkl
          draw --n neutral single-fragment GEOM-Drugs molecules (seeded) and keep, for each, the LOWEST-ENERGY
          conformer (GFN2-xTB geometry, hydrogens included), heavy atoms first
  charges --extracted geom_confs.pkl
          the charge-separation control: the same perception as for generated molecules (validate_3D on the
          xTB geometry, neutral Lewis structure, heavy-atom bonds equal to GEOM's own graph), then the share of
          molecules with an S-mediated / a non-S formal-charge pair
"""
import os, sys, json, random, argparse, pickle
from collections import Counter
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, QED
RDLogger.DisableLog("rdApp.*")

SCAFFOLDS = [("benzene", "c1ccccc1"), ("pyridine", "c1ccncc1"), ("pyrimidine", "c1cncnc1"),
             ("pyrazine", "c1cnccn1"), ("furan", "c1ccoc1"), ("thiophene", "c1ccsc1"),
             ("cyclohexane", "C1CCCCC1"), ("bootstrap3", "")]
ELEM = {6: "C", 7: "N", 8: "O", 16: "S", 9: "F", 17: "Cl", 35: "Br"}


def read_smiles(path):
    return [l.split()[0] for l in open(path) if l.strip()]


def _props(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        return (Descriptors.MolWt(m), Crippen.MolLogP(m), QED.qed(m), m.GetNumHeavyAtoms(),
                [ELEM.get(a.GetAtomicNum(), "other") for a in m.GetAtoms()])
    except Exception:
        return None


def cmd_props(a):
    smis = read_smiles(a.smi)
    with Pool(a.procs) as p:
        res = [x for x in p.map(_props, smis, chunksize=500) if x]
    mw, lp, qd, sz = (np.array([x[i] for x in res]) for i in range(4))
    ec = Counter(e for x in res for e in x[4]); tot = sum(ec.values())
    print("GEOM-Drugs: %d lines, %d valid molecules" % (len(smis), len(res)))
    print("MW %.1f +- %.1f   logP %+.2f +- %.2f   QED %.3f +- %.3f" % (mw.mean(), mw.std(), lp.mean(), lp.std(), qd.mean(), qd.std()))
    print("size (heavy) %.2f +- %.2f (median %d)" % (sz.mean(), sz.std(), int(np.median(sz))))
    print("C/N/O = %.2f / %.2f / %.2f;  S/F/Cl/Br = %.2f / %.2f / %.2f / %.2f (%%)"
          % tuple(100 * ec.get(k, 0) / tot for k in ("C", "N", "O", "S", "F", "Cl", "Br")))


def cmd_intdiv(a):
    from paper_tables import intdiv
    smis = read_smiles(a.smi)
    for name, smarts in SCAFFOLDS:
        pat = Chem.MolFromSmarts(smarts) if smarts else None
        u = {}
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is None or (pat is not None and not m.HasSubstructMatch(pat)):
                continue
            u.setdefault(Chem.MolToSmiles(m, isomericSmiles=False), None)
        pool = [Chem.MolFromSmiles(k) for k in sorted(u)]
        idv, tmed = intdiv([m for m in pool if m is not None])
        print("%-12s pool %6d  IntDiv %.3f  (median pair Tanimoto %.2f)" % (name, len(pool), idv, tmed), flush=True)


def cmd_extract(a):
    S = json.load(open(os.path.join(a.rdkit_folder, "summary_drugs.json")))
    keys = sorted(S); random.Random(a.seed).shuffle(keys)
    out, n_first_lowest, seen = [], 0, 0
    for k in keys:
        if len(out) >= a.n:
            break
        m0 = Chem.MolFromSmiles(k)
        if m0 is None or "." in k:
            continue                                            # single-fragment molecules only
        if not a.allow_charged and any(at.GetFormalCharge() != 0 for at in m0.GetAtoms()):
            continue                                            # by default also formally neutral
        pp = S[k].get("pickle_path")
        if not pp:
            continue
        try:
            confs = pickle.load(open(os.path.join(a.rdkit_folder, pp), "rb"))["conformers"]
        except Exception:
            continue
        if not confs:
            continue
        e = [c.get("totalenergy") for c in confs]
        i = int(np.argmin(e)); seen += 1; n_first_lowest += (i == 0)
        m = confs[i]["rd_mol"]
        z = [at.GetAtomicNum() for at in m.GetAtoms()]
        order = [j for j in range(len(z)) if z[j] != 1] + [j for j in range(len(z)) if z[j] == 1]
        new = {old: nw for nw, old in enumerate(order)}
        xyz = m.GetConformer().GetPositions()[order]
        nh = sum(1 for v in z if v != 1)
        bonds = sorted({(min(new[b.GetBeginAtomIdx()], new[b.GetEndAtomIdx()]), max(new[b.GetBeginAtomIdx()], new[b.GetEndAtomIdx()]))
                        for b in m.GetBonds() if z[b.GetBeginAtomIdx()] != 1 and z[b.GetEndAtomIdx()] != 1})
        out.append({"smiles": k, "anums": [z[j] for j in order], "coords": xyz.tolist(), "n_heavy": nh, "bonds": bonds,
                    "has_S": any(v == 16 for v in z)})
    pickle.dump(out, open(a.out, "wb"))
    print("extracted %d molecules (lowest-energy conformer); the first listed conformer was the lowest-energy one in %d of %d (%.1f%%)"
          % (len(out), n_first_lowest, seen, 100.0 * n_first_lowest / max(seen, 1)))


def _perceive(r):
    from util_validation import validate_3D
    try:
        mol, _, _ = validate_3D(list(r["anums"]), np.asarray(r["coords"], float), charge=0)
    except Exception:
        return "exception", None
    if mol is None:
        return "no_lewis", None
    na = r["n_heavy"]
    per = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i < na and j < na:
            per.add((min(i, j), max(i, j)))
    if per != set(tuple(b) for b in r["bonds"]):
        return "bond_mismatch", None
    h = Chem.RemoveAllHs(mol)
    chg = [at for at in h.GetAtoms() if at.GetFormalCharge() != 0]
    if not chg:
        return "neutral", None
    return ("S-mediated" if any(at.GetSymbol() == "S" for at in chg) else "non-S"), \
        "/".join(sorted("%s%+d" % (at.GetSymbol(), at.GetFormalCharge()) for at in chg))


def cmd_charges(a):
    recs = pickle.load(open(a.extracted, "rb"))
    with Pool(a.procs) as p:
        res = p.map(_perceive, recs, chunksize=50)
    c = Counter(x[0] for x in res)
    ok = c["neutral"] + c["S-mediated"] + c["non-S"]
    nS = sum(1 for r, x in zip(recs, res) if r["has_S"] and x[0] in ("neutral", "S-mediated", "non-S"))
    print("GEOM-Drugs control: %d molecules; %s" % (len(recs), dict(c)))
    print("read back as the same molecule: %d (%.1f%%)" % (ok, 100.0 * ok / len(recs)))
    print("S-mediated charge pair: %d (%.2f%% of read-back; %.2f%% of %d S-containing)"
          % (c["S-mediated"], 100.0 * c["S-mediated"] / max(ok, 1), 100.0 * c["S-mediated"] / max(nS, 1), nS))
    print("non-S charge pair:      %d (%.2f%% of read-back)" % (c["non-S"], 100.0 * c["non-S"] / max(ok, 1)))
    pats = Counter(x[1] for x in res if x[1])
    for pat, n in pats.most_common(8):
        print("   %5d  %s" % (n, pat))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("props", "intdiv"):
        s = sub.add_parser(name); s.add_argument("--smi", required=True); s.add_argument("--procs", type=int, default=16)
    s = sub.add_parser("extract"); s.add_argument("--rdkit_folder", required=True); s.add_argument("--n", type=int, default=5000)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--out", required=True)
    s.add_argument("--allow_charged", action="store_true",
                   help="also keep molecules whose SMILES carries formal charges (nitro, N-oxides, zwitterions); single fragment only")
    s = sub.add_parser("charges"); s.add_argument("--extracted", required=True); s.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    {"props": cmd_props, "intdiv": cmd_intdiv, "extract": cmd_extract, "charges": cmd_charges}[a.cmd](a)


if __name__ == "__main__":
    main()
