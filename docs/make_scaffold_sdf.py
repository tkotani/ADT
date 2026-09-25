#!/usr/bin/env python3
"""make_scaffold_sdf.py — pick the molecules shown by the 3D browser from the molrecord banks.

For each ring scaffold, draws --n molecules (fixed seed) from the paper's N^gen set: XTP-accepted and
read back by RDKit as the declared molecule (same definition as common/paper_tables.py), and whose
first atoms are the scaffold ring, so the page can show the growth from the ring. Each is
written heavy-atom only, in the order the model generated the atoms (the scaffold ring first), with
the RAW network coordinates (before any xTB relaxation) and the bond orders RDKit perceived on the
relaxed structure. export_html.py turns data/<scaffold>_50.sdf into <scaffold>.html.

usage: python3 docs/make_scaffold_sdf.py <bank_dir> [--n 50] [--seed 0] [--out docs/data]
"""
import os, sys, glob, argparse, random
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
RDLogger.DisableLog("rdApp.*")
from util_validation import validate_3D

SCAFFOLDS = {"benzene": 6, "pyridine": 6, "pyrimidine": 6, "pyrazine": 6,
             "furan": 5, "thiophene": 5, "cyclohexane": 6}      # name -> ring size


def declared_mol(r):
    """Heavy-atom RDKit mol in generation order with raw coordinates, or None."""
    si, sr = r.get("struct_init"), r.get("struct_relaxed")
    if not (isinstance(si, dict) and isinstance(sr, dict)):
        return None
    ai, ci = np.asarray(si["anums"]), np.asarray(si["coords"], float)
    na = int(r["n_heavy"])
    if not (ai[:na] > 1).all():
        return None
    try:
        mol, _, _ = validate_3D(list(sr["anums"]), np.asarray(sr["coords"], float), charge=0)
    except Exception:
        return None
    if mol is None:
        return None
    per = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i < na and j < na:
            per.add((min(i, j), max(i, j)))
    if per != set((min(int(a), int(b)), max(int(a), int(b))) for a, b in r["bonds0"]):
        return None
    m = Chem.RWMol(Chem.RemoveAllHs(mol))
    if m.GetNumAtoms() != na:
        return None
    conf = m.GetConformer()
    for k in range(na):
        conf.SetAtomPosition(k, Point3D(*ci[k]))
    return m.GetMol()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bank_dir")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, ring in SCAFFOLDS.items():
        f = os.path.join(args.bank_dir, name + "_real", "pfree_bank_%s_real.pt" % name)
        recs = [r for r in torch.load(f, weights_only=False)["mols"] if r.get("xvr")]
        random.Random(args.seed).shuffle(recs)
        picked, seen = [], set()
        for r in recs:
            m = declared_mol(r)
            if m is None:
                continue
            smi = Chem.MolToSmiles(m, isomericSmiles=False)
            if smi in seen:
                continue
            sub = [b for b in m.GetBonds() if b.GetBeginAtomIdx() < ring and b.GetEndAtomIdx() < ring]
            deg = [sum(1 for b in sub if k in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for k in range(ring)]
            if len(sub) != ring or any(d != 2 for d in deg):
                continue                      # the growth must start from the scaffold ring itself
            seen.add(smi); m.SetProp("SMILES", smi); m.SetProp("scaffold", name)
            picked.append(m)
            if len(picked) == args.n:
                break
        path = os.path.join(args.out, "%s_%d.sdf" % (name, args.n))
        w = Chem.SDWriter(path)
        for m in picked:
            w.write(m)
        w.close()
        print("%-12s %d molecules -> %s" % (name, len(picked), path))


if __name__ == "__main__":
    main()
