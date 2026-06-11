#!/usr/bin/env python3
"""regen_structfidelity.py - regenerate the pre/post-xTB coordinates for Fig. 6.

The original evaluation retained xtb_work/*.xtbopt.xyz for only ~18 molecules
per scaffold, so the Fig. 6 / Section 4.5 numbers could not be recomputed
exactly. This script re-runs xTB --opt on every XVR-positive (ok & same)
molecule of the saved pre-structures (mol_stable_kekulized.sdf) and writes the
xtb_work/{mol_N.xyz, mol_N.xtbopt.xyz} files that analyze_structfidelity.py and
bond_angle_errors.py read. It reuses gen_eval_lib.xtb_relax -- the exact same
relaxation procedure as the original pipeline.

Output: Drugs/data/freeorder_v26/structfidelity/{qm9,drugs30,drugs50}/<scaffold>/
        xtb_work/ + xtb_results.json (copied). Resumable: existing *.xtbopt.xyz
        are skipped.
Usage:  ADT_ROOT=/path/to/ADT python3 regen_structfidelity.py [--procs N]
"""
import os, sys, json, shutil, tempfile, argparse
from multiprocessing import Pool
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

_HERE = os.path.dirname(os.path.abspath(__file__))
ADT_ROOT = os.environ.get("ADT_ROOT", os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.join(ADT_ROOT, "common"))
from gen_eval_lib import xtb_relax

XTB = os.environ.get("XTB_BIN", os.path.expanduser("~/miniconda3/bin/xtb"))
DRUGS = os.path.join(ADT_ROOT, "Drugs", "data", "freeorder_v26")
QM9 = os.path.join(ADT_ROOT, "QM9", "data", "freeorder")
OUT = os.path.join(DRUGS, "structfidelity")
SC = ["benzene", "pyridine", "pyrimidine", "pyrazine", "furan", "thiophene", "cyclohexane"]

DATASETS = [
    ("drugs30", os.path.join(DRUGS, "v26s_scaffolds_n10k"), SC),
    ("drugs50", os.path.join(DRUGS, "v26g_scaffolds_n10k"), SC),
    ("qm9", QM9 + "/eval_v26_E562_N10000", ["all"]),
]


def build_tasks(dataset, base, scaffolds, limit=None):
    tasks = []
    for sc in scaffolds:
        sdir = base if sc == "all" else os.path.join(base, sc)
        sdf = os.path.join(sdir, "mol_stable_kekulized.sdf")
        rj = os.path.join(sdir, "xtb_results.json")
        if not (os.path.exists(sdf) and os.path.exists(rj)):
            print("  SKIP (no data):", dataset, sc); continue
        res = json.load(open(rj))
        kek = sorted([r for r in res if r.get("sdf_tag") in ("kekulized", None)], key=lambda r: r["idx"])
        sup = Chem.SDMolSupplier(sdf, removeHs=False, sanitize=False)
        outdir = os.path.join(OUT, dataset, sc, "xtb_work")
        os.makedirs(outdir, exist_ok=True)
        shutil.copy(rj, os.path.join(OUT, dataset, sc, "xtb_results.json"))
        n = 0
        for k, mol in enumerate(sup):
            if k >= len(kek):
                break
            r = kek[k]
            if not (r.get("ok") and r.get("same")):
                continue
            if mol is None:
                continue
            idx = r["idx"]
            if os.path.exists(os.path.join(outdir, f"mol_{idx}.xtbopt.xyz")):
                continue
            try:
                mb = Chem.MolToMolBlock(mol)
                conf = mol.GetConformer()
                hv = [[conf.GetAtomPosition(a.GetIdx()).x, conf.GetAtomPosition(a.GetIdx()).y,
                       conf.GetAtomPosition(a.GetIdx()).z]
                      for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
            except Exception:
                continue
            tasks.append((dataset, sc, idx, mb, hv, outdir))
            n += 1
            if limit and n >= limit:
                break
    return tasks


def work(t):
    dataset, sc, idx, mb, hv, outdir = t
    tmp = tempfile.mkdtemp(prefix=f"sf_{idx}_")
    try:
        r = xtb_relax(idx, mb, hv, tmp, XTB, charge=0, timeout=300)
        if not r.get("ok"):
            return (dataset, sc, idx, False)
        for fn in (f"mol_{idx}.xyz", f"mol_{idx}.xtbopt.xyz"):
            src = os.path.join(tmp, fn)
            if os.path.exists(src):
                shutil.move(src, os.path.join(outdir, fn))
        return (dataset, sc, idx, True)
    except Exception:
        return (dataset, sc, idx, False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="per-scaffold molecule cap (dry-run)")
    ap.add_argument("--procs", type=int, default=30)
    ap.add_argument("--only", default=None, help="comma dataset filter e.g. drugs30")
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    sets = DATASETS if not args.only else [d for d in DATASETS if d[0] in args.only.split(",")]
    all_tasks = []
    for name, base, scaf in sets:
        t = build_tasks(name, base, scaf, limit=args.limit)
        print(f"[{name}] tasks={len(t)}", flush=True)
        all_tasks += t
    print(f"TOTAL tasks={len(all_tasks)}  procs={args.procs}", flush=True)
    done = ok = 0
    with Pool(args.procs) as p:
        for res in p.imap_unordered(work, all_tasks, chunksize=8):
            done += 1
            if res[3]:
                ok += 1
            if done % 500 == 0:
                print(f"  progress {done}/{len(all_tasks)}  ok={ok}", flush=True)
    print(f"DONE total={done} ok={ok}", flush=True)


if __name__ == "__main__":
    main()
