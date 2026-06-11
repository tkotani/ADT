"""QM9 generation + xTB optimization + SMILES consistency check.

Pipeline:
  Phase 1: Generate N molecules → mol_stable_kekulized.sdf + metadata
  Phase 2: xTB GFN2 relax → xtb_results.json + xyz files
  Phase 3: SMILES before/after consistency check (via consistency_check)

Env vars:
  GEN_N=10000            Number of molecules to generate
  CKPT=...               Checkpoint (default checkpoints_fo_wide_200bin_repro/best.pt)
  FRAME_CACHE=...        Frame cache (default ../frame_cache_200bin.pt)
  QM9_CACHE=...          QM9 molecule cache for novelty (default ../qm9_mols_cache_v3b_noh.pkl)
  OUTDIR=...             Output dir (default eval_N${GEN_N})
  XTB_BIN=...            xtb binary (default ~/miniconda3/bin/xtb)
  AROMATIZE_RINGS=0      (default; QM9 molecules are small, aromatization hurts)
"""
import sys, os, time, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(1, os.path.join(_HERE, ".."))
_common_dir = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "common"))
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)

os.environ["ADT_R_BINS"] = os.environ.get("ADT_R_BINS", "200")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["AROMATIZE_RINGS"] = os.environ.get("AROMATIZE_RINGS", "0")

import torch, numpy as np
from collections import Counter
from itertools import combinations

from fo_train_v26 import generate_one, COLLISION_R
from adt_model import build_model
from adt_dataset import FrameSampler
from util_validation import validate_3D
from gen_eval_lib import try_write_sdf, xtb_relax, consistency_check, print_summary

from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops, rdMolDescriptors, Descriptors
RDLogger.logger().setLevel(RDLogger.ERROR)

ALLOWED_ATOMS = {6, 7, 8, 9}
N = int(os.environ.get("GEN_N", "10000"))
device = torch.device("cuda")

ckpt_path = os.environ.get("CKPT", "checkpoints_fo_wide_200bin_repro/best.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = build_model(ckpt["config"])
max_pointer_ckpt = ckpt["config"].get("max_pointer", 30)
print(f"ckpt max_pointer = {max_pointer_ckpt}", flush=True)
model.load_state_dict(ckpt["model"])
model.to(device); model.eval()
print(f"Loaded {ckpt_path} epoch={ckpt['epoch']} val={ckpt['val_loss']:.4f}")
sys.stdout.flush()

frame_sampler = FrameSampler.load(os.environ.get("FRAME_CACHE", "../frame_cache_200bin.pt"))
print(f"FrameSampler loaded")

qm9_cache_path = os.environ.get("QM9_CACHE", "../qm9_mols_cache_v3b_noh.pkl")
qm9_smiles = set()
if os.path.exists(qm9_cache_path):
    import pickle
    with open(qm9_cache_path, "rb") as f:
        qm9_data = pickle.load(f)
    for mol, pos, smi in qm9_data:
        try:
            m2 = Chem.MolFromSmiles(smi)
            if m2:
                Chem.RemoveStereochemistry(m2)
                qm9_smiles.add(Chem.MolToSmiles(m2))
        except Exception:
            pass
    print(f"QM9 reference: {len(qm9_smiles)} unique SMILES (no_stereo)")

outdir = os.environ.get("OUTDIR", f"eval_N{N}")
os.makedirs(outdir, exist_ok=True)
xtb_workdir = os.path.join(outdir, "xtb_work")
os.makedirs(xtb_workdir, exist_ok=True)

# =============================================================
# Phase 1: Generation
# =============================================================
stats = Counter()
stable_mols = []
all_smiles_pre = []
t0 = time.time()

sdf_kek_f = open(os.path.join(outdir, "mol_stable_kekulized.sdf"), "w")
sdf_arom_f = open(os.path.join(outdir, "mol_stable_aromatic.sdf"), "w")
smiles_f = open(os.path.join(outdir, "mol_stable_smiles.txt"), "w")
meta_all = []

print(f"\n=== Phase 1: generating N={N} molecules ===")
sys.stdout.flush()

for i in range(N):
    try:
        at, bonds, na = generate_one(model, device, frame_sampler=frame_sampler, temperature=1.0)
    except Exception:
        stats["gen_fail"] += 1; continue
    if na < 2:
        stats["too_small"] += 1; continue
    anums = [at[k].atomic_num for k in range(na)]
    coords = [list(at[k].pos) for k in range(na)]
    if any(a not in ALLOWED_ATOMS for a in anums):
        stats["bad_atom"] += 1; continue
    has_coll = any(np.linalg.norm(np.array(coords[a])-np.array(coords[b])) < COLLISION_R
                   for a, b in combinations(range(na), 2))
    if has_coll:
        stats["collision"] += 1; continue
    mol, smi, info = validate_3D(anums, coords)
    if not info.get("valid"):
        stats["invalid_3d"] += 1; continue
    stats["valid_3d"] += 1
    is_conn = len(rdmolops.GetMolFrags(mol)) == 1
    is_stable = info.get("mol_stable", False)
    if not is_conn:
        stats["disconnected"] += 1; continue
    if not is_stable:
        stats["unstable"] += 1; continue
    stats["mol_stable"] += 1

    try:
        inchi_pre = Chem.MolToInchiKey(mol).split('-')[0]
        topo_pre = Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        inchi_pre = ""
        topo_pre = ""

    all_smiles_pre.append(smi)
    smiles_f.write(smi + "\n"); smiles_f.flush()

    try:
        arom = rdMolDescriptors.CalcNumAromaticRings(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
    except Exception:
        arom = rings = 0; mw = logp = 0.0

    props = {"idx": i, "smi": smi, "n_atoms": na, "arom": arom, "rings": rings,
             "mw": round(mw, 1), "logp": round(logp, 2),
             "inchi": inchi_pre, "topo": topo_pre}

    sdf, sdf_tag = try_write_sdf(mol, sdf_kek_f, sdf_arom_f, stats)
    props["sdf_tag"] = sdf_tag
    meta_all.append(props)
    if sdf is not None and sdf_tag == "kekulized":
        stable_mols.append({**props, "sdf": sdf, "anums": anums, "coords": coords})

    if (i+1) % 1000 == 0:
        print(f"  [{i+1}/{N}] {time.time()-t0:.0f}s stable={stats['mol_stable']} unique={len(set(all_smiles_pre))}")
        sys.stdout.flush()

sdf_kek_f.close(); sdf_arom_f.close(); smiles_f.close()
gen_elapsed = time.time() - t0
unique_pre = set(all_smiles_pre)

kek_mols = [m for m in stable_mols if m.get("sdf_tag") == "kekulized"]
arom_n = stats.get("sdf_aromatic", 0)

print(f"\n=== Phase 1 results (N={N}, {gen_elapsed/60:.1f} min) ===")
for k in ["gen_fail","too_small","bad_atom","collision","invalid_3d","valid_3d",
          "disconnected","unstable","mol_stable"]:
    v = stats.get(k, 0)
    print(f"  {k:<20s} {v:>6d} ({v/N*100:>5.1f}%)")
print(f"\n  mol_stable/N = {stats.get('mol_stable', 0)}/{N} = {stats.get('mol_stable',0)/N*100:.1f}%")
print(f"  unique       = {len(unique_pre)}")
print(f"  SDF saved    = {len(stable_mols)} (kek={len(kek_mols)} + arom={arom_n})")
if qm9_smiles:
    n_in_qm9 = sum(1 for s in unique_pre if s in qm9_smiles)
    print(f"  in QM9       = {n_in_qm9}/{len(unique_pre)} ({n_in_qm9/max(len(unique_pre),1)*100:.1f}%)")
    print(f"  novel        = {len(unique_pre)-n_in_qm9}")
sys.stdout.flush()

with open(os.path.join(outdir, "mol_stable_meta.json"), "w") as f:
    json.dump(meta_all, f, indent=1)
with open(os.path.join(outdir, "gen_stats.json"), "w") as f:
    json.dump({
        "N": N, "gen_elapsed_min": round(gen_elapsed/60, 1),
        "gen_stats": dict(stats), "n_unique": len(unique_pre),
        "n_sdf_total": len(stable_mols), "n_kekulizable": len(kek_mols),
        "n_aromatic": arom_n, "ckpt": ckpt_path, "epoch": ckpt["epoch"],
        "val_loss": float(ckpt["val_loss"]),
    }, f, indent=2)

# =============================================================
# Phase 2: xTB optimization + consistency check
# =============================================================
print(f"\n=== Phase 2: xTB + consistency ({len(kek_mols)} molecules) ===")
sys.stdout.flush()

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
XTB = os.environ.get("XTB_BIN", os.path.expanduser("~/miniconda3/bin/xtb"))

xtb_results = []
n_xtb_ok = 0; n_xtb_fail = 0
n_topo_same = 0; n_inchi_same = 0; n_either_same = 0
t1 = time.time()

xtb_inc_f = open(os.path.join(outdir, "xtb_results_incremental.jsonl"), "w")

for idx, md in enumerate(kek_mols):
    mol_h = Chem.MolFromMolBlock(md["sdf"], sanitize=False)
    if mol_h is None:
        n_xtb_fail += 1; continue
    try:
        Chem.SanitizeMol(mol_h)
    except Exception:
        pass
    mol_h = Chem.AddHs(mol_h, addCoords=True)
    init_heavy = []
    conf = mol_h.GetConformer()
    for a in mol_h.GetAtoms():
        if a.GetAtomicNum() != 1:
            p = conf.GetAtomPosition(a.GetIdx())
            init_heavy.append([p.x, p.y, p.z])
    charge = sum(a.GetFormalCharge() for a in mol_h.GetAtoms())

    res = xtb_relax(idx, md["sdf"], init_heavy, xtb_workdir, XTB, charge=charge)

    if res.get("ok"):
        cc = consistency_check(md.get("inchi", ""), md.get("topo", ""),
                               res.get("opt_heavy_anums"), res.get("opt_heavy_coords"),
                               validate_3D)
        if cc["same_topo"]: n_topo_same += 1
        if cc["same_inchi"]: n_inchi_same += 1
        if cc["same"]: n_either_same += 1
        rec = {"idx": idx, "smi": md["smi"], "e_gain": res["e_gain"],
               "rmsd": res["rmsd"], "rmsd_heavy": res.get("rmsd_heavy"),
               "topo_post": cc["topo_post"], "inchi_post": cc["inchi_post"],
               "same_topo": cc["same_topo"], "same_inchi": cc["same_inchi"],
               "same": cc["same"], "ok": True}
        n_xtb_ok += 1
    else:
        rec = {"idx": idx, "smi": md["smi"], "ok": False,
               "reason": res.get("reason", "unknown")}
        n_xtb_fail += 1
    xtb_results.append(rec)
    xtb_inc_f.write(json.dumps(rec) + "\n"); xtb_inc_f.flush()

    if idx >= int(os.environ.get("XYZ_KEEP", "20")):
        for pat in ["mol_%d.xyz", "mol_%d.out", "mol_%d.xtbtopo.mol",
                    "mol_%d.xtbopt.xyz", "mol_%d.xtbopt.log",
                    "mol_%d.charges", "mol_%d.wbo", "mol_%d.xtbrestart"]:
            p = os.path.join(xtb_workdir, pat % idx)
            if os.path.exists(p): os.remove(p)
        for fn in os.listdir(xtb_workdir):
            if fn.startswith(f".mol_{idx}"):
                os.remove(os.path.join(xtb_workdir, fn))

    if (idx+1) % 200 == 0:
        print(f"  xTB [{idx+1}/{len(kek_mols)}] {time.time()-t1:.0f}s ok={n_xtb_ok} same={n_either_same}")
        sys.stdout.flush()

xtb_inc_f.close()
xtb_elapsed = time.time() - t1

# =============================================================
# Summary
# =============================================================
print(f"\n=== xTB Results ({xtb_elapsed/60:.1f} min) ===")
print(f"  Converged: {n_xtb_ok}/{len(kek_mols)} ({n_xtb_ok/max(len(kek_mols),1)*100:.1f}%)")

ok = [r for r in xtb_results if r.get("ok")]
if ok:
    energies = np.array([r["e_gain"] for r in ok])
    rmsds_h = np.array([r["rmsd_heavy"] for r in ok if r.get("rmsd_heavy") is not None])
    print(f"  Energy gain: mean={np.mean(energies):.1f} median={np.median(energies):.1f} kcal/mol")
    if len(rmsds_h) > 0:
        print(f"  RMSD heavy : mean={np.mean(rmsds_h):.3f} median={np.median(rmsds_h):.3f} A (n={len(rmsds_h)})")
        for t in [0.1, 0.25, 0.5, 1.0]:
            n_ = np.sum(rmsds_h < t)
            print(f"    <{t:.2f}A: {n_} ({n_/len(rmsds_h)*100:.1f}%)")

print_summary(stats, xtb_results, n_topo_same, n_inchi_same, n_either_same, n_xtb_ok)

with open(os.path.join(outdir, "xtb_results.json"), "w") as f:
    json.dump(xtb_results, f, indent=1)
with open(os.path.join(outdir, "stats.json"), "w") as f:
    json.dump({
        "N": N, "gen_elapsed_min": round(gen_elapsed/60, 1),
        "xtb_elapsed_min": round(xtb_elapsed/60, 1),
        "total_elapsed_min": round((time.time()-t0)/60, 1),
        "gen_stats": dict(stats), "n_unique": len(unique_pre),
        "n_sdf": sum(stats.get(k, 0) for k in ["sdf_kekulized", "sdf_aromatic"]),
        "n_kekulizable": len(kek_mols),
        "n_aromatic_fallback": arom_n,
        "xtb_ok": n_xtb_ok, "xtb_fail": n_xtb_fail,
        "topo_same": n_topo_same, "inchi_same": n_inchi_same,
        "either_same": n_either_same,
        "ckpt": ckpt_path, "epoch": ckpt["epoch"],
        "val_loss": float(ckpt["val_loss"]),
    }, f, indent=2)

print(f"\n=== Total: {(time.time()-t0)/60:.1f} min ===")
print(f"All saved to {outdir}")
