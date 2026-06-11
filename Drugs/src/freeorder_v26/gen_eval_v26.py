"""Drugs v26 generation + xTB evaluation.

Uses fo_train_v26.generate_one with benzene frame cache.
ALLOWED_ATOMS matches v21: {6,7,8,9,15,16,17,35,53} (C,N,O,F,P,S,Cl,Br,I).
"""
import sys, os, time, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_v21_dir = os.path.abspath(os.path.join(_HERE, "..", "freeorder_v21"))
if _v21_dir not in sys.path:
    sys.path.insert(0, _v21_dir)
_common_dir = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "common"))
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)

os.environ["ADT_R_BINS"] = os.environ.get("ADT_R_BINS", "200")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["AROMATIZE_RINGS"] = os.environ.get("AROMATIZE_RINGS", "1")  # Drugs default

import torch, numpy as np
from collections import Counter
from itertools import combinations

from fo_train_v26 import generate_one, COLLISION_R
from adt_model import build_model
from adt_dataset import FrameSampler
from util_validation import validate_3D
from gen_eval_lib import try_write_sdf, xtb_relax, consistency_check, print_summary
from collision_check import check_collisions

from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops, rdMolDescriptors, Descriptors
RDLogger.logger().setLevel(RDLogger.ERROR)

ALLOWED_ATOMS = {6, 7, 8, 9, 15, 16, 17, 35, 53}  # match v21
N = int(os.environ.get("GEN_N", "1000"))
device = torch.device("cuda")

ckpt_path = os.environ.get("CKPT", "/tmp/drugs_v23_best.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = build_model(ckpt["config"])
max_pointer_ckpt = ckpt["config"].get("max_pointer", 30)
print(f"ckpt max_pointer = {max_pointer_ckpt}", flush=True)
model.load_state_dict(ckpt["model"])
model.to(device); model.eval()
print(f"Loaded {ckpt_path} epoch={ckpt['epoch']} val={ckpt['val_loss']:.4f}")
sys.stdout.flush()

frame_sampler = FrameSampler.load(os.environ.get(
    "FRAME_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "freeorder_v26", "frame_cache_v21_benzene.pt")))
print("FrameSampler loaded")

drugs_ref_path = os.environ.get("DRUGS_REF", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "freeorder_v26", "drugs_mols_v21_nolimit.pkl"))
drugs_smiles = set()
if os.path.exists(drugs_ref_path):
    import pickle
    with open(drugs_ref_path, "rb") as f:
        drugs_data = pickle.load(f)
    for mol, pos, smi in drugs_data:
        try:
            m2 = Chem.MolFromSmiles(smi)
            if m2:
                Chem.RemoveStereochemistry(m2)
                drugs_smiles.add(Chem.MolToSmiles(m2))
        except Exception:
            pass
    print(f"Drugs reference: {len(drugs_smiles)} unique SMILES (no_stereo)")

outdir = os.environ.get("OUTDIR", f"/tmp/v23_drugs_eval_N{N}")
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
    # Collision: info only, do NOT filter (matches Drugs v21 gen_eval)
    bonds_0idx = set((e1-1, e2-1) for e1, e2 in bonds)
    has_coll, _ = check_collisions(coords, anums, bonds_0idx)
    if has_coll:
        stats["collision"] += 1  # count, no skip
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
        inchi_pre = ""; topo_pre = ""

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
    # Include both kekulized and aromatic-fallback SDFs in xTB pipeline.
    # xTB uses 3D coords only and is insensitive to bond-order representation;
    # aromatic-fallback mols are chemically valid (mol_stable passed), they
    # just failed RDKit kekulize due to ambiguous bond order assignment.
    if sdf is not None and sdf_tag in ("kekulized", "aromatic"):
        stable_mols.append({**props, "sdf": sdf, "anums": anums, "coords": coords})

    if (i+1) % 500 == 0:
        print(f"  [{i+1}/{N}] {time.time()-t0:.0f}s stable={stats['mol_stable']} unique={len(set(all_smiles_pre))}")
        sys.stdout.flush()

sdf_kek_f.close(); sdf_arom_f.close(); smiles_f.close()
gen_elapsed = time.time() - t0
unique_pre = set(all_smiles_pre)

kek_mols = [m for m in stable_mols if m.get("sdf_tag") == "kekulized"]
arom_mols = [m for m in stable_mols if m.get("sdf_tag") == "aromatic"]
arom_n = stats.get("sdf_aromatic", 0)

print(f"\n=== Phase 1 results (N={N}, {gen_elapsed/60:.1f} min) ===")
for k in ["gen_fail","too_small","bad_atom","collision","invalid_3d","valid_3d",
          "disconnected","unstable","mol_stable"]:
    v = stats.get(k, 0)
    print(f"  {k:<20s} {v:>6d} ({v/N*100:>5.1f}%)")
print(f"\n  mol_stable/N = {stats.get('mol_stable', 0)}/{N} = {stats.get('mol_stable',0)/N*100:.1f}%")
print(f"  unique       = {len(unique_pre)}")
print(f"  SDF saved    = {len(stable_mols)} (kek={len(kek_mols)} + arom={arom_n})")
if drugs_smiles:
    n_in_ref = sum(1 for s in unique_pre if s in drugs_smiles)
    print(f"  in Drugs ref = {n_in_ref}/{len(unique_pre)} ({n_in_ref/max(len(unique_pre),1)*100:.1f}%)")
    print(f"  novel        = {len(unique_pre)-n_in_ref}")
sys.stdout.flush()

with open(os.path.join(outdir, "gen_stats.json"), "w") as f:
    json.dump({"N": N, "gen_elapsed_min": round(gen_elapsed/60, 1),
               "gen_stats": dict(stats), "n_unique": len(unique_pre),
               "n_sdf_total": len(stable_mols), "n_kekulizable": len(kek_mols),
               "n_aromatic": arom_n, "ckpt": ckpt_path, "epoch": ckpt["epoch"],
               "val_loss": float(ckpt["val_loss"])}, f, indent=2)

# =============================================================
# Phase 2: xTB + consistency
# =============================================================
print(f"\n=== Phase 2: xTB + consistency ({len(stable_mols)} molecules: "
      f"{len(kek_mols)} kekulized + {len(arom_mols)} aromatic-fallback) ===")
sys.stdout.flush()

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
XTB = os.environ.get("XTB_BIN", os.path.expanduser("~/miniconda3/bin/xtb"))

xtb_results = []
n_xtb_ok = 0; n_xtb_fail = 0
n_topo_same = 0; n_inchi_same = 0; n_either_same = 0
# Per-tag counters (kekulized vs aromatic-fallback)
xtb_ok_by_tag = {"kekulized": 0, "aromatic": 0}
xtb_fail_by_tag = {"kekulized": 0, "aromatic": 0}
topo_same_by_tag = {"kekulized": 0, "aromatic": 0}
inchi_same_by_tag = {"kekulized": 0, "aromatic": 0}
either_by_tag = {"kekulized": 0, "aromatic": 0}
t1 = time.time()

xtb_inc_f = open(os.path.join(outdir, "xtb_results_incremental.jsonl"), "w")

for idx, md in enumerate(stable_mols):
    tag = md.get("sdf_tag", "kekulized")
    mol_h = Chem.MolFromMolBlock(md["sdf"], sanitize=False)
    if mol_h is None:
        n_xtb_fail += 1; xtb_fail_by_tag[tag] = xtb_fail_by_tag.get(tag, 0) + 1
        continue
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
        if cc["same_topo"]: n_topo_same += 1; topo_same_by_tag[tag] = topo_same_by_tag.get(tag, 0) + 1
        if cc["same_inchi"]: n_inchi_same += 1; inchi_same_by_tag[tag] = inchi_same_by_tag.get(tag, 0) + 1
        if cc["same"]: n_either_same += 1; either_by_tag[tag] = either_by_tag.get(tag, 0) + 1
        rec = {"idx": idx, "smi": md["smi"], "sdf_tag": tag,
               "e_gain": res["e_gain"],
               "rmsd": res["rmsd"], "rmsd_heavy": res.get("rmsd_heavy"),
               "topo_post": cc["topo_post"], "inchi_post": cc["inchi_post"],
               "same_topo": cc["same_topo"], "same_inchi": cc["same_inchi"],
               "same": cc["same"], "ok": True}
        n_xtb_ok += 1
        xtb_ok_by_tag[tag] = xtb_ok_by_tag.get(tag, 0) + 1
    else:
        rec = {"idx": idx, "smi": md["smi"], "sdf_tag": tag, "ok": False,
               "reason": res.get("reason", "unknown")}
        n_xtb_fail += 1
        xtb_fail_by_tag[tag] = xtb_fail_by_tag.get(tag, 0) + 1
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
        print(f"  xTB [{idx+1}/{len(stable_mols)}] {time.time()-t1:.0f}s ok={n_xtb_ok} same={n_either_same}")
        sys.stdout.flush()

xtb_inc_f.close()
xtb_elapsed = time.time() - t1

print(f"\n=== xTB Results ({xtb_elapsed/60:.1f} min) ===")
print(f"  Converged: {n_xtb_ok}/{len(stable_mols)} ({n_xtb_ok/max(len(stable_mols),1)*100:.1f}%)")
# Breakdown by SDF type: kekulized (preferred, unambiguous bond orders) vs
# aromatic-fallback (still chemically valid, but kekulize failed — see note below)
print(f"    kekulized (preferred): {xtb_ok_by_tag.get('kekulized',0)}/{len(kek_mols)} converged")
if len(arom_mols) > 0:
    print(f"    aromatic-fallback:     {xtb_ok_by_tag.get('aromatic',0)}/{len(arom_mols)} converged")

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

# Topology preservation breakdown
print(f"\n  Topology preserved breakdown (kekulized is preferred):")
kek_tot = len(kek_mols); arom_tot = len(arom_mols)
print(f"    kekulized  : topo_same={topo_same_by_tag.get('kekulized',0)}/{kek_tot} "
      f"inchi_same={inchi_same_by_tag.get('kekulized',0)}/{kek_tot}")
if arom_tot > 0:
    print(f"    aromatic   : topo_same={topo_same_by_tag.get('aromatic',0)}/{arom_tot} "
          f"inchi_same={inchi_same_by_tag.get('aromatic',0)}/{arom_tot}")
print(f"    combined   : topo_same={n_topo_same}/{len(stable_mols)} "
      f"inchi_same={n_inchi_same}/{len(stable_mols)}")
print(f"  Note: aromatic-fallback mols are mol_stable but failed RDKit kekulize; "
      f"xTB still runs on 3D coords. Kekulized is preferred for unambiguous bond orders.")

print_summary(stats, xtb_results, n_topo_same, n_inchi_same, n_either_same, n_xtb_ok)

with open(os.path.join(outdir, "xtb_results.json"), "w") as f:
    json.dump(xtb_results, f, indent=1)
with open(os.path.join(outdir, "stats.json"), "w") as f:
    json.dump({"N": N, "gen_elapsed_min": round(gen_elapsed/60, 1),
               "xtb_elapsed_min": round(xtb_elapsed/60, 1),
               "total_elapsed_min": round((gen_elapsed+xtb_elapsed)/60, 1),
               "gen_stats": dict(stats), "n_unique": len(unique_pre),
               "n_sdf_total": len(stable_mols), "n_kekulizable": len(kek_mols),
               "n_aromatic_fallback": arom_n,
               "xtb_converged": n_xtb_ok, "xtb_failed": n_xtb_fail,
               "topo_same": n_topo_same, "inchi_same": n_inchi_same,
               "either_same": n_either_same,
               "by_tag": {
                   "xtb_ok": xtb_ok_by_tag,
                   "xtb_fail": xtb_fail_by_tag,
                   "topo_same": topo_same_by_tag,
                   "inchi_same": inchi_same_by_tag,
                   "either": either_by_tag,
               }}, f, indent=2)

print(f"\n=== Total: {(time.time()-t0)/60:.1f} min ===")
print(f"All saved to {outdir}")
