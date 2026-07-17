"""XVR reward (common): xTB convergence + topology preservation.

Graded shaping (dense signal for REINFORCE):
  0.0  RDKit fail (invalid / unstable / disconnected)
  0.3  RDKit pass, xTB does not converge
  0.6  xTB converges, topology changed after relaxation
  1.0  xTB converges AND topology preserved  (= true XVR)

Batch evaluation parallelizes xTB calls over a thread pool (xtb is a
subprocess, so threads release the GIL during subprocess.run).
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/ADT/common"))

from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops
RDLogger.DisableLog("rdApp.*")

from util_validation import validate_3D
from gen_eval_lib import xtb_relax, consistency_check, extract_metrics

ALLOWED_ATOMS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53}

# Reward tiers
R_FAIL = 0.0      # RDKit fail
R_RDKIT = 0.0     # (was 0.3) no credit for RDKit-pass-but-xTB-fail: focus on xTB convergence
R_XTB = 0.6       # xTB converge, topo changed
R_XVR = 1.0       # xTB converge + topo preserved

# --- e_gain-shaped XVR (experiment, 2026-07-01): reward LOW per-atom relaxation
# energy (strain). When XVR_ESTRAIN_TAU>0, the XVR tier (topology preserved) is
# shaped to  R_XTB + (R_XVR-R_XTB)*exp(-(|e_gain|/n_heavy)/TAU)  in [R_XTB, R_XVR],
# so among topology-preserving molecules, lower per-atom strain -> reward closer
# to 1.0 (topology-changed stays 0.6, fail 0). |e_gain| is the xTB relaxation
# energy in kcal/mol, normalized per heavy atom to avoid a size penalty.
# Default 0 = OFF: identical to the original XVR reward (safe for all runs/evals).
import math
XVR_ESTRAIN_TAU = float(os.environ.get("XVR_ESTRAIN_TAU", "0") or 0)


def _build_molblock(atoms, na):
    """Run validate_3D, return (mol_block, inchi_pre, topo_pre, charge) or None."""
    anums = [atoms[k].atomic_num for k in range(na)]
    if any(a not in ALLOWED_ATOMS for a in anums):
        return None
    coords = [list(atoms[k].pos) for k in range(na)]
    mol, smi, info = validate_3D(anums, coords)
    if not info.get("valid", False):
        return None
    if len(rdmolops.GetMolFrags(mol)) != 1:
        return None
    if not info.get("mol_stable", False):
        return None
    try:
        inchi_pre, topo_pre = extract_metrics(mol)
        mol_block = Chem.MolToMolBlock(mol)
    except Exception:
        return None
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    return mol_block, inchi_pre, topo_pre, charge, smi


def _angle_deg(a, b, c):
    """Angle at vertex b (degrees) for points a-b-c."""
    import numpy as np
    v1 = np.asarray(a, float) - np.asarray(b, float)
    v2 = np.asarray(c, float) - np.asarray(b, float)
    n1 = float(np.linalg.norm(v1)); n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosv = max(-1.0, min(1.0, float(np.dot(v1, v2)) / (n1 * n2)))
    return float(np.degrees(np.arccos(cosv)))


def _relax_geom_stats(molh, init_heavy, opt_heavy):
    """Per-molecule geometry change during xTB relaxation. init_heavy/opt_heavy are heavy-atom
    coords in molh's heavy-atom order (idx 0..nheavy-1; AddHs appends H AFTER the heavy atoms),
    so a bond with both endpoints < nheavy is heavy-heavy. Returns max heavy-atom displacement (A),
    mean |bond-length delta| (A), mean/max |bond-angle delta| (deg). None on shape mismatch."""
    import numpy as np
    if opt_heavy is None:
        return None
    I = np.asarray(init_heavy, float); O = np.asarray(opt_heavy, float)
    if I.shape != O.shape or len(I) < 3:
        return None
    disp = np.sqrt(((O - I) ** 2).sum(1))
    nh = len(I)
    nbr = [[] for _ in range(nh)]
    dlen = []
    for b in molh.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 < nh and a2 < nh:
            nbr[a1].append(a2); nbr[a2].append(a1)
            dlen.append(abs(float(np.linalg.norm(O[a1] - O[a2]) - np.linalg.norm(I[a1] - I[a2]))))
    dang = []
    for j in range(nh):
        ns = nbr[j]
        for x in range(len(ns)):
            for y in range(x + 1, len(ns)):
                dang.append(abs(_angle_deg(I[ns[x]], I[j], I[ns[y]]) - _angle_deg(O[ns[x]], O[j], O[ns[y]])))
    return {"max_disp": float(disp.max()),
            "dlen_mean": float(np.mean(dlen)) if dlen else 0.0,
            "dang_mean": float(np.mean(dang)) if dang else 0.0,
            "dang_max": float(np.max(dang)) if dang else 0.0}


def xvr_reward_single(atoms, bonds, na, xtb_bin, workdir, idx, collect_relax=False):
    """Returns dict: {reward, rdkit_ok, xtb_ok, same_topo, smi, topo_post} (+ e_gain / rmsd* /
    max_disp / dlen_mean / dang_mean / dang_max when collect_relax=True). collect_relax is
    EVAL-ONLY; the training reward path leaves it False so per-call cost is unchanged."""
    out = {"reward": R_FAIL, "rdkit_ok": False, "xtb_ok": False,
           "same_topo": False, "smi": "", "topo_post": ""}
    if na is None or na < 3:
        return out
    try:
        built = _build_molblock(atoms, na)
    except Exception:
        built = None
    if built is None:
        return out
    mol_block, inchi_pre, topo_pre, charge, smi = built
    out["rdkit_ok"] = True
    out["smi"] = smi
    out["reward"] = R_RDKIT

    # init heavy coords from mol_block (heavy atoms, in order)
    try:
        molh = Chem.MolFromMolBlock(mol_block, sanitize=False)
        molh = Chem.AddHs(molh, addCoords=True)
        conf = molh.GetConformer()
        init_heavy = []
        for a in molh.GetAtoms():
            if a.GetAtomicNum() != 1:
                p = conf.GetAtomPosition(a.GetIdx())
                init_heavy.append([p.x, p.y, p.z])
    except Exception:
        return out  # stays at R_RDKIT

    try:
        res = xtb_relax(idx, mol_block, init_heavy, workdir, xtb_bin, charge=charge)
    except Exception:
        res = {"ok": False}

    if not res.get("ok"):
        # cleanup partial files
        _cleanup(workdir, idx)
        return out  # R_RDKIT

    out["xtb_ok"] = True
    out["reward"] = R_XTB
    # always capture relaxation energy (kcal/mol) + per-atom strain (size-normalized)
    e_gain = res.get("e_gain")
    strain_pa = (abs(e_gain) / max(na, 1)) if (e_gain is not None) else None
    out["e_gain"] = e_gain
    out["strain_pa"] = strain_pa
    if collect_relax:
        out["rmsd_all"] = res.get("rmsd"); out["rmsd_heavy"] = res.get("rmsd_heavy")
        try:
            _gs = _relax_geom_stats(molh, init_heavy, res.get("opt_heavy_coords"))
            if _gs:
                out.update(_gs)
        except Exception:
            pass
    try:
        cc = consistency_check(inchi_pre, topo_pre,
                               res.get("opt_heavy_anums"), res.get("opt_heavy_coords"),
                               validate_3D)
        out["topo_post"] = cc.get("topo_post", "")  # post-relax SMILES (for post-relax cumulene)
        if cc.get("same_topo") or cc.get("same_inchi"):
            out["same_topo"] = True
            if XVR_ESTRAIN_TAU > 0 and strain_pa is not None:
                out["reward"] = R_XTB + (R_XVR - R_XTB) * math.exp(-strain_pa / XVR_ESTRAIN_TAU)
            else:
                out["reward"] = R_XVR
    except Exception:
        pass
    _cleanup(workdir, idx)
    return out


def _cleanup(workdir, idx):
    for pat in ["mol_%d.xyz", "mol_%d.out", "mol_%d.xtbtopo.mol",
                "mol_%d.xtbopt.xyz", "mol_%d.xtbopt.log",
                "mol_%d.charges", "mol_%d.wbo", "mol_%d.xtbrestart"]:
        p = os.path.join(workdir, pat % idx)
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    try:
        for fn in os.listdir(workdir):
            if fn.startswith(f".mol_{idx}"):
                os.remove(os.path.join(workdir, fn))
    except Exception:
        pass


def xvr_reward_batch(mols, xtb_bin, workdir, max_workers=16, collect_relax=False):
    """mols: list of (atoms, bonds, na). Returns list of result dicts (same order).
    collect_relax=True additionally returns per-molecule xTB relaxation stats (eval-only)."""
    if os.environ.get("XVR_PFREE") == "1":                     # perception-free (RDKit-free) path
        from reward_pfree import pfree_reward_batch
        return pfree_reward_batch(mols, xtb_bin, workdir, max_workers, collect_relax)
    os.makedirs(workdir, exist_ok=True)
    results = [None] * len(mols)

    def _task(i):
        atoms, bonds, na = mols[i]
        return i, xvr_reward_single(atoms, bonds, na, xtb_bin, workdir, i, collect_relax=collect_relax)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, r in ex.map(_task, range(len(mols))):
            results[i] = r
    if os.environ.get("XVR_ESTRAIN_MEASURE") == "1":
        try:
            import numpy as _np
            _sp = [r["strain_pa"] for r in results if r and r.get("strain_pa") is not None]
            if _sp:
                _a = _np.asarray(_sp, float)
                print(f"[ESTRAIN] n={len(_a)}/{len(results)} strain_pa(kcal/mol/atom) "
                      f"p10={_np.percentile(_a,10):.3f} p25={_np.percentile(_a,25):.3f} "
                      f"median={_np.median(_a):.3f} mean={_a.mean():.3f} "
                      f"p75={_np.percentile(_a,75):.3f} p90={_np.percentile(_a,90):.3f} "
                      f"max={_a.max():.3f}", flush=True)
        except Exception:
            pass
    return results
