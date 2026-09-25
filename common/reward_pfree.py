"""Perception-free XVR reward (RDKit valence/kekulize/AddHs excluded).

Drop-in for reward_xtb.xvr_reward_batch when XVR_PFREE=1:
  generate heavy (atoms,bonds,na)
    -> screen ② disconnection (ADT bond graph connected components==1)
    -> screen ① clash (check_collisions)
    -> completer n_H (COMPLETER_CKPT, MAIN-thread pre-pass=GPU)
    -> VSEPR H placement -> all-atom molblock (RDKit=graph container only)
    -> xtb_relax (reused: e_gain/opt_heavy proven) -> distance topology preserved
    -> reward: R_FAIL / R_XTB / estrain-shaped R_XVR (strain_pa=|e_gain|/n_heavy, same as current)

Completer runs in the MAIN thread (GPU); xTB in the thread pool (subprocess).
"""
import os, sys, math, subprocess, shutil, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Hcompleter"))
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from train_completer import HCompleter, N_SLOTS
import adt_tokenizer as tk
from relative_pointer import absolute_to_relative
from collision_check import check_collisions
from gen_eval_lib import xtb_relax, xtb_hrelax

ALLOWED_ATOMS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53}
R_FAIL, R_CLASH, R_XTB, R_XVR = 0.0, 0.3, 0.6, 1.0     # clashVR: 0.3 credit for clash-pass (structurally valid)
CLASHVR = os.environ.get("CLASHVR", "1") == "1"        # clashVR tier on/off (default on)
H_PLACER = os.environ.get("H_PLACER", "rdkit")         # rdkit (AddHs driven by completer n_H) / vsepr
CLASHVR_SWITCH = float(os.environ.get("CLASHVR_SWITCH", "0.90"))  # clash-pass EMA threshold -> switch to pure XVR
_clash_ema = None                                      # rolling clash-pass rate
_switched = False                                      # True after auto-switch to pure XVR
XVR_ESTRAIN_TAU = float(os.environ.get("XVR_ESTRAIN_TAU", "0") or 0)
H_PRERELAX = os.environ.get("H_PRERELAX") == "1"       # insert H-only xTB prerelax (freeze heavy) before full relax
BL = {6: 1.09, 7: 1.01, 8: 0.96, 16: 1.34, 15: 1.42, 9: 0.92, 17: 1.27, 35: 1.41, 53: 1.61}
COV = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39}
_PT = Chem.GetPeriodicTable()
MLNH_PARITY = os.environ.get("MLNH_PARITY", "1") == "1"   # corrected MLnH: fix completer n_H parity so neutral molecule is closed-shell (even electrons). default ON
STD_VAL = {5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 3, 16: 2, 17: 1, 33: 3, 35: 1, 53: 1}
_PARITY_STATS = {"n": 0, "odd": 0, "remove": 0, "add": 0, "fail": 0}
H_INTEGRITY = os.environ.get("H_INTEGRITY", "1") == "1"   # handle a molecule whose H-prerelaxed structure has a detached/stray H (fragmentation). default ON
H_INTEGRITY_MODE = os.environ.get("H_INTEGRITY_MODE", "correct")  # "correct" = strip the detached(excess) H + re-prerelax, recover if intact; "reject" = drop
_H_STRAY_A = float(os.environ.get("H_STRAY_A", "1.6"))    # an H farther than this from every heavy atom = detached
# Stopper: max number of returns to H prerelax (first try + up to MAX_RETRY retries). When reached, record as F1b and reject.
H_INTEGRITY_MAX_RETRY = int(os.environ.get("H_INTEGRITY_MAX_RETRY", "3"))
_INTEGRITY_STATS = {"n": 0, "correct": 0, "reject": 0, "n_full": 0, "correct_full": 0, "reject_full": 0,
                    "F1a_odd_parity": 0, "F1b_retry_exhausted": 0}   # F1a: odd number of removals needed / F1b: retry limit exceeded
# MLnH(+MLHplacer) performance: distribution of how many H (even) were rejected, and in how many rounds, before success. No strip and no parity fix = H count right on the first try.
_MLNH_PERF = {"ok_first": 0, "parity_only": 0, "strip_hist": {}, "attempt_hist": {}, "at_prerelax": 0, "at_full": 0,
              "odd_extra_removed": 0}   # xTB rejected an odd number (1H) -> number of times one more H was removed to make it even
# Breakdown of why molecules were screened out in _prep (at the cliff, screened is the largest failure group; tells which screen is active)
_SCREEN_STATS = {"atom": 0, "disconnect": 0, "clash": 0, "completer": 0, "placer": 0}
# --- realizable-XTP via clamp->unclamp (2026-07-10): when free relax FLIPS heavy topology, try to RESCUE it
# by constraining the generated heavy bonds (clamp) -> relax -> release (unclamp) -> relax; realizable if the
# unclamp result still preserves the generated topology (= a stable topo-preserving minimum EXISTS = honest XTP).
# --- Reward terms to reduce accumulated error on the generator side (2026-07-10, orthogonal to IKT) ---
# λ1: self-consistency penalty = |bonds0 △ _heavy_conn(generated coords)| / |bonds0|
#     "Do ADT's own 3D coordinates realize the graph ADT declared?" No xtb needed, so
#     **gradient reaches even xtb-nonconverged molecules (= largest accumulated error)**; that is the key (currently reward 0, zero information).
XVR_SELFMIS_LAM = float(os.environ.get("XVR_SELFMIS_LAM", "0") or 0)
# λ2: penalty on relaxation displacement RMSD (gen_heavy -> relaxed heavy) = direct measure of accumulated error (measured median ~0.9Å)
XVR_RMSD_LAM = float(os.environ.get("XVR_RMSD_LAM", "0") or 0)
# λ_c: **graded clash penalty**. At the cliff, 100% of screened are clashes, the largest failure group (~30% of all molecules).
#      Currently screened with reward 0 (zero gradient). Give a negative reward proportional to penetration depth Σ(thr - d)/na.
#      No xtb needed. A severe form of the same phenomenon as selfmis "spurious contacts", so it pushes accumulated error down most directly.
XVR_CLASH_LAM = float(os.environ.get("XVR_CLASH_LAM", "0") or 0)
XVR_CLAMP = os.environ.get("XVR_CLAMP") == "1"          # env-gate; default OFF (proven free-relax). ON for the step2b kt1 run.
XVR_CLAMP_FC = os.environ.get("XVR_CLAMP_FC", "0.5")    # $constrain force constant
# `distance: i,j,auto` fixes the "current distance". If accumulated error has stretched a bonds0 bond beyond the bonding threshold,
# auto **freezes the broken bond as broken**, and it does not form on release -> misjudged as F3_unclamp_flip.
# With XVR_CLAMP_IDEAL=1, only stretched bonds (d > 1.3*(cov_i+cov_j)) are **pulled in by explicitly setting the bond length cov_i+cov_j**.
XVR_CLAMP_IDEAL = os.environ.get("XVR_CLAMP_IDEAL") == "1"
# --- 2026-07-10: 3-part set to raise XTP (all default off = previous behavior) --------------------
# ① CLAMP_ONLY: drop free relax; judge every molecule by H-prerelax -> clamp(bonds0) -> unclamp.
#    Previously only molecules whose free relax converged were tried with clamp, so clamp never reached
#    relax_fail (full relax nonconverged = largest accumulated error), where clamp should help most.
XVR_CLAMP_ONLY = os.environ.get("XVR_CLAMP_ONLY") == "1"
# ② STRAIN_HPRE: unify the strain_pa reference point to the H-prerelax structure for all molecules.
#    Previously clamp-rescued molecules were measured by "gain from the post-clamp minimum", underestimating strain -> overestimated reward.
XVR_STRAIN_HPRE = os.environ.get("XVR_STRAIN_HPRE") == "1"
#    Note: the union (pass if free passes / else clamp) was rejected: scoring whose procedure varies per molecule is
#      unsound both as a measurement and as a reward. Every molecule always goes through the same procedure.
# ①'' CLAMP_FADE: instead of cutting restraints at once, lower the force constant stepwise to 0 (continuous deformation, homotopy).
#    Target distance d0 is set once at the start and kept fixed; only k is lowered: E_k = E_xtb + k*Σ(d-d0)^2, k -> 0.
#    The last stage is k=0 = plain free relax, so the pass certificate stays "the unrestrained xTB minimum keeps bonds0".
#    Aim: reduce F3_unclamp_flip caused by abrupt release (largest failure of B, 25-33/192).
#    Since the last stage is free relax, molecules passed by A(free) also pass in principle; union C is unnecessary = single procedure for all molecules.
XVR_CLAMP_FADE = os.environ.get("XVR_CLAMP_FADE", "")   # e.g. "1.0,0.3,0.1,0.03" (empty = previous single-stage clamp)
# ①''' CLAMP_LOOSE: converge the clamp (restrained) stage coarsely with --opt loose for speed. Final unclamp stays full --opt
#    = the strictness of the pass criterion (unrestrained minimum keeping bonds0) is unchanged. The clamp stage only has to "enter the bonds0 basin";
#    final refinement is done by unclamp, so quality is expected to barely drop even if coarse. For speed testing (default off).
XVR_CLAMP_LOOSE = os.environ.get("XVR_CLAMP_LOOSE") == "1"
# ③ FAIL_CREDIT: credit for molecules where "xtb converged but bonds0 is not realized" (role ① of the old R_XTB=0.6).
#    Success reward is R_XTB + (R_XVR-R_XTB)exp(-strain/tau), approaching 0.6 for large strain, so
#    a 0.6 failure credit kills the XTP gradient at the cliff (large strain). With 0, a 0.6 gap always stands.
#    Dense gradient on the failure side comes from the selfmis/clash penalties (no xtb needed, also reach screened).
XVR_FAIL_CREDIT = float(os.environ.get("XVR_FAIL_CREDIT", "0.6"))
# ④ RMSD_RHO: put geometric accumulated error (ADT generated heavy coords -> final relaxed) into the reward.
#    Energy is blind to soft modes (torsions) = large motions cost almost nothing. RMSD looks at that directly.
#    ★Use it as a shaping factor, not a subtraction: so the "0.6 gap between success and failure" created by FAIL_CREDIT=0
#      is not collapsed by subtracting from the success side. reward always stays within [R_XTB, R_XVR].
#      success = R_XTB + (R_XVR-R_XTB)*exp(-strain/tau)*exp(-rmsd/rho)
XVR_RMSD_RHO = float(os.environ.get("XVR_RMSD_RHO", "0") or 0)   # 0 = off. Recommended 0.3-0.5 (A)
PFREE_DUMP = os.environ.get("PFREE_DUMP")                # set -> append-pickle generated molecules (Z, coords, bonds, na)
HARTREE2KCAL = 627.5094740631


def _kabsch_rmsd(P, Q):
    """Heavy-atom RMSD after removing translation/rotation. xtb moves the center of mass/principal axes, so raw coordinate differences cannot be used."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    if P.shape != Q.shape or len(P) == 0:
        return None
    P = P - P.mean(0); Q = Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    P = P @ (V @ D @ Wt)
    return float(np.sqrt(((P - Q) ** 2).sum(1).mean()))


def _cu_energy_kcal(path):
    """Absolute total energy (Hartree) from the comment line of xtbopt.xyz -> kcal/mol. Raw GFN2 value without restraint bias."""
    try:
        with open(path) as f:
            comment = f.readlines()[1]
        m = re.search(r"energy:\s*(-?[\d.]+)", comment)
        return float(m.group(1)) * HARTREE2KCAL if m else None
    except Exception:
        return None
_CLAMP_STATS = {"tried": 0, "rescued": 0}


def _bank_struct():
    """Dynamic gate (read per call, no import-order dependency): freeze full HADD + relaxed
    all-atom structures into the reward dict so a molecule bank can store raw/HADD/relaxed.

    DEFAULT ON (2026-07-13): a record without the relaxed structure cannot yield a SMILES, and without a
    SMILES there is no diversity, no novelty and no rdkit_valid -- i.e. the record is unusable for the
    paper. It was opt-in before, and one forgotten env var cost a 80,000-molecule run. Set BANK_STRUCT=0
    to switch it off deliberately (RL speed)."""
    return os.environ.get("BANK_STRUCT", "1") != "0"


def _parse_xyz_all(path_or_text):
    """Full all-atom parse of an xyz FILE PATH *or* of the xyz TEXT itself -> (anums, coords) or (None, None).

    _clamp_unclamp returns the xyz CONTENT in "opt_xyz" (not a path); passing that to a path-only parser
    failed silently, which nulled struct_relaxed in every bank record under CLAMP_ONLY (= the current XTP
    definition) and disabled the post-relax H-integrity check. Accept both."""
    try:
        if isinstance(path_or_text, str) and "\n" in path_or_text:
            lines = path_or_text.splitlines()                     # already the file content
        else:
            with open(path_or_text) as f:
                lines = f.read().splitlines()
        n = int(lines[0].split()[0])
        zs, xs = [], []
        for ln in lines[2:2 + n]:
            p = ln.split()
            zs.append(_PT.GetAtomicNumber(p[0]))
            xs.append([float(p[1]), float(p[2]), float(p[3])])
        return zs, np.asarray(xs, np.float32)
    except Exception:
        return None, None

_dev = "cuda" if torch.cuda.is_available() else "cpu"
_cck = torch.load(os.environ["COMPLETER_CKPT"], weights_only=False)
_comp = HCompleter(_cck["cfg"]["d_model"], _cck["cfg"]["n_layers"]).to(_dev).eval()
_comp.load_state_dict(_cck["model"])
print("[reward_pfree] completer loaded: %s (d=%d L=%d) dev=%s tau=%s h_prerelax=%s"
      % (os.environ["COMPLETER_CKPT"], _cck["cfg"]["d_model"], _cck["cfg"]["n_layers"], _dev, XVR_ESTRAIN_TAU, H_PRERELAX), flush=True)
print("[reward_pfree] XTP mode: clamp_only=%s fade=%s loose=%s strain_hpre=%s fail_credit=%.2f rho=%.2f clamp=%s ideal=%s | lam selfmis=%.2f clash=%.2f rmsd=%.2f"
      % (XVR_CLAMP_ONLY, XVR_CLAMP_FADE or "off", XVR_CLAMP_LOOSE, XVR_STRAIN_HPRE, XVR_FAIL_CREDIT, XVR_RMSD_RHO, XVR_CLAMP, XVR_CLAMP_IDEAL,
         XVR_SELFMIS_LAM, XVR_CLASH_LAM, XVR_RMSD_LAM), flush=True)


def _bonded(z1, z2, d):
    return d < 1.3 * (COV.get(z1, 0.75) + COV.get(z2, 0.75))


def _ncomp(n, bonds):
    adj = [[] for _ in range(n)]
    for a, b in bonds:
        adj[a].append(b); adj[b].append(a)
    seen = [False] * n; c = 0
    for s in range(n):
        if seen[s]:
            continue
        c += 1; dq = deque([s]); seen[s] = True
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True; dq.append(v)
    return c


def _hdirs(nbr, nH, seed):
    rng = np.random.RandomState(seed)
    fixed = (np.array([np.asarray(d, float) / (np.linalg.norm(d) + 1e-9) for d in nbr])
             if nbr else np.zeros((0, 3)))
    H = rng.randn(nH, 3); H /= (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
    nf = len(fixed)
    for _ in range(400):
        pts = np.vstack([fixed, H]) if nf else H
        g = np.zeros((nH, 3))
        for i in range(nH):
            for k in range(len(pts)):
                if k == nf + i:
                    continue
                diff = H[i] - pts[k]; d2 = (diff * diff).sum() + 1e-6
                g[i] += diff / d2 ** 1.5
        H = H + 0.03 * g; H /= (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
    return H


def _heavy_conn(anums, coords, nh):
    s = set()
    for i in range(nh):
        for j in range(i + 1, nh):
            if _bonded(anums[i], anums[j], float(np.linalg.norm(coords[i] - coords[j]))):
                s.add((i, j))
    return s


def _h_intact(mb):
    """True iff every H in the molblock is within _H_STRAY_A of some heavy atom
    (no detached/stray H). H detachment = the completer over-counted H and xTB ejected
    the excess -> fragmentation. Permissive (True) if the block can't be parsed."""
    try:
        m = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=False)
        if m is None or m.GetNumConformers() == 0:
            return True
        P = m.GetConformer().GetPositions()
        Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
        heavy = np.where(Z > 1)[0]
        if len(heavy) == 0:
            return True
        hp = P[heavy]
        for i in np.where(Z == 1)[0]:
            if np.linalg.norm(hp - P[i], axis=1).min() > _H_STRAY_A:
                return False
        return True
    except Exception:
        return True


def _strip_detached_h(mb):
    """Remove detached (excess, rejected by xTB) H from the molblock -> (corrected_molblock, n_removed).
    If the number rejected is **odd**, the electron parity breaks and gives a radical, so **additionally remove the one most loosely bound H
    (the H farthest from its nearest heavy atom) to make it even** (2026-07-10, per user instruction).
    Returns (None,0) only if no H is left for the extra removal; the caller then rejects as F1a.
    Heavy atoms are unchanged, so `na` does not change (= bonds0 is preserved)."""
    try:
        m = Chem.MolFromMolBlock(mb, sanitize=False, removeHs=False)
        if m is None or m.GetNumConformers() == 0:
            return None, 0
        P = m.GetConformer().GetPositions()
        Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
        heavy = np.where(Z > 1)[0]
        if len(heavy) == 0:
            return None, 0
        hp = P[heavy]
        rem = [int(i) for i in np.where(Z == 1)[0]
               if np.linalg.norm(hp - P[i], axis=1).min() > _H_STRAY_A]
        if not rem:
            return mb, 0
        if len(rem) % 2 != 0:            # odd number detached -> remove one more to make it even (keep closed shell)
            _rs = set(rem)
            cand = [(float(np.linalg.norm(hp - P[i], axis=1).min()), int(i))
                    for i in np.where(Z == 1)[0] if int(i) not in _rs]
            if not cand:
                return None, 0           # no H available for extra removal -> F1a reject
            cand.sort(reverse=True)      # H farthest from nearest heavy = most loosely bound = most likely excess
            rem.append(cand[0][1])
            _MLNH_PERF["odd_extra_removed"] += 1
        ed = Chem.RWMol(m)
        for i in sorted(rem, reverse=True):
            ed.RemoveAtom(i)
        return Chem.MolToMolBlock(ed.GetMol()), len(rem)
    except Exception:
        return None, 0


def _intact_zc(Z, P):
    """(is_intact, detached_H_indices): an H is 'detached' if farther than _H_STRAY_A from every heavy atom.
    Operates directly on atomic numbers + coords (for the full-relax structure, parsed from xyz)."""
    Z = np.asarray(Z); P = np.asarray(P, dtype=np.float64)
    heavy = np.where(Z > 1)[0]
    if len(heavy) == 0:
        return True, []
    hp = P[heavy]
    det = [int(i) for i in np.where(Z == 1)[0]
           if np.linalg.norm(hp - P[i], axis=1).min() > _H_STRAY_A]
    return (len(det) == 0), det


def _zc_to_mb(Z, P):
    """Bondless molblock from atomic numbers + coords (xTB uses coords/elements only; bonds irrelevant)."""
    rw = Chem.RWMol()
    for z in Z:
        rw.AddAtom(Chem.Atom(int(z)))
    conf = Chem.Conformer(len(Z))
    for j in range(len(Z)):
        conf.SetAtomPosition(j, [float(P[j][0]), float(P[j][1]), float(P[j][2])])
    rw.AddConformer(conf, assignId=True)
    return Chem.MolToMolBlock(rw.GetMol())


def _strip_detached_zc(Z, P):
    """Remove detached (excess) H from the Z/coords structure -> (bondless molblock, n_removed).
    If odd, **additionally remove the one most loosely bound H to make it even** (keep closed shell).
    Returns (None,0) only if there is nothing to remove / no H available for the extra removal."""
    Z = np.asarray(Z); P = np.asarray(P, dtype=np.float64)
    _, det = _intact_zc(Z, P)
    if not det:
        return None, 0
    if len(det) % 2 != 0:                # odd number detached -> remove one more to make it even
        heavy = np.where(Z > 1)[0]
        if len(heavy) == 0:
            return None, 0
        hp = P[heavy]; _ds = set(det)
        cand = [(float(np.linalg.norm(hp - P[i], axis=1).min()), int(i))
                for i in np.where(Z == 1)[0] if int(i) not in _ds]
        if not cand:
            return None, 0               # no H available for extra removal -> F1a reject
        cand.sort(reverse=True)
        det = list(det) + [cand[0][1]]
        _MLNH_PERF["odd_extra_removed"] += 1
    keep = [i for i in range(len(Z)) if i not in set(det)]
    return _zc_to_mb(Z[keep], P[keep]), len(det)


def _completer_nH(anums, coords, bonds):
    rw = Chem.RWMol()
    for z in anums:
        rw.AddAtom(Chem.Atom(int(z)))
    for a, b in bonds:
        rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
    conf = Chem.Conformer(len(anums))
    for j in range(len(anums)):
        conf.SetAtomPosition(j, [float(x) for x in coords[j]])
    rw.AddConformer(conf, assignId=True); mol = rw.GetMol()
    try:
        Chem.FastFindRings(mol)
    except Exception:
        pass
    pos = np.asarray(coords, dtype=np.float64)
    for _ in range(4):
        try:
            tokd = tk.tokenize_molecule(mol, pos)
        except Exception:
            tokd = None
        if tokd is None:
            continue
        arr = tk.tokens_to_array(tokd.tokens).astype(np.int64); off = absolute_to_relative(arr); L = len(off)
        v = torch.tensor(off, device=_dev).unsqueeze(0)
        sl = (torch.arange(L, device=_dev) % N_SLOTS).unsqueeze(0)
        ss = (torch.arange(L, device=_dev) // N_SLOTS) * N_SLOTS
        ac = v.gather(1, ss.unsqueeze(0)).clamp(min=0)
        pm = torch.zeros(1, L, dtype=torch.bool, device=_dev)
        with torch.no_grad():
            pr = _comp(v, sl, ac, pm).argmax(-1)[0]
        steps = sorted(tokd.atom_table.keys()); out = [-1] * len(anums); j = 0
        for t in range(L // N_SLOTS):
            if int(arr[t * N_SLOTS]) <= 3:
                out[tokd.atom_table[steps[j]].original_idx] = int(pr[t * N_SLOTS + 2]); j += 1
        if all(x >= 0 for x in out):
            return _parity_correct_nH(anums, bonds, out) if MLNH_PARITY else out
    return None


def _parity_correct_nH(anums, bonds, nH):
    """Corrected MLnH parity fix. A neutral closed-shell molecule needs an even electron count
    (sum Z_all = sum Z_heavy + sum nH). If odd, the completer mis-counted H parity ->
    forced radical (RDKit rejects it; xTB still relaxes it). Adjust ONE atom's H by +-1
    to restore even parity, preferring REMOVAL (the completer over-counts in ~98% of errors:
    the excess H is ejected as stray H/H2 in relaxation)."""
    _PARITY_STATS["n"] += 1
    nH = list(nH)
    if (sum(int(z) for z in anums) + sum(int(h) for h in nH)) % 2 == 0:
        return nH                                              # already even -> unchanged
    _PARITY_STATS["odd"] += 1
    na = len(anums)
    deg = [0] * na
    for a, b in bonds:
        deg[int(a)] += 1; deg[int(b)] += 1
    deficit = [STD_VAL.get(int(anums[i]), 4) - deg[i] - nH[i] for i in range(na)]
    adj = [[] for _ in range(na)]
    for a, b in bonds:
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    # REMOVE candidate (nH_i -= 1): over-valent(deficit<0) first, then an H-bearing atom next
    # to an unsatisfied (deficit>=1) neighbor (removing frees a double bond), then any H-bearer.
    best = None; bestkey = None
    for i in range(na):
        if nH[i] < 1:
            continue
        if deficit[i] < 0:
            key = (0, deficit[i])
        elif any(deficit[j] >= 1 for j in adj[i]):
            key = (1, deficit[i])
        else:
            key = (2, deficit[i])
        if bestkey is None or key < bestkey:
            bestkey = key; best = i
    if best is not None:
        nH[best] -= 1; _PARITY_STATS["remove"] += 1
        return nH
    add = [(deficit[i], i) for i in range(na) if deficit[i] >= 1]   # else ADD to most-deficient
    if add:
        add.sort(reverse=True)
        nH[add[0][1]] += 1; _PARITY_STATS["add"] += 1
        return nH
    _PARITY_STATS["fail"] += 1
    return nH


def _prep(atoms, bonds, na):
    """MAIN-thread: screen + completer + placeH -> all-atom molblock. None if screened/failed."""
    anums = [atoms[k].atomic_num for k in range(na)]
    if any(a not in ALLOWED_ATOMS for a in anums):
        _SCREEN_STATS["atom"] += 1
        return None
    coords = np.array([list(atoms[k].pos) for k in range(na)], dtype=np.float64)
    bonds0 = set()
    for e1, e2 in bonds:
        a, b = int(e1) - 1, int(e2) - 1
        if a != b and 0 <= a < na and 0 <= b < na:
            bonds0.add((min(a, b), max(a, b)))
    bonds0 = list(bonds0)
    # === Plan B (2026-07-10): topology respects the bond graph `bonds0` declared by ADT ===
    # The old `ref = _heavy_conn(generated coords)` is distance-based, so 1-3 (geminal) atoms etc. brought close by accumulated error
    # are mistaken for "bonds" (spurious bonds). As a result (a) when relaxation resolves a spurious contact it is misjudged as "flip = failure", underestimating
    # XTP, and (b) clamp restrains that spurious bond and forces an impossible geometry, so rescue did not work.
    # With bonds0 as reference, H addition -> H relax -> full relax are all geometric operations that do not change topology (end to end), so
    # "topology changed" failures vanish; only "could not relax stably while keeping bonds0" failures remain.
    ref = set(bonds0)                                              # reference for XTP judgement and clamp restraints = ADT bond graph
    ref_dist = _heavy_conn(anums, coords, na)                      # distance topology of the generated coords (also the old-definition ref)
    # Self-consistency miss: symmetric difference between ADT's declared graph (bonds0) and the distance topology of the generated geometry.
    # >0 means "its own 3D coordinates do not realize the declared graph" = direct symptom of accumulated error
    # (stretched, broken bonds = bonds0\ref_dist / spurious contacts from compressed angles etc. = ref_dist\bonds0). No xtb needed.
    selfmis_miss = len(ref - ref_dist)    # in bonds0 but not bonded in the generated geometry = stretched, broken bond
    selfmis_spur = len(ref_dist - ref)    # proximity mistaken for a bond in the generated geometry = spurious contact (1-3 compressed angle etc.)
    selfmis = selfmis_miss + selfmis_spur
    if _ncomp(na, list(ref)) != 1:                                 # ② disconnection: is the ADT bond graph a single molecule
        _SCREEN_STATS["disconnect"] += 1
        return None
    _det = []
    if os.environ.get("XVR_NOCLASH") == "1":
        hc = False                                                 # XVR_NOCLASH: skip clash screen (ablation; eval leaves it unset -> honest)
    else:
        try:
            hc, _det = check_collisions([list(c) for c in coords], anums, set(bonds0))
        except Exception:
            hc = False; _det = []
    if hc:                                                          # ① clash (largest failure group at the cliff)
        _SCREEN_STATS["clash"] += 1
        # Return a "screened dict" instead of None -> reward stays 0, but penetration depth and selfmis are carried back.
        # If the batch side applies the graded penalty λ_c·Σ(thr-d)/na, a dense signal reaches the ~30% that had zero gradient.
        _depth = float(sum(max(0.0, t - d) for (_a, _b, d, t) in _det))
        return dict(screened="clash", clash_n=len(_det), clash_depth=_depth, clash_max=float(max([t - d for (_a, _b, d, t) in _det], default=0.0)),
                    selfmis=selfmis, selfmis_miss=selfmis_miss, selfmis_spur=selfmis_spur, nbond0=len(ref), na=na)
    nH = _completer_nH(anums, coords, bonds0)
    if nH is None:
        _SCREEN_STATS["completer"] += 1
        return None
    # --- H placement (M): MLHadd learned H positions (perception-free H GEOMETRY; RDKit only best-effort
    # for the diversity SMILES side-channel, not for the XVR judgment). Falls through to VSEPR if it fails.
    if H_PLACER == "mlhadd":
        try:
            import mlhadd
            mb = mlhadd.place_h(anums, coords, bonds0, [int(x) for x in nH])
        except Exception:
            mb = None
        if mb is not None:
            smi = ""
            try:                                                    # real SMILES (diversity only), crude fallback
                rw = Chem.RWMol()
                for i, z in enumerate(anums):
                    a = Chem.Atom(int(z)); a.SetNumExplicitHs(int(nH[i])); a.SetNoImplicit(True); rw.AddAtom(a)
                for a, b in bonds0:
                    rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
                m = rw.GetMol(); Chem.SanitizeMol(m); smi = Chem.MolToSmiles(m)
            except Exception:
                try:
                    m2 = Chem.RWMol()
                    for z in anums:
                        m2.AddAtom(Chem.Atom(int(z)))
                    for a, b in bonds0:
                        m2.AddBond(int(a), int(b), Chem.BondType.SINGLE)
                    smi = Chem.MolToSmiles(Chem.RemoveHs(m2.GetMol(), sanitize=False))
                except Exception:
                    smi = ""
            return dict(mb=mb, init_heavy=[list(coords[i]) for i in range(na)], anums=anums, na=na,
                        ref=ref, ref_dist=ref_dist, selfmis=selfmis, nbond0=len(ref),
                        selfmis_miss=selfmis_miss, selfmis_spur=selfmis_spur, placer="mlhadd", smi=smi, nH=[int(x) for x in nH], bonds0=bonds0)
        # mlhadd failed -> fall through to VSEPR
    # --- H placement (A): RDKit AddHs, driven by completer n_H (forces valence -> kekulize -> good geom)
    if H_PLACER == "rdkit":
        try:
            rw = Chem.RWMol()
            for i, z in enumerate(anums):
                a = Chem.Atom(int(z)); a.SetNumExplicitHs(int(nH[i])); a.SetNoImplicit(True)
                rw.AddAtom(a)
            for a, b in bonds0:
                rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
            conf = Chem.Conformer(na)
            for j in range(na):
                conf.SetAtomPosition(j, [float(x) for x in coords[j]])
            rw.AddConformer(conf, assignId=True)
            mol = rw.GetMol()
            Chem.SanitizeMol(mol)                                  # kekulize/hybridization from forced n_H
            try:
                smi = Chem.MolToSmiles(mol)                        # for diversity (tani/scaffold/qed/logP)
            except Exception:
                smi = ""
            molH = Chem.AddHs(mol, addCoords=True)                 # RDKit's good H placement
            c = molH.GetConformer()
            init_heavy = [[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z]
                          for i in range(na)]
            d = dict(mb=Chem.MolToMolBlock(molH), init_heavy=init_heavy, anums=anums, na=na,
                     ref=ref, ref_dist=ref_dist, selfmis=selfmis, nbond0=len(ref),
                        selfmis_miss=selfmis_miss, selfmis_spur=selfmis_spur, placer="rdkit", smi=smi, nH=[int(x) for x in nH], bonds0=bonds0)
            if _bank_struct():                                     # HADD (all-atom, pre-relax)
                d["hadd_anums"] = [a.GetAtomicNum() for a in molH.GetAtoms()]
                d["hadd_coords"] = np.array(
                    [[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z]
                     for i in range(molH.GetNumAtoms())], np.float32)
            return d
        except Exception:
            pass                                                   # -> VSEPR fallback
    # --- H placement (B): VSEPR repulsion fallback ---
    adj = [[] for _ in range(na)]
    for a, b in bonds0:
        adj[a].append(b); adj[b].append(a)
    rw = Chem.RWMol()
    for z in anums:
        rw.AddAtom(Chem.Atom(int(z)))
    for a, b in bonds0:
        rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
    cpos = [list(coords[i]) for i in range(na)]                     # heavy first, then H
    for i in range(na):
        k = int(nH[i])
        if k <= 0:
            continue
        nbr = [coords[j] - coords[i] for j in adj[i]]
        for hd in _hdirs(nbr, k, i):
            hi = rw.AddAtom(Chem.Atom(1)); rw.AddBond(hi, i, Chem.BondType.SINGLE)
            cpos.append(list(coords[i] + BL.get(int(anums[i]), 1.09) * hd))
    conf = Chem.Conformer(rw.GetNumAtoms())
    for j, p in enumerate(cpos):
        conf.SetAtomPosition(j, [float(x) for x in p])
    rw.AddConformer(conf, assignId=True)
    vmol = rw.GetMol()
    try:
        mb = Chem.MolToMolBlock(vmol)
    except Exception:
        _SCREEN_STATS["placer"] += 1                                  # neither mlhadd nor VSEPR could produce a molblock
        return None
    try:
        smi = Chem.MolToSmiles(Chem.RemoveHs(vmol, sanitize=False))   # crude (single-bond) smiles
    except Exception:
        smi = ""
    d = dict(mb=mb, init_heavy=cpos[:na], anums=anums, na=na, ref=ref, ref_dist=ref_dist, selfmis=selfmis, nbond0=len(ref),
                        selfmis_miss=selfmis_miss, selfmis_spur=selfmis_spur, placer="vsepr", smi=smi,
             nH=[int(x) for x in nH], bonds0=bonds0)
    if _bank_struct():                                                # HADD (all-atom, pre-relax)
        d["hadd_anums"] = [a.GetAtomicNum() for a in vmol.GetAtoms()]
        d["hadd_coords"] = np.asarray(cpos, np.float32)
    return d


def _cu_write_xyz(path, z, c):
    with open(path, "w") as f:
        f.write("%d\n\n" % len(z))
        for zz, xyz in zip(z, c):
            f.write("%s %.6f %.6f %.6f\n" % (_PT.GetElementSymbol(int(zz)), xyz[0], xyz[1], xyz[2]))


_CU_FLIP_CACHE = {}     # FLIP_KEEP=1: idx -> (anums, coords) of the post-unclamp geometry
_CU_CLAMP_CACHE = {}    # CLAMP_KEEP=1: idx -> (anums, coords) of the CLAMPED geometry, i.e. the one
                        # that DOES realise bonds0 (the rings are closed) but is metastable: releasing
                        # the restraints lets it fall back. It is the natural starting point for an
                        # alternating search -- physics closes the rings, the corrector then moves the
                        # global conformer, physics closes them again, and so on.


def _cu_read_xyz(path):
    with open(path) as f:
        L = f.read().split("\n")
    n = int(L[0].split()[0]); z = []; c = []
    for ln in L[2:2 + n]:
        pp = ln.split(); z.append(_PT.GetAtomicNumber(pp[0])); c.append([float(pp[1]), float(pp[2]), float(pp[3])])
    return np.array(z), np.array(c, float)


def _cu_xtbopt(wd, xtb_bin, extra, inp=None, timeout=180):
    args = [xtb_bin, "m.xyz", "--chrg", "0", "--namespace", "m", "--parallel", "1"] + extra
    if inp:
        args += ["--input", inp]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        r = subprocess.run(args, cwd=wd, capture_output=True, text=True, timeout=timeout, env=env)
    except Exception:
        return None
    conv = "GEOMETRY OPTIMIZATION CONVERGED" in (r.stdout or "")
    m = re.search(r"total energy gain.*?(-?[\d.]+)\s+kcal/mol", r.stdout or "")
    eg = float(m.group(1)) if m else None
    optf = os.path.join(wd, "m.xtbopt.xyz")
    if not os.path.exists(optf):
        return None
    try:
        z, c = _cu_read_xyz(optf)
    except Exception:
        return None
    return conv, z, c, eg


def _shape_reward(strain_pa, rmsd_heavy):
    """Reward for XTP-successful molecules: modulated by strain and RMSD between R_XTB (=0.6, success floor) and R_XVR (=1.0).
    Multiplicative, so it never falls below 0.6 -> keeps the "0.6 success/failure gap" created by FAIL_CREDIT=0."""
    shape = 1.0
    if XVR_ESTRAIN_TAU > 0:
        if strain_pa is None:
            return R_XVR
        shape *= math.exp(-float(strain_pa) / XVR_ESTRAIN_TAU)
    if XVR_RMSD_RHO > 0 and rmsd_heavy:
        shape *= math.exp(-float(rmsd_heavy) / XVR_RMSD_RHO)
    return R_XTB + (R_XVR - R_XTB) * shape


def _clamp_unclamp(mb_hprerelax, ref, anums, na, idx, workdir, xtb_bin, E_start=None, init_heavy=None):
    """Test "can it relax stably while keeping bonds0" (Plan B realizable judgement):
    relax with ref (= actual bonds of bonds0) distance-restrained (clamp) -> remove restraints and relax again (unclamp) ->
    realizable if the final geometry realizes bonds0.
      success -> xtb_relax-compatible dict (ok/opt_xyz/opt_heavy_coords/e_full/e_gain/rmsd_heavy/strain_pa/opt_heavy)
      failure -> reason string (F2_* = clamp stage / F3_* = unclamp stage)
    If E_start (kcal/mol, energy of the H-prerelax structure) is given, strain_pa is returned as
    (E_start - E_final)/na = common reference for all molecules. Otherwise, as before, only the unclamp gain."""
    m = Chem.MolFromMolBlock(mb_hprerelax, sanitize=False, removeHs=False)
    if m is None or m.GetNumConformers() == 0:
        return "F2_clamp_parse"
    cf = m.GetConformer()
    z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    c = np.array([[cf.GetAtomPosition(i).x, cf.GetAtomPosition(i).y, cf.GetAtomPosition(i).z]
                  for i in range(m.GetNumAtoms())], float)
    wd = os.path.join(workdir, "clamp_%d" % idx)
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd, exist_ok=True)
    try:
        _cu_write_xyz(os.path.join(wd, "m.xyz"), z, c)
        hc = c[:na]                                                      # heavy coords (heavy-first)
        # Target distance d0 is set once from the initial geometry and kept fixed (only k is lowered during fade)
        _npull = 0
        targets = []
        for (i, j) in sorted(ref):
            d = float(np.linalg.norm(hc[i] - hc[j]))
            if XVR_CLAMP_IDEAL:
                csum = COV.get(int(anums[i]), 0.75) + COV.get(int(anums[j]), 0.75)
                if d > 1.3 * csum:                                       # stretched, not bonded -> pull in to bond length
                    targets.append((i, j, csum)); _npull += 1
                    continue
            targets.append((i, j, d))                                    # heavy-first: xyz idx = heavy idx + 1
        if _npull:
            _CLAMP_STATS["ideal_pulled"] = _CLAMP_STATS.get("ideal_pulled", 0) + _npull

        def _write_inp(fc):
            with open(os.path.join(wd, "c.inp"), "w") as f:
                f.write("$constrain\n force constant=%s\n" % fc)
                for (i, j, d0) in targets:
                    f.write(" distance: %d,%d,%.4f\n" % (i + 1, j + 1, d0))
                f.write("$end\n")

        fcs = [s for s in XVR_CLAMP_FADE.split(",") if s.strip()] or [XVR_CLAMP_FC]
        cz = cc = None
        for _si, fc in enumerate(fcs):                                   # restrained stages: lower the force constant stepwise
            _write_inp(fc.strip())
            _last = (_si == len(fcs) - 1)
            _lvl = ["--opt"] if (_last and not XVR_CLAMP_LOOSE) else ["--opt", "loose"]   # intermediate stages coarse; with LOOSE the last restrained stage is also coarse (speed). unclamp below is full
            rc = _cu_xtbopt(wd, xtb_bin, ["--gfn", "2"] + _lvl, inp="c.inp")
            if rc is None:
                _CLAMP_STATS["fade_fail_stage_%d" % _si] = _CLAMP_STATS.get("fade_fail_stage_%d" % _si, 0) + 1
                return "F2_clamp_fail"                                    # cannot reach a geometry keeping bonds0
            _cv, cz, cc, _ = rc
            if not _cv and _last:
                _CLAMP_STATS["clamp_nonconv"] = _CLAMP_STATS.get("clamp_nonconv", 0) + 1  # recorded for information (continue)
            _cu_write_xyz(os.path.join(wd, "m.xyz"), cz, cc)
        if os.environ.get("CLAMP_KEEP") == "1" and cz is not None:
            _CU_CLAMP_CACHE[idx] = (cz.tolist(), cc.tolist())               # bonds0 IS realised here
        ru = _cu_xtbopt(wd, xtb_bin, ["--gfn", "2", "--opt"])                # k=0: unrestrained free relax (the pass certificate is here)
        if ru is None:
            return "F3_unclamp_fail"
        uconv, uz, uc, ueg = ru
        if not uconv:
            return "F3_unclamp_nonconv"
        heavy = uc[uz != 1]
        if len(heavy) != na:
            return "F3_atom_mismatch"
        if _heavy_conn(anums, heavy, na) != ref:
            if os.environ.get("FLIP_KEEP") == "1":                          # diagnostics: keep the FLIPPED geometry
                _CU_FLIP_CACHE[idx] = (uz.tolist(), uc.tolist())            # (all atoms incl. H) so the mechanism
            return "F3_unclamp_flip"                                        # bonds0 breaks on release = metastable, not realizable
        optf = os.path.join(wd, "m.xtbopt.xyz")
        E_final = _cu_energy_kcal(optf)                                     # raw GFN2 total energy after unclamp (no restraint bias)
        try:
            with open(optf) as f:
                opt_xyz_txt = f.read()                                      # for H-integrity check #2 (no H detached?)
        except Exception:
            opt_xyz_txt = ""
        if E_start is not None and E_final is not None:
            strain = abs(E_start - E_final) / na                            # unified reference: H-prerelax structure -> final minimum
        else:
            strain = abs(ueg) / na if ueg is not None else None             # old: only the gain over the unclamp segment (underestimate)
        rmsd_h = None
        if init_heavy is not None:
            try:
                rmsd_h = _kabsch_rmsd(init_heavy, heavy)   # accumulated error: generated heavy atoms -> final minimum
            except Exception:
                rmsd_h = None
        return {"ok": True, "realizable": True, "strain_pa": strain, "opt_heavy": heavy.tolist(),
                "opt_heavy_coords": heavy.tolist(), "opt_xyz": opt_xyz_txt,
                "e_full": E_final, "e_gain": (None if (E_start is None or E_final is None) else (E_final - E_start)),
                "rmsd_heavy": rmsd_h, "clamped": True}
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def _xtb_reward(p, xtb_bin, workdir, idx, collect_relax, use_clashvr):
    out = {"reward": R_FAIL, "rdkit_ok": False, "xtb_ok": False, "same_topo": False,
           "smi": "", "topo_post": "", "strain_pa": None, "clash_pass": False}
    if p is None:
        return out                                                 # disconnected/completer/placer -> 0.0 (no information)
    if p.get("screened"):                                          # screened by clash (reward stays 0; graded penalty on the batch side)
        for _k in ("screened", "clash_n", "clash_depth", "clash_max", "selfmis", "selfmis_miss", "selfmis_spur", "nbond0", "na"):
            out[_k] = p.get(_k)
        return out
    out["rdkit_ok"] = True; out["clash_pass"] = True               # clash-pass + completer (clashVR)
    out["placer"] = p.get("placer"); out["smi"] = p.get("smi", "")
    # Self-consistency miss (no xtb needed) is attached to every molecule passing the screen -> penalty gradient reaches even xtb-nonconverged ones
    out["selfmis"] = p.get("selfmis"); out["nbond0"] = p.get("nbond0"); out["na"] = p["na"]
    out["selfmis_miss"] = p.get("selfmis_miss"); out["selfmis_spur"] = p.get("selfmis_spur")
    if use_clashvr:
        out["reward"] = R_CLASH                                     # 0.3 clashVR credit
    hprerelax_ok = None; E_hpre = None; res = None; n_corr = 0
    mb_start = p["mb"]                                              # placement struct; re-stripped + restarted on fragmentation
    mb_use = mb_start
    _where = None                                                  # stage at which a detached H was detected ("prerelax" / "full")
    for _att in range(1 + H_INTEGRITY_MAX_RETRY):                  # stopper: return to H prerelax at most H_INTEGRITY_MAX_RETRY (default 3) times
        mb_use = mb_start
        if H_PRERELAX:                                             # H-only prerelax (freeze heavy) -> isolate heavy strain
            try:
                mb_h, E_hpre = xtb_hrelax(idx, mb_start, p["na"], workdir, xtb_bin, charge=0)
            except Exception:
                mb_h, E_hpre = None, None
            hprerelax_ok = mb_h is not None
            out["hprerelax_ok"] = hprerelax_ok; out["E_hprerelax"] = E_hpre
            if mb_h is None:
                return out                                         # H-prerelax REQUIRED (no fallback): drop here
            if H_INTEGRITY and not _h_intact(mb_h):                # CHECK #1 (end of H-prerelax): detached/stray H = excess H ejected by xTB
                _INTEGRITY_STATS["n"] += 1
                mb_s, nrem = _strip_detached_h(mb_h) if H_INTEGRITY_MODE == "correct" else (None, 0)
                if mb_s is not None:
                    mb_start = mb_s; n_corr += nrem; _INTEGRITY_STATS["correct"] += 1
                    _where = "prerelax"; _MLNH_PERF["at_prerelax"] += 1
                    continue                                       # remove excess H (even count) -> redo from H-prerelax
                out["h_intact"] = False; _INTEGRITY_STATS["reject"] += 1
                _INTEGRITY_STATS["F1a_odd_parity"] += 1; out["F"] = "F1a_odd_parity"   # odd removal needed = radical -> reject
                return out
            mb_use = mb_h                                          # heavy frozen; H at optimum -> full relax start point
        if XVR_CLAMP_ONLY:                                          # drop free relax; restrain bonds0 from the start -> release
            _rz = _clamp_unclamp(mb_use, p["ref"], p["anums"], p["na"], idx, workdir, xtb_bin,
                                 E_start=(E_hpre if XVR_STRAIN_HPRE else None), init_heavy=p["init_heavy"])
            _CLAMP_STATS["tried"] += 1
            if not isinstance(_rz, dict):                           # F2_* (clamp stage) / F3_* (unclamp stage)
                _CLAMP_STATS[_rz] = _CLAMP_STATS.get(_rz, 0) + 1
                out["F"] = _rz
                out["xtb_ok"] = _rz.startswith("F3")                # F3 = clamp geometry was obtained -> xtb can relax it
                out["reward"] = XVR_FAIL_CREDIT if out["xtb_ok"] else R_FAIL
                if os.environ.get("CLAMP_KEEP") == "1" and idx in _CU_CLAMP_CACHE:
                    _cz, _cc = _CU_CLAMP_CACHE.pop(idx)
                    _cz = np.asarray(_cz); _cc = np.asarray(_cc)
                    out["clamp_heavy"] = _cc[_cz != 1][:p["na"]].tolist()   # the geometry that DOES close
                    out["clamp_all_z"] = _cz.tolist()                       # the rings (metastable)
                    out["clamp_all_xyz"] = _cc.tolist()
                if os.environ.get("FLIP_KEEP") == "1" and idx in _CU_FLIP_CACHE:
                    _fz, _fc = _CU_FLIP_CACHE.pop(idx)              # diagnostics: the structure xTB actually
                    out["flip_anums"] = _fz; out["flip_coords"] = _fc   # relaxed this FAILED molecule to
                return out
            res = _rz
        else:
            try:
                res = xtb_relax(idx, mb_use, p["init_heavy"], workdir, xtb_bin, charge=0)
            except Exception:
                res = {"ok": False}
        if not res.get("ok"):
            return out                                             # clashVR: 0.3 / pure XVR: 0.0
        if H_INTEGRITY:                                            # CHECK #2 (end of full relax, REQUIRED): full relax moves everything -> H can still detach
            _ra, _rc = _parse_xyz_all(res.get("opt_xyz", ""))
            if _ra is not None and not _intact_zc(_ra, _rc)[0]:
                _INTEGRITY_STATS["n_full"] += 1
                mb_s, nrem = _strip_detached_zc(_ra, _rc) if H_INTEGRITY_MODE == "correct" else (None, 0)
                if mb_s is not None:
                    mb_start = mb_s; n_corr += nrem; _INTEGRITY_STATS["correct_full"] += 1
                    _where = "full"; _MLNH_PERF["at_full"] += 1
                    continue                                       # remove excess H (even count) -> redo from H-prerelax
                out["h_intact"] = False; _INTEGRITY_STATS["reject_full"] += 1
                _INTEGRITY_STATS["F1a_odd_parity"] += 1; out["F"] = "F1a_odd_parity"
                return out
        break                                                      # intact at BOTH H-prerelax and full relax
    else:
        _INTEGRITY_STATS["F1b_retry_exhausted"] += 1               # stopper triggered: not intact even after max retries
        out["F"] = "F1b_retry_exhausted"; out["n_strip"] = n_corr; out["n_attempt"] = H_INTEGRITY_MAX_RETRY
        return out
    out["h_intact"] = True
    out["n_strip"] = n_corr                                        # total number of rejected H (always even)
    out["n_attempt"] = _att                                        # number of returns to H-prerelax (0 = intact on first try)
    out["where"] = _where                                          # stage at which detachment was detected (prerelax / full / None)
    _MLNH_PERF["strip_hist"][n_corr] = _MLNH_PERF["strip_hist"].get(n_corr, 0) + 1
    _MLNH_PERF["attempt_hist"][_att] = _MLNH_PERF["attempt_hist"].get(_att, 0) + 1
    if n_corr == 0:
        _MLNH_PERF["ok_first"] += 1                                # success without strip = MLnH H count passed as is
    if n_corr:
        out["h_corrected"] = n_corr
    out["xtb_ok"] = True; out["reward"] = XVR_FAIL_CREDIT           # provisional value for "relaxed but bonds0 not realized"; overwritten below on success
    eg = res.get("e_gain"); na = p["na"]
    if XVR_STRAIN_HPRE and E_hpre is not None and res.get("e_full") is not None:
        strain_pa = abs(E_hpre - res["e_full"]) / na                # unified reference: H-prerelax structure -> final minimum (common to free/clamp)
    elif eg is not None:
        strain_pa = abs(eg) / na
    else:
        strain_pa = res.get("strain_pa")                            # clamp-only with unified reference off: gain over the unclamp segment (underestimate)
    out["strain_pa"] = strain_pa
    out["clamped"] = bool(res.get("clamped"))
    out["rmsd_heavy"] = res.get("rmsd_heavy")                      # direct measure of accumulated error: how far heavy atoms moved in relaxation
    if _bank_struct():                                             # freeze HADD + relaxed (all-atom) + funnel record
        out["e_gain"] = eg                                         # (hprerelax_ok/E_hprerelax set above, pre-gate)
        out["E_full"] = res.get("e_full")
        out["nH"] = p.get("nH"); out["bonds0"] = p.get("bonds0")
        out["rmsd_heavy"] = res.get("rmsd_heavy")
        _him = Chem.MolFromMolBlock(mb_use, sanitize=False)        # mb_use = H-prerelaxed struct = structure of E_hprerelax = starting point of full relax
        if _him is not None:
            _hc = _him.GetConformer()
            out["hadd_anums"] = [a.GetAtomicNum() for a in _him.GetAtoms()]
            out["hadd_coords"] = np.array([[_hc.GetAtomPosition(i).x, _hc.GetAtomPosition(i).y, _hc.GetAtomPosition(i).z]
                                           for i in range(_him.GetNumAtoms())], np.float32)
        ra, rc = _parse_xyz_all(res.get("opt_xyz", ""))            # full relaxed geometry (heavy+H)
        out["relaxed_anums"] = ra; out["relaxed_coords"] = rc
    if collect_relax:
        out["e_gain"] = eg; out["rmsd_all"] = res.get("rmsd"); out["rmsd_heavy"] = res.get("rmsd_heavy")
    oc = res.get("opt_heavy_coords")
    if oc and len(oc) == na:
        post = _heavy_conn(p["anums"], np.asarray(oc, float), na)   # distance topology after relaxation
        out["same_topo_old"] = (post == p.get("ref_dist"))          # XTP by the old definition (distance ref): for comparison/side-by-side only
        out["bonds0_subset"] = set(p["ref"]) <= post                # loose criterion (ii): are all bonds0 bonds present (extra contacts allowed)
        if post == p["ref"]:                                        # strict criterion (i): relaxed distance topology == bonds0
            out["same_topo"] = True
            if res.get("clamped"):                                  # bonds0 minimum reached via clamp (common to clamp-only / fallback)
                _CLAMP_STATS["rescued"] += 1
                out["rescued"] = True; out["realizable_heavy"] = oc   # IKT target (gen_heavy -> realizable_heavy)
            out["reward"] = _shape_reward(strain_pa, out.get("rmsd_heavy"))
        elif XVR_CLAMP:                                            # bonds0 not realized -> rescue by restraining bonds0 with clamp->unclamp
            _CLAMP_STATS["tried"] += 1
            rz = _clamp_unclamp(mb_use, p["ref"], p["anums"], na, idx, workdir, xtb_bin,
                                E_start=(E_hpre if XVR_STRAIN_HPRE else None), init_heavy=p["init_heavy"])
            if isinstance(rz, dict):                              # a stable minimum keeping bonds0 exists = realizable
                _CLAMP_STATS["rescued"] += 1
                out["same_topo"] = True; out["rescued"] = True
                out["bonds0_subset"] = True                       # final adopted geometry (unclamp) realizes bonds0, so (ii) also holds
                out["realizable_heavy"] = rz["opt_heavy"]         # IKT target (gen_heavy -> realizable_heavy), on-distribution
                out["strain_pa"] = rz.get("strain_pa"); out["rmsd_heavy"] = rz.get("rmsd_heavy")
                out["reward"] = _shape_reward(rz.get("strain_pa"), rz.get("rmsd_heavy"))
            else:
                out["F"] = rz or "F3_not_realizable"              # F2_clamp_* / F3_unclamp_* (_clamp_unclamp returns the reason)
                if os.environ.get("FLIP_KEEP") == "1" and idx in _CU_FLIP_CACHE:
                    _fz, _fc = _CU_FLIP_CACHE.pop(idx)                # the geometry xTB actually relaxed to
                    out["flip_anums"] = _fz; out["flip_coords"] = _fc  # -> lets us diff the graph and see WHAT changed
                _CLAMP_STATS[out["F"]] = _CLAMP_STATS.get(out["F"], 0) + 1
        else:
            out["F"] = "F3_not_realizable"                        # clamp disabled: bonds0 could not be realized
    return out


def pfree_reward_batch(mols, xtb_bin, workdir, max_workers=16, collect_relax=False):
    global _clash_ema, _switched
    if PFREE_DUMP:                                                  # dump generated molecules in raw form for A/B comparison of scoring schemes
        import pickle
        with open(PFREE_DUMP, "ab") as _f:
            for (atoms, bonds, na) in mols:
                if not na or na < 3:
                    continue
                pickle.dump(([int(atoms[k].atomic_num) for k in range(na)],
                             [list(map(float, atoms[k].pos)) for k in range(na)],
                             [(int(a), int(b)) for (a, b) in bonds], int(na)), _f)
    os.makedirs(workdir, exist_ok=True)
    results = [None] * len(mols)
    prep = [None] * len(mols)
    n_proc = 0
    for i, (atoms, bonds, na) in enumerate(mols):                   # MAIN-thread completer pre-pass
        if na and na >= 3:
            n_proc += 1
            try:
                prep[i] = _prep(atoms, bonds, na)
            except Exception:
                prep[i] = None
    # clashVR auto-switch: clash-pass EMA >= threshold -> drop 0.3 tier -> pure XVR
    n_valid = sum(1 for p in prep if p is not None)
    rate = n_valid / max(n_proc, 1)
    _clash_ema = rate if _clash_ema is None else 0.9 * _clash_ema + 0.1 * rate
    if CLASHVR and not _switched and _clash_ema >= CLASHVR_SWITCH:
        _switched = True
        print("[CLASHVR] clash-pass EMA %.3f >= %.2f -> SWITCH to pure XVR (drop 0.3 tier)"
              % (_clash_ema, CLASHVR_SWITCH), flush=True)
    use_clashvr = CLASHVR and not _switched

    def _task(i):
        return i, _xtb_reward(prep[i], xtb_bin, workdir, i, collect_relax, use_clashvr)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:         # parallel xTB
        for i, r in ex.map(_task, range(len(mols))):
            results[i] = r
    if os.environ.get("XVR_ESTRAIN_MEASURE") == "1":
        try:
            _sp = [r["strain_pa"] for r in results if r and r.get("strain_pa") is not None]
            if _sp:
                _a = np.asarray(_sp, float)
                print("[ESTRAIN] n=%d/%d strain_pa median=%.3f mean=%.3f p90=%.3f max=%.3f"
                      % (len(_a), len(results), np.median(_a), _a.mean(),
                         np.percentile(_a, 90), _a.max()), flush=True)
        except Exception:
            pass
    if MLNH_PARITY and _PARITY_STATS["odd"]:
        s = _PARITY_STATS
        print("[MLNH_PARITY] odd-parity corrected %d/%d calls (remove %d / add %d / unfixed %d)"
              % (s["odd"], s["n"], s["remove"], s["add"], s["fail"]), flush=True)
    if H_INTEGRITY and (_INTEGRITY_STATS["n"] or _INTEGRITY_STATS["n_full"]):
        s = _INTEGRITY_STATS
        print("[H_INTEGRITY] detached-H at H-prerelax: %d (corrected %d / rejected %d) | at full-relax: %d (corrected %d / rejected %d)"
              % (s["n"], s["correct"], s["reject"], s["n_full"], s["correct_full"], s["reject_full"]), flush=True)
    if XVR_CLAMP and _CLAMP_STATS["tried"]:
        s = _CLAMP_STATS
        print("[XVR_CLAMP] bonds0-clamp rescued (clamp->unclamp realizable): %d/%d tried | fail: %s"
              % (s["rescued"], s["tried"],
                 {k: v for k, v in s.items() if k.startswith(("F2", "F3", "clamp_"))}), flush=True)
    # --- accumulated-error penalty: self-consistency miss (no xtb, all molecules) + relaxation displacement RMSD (xtb-converged only) ---
    if XVR_SELFMIS_LAM > 0 or XVR_RMSD_LAM > 0 or XVR_CLASH_LAM > 0:
        for r in results:
            if not r or r.get("selfmis") is None:
                continue                                            # uninformative screen-outs (p=None) stay at 0
            if r.get("same_topo"):
                continue                     # success side is already graded by shaping (strain, rmsd); subtracting would collapse the 0.6 gap
            pen = 0.0
            if XVR_SELFMIS_LAM > 0:
                pen += XVR_SELFMIS_LAM * (r["selfmis"] / max(int(r.get("nbond0") or 1), 1))
            if XVR_RMSD_LAM > 0 and r.get("rmsd_heavy"):
                pen += XVR_RMSD_LAM * float(r["rmsd_heavy"])
            if XVR_CLASH_LAM > 0 and r.get("clash_depth"):          # graded clash penalty (dense gradient for the largest failure group at the cliff)
                pen += XVR_CLASH_LAM * (float(r["clash_depth"]) / max(int(r.get("na") or 1), 1))
            r["reward"] = r["reward"] - pen                          # negative values allowed (absorbed by baseline; strongest push on the worst molecules)
    # --- clash severity distribution (for choosing the λ_c scale; no xtb needed) ---
    _cd = [r for r in results if r and r.get("clash_depth") is not None]
    if _cd:
        _dep = np.asarray([float(r["clash_depth"]) for r in _cd])
        _dpa = np.asarray([float(r["clash_depth"]) / max(int(r.get("na") or 1), 1) for r in _cd])
        _cn = np.asarray([int(r["clash_n"]) for r in _cd])
        print("[CLASH] clash molecules %d/%d | clashing pairs mean=%.1f max=%d | Σpenetration depth(Å) mean=%.2f p90=%.2f max=%.2f | depth/atom mean=%.4f p90=%.4f"
              % (len(_cd), len(results), float(_cn.mean()), int(_cn.max()), float(_dep.mean()),
                 float(np.percentile(_dep, 90)), float(_dep.max()), float(_dpa.mean()), float(np.percentile(_dpa, 90))), flush=True)
    # --- self-consistency statistics (direct indicator of accumulated error, no xtb): does the generated geometry realize bonds0 ---
    _sm = [int(r["selfmis"]) for r in results if r and r.get("selfmis") is not None]
    if _sm:
        _a = np.asarray(_sm)
        _ms = np.asarray([int(r["selfmis_miss"]) for r in results if r and r.get("selfmis") is not None])
        _sp = np.asarray([int(r["selfmis_spur"]) for r in results if r and r.get("selfmis") is not None])
        _nb = np.asarray([max(int(r.get("nbond0") or 1), 1) for r in results if r and r.get("selfmis") is not None])
        print("[SELFCONSIST] generated geometry realizes bonds0: %d/%d (%.1f%%) | |Δ| mean=%.2f p90=%.0f max=%.0f | "
              "breakdown: stretched-broken bonds mean=%.2f (molecules with >0 %.1f%%) / spurious contacts mean=%.2f (molecules with >0 %.1f%%) | |Δ|/|bonds0| mean=%.4f p90=%.4f"
              % (int((_a == 0).sum()), len(_a), 100 * float((_a == 0).mean()), float(_a.mean()),
                 float(np.percentile(_a, 90)), int(_a.max()),
                 float(_ms.mean()), 100 * float((_ms > 0).mean()), float(_sp.mean()), 100 * float((_sp > 0).mean()),
                 float((_a / _nb).mean()), float(np.percentile(_a / _nb, 90))), flush=True)
    # --- at which stage failures occurred (to identify the main cause of the cliff) ---
    # screened      : dropped in _prep (connectivity / clash / completer nH / mlhadd H placement failure)
    # hprerelax_fail: H placed, but relaxing only H with heavy atoms frozen does not converge (skeleton too bad to accommodate H)
    # relax_fail    : passed H placement and H relax. **full relax moving everything does not converge** = mainly heavy-skeleton accumulated error
    # not_realizable: full relax converged but bonds0 not realized (F3)
    _stage = {"ok": 0, "not_realizable": 0, "relax_fail": 0, "hprerelax_fail": 0, "F1a": 0, "F1b": 0, "screened": 0}
    _stage_sm = {k: [] for k in _stage}                              # self-consistency miss |Δ| per stage
    for r in results:
        if not r:
            continue
        if r.get("same_topo"):
            _b = "ok"
        elif not r.get("rdkit_ok"):
            _b = "screened"
        elif r.get("F") == "F1a_odd_parity":
            _b = "F1a"
        elif r.get("F") == "F1b_retry_exhausted":
            _b = "F1b"
        elif r.get("hprerelax_ok") is False:
            _b = "hprerelax_fail"
        elif not r.get("xtb_ok"):
            _b = "relax_fail"
        else:
            _b = "not_realizable"
        _stage[_b] += 1
        if r.get("selfmis") is not None:
            _stage_sm[_b].append(int(r["selfmis"]))
    print("[FAILSTAGE] %s   note: relax_fail = H placement and H relax succeeded but full relax did not converge = accumulated error in heavy-atom geometry" % _stage, flush=True)
    # At the cliff, screened is the largest failure group. Its breakdown (cumulative): which screen is active
    if sum(_SCREEN_STATS.values()):
        print("[SCREEN] breakdown (cumulative): %s   note: these drop out in _prep, so they get no selfmis and reward 0 (zero gradient)" % _SCREEN_STATS, flush=True)
    # ★Decisive diagnostic of whether λ1 can target relax_fail: mean |Δ| per stage and the "fraction with |Δ|>0"
    print("[FAILSTAGE|Δ] self-consistency miss |Δ| per stage, mean (n, fraction |Δ|>0): %s" %
          {k: (round(float(np.mean(v)), 2), len(v), "%.0f%%" % (100 * float(np.mean(np.asarray(v) > 0))))
           for k, v in _stage_sm.items() if v}, flush=True)
    # --- Plan B: new and old XTP side by side + failure breakdown (F1a/F1b/F2/F3) ---
    _n = len(results)
    _new = sum(1 for r in results if r and r.get("same_topo"))
    _old = sum(1 for r in results if r and r.get("same_topo_old"))
    _loose = sum(1 for r in results if r and r.get("bonds0_subset"))
    _F = {}
    for r in results:
        if r and r.get("F"):
            _F[r["F"]] = _F.get(r["F"], 0) + 1
    print("[XTP] bonds0-strict(i)=%d/%d (%.1f%%) | bonds0-loose(ii)=%d (%.1f%%) | old definition (distance ref)=%d (%.1f%%) | failure breakdown %s"
          % (_new, _n, 100 * _new / max(_n, 1), _loose, 100 * _loose / max(_n, 1),
             _old, 100 * _old / max(_n, 1), _F), flush=True)
    # --- XTP by size: the population mean is pulled by the 40+ tail, so split by size band ---
    _BUCKETS = [(0, 24), (25, 29), (30, 34), (35, 39), (40, 999)]
    _sz_tot = {b: 0 for b in _BUCKETS}
    _sz_ok = {b: 0 for b in _BUCKETS}
    for r in results:
        if not r or r.get("na") is None:
            continue
        _na = int(r["na"])
        for (lo, hi) in _BUCKETS:
            if lo <= _na <= hi:
                _sz_tot[(lo, hi)] += 1
                if r.get("same_topo"):
                    _sz_ok[(lo, hi)] += 1
                break
    _parts = []
    for (lo, hi) in _BUCKETS:
        t = _sz_tot[(lo, hi)]
        if t == 0:
            continue
        _lab = "%d-%d" % (lo, hi) if hi < 999 else "%d+" % lo
        _parts.append("%s=%d/%d(%.0f%%)" % (_lab, _sz_ok[(lo, hi)], t, 100 * _sz_ok[(lo, hi)] / t))
    print("[XTP|size] " + " | ".join(_parts), flush=True)
    _ct={}; _co={}
    for r in results:
        if not r or r.get("na") is None:
            continue
        _n=int(r["na"]); _ct[_n]=_ct.get(_n,0)+1
        if r.get("same_topo"): _co[_n]=_co.get(_n,0)+1
    print("[XTP|count] "+" ".join("%d:%d/%d"%(k,_co.get(k,0),_ct[k]) for k in sorted(_ct)), flush=True)
    _sp = sorted(float(r["strain_pa"]) for r in results if r and r.get("same_topo") and r.get("strain_pa") is not None)
    if _sp:
        def _q(f):
            return _sp[min(len(_sp) - 1, int(f * len(_sp)))]
        _rw = [float(r["reward"]) for r in results if r and r.get("same_topo")]
        _rm = sorted(float(r["rmsd_heavy"]) for r in results
                     if r and r.get("same_topo") and r.get("rmsd_heavy"))
        _rs = ("| relaxation displacement RMSD(Kabsch,heavy) mean=%.3f p50=%.3f p90=%.3f max=%.3f Å"
               % (sum(_rm) / len(_rm), _rm[len(_rm) // 2], _rm[min(len(_rm) - 1, int(.9 * len(_rm)))], _rm[-1])) if _rm else ""
        print("[STRAIN] strain/heavy of XTP-successful molecules (kcal/mol/atom, reference=%s): mean=%.2f p50=%.2f p90=%.2f max=%.2f | reward mean=%.3f min=%.3f %s"
              % ("H-prerelax" if XVR_STRAIN_HPRE else "gain over relaxation segment",
                 sum(_sp) / len(_sp), _q(0.5), _q(0.9), _sp[-1],
                 sum(_rw) / max(len(_rw), 1), min(_rw) if _rw else 0.0, _rs), flush=True)
    # --- MLnH(+MLHplacer) performance: how many H (even) were rejected, and in how many rounds, before success ---
    m = _MLNH_PERF
    if m["ok_first"] or m["strip_hist"]:
        _tot = sum(m["strip_hist"].values()) or 1
        print("[MLNH_PERF] success without strip=%d/%d (%.1f%%) | strip distribution (nH:count)=%s | retry distribution (rounds:count)=%s | detection stage prerelax=%d full=%d | odd->extra removal=%d | F1a_odd_reject=%d F1b_exhausted=%d"
              % (m["ok_first"], _tot, 100 * m["ok_first"] / _tot, dict(sorted(m["strip_hist"].items())),
                 dict(sorted(m["attempt_hist"].items())), m["at_prerelax"], m["at_full"], m["odd_extra_removed"],
                 _INTEGRITY_STATS["F1a_odd_parity"], _INTEGRITY_STATS["F1b_retry_exhausted"]), flush=True)
    return results


def _completer_mol_stable(anums, coords, bonds0):
    """Perception-free mol_stable (HADD MVR): force completer n_H -> RDKit SanitizeMol succeeds.
    completer decides n_H (learned, not RDKit valence-from-geometry); RDKit only checks consistency +
    kekulizes. Recovers aromatics that RDKit-own kekulize drops. Returns (ok: bool, smi: str)."""
    nH = _completer_nH(anums, coords, bonds0)
    if nH is None:
        return False, ""
    try:
        rw = Chem.RWMol()
        for i, z in enumerate(anums):
            a = Chem.Atom(int(z)); a.SetNumExplicitHs(int(nH[i])); a.SetNoImplicit(True); rw.AddAtom(a)
        for a, b in bonds0:
            rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return True, Chem.MolToSmiles(m)
    except Exception:
        return False, ""


def _rdkit_mol_stable(anums, coords):
    """RDKit's OWN mol_stable (validate_3D = RDKit adds H / assigns valence from geometry). For the
    RDKit-H MVR variant (MVR_HMODE=rdkit): same clash+connected screen as the completer MVR, but
    RDKit (not the completer) decides H/valence -> isolates completer-vs-RDKit at fixed screen.
    Returns (ok, smi)."""
    try:
        from util_validation import validate_3D
        mol, smi, info = validate_3D(list(anums), [list(c) for c in coords])
        if mol is not None and info.get("mol_stable", False):
            return True, (smi or "")
    except Exception:
        pass
    return False, ""


def pfree_mvr_batch(mols, xtb_bin=None, workdir=None, max_workers=None, collect_relax=False):
    """Perception-free MVR proxy (cheap dense Stage-1 signal, NO xTB): reward 1.0 iff
    clash-free + connected(distance) + mol_stable; else 0.0. mol_stable via completer n_H ->
    SanitizeMol (default, "new HADD mol_stable") OR via RDKit validate_3D (MVR_HMODE=rdkit, ablation:
    same screen, RDKit decides H). Interface matches pfree_reward_batch."""
    hmode = os.environ.get("MVR_HMODE", "completer")
    noclash = os.environ.get("MVR_NOCLASH") == "1"     # ablation: drop the clash requirement from the MVR reward only
    results = []
    for atoms, bonds, na in mols:
        out = {"reward": 0.0, "smi": "", "same_topo": False, "xtb_ok": False,
               "clash_pass": False, "rdkit_ok": False, "strain_pa": None}
        if na and na >= 3:
            anums = [atoms[k].atomic_num for k in range(na)]
            if all(a in ALLOWED_ATOMS for a in anums):
                coords = np.array([list(atoms[k].pos) for k in range(na)], dtype=np.float64)
                bonds0 = set()
                for e1, e2 in bonds:
                    a, b = int(e1) - 1, int(e2) - 1
                    if a != b and 0 <= a < na and 0 <= b < na:
                        bonds0.add((min(a, b), max(a, b)))
                bonds0 = list(bonds0)
                if _ncomp(na, list(_heavy_conn(anums, coords, na))) == 1:      # connected (distance)
                    if noclash:
                        hc = False                                             # MVR_NOCLASH: skip clash check
                    else:
                        try:
                            hc, _ = check_collisions([list(c) for c in coords], anums, set(bonds0))
                        except Exception:
                            hc = False
                    if not hc:                                                 # clash-free (or skipped)
                        out["clash_pass"] = True
                        if hmode == "rdkit":
                            ok, smi = _rdkit_mol_stable(anums, coords)         # RDKit-H ablation
                        else:
                            ok, smi = _completer_mol_stable(anums, coords, bonds0)
                        if ok:
                            out.update(reward=1.0, smi=smi, same_topo=True, rdkit_ok=True)
        results.append(out)
    return results
