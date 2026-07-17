"""Perception-free Table-4 measurement (NOT a funnel; independent rates over N).

Over N generated molecules, report:
  noclash-connected/N : distance-connectivity == 1 component AND clash-free (geometry)
  mol_stable/N        : completer n_H forced -> RDKit SanitizeMol succeeds (chemistry/valence)
  XVR/N               : H placed (RDKit if mol_stable else VSEPR) -> xTB -> topology preserved (physics)
and (computed, optional for table) the RDKit-dropout / recovery measure:
  kekulized/N         : RDKit's OWN perception (validate_3D mol_stable) + kekulize succeeds
  kekulized/mol_stable: fraction of completer-mol_stable that RDKit-own also handles
                        (1 - this = aromatic-fallback RDKit dropped = the recovery)

mol_stable and XVR are INDEPENDENT (a VSEPR-fallback molecule can be XVR without mol_stable),
so this is a set of rates, not a nested funnel.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("AROMATIZE_RINGS", "1")
sys.path.insert(0, os.path.expanduser("~/ADT/common"))
sys.path.insert(0, os.path.expanduser("~/ADT/Drugs/vtakao202606231610"))
import numpy as np
import torch
import train as T
from adt_dataset import FrameSampler
from util_validation import validate_3D
from collision_check import check_collisions
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import reward_pfree                                    # loads completer (COMPLETER_CKPT)

dev = "cuda" if torch.cuda.is_available() else "cpu"
GEN = os.environ.get("GEN_CKPT", os.path.expanduser("~/ADT/Drugs/vtakao202606231610/ckpts/scratch_epoch240.pt"))
import hashlib
GEN_HASH = "sha256:" + hashlib.sha256(open(GEN, "rb").read()).hexdigest()   # ckpt 内容の hash (path非依存・cp不変)
ck = torch.load(GEN, map_location="cpu", weights_only=False)
model = T.build_model(ck.get("config", ck.get("cfg")))
model.load_state_dict(ck["model"]); model.to(dev).eval()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SCAF = sys.argv[2] if len(sys.argv) > 2 else "bootstrap3"
FRAME_DIR = os.environ.get("FRAME_DIR", os.path.expanduser("~/ADT/Drugs/data/geom/scaffolds"))  # portable: set FRAME_DIR on other hosts
fs = FrameSampler.load(os.path.join(FRAME_DIR, "frame_cache_%s.pt" % SCAF))
XTB = os.environ.get("XTB_BIN", os.path.expanduser("~/xtb/bin/xtb"))  # portable: set XTB_BIN on other hosts
WORKERS = int(os.environ.get("XTB_WORKERS", "24"))


def rdkit_kekulized(anums, coords):
    """RDKit's OWN perception: validate_3D mol_stable AND kekulizable."""
    try:
        mol, _, info = validate_3D(anums, coords)
    except Exception:
        return False
    if mol is None or not info.get("mol_stable", False):
        return False
    try:
        m = Chem.Mol(mol); Chem.Kekulize(m, clearAromaticFlags=True)
        return True
    except Exception:
        return False


def completer_mol_stable(anums, coords, bonds0, nH=None):
    """rdkit_stable: completer n_H forced -> RDKit SanitizeMol succeeds (valence valid, aromatics recovered).
    (NOTE: this uses RDKit -> it is rdkit_stable, NOT the perception-free EDM mol_stable.)"""
    if nH is None:
        nH = reward_pfree._completer_nH(anums, coords, bonds0)
    if nH is None:
        return False
    try:
        rw = Chem.RWMol()
        for i, z in enumerate(anums):
            a = Chem.Atom(int(z)); a.SetNumExplicitHs(int(nH[i])); a.SetNoImplicit(True); rw.AddAtom(a)
        for a, b in bonds0:
            rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
        Chem.SanitizeMol(rw.GetMol())
        return True
    except Exception:
        return False


import time as _time
from rollout_batched import rollout_batch_kv                       # batched KV-cache rollout (same as training)
GEN_BATCH = int(os.environ.get("GEN_BATCH", "128"))               # GPU batch size for generation
GEN_MAXSTEPS = int(os.environ.get("GEN_MAXSTEPS", "60"))          # match training rollout (not generate_one's 200)
GEN_SIZENMAX = int(os.environ.get("GEN_SIZENMAX", "56"))          # size ceiling (heavy-atom cap)
mols = []
_t_gen0 = _time.time()
while len(mols) < N:                                              # generate full batches on GPU until >= N
    for tokens, n_frame, at, bonds, na, done in rollout_batch_kv(
            model, fs, dev, GEN_BATCH, max_steps=GEN_MAXSTEPS, temperature=1.0, size_ceiling=GEN_SIZENMAX):
        if na and na >= 3:
            mols.append((at, bonds, na))
mols = mols[:N]
_T_GEN = _time.time() - _t_gen0
print("[gen] %d mols via rollout_batch_kv (B=%d max_steps=%d) in %.1fs = %.1f mol/s"
      % (len(mols), GEN_BATCH, GEN_MAXSTEPS, _T_GEN, len(mols) / max(_T_GEN, 0.01)), flush=True)

# --- per-molecule cheap rates (geometry / chemistry), independent of clash where noted ---
# SAVE_BANK=<dir> freezes every generated heavy structure (anums/coords/bonds0) + its cheap
# flags so downstream figures/tables (Fig 6 ΔE, per-size XVR, composition, diversity) re-derive
# from the SAME molecules WITHOUT regeneration (generation is the bottleneck) and stay internally
# consistent. Aligned index-for-index with `res` below. Default off = unchanged behavior.
SAVE_BANK = os.environ.get("SAVE_BANK", "")
n_ncc = n_ms = n_kek = 0
bank = []
for at, bonds, na in mols:
    anums = [at[k].atomic_num for k in range(na)]
    coords = np.array([list(at[k].pos) for k in range(na)], dtype=np.float64)
    bonds0 = set()
    for e1, e2 in bonds:
        a, b = int(e1) - 1, int(e2) - 1
        if a != b and 0 <= a < na and 0 <= b < na:
            bonds0.add((min(a, b), max(a, b)))
    bonds0 = list(bonds0)
    ref = reward_pfree._heavy_conn(anums, coords, na)          # distance bonds
    connected = reward_pfree._ncomp(na, list(ref)) == 1
    try:
        clash, _ = check_collisions([list(c) for c in coords], anums, set(bonds0))
    except Exception:
        clash = False
    nH_ml = reward_pfree._completer_nH(anums, coords, bonds0)   # ML nH (perception-free); reused for rdkit_stable
    mlnh_ok = nH_ml is not None
    ms = completer_mol_stable(anums, coords, bonds0, nH=nH_ml)  # rdkit_stable (RDKit SanitizeMol given completer nH)
    kek = rdkit_kekulized(anums, coords)
    if connected and not clash:
        n_ncc += 1
    if ms:
        n_ms += 1
    if kek:
        n_kek += 1
    if SAVE_BANK:
        bank.append({"anums": anums, "coords": coords.astype(np.float32), "bonds0": bonds0,
                     "na": na, "connected": bool(connected), "clash": bool(clash),
                     "mlnh_ok": bool(mlnh_ok), "nH": ([int(x) for x in nH_ml] if nH_ml is not None else None),
                     "rdkit_stable": bool(ms), "kekulized": bool(kek)})

# --- XVR via the reward pipeline (H placed -> xTB -> topology) ---
COLLECT = os.environ.get("BANK_RELAX", "0") == "1"            # richer xTB stats (rmsd/geom) if banking
if SAVE_BANK:
    os.environ["BANK_STRUCT"] = "1"                           # make reward_pfree freeze HADD + 緩和 structs
_t_xtb0 = _time.time()
res = reward_pfree.pfree_reward_batch(mols, XTB, "/tmp/measure_pfree_work", max_workers=WORKERS, collect_relax=COLLECT)
_T_XTB = _time.time() - _t_xtb0
n_xvr = sum(1 for r in res if r.get("same_topo"))
print("[xtb] %d mols pipeline (H-prerelax+full relax+XVR, workers=%d) in %.1fs = %.2f mol/s | [total gen+xtb] %.1fs"
      % (len(mols), WORKERS, _T_XTB, len(mols) / max(_T_XTB, 0.01), _T_GEN + _T_XTB), flush=True)

if SAVE_BANK:
    os.makedirs(SAVE_BANK, exist_ok=True)
    records = []                                             # molrecord_v2 (molrecord_design.md)
    for i, r in enumerate(res):                              # merge pipeline result into per-molecule record
        b = bank[i]
        placer = r.get("placer")
        hadd_a, hadd_c = r.get("hadd_anums"), r.get("hadd_coords")
        rel_a, rel_c = r.get("relaxed_anums"), r.get("relaxed_coords")
        Eh, Ef = r.get("E_hprerelax"), r.get("E_full")          # clean strain = E_hprerelax - E_full (>0)
        # H-prerelax 失敗(Eh=None)なら full relax は生構造から走り strain が汚染される -> null(汚染値は入れない)
        strain_dE = (Eh - Ef) if (Eh is not None and Ef is not None) else None
        records.append({
            "scaffold": SCAF, "gen_ckpt": GEN, "gen_ckpt_hash": GEN_HASH,          # 来歴
            "n_heavy": b["na"], "bonds0": b["bonds0"],                              # グラフ(LINK)
            "struct_init":    ({"anums": hadd_a, "coords": hadd_c} if hadd_a is not None else None),  # 生H(緩和なし)
            "struct_relaxed": ({"anums": rel_a, "coords": rel_c} if rel_a is not None else None),     # full後
            "mlnh_ok": b.get("mlnh_ok"), "nH": b.get("nH"),                         # funnel: MLnH
            "hplace_method": placer, "hplace_ok": placer is not None,               #         MLHplace(3値)
            "hprerelax_ok": r.get("hprerelax_ok"), "E_hprerelax": r.get("E_hprerelax"),  #     H-prerelax
            "h_intact": r.get("h_intact"),                                          # H整合ゲート: 離脱H無し(fragmentation reject)
            "full_ok": bool(r.get("xtb_ok")), "E_full": r.get("E_full"),            #         full relax
            "xvr": bool(r.get("same_topo")),                                        #         XVR
            "strain_dE": strain_dE,                                                 # 導出: ΔE=E_hpre-E_full(>0)
            "strain_pa": (strain_dE / b["na"] if strain_dE is not None else None),  # =ΔE/n_heavy (汚染時null)
            "rmsd_heavy": r.get("rmsd_heavy"),
            "connected": b["connected"], "clash": b["clash"],                       # 幾何ゲート(pre-funnel)
        })
    bank_path = os.path.join(SAVE_BANK, "pfree_bank_%s.pt" % SCAF)
    torch.save({"scaffold": SCAF, "gen_ckpt": GEN, "gen_ckpt_hash": GEN_HASH, "n": len(mols),
                "completer": os.environ.get("COMPLETER_CKPT", ""),
                "h_placer": os.environ.get("H_PLACER", "rdkit"),
                "estrain_tau": float(os.environ.get("XVR_ESTRAIN_TAU", "0") or 0),
                "schema": "molrecord_v2",
                "mols": records}, bank_path)
    sz = os.path.getsize(bank_path) / 1e6
    ni = sum(1 for e in records if e["struct_init"] is not None)
    nr = sum(1 for e in records if e["struct_relaxed"] is not None)
    nx = sum(1 for e in records if e["xvr"])
    print("  [SAVE_BANK] molrecord_v2: %d recs (init %d / relaxed %d / xvr %d) -> %s (%.2f MB, %.0f B/rec)"
          % (len(records), ni, nr, nx, bank_path, sz, sz * 1e6 / max(len(records), 1)))

Nt = len(mols)
print("========== PERCEPTION-FREE Table-4 (%s, N=%d) ==========" % (SCAF, Nt))
print("  noclash-connected/N = %d/%d = %.2f%%" % (n_ncc, Nt, 100 * n_ncc / max(Nt, 1)))
print("  mol_stable/N        = %d/%d = %.2f%%  (completer n_H forced, SanitizeMol)" % (n_ms, Nt, 100 * n_ms / max(Nt, 1)))
print("  XVR/N               = %d/%d = %.2f%%" % (n_xvr, Nt, 100 * n_xvr / max(Nt, 1)))
print("  --- RDKit dropout (計算のみ, 表には任意) ---")
print("  kekulized/N (RDKit own) = %d/%d = %.2f%%" % (n_kek, Nt, 100 * n_kek / max(Nt, 1)))
print("  kekulized/mol_stable    = %d/%d = %.2f%%   (残り %.1f%% = RDKitが落とした芳香 = 回収)"
      % (n_kek, n_ms, 100 * n_kek / max(n_ms, 1), 100 * (1 - n_kek / max(n_ms, 1))))
