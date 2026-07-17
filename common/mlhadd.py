"""MLHadd — learned H placement (replaces RDKit AddHs / H-prerelax).

Reuses the trained H-position head (train_hpos.HPosModel): frozen completer encoder + pos_head.
place_h(anums, coords, bonds, nH) tokenizes the heavy skeleton (ADT), runs the pos_head to get
per-atom H directions in the ADT tree-local frame (atom_table[i].frame), rotates them to global
(frame.T @ local), and places each H at heavy + BL[Z]*dir. Returns an all-atom molblock (H explicit,
SetNoImplicit so downstream AddHs is a no-op), or None on tokenization failure.

Env: MLHADD_CKPT (default ~/ADT/Hcompleter/ckpt_hpos/best.pt).
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/ADT/common"))
sys.path.insert(0, os.path.expanduser("~/ADT/Hcompleter"))
import numpy as np
import torch
from rdkit import Chem
import adt_tokenizer as tk
from relative_pointer import absolute_to_relative
from train_hpos import HPosModel, MAX_H
from train_completer import N_SLOTS

# standard X-H bond lengths (A) per heavy element
BL = {5: 1.19, 6: 1.09, 7: 1.01, 8: 0.96, 9: 0.92, 14: 1.48, 15: 1.42,
      16: 1.34, 17: 1.27, 33: 1.52, 35: 1.41, 53: 1.61}

_dev = "cuda" if torch.cuda.is_available() else "cpu"
_model = None


def load(ckpt=None):
    global _model
    ckpt = ckpt or os.environ.get("MLHADD_CKPT", os.path.expanduser("~/ADT/Hcompleter/ckpt_hpos/best.pt"))
    ck = torch.load(ckpt, weights_only=False, map_location=_dev)
    cfg = ck["cfg"]
    _model = HPosModel(cfg["d_model"], cfg["n_layers"]).to(_dev).eval()
    _model.load_state_dict(ck["model"])
    print("[mlhadd] H-position head loaded: %s (d=%d L=%d val_ang=%.2f)"
          % (ckpt, cfg["d_model"], cfg["n_layers"], ck.get("val_ang", -1)), flush=True)


def place_h(anums, coords, bonds, nH):
    """anums:(N,) heavy Z; coords:(N,3); bonds: list of (i,j) heavy edges; nH:(N,) per-atom H count.
    Returns an all-atom molblock string, or None."""
    if _model is None:
        load()
    n = len(anums)
    rw = Chem.RWMol()
    for z in anums:
        rw.AddAtom(Chem.Atom(int(z)))
    for a, b in bonds:
        rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
    conf = Chem.Conformer(n)
    for j in range(n):
        conf.SetAtomPosition(j, [float(x) for x in coords[j]])
    rw.AddConformer(conf, assignId=True)
    mol = rw.GetMol()
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
        arr = tk.tokens_to_array(tokd.tokens).astype(np.int64)
        off = absolute_to_relative(arr); L = len(off)
        v = torch.tensor(off, device=_dev).unsqueeze(0)
        sl = (torch.arange(L, device=_dev) % N_SLOTS).unsqueeze(0)
        ss = (torch.arange(L, device=_dev) // N_SLOTS) * N_SLOTS
        ac = v.gather(1, ss.unsqueeze(0)).clamp(min=0)
        pm = torch.zeros(1, L, dtype=torch.bool, device=_dev)
        nhc = torch.zeros(1, L, dtype=torch.long, device=_dev)
        steps = sorted(tokd.atom_table.keys()); j = 0
        heavy_step = []                                            # (z-slot pos, original_idx, frame)
        ok = True
        for t in range(L // N_SLOTS):
            if int(arr[t * N_SLOTS]) <= 3:
                if j >= len(steps):
                    ok = False; break
                s = steps[j]; oi = tokd.atom_table[s].original_idx
                nhc[0, t * N_SLOTS + 2] = min(int(nH[oi]), MAX_H)
                heavy_step.append((t * N_SLOTS + 2, oi, np.asarray(tokd.atom_table[s].frame, dtype=np.float64)))
                j += 1
        if not ok or len(heavy_step) != n:
            continue
        with torch.no_grad():
            dirs = _model(v, sl, ac, pm, nhc)[0].cpu().numpy()     # (L,3,3) local unit-ish dirs
        # build all-atom mol (heavy + learned H), single bonds, no implicit H
        rw2 = Chem.RWMol()
        for z in anums:
            a = Chem.Atom(int(z)); a.SetNoImplicit(True); rw2.AddAtom(a)
        for a, b in bonds:
            rw2.AddBond(int(a), int(b), Chem.BondType.SINGLE)
        hpos = []                                                 # (h_atom_idx, xyz)
        for (zp, oi, fr) in heavy_step:
            k = min(int(nH[oi]), MAX_H)
            bl = BL.get(int(anums[oi]), 1.09)
            for h in range(k):
                d = dirs[zp, h]; nn = np.linalg.norm(d)
                if nn < 1e-6:
                    continue
                g = fr.T @ (d / nn)                               # local -> global unit dir
                hp = pos[oi] + bl * g
                ha = Chem.Atom(1); ha.SetNoImplicit(True); hi = rw2.AddAtom(ha)
                rw2.AddBond(hi, int(oi), Chem.BondType.SINGLE)
                hpos.append((hi, hp))
        conf2 = Chem.Conformer(rw2.GetNumAtoms())
        for i in range(n):
            conf2.SetAtomPosition(i, [float(x) for x in coords[i]])
        for (hi, hp) in hpos:
            conf2.SetAtomPosition(hi, [float(hp[0]), float(hp[1]), float(hp[2])])
        rw2.AddConformer(conf2, assignId=True)
        try:
            return Chem.MolToMolBlock(rw2.GetMol(), kekulize=False)
        except Exception:
            return None
    return None
