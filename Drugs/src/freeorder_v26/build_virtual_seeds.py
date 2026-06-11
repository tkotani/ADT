"""Build virtual frame caches for scaffold seeds (pyridine etc.).

Approach: canonical 3D embedding + small jitter; tokenize from each ring atom.
Keep first N_SLOTS * ring_steps tokens (FRAME_LEN=49 for 6-rings, 42 for 5-rings).
"""
import os, sys, argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "ADT/Drugs/freeorder_v21"))
sys.path.insert(1, os.path.join(os.path.expanduser("~"), "ADT/common"))

from fo_tokenizer_v21 import tokenize_molecule
from adt_tokenizer import tokens_to_array
from adt_model import ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD, LINK

from rdkit import Chem
from rdkit.Chem import AllChem

N_SLOTS = 7
ADD_ACTIONS = {ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD}

SEEDS = {
    "pyridine":    ("c1ccncc1",   6),
    "pyrimidine":  ("c1cncnc1",   6),
    "pyrazine":    ("c1cnccn1",   6),
    "cyclohexane": ("C1CCCCC1",   6),
    "furan":       ("c1ccoc1",    5),
    "thiophene":   ("c1ccsc1",    5),
}


def build_virtual_frames(smi, ring_size, n_frames=200, jitter=0.03, seed=42):
    rng = np.random.default_rng(seed)
    frame_steps = ring_size + 1          # ring_size ADD + 1 LINK
    frame_len = N_SLOTS * frame_steps

    mol = Chem.MolFromSmiles(smi)
    mol_h = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol_h, randomSeed=seed) != 0:
        raise RuntimeError(f"Embed failed for {smi}")
    try:
        AllChem.MMFFOptimizeMolecule(mol_h)
    except Exception:
        pass
    mol = Chem.RemoveHs(mol_h)
    pos = mol.GetConformer().GetPositions().astype(np.float64)

    ri = mol.GetRingInfo()
    ring_atoms = []
    for r in ri.AtomRings():
        if len(r) == ring_size:
            ring_atoms.extend(r)
    ring_atoms = list(set(ring_atoms))
    if not ring_atoms:
        raise RuntimeError(f"No {ring_size}-ring found in {smi}")

    frames = []
    attempts = 0
    while len(frames) < n_frames and attempts < n_frames * 20:
        attempts += 1
        pos_j = pos + rng.normal(0.0, jitter, pos.shape)
        root = int(rng.choice(ring_atoms))
        try:
            result = tokenize_molecule(mol, pos_j, root=root)
        except Exception:
            continue
        if result is None:
            continue
        arr = tokens_to_array(result.tokens)
        if len(arr) < frame_len:
            continue
        frame = arr[:frame_len]
        ok = True
        for i in range(ring_size):
            if int(frame[i * N_SLOTS]) not in ADD_ACTIONS:
                ok = False; break
        if not ok:
            continue
        link_base = ring_size * N_SLOTS
        if int(frame[link_base]) != LINK:
            continue
        link_from = int(frame[link_base + 1])
        link_to = int(frame[link_base + 2])
        if not (1 <= link_from <= ring_size and 1 <= link_to <= ring_size):
            continue
        frames.append(np.array(frame, dtype=np.int32))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "freeorder_v26", "scaffolds"))
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for name, (smi, rs) in SEEDS.items():
        print(f"[{name}] smi={smi} ring={rs}")
        try:
            frames = build_virtual_frames(smi, rs, n_frames=args.n)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        out = os.path.join(args.outdir, f"frame_cache_{name}.pt")
        torch.save({"frames": frames}, out)
        print(f"  saved {len(frames)} frames → {out}")


if __name__ == "__main__":
    main()
