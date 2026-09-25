#!/usr/bin/env python3
"""build_scaffold_frames.py — the per-scaffold frame caches used for scaffold-conditional generation.

A scaffold frame is the opening of a tokenized GEOM-Drugs molecule that lays down exactly the named
ring: ring_size ADD steps that place the ring atoms, then the LINK that closes the ring. Generation
starts from such a frame, so every molecule of that row is grown from the named ring.

For each GEOM molecule containing the scaffold (SMARTS match), the molecule is tokenized from a ring
atom as root, and the frame is kept only if
  (1) the first ring_size steps are ADD actions and step ring_size is a LINK, and
  (2) the first ring_size PLACED atoms are exactly the atoms of the matched scaffold ring.
Check (2) is what makes the frame the named ring. Without it (earlier caches) a root on a fused
or substituted ring could close a neighbouring ring first, and 0.6-14% of the frames of some
scaffolds were another ring.

usage: python3 build_scaffold_frames.py --pkl drugs_mols.pkl --outdir <dir> [--n_frames 1000] [--seed 0]
       (--pkl: list of (rdkit mol, positions, smiles), e.g. from Drugs/data/geom/load_drugs.py)
"""
import os, sys, pickle, argparse, random
from collections import Counter
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, HERE)
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from tokenizer import tokenize_molecule
from adt_tokenizer import tokens_to_array
from adt_model import ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD, LINK

N_SLOTS = 7
ADD_ACTIONS = {ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD}
SCAFFOLDS = {"benzene": ("c1ccccc1", 6), "pyridine": ("c1ccncc1", 6), "pyrimidine": ("c1cncnc1", 6),
             "pyrazine": ("c1cnccn1", 6), "cyclohexane": ("C1CCCCC1", 6),
             "furan": ("c1ccoc1", 5), "thiophene": ("c1ccsc1", 5)}


def ring_frames(mols, query, ring_size, n_frames):
    frame_len = N_SLOTS * (ring_size + 1)
    frames, stats = [], Counter()
    for item in mols:
        if len(frames) >= n_frames:
            break
        mol, pos = item[0], item[1]
        if mol is None:
            continue
        try:
            matches = mol.GetSubstructMatches(query)
        except Exception:
            continue
        if not matches:
            continue
        stats["molecules"] += 1
        pos = np.asarray(pos, dtype=np.float64)
        for ring in matches[:2]:                       # at most two rings per molecule
            ring_set = set(int(a) for a in ring)
            for root in ring:                          # any ring atom may start the frame
                stats["tries"] += 1
                try:
                    res = tokenize_molecule(mol, pos, root=int(root))
                except Exception:
                    continue
                if res is None:
                    continue
                arr = tokens_to_array(res.tokens)
                if len(arr) < frame_len:
                    continue
                frame = arr[:frame_len]
                if not all(int(frame[i * N_SLOTS]) in ADD_ACTIONS for i in range(ring_size)) \
                        or int(frame[ring_size * N_SLOTS]) != LINK:
                    stats["not_add_then_link"] += 1
                    continue
                placed = sorted(res.idx_map, key=lambda a: res.idx_map[a])[:ring_size]
                if set(int(a) for a in placed) != ring_set:
                    stats["other_ring_first"] += 1     # the check the earlier caches lacked
                    continue
                frames.append(np.array(frame, dtype=np.int32))
                stats["kept"] += 1
                break                                  # one frame per matched ring
            if len(frames) >= n_frames:
                break
    return frames, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_frames", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated scaffold names")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)   # the tokenizer's free order uses `random`
    mols = pickle.load(open(args.pkl, "rb"))
    print("loaded %d molecules" % len(mols))
    os.makedirs(args.outdir, exist_ok=True)
    names = args.only.split(",") if args.only else list(SCAFFOLDS)
    for name in names:
        smarts, rs = SCAFFOLDS[name]
        fr, st = ring_frames(mols, Chem.MolFromSmarts(smarts), rs, args.n_frames)
        out = os.path.join(args.outdir, "frame_cache_%s_real.pt" % name)
        torch.save({"frames": fr}, out)
        print("%-12s %5d frames  (%s) -> %s" % (name, len(fr), dict(st), out), flush=True)


if __name__ == "__main__":
    main()
