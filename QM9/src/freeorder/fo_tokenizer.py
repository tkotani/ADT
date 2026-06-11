"""
Free-Order Tokenizer for 3D Molecule Generation

Based on ADT v4 (tree-local frame), but with free ordering:
  - Bootstrap triple (atomL, atomC, atomR) same as DFS version
  - After bootstrap, atoms are added in RANDOM order (not DFS)
  - Each ADD step specifies 'from' (parent) explicitly
  - Tree-local frame construction: same as DFS version
  - Random ordering provides massive data augmentation

Actions: ADD_INIT(0), ADD_CHAIN(1), ADD_ANGLE(2), ADD(3), LINK(4), END(5)
Token:   [action, from, atom/to, r, hp0/θc, hp1/θf, hp2]
"""

import numpy as np
import healpy as hp
import networkx as nx
from rdkit import Chem
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict, deque
import random

# ============================================================
# Import shared components from parent adt_tokenizer
# ============================================================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# [common] ensure ADT/common is on path
import os as _os_adt, sys as _sys_adt
_common_dir = _os_adt.path.abspath(_os_adt.path.join(_os_adt.path.dirname(__file__), "../../../common"))
if _common_dir not in _sys_adt.path:
    _sys_adt.path.insert(0, _common_dir)
from adt_tokenizer import (
    # Constants
    ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD, LINK, END, N_ACTIONS,
    ACTION_NAMES, NULL_POINTER, SLOT_NAMES,
    # Distance
    R_MIN, R_MAX, R_BINS, r_to_bin, bin_to_r,
    # Angle
    THETA_BINS, theta_to_bins, bins_to_theta,
    # HEALPix
    HEALPIX_NSIDE, HEALPIX_NPIX, vec_to_healpix, healpix_to_vec,
    # Frame
    is_parallel, arbitrary_perpendicular, build_frame,
    compute_tree_frame, PARALLEL_THRESHOLD, COLLINEAR_COS,
    # Encoding/decoding
    encode_direction, decode_direction,
    # Atom table
    AtomEntry,
    # Bootstrap
    find_bootstrap_triple,
    # Reconstruction (reuse DFS version — same token format)
    reconstruct_from_tokens,
    _ancestor_positions_orig, _ancestor_positions_recon,
    # Token conversion
    tokens_to_array, array_to_tokens,
)


# ============================================================
# Free-order decomposition (replaces DFS bootstrap_decompose)
# ============================================================

def freeorder_decompose(mol, chain, atomR):
    """
    Free-order decomposition: after bootstrap triple, add remaining
    atoms in random order (BFS-like with random frontier selection).

    Returns atom_order, parent_map, link_edges.
    """
    adj = defaultdict(set)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].add(j)
        adj[j].add(i)

    # Bootstrap atoms are placed first (same as DFS)
    visited = set(chain) | {atomR}
    atom_order = list(chain) + [atomR]
    parent_map = {chain[0]: -1}
    for k in range(1, len(chain)):
        parent_map[chain[k]] = chain[k - 1]
    parent_map[atomR] = chain[-1]

    # Build frontier: unvisited atoms adjacent to visited atoms
    frontier = []  # list of (atom, parent) pairs
    def refresh_frontier():
        frontier.clear()
        for v in visited:
            for nb in adj[v]:
                if nb not in visited:
                    frontier.append((nb, v))

    refresh_frontier()

    # Randomly pick from frontier until all atoms are placed
    while frontier:
        random.shuffle(frontier)
        # Pick first valid entry (atom might have been visited by now)
        placed_any = False
        new_frontier = []
        for atom, parent in frontier:
            if atom in visited:
                continue
            visited.add(atom)
            atom_order.append(atom)
            parent_map[atom] = parent
            placed_any = True
            # Add new neighbors to frontier
            for nb in adj[atom]:
                if nb not in visited:
                    new_frontier.append((nb, atom))
            break

        if not placed_any:
            break

        # Rebuild frontier: keep unvisited entries + new ones
        remaining = [(a, p) for a, p in frontier if a not in visited]
        # Also add alternative parents for unvisited atoms from newly visited
        for v in visited:
            for nb in adj[v]:
                if nb not in visited:
                    # Check if already in frontier with some parent
                    existing = {a for a, _ in remaining}
                    if nb not in existing:
                        remaining.append((nb, v))
                    else:
                        # Randomly decide whether to replace parent
                        if random.random() < 0.5:
                            remaining = [(a, p) if a != nb else (a, v)
                                        for a, p in remaining]
        frontier = remaining + [(a, p) for a, p in new_frontier
                                if a not in visited and a not in {x for x, _ in remaining}]

    # Link edges (non-tree)
    tree_set = set()
    for c, p in parent_map.items():
        if p >= 0:
            tree_set.add((min(c, p), max(c, p)))
    all_edges = set((min(i, j), max(i, j)) for i in adj for j in adj[i] if i < j)
    link_edges = all_edges - tree_set

    return atom_order, parent_map, list(link_edges)


# ============================================================
# Tokenization (same structure as DFS, different ordering)
# ============================================================

@dataclass
class TokenizedMolecule:
    tokens: list = field(default_factory=list)
    atom_table: dict = field(default_factory=dict)
    idx_map: dict = field(default_factory=dict)
    n_atoms: int = 0
    n_links: int = 0
    n_tokens: int = 0
    chain_length: int = 0
    theta_bins: tuple = None
    bootstrap_triple: tuple = None


def tokenize_molecule(mol, positions, root=None, seed=None):
    """
    Free-order SE(3)-invariant tokenization.
    Same token format as DFS version, but with random atom ordering.
    """
    positions = positions.copy()
    n_atoms = mol.GetNumAtoms()

    if n_atoms < 3:
        return None

    # --- Find bootstrap triple ---
    start = root if root is not None else (
        random.Random(seed).randint(0, n_atoms - 1) if seed is not None
        else random.randint(0, n_atoms - 1)
    )
    atomL, chain, atomR = find_bootstrap_triple(mol, positions, start)
    atomC = chain[-1]

    if atomR is None:
        return None

    n_chain = len(chain)
    n_bootstrap = n_chain + 1
    bootstrap_set = set(chain) | {atomR}

    # --- Free-order decomposition ---
    atom_order, parent_map, link_edges = freeorder_decompose(mol, chain, atomR)
    assert atom_order[:n_chain] == chain
    assert atom_order[n_chain] == atomR

    # --- Bootstrap geometry ---
    v_LC = positions[atomL] - positions[atomC]
    v_RC = positions[atomR] - positions[atomC]
    cos_theta = np.dot(v_LC / np.linalg.norm(v_LC), v_RC / np.linalg.norm(v_RC))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    theta_c, theta_f = theta_to_bins(theta)

    # --- Bootstrap frame ---
    v_chain = positions[atomC] - positions[atomL]
    v_depart = positions[atomR] - positions[atomC]
    bootstrap_frame = build_frame(v_depart, v_chain)

    # --- Build token sequence ---
    placed = set()
    pending_links = list(link_edges)
    atom_table = {}
    idx_map = {}

    tokens = []
    n_links = 0

    for step, atom_idx in enumerate(atom_order):
        from_orig = parent_map[atom_idx]
        from_id = idx_map.get(from_orig, NULL_POINTER)
        pos_j = positions[atom_idx]
        atom_type = mol.GetAtomWithIdx(atom_idx).GetAtomicNum()
        is_bs = atom_idx in bootstrap_set

        if step == 0:
            tokens.extend([
                ('action', ADD_INIT),
                ('from', NULL_POINTER),
                ('atom', atom_type),
                ('r', 0),
                ('hp0', 0),
                ('hp1', 0),
                ('hp2', 0),
            ])
            frame = bootstrap_frame.copy()

        elif step < n_chain:
            r = np.linalg.norm(pos_j - positions[from_orig])
            tokens.extend([
                ('action', ADD_CHAIN),
                ('from', from_id),
                ('atom', atom_type),
                ('r', r_to_bin(r)),
                ('hp0', 0),
                ('hp1', 0),
                ('hp2', 0),
            ])
            frame = bootstrap_frame.copy()

        elif step == n_chain:
            r = np.linalg.norm(pos_j - positions[from_orig])
            tokens.extend([
                ('action', ADD_ANGLE),
                ('from', from_id),
                ('atom', atom_type),
                ('r', r_to_bin(r)),
                ('hp0', theta_c),
                ('hp1', theta_f),
                ('hp2', 0),
            ])
            frame = bootstrap_frame.copy()

        else:
            from_frame = atom_table[from_id - 1].frame
            d_vec = pos_j - positions[from_orig]
            r_bin, hp0, hp1, hp2 = encode_direction(d_vec, from_frame)

            anc_pos = _ancestor_positions_orig(atom_idx, parent_map, positions)
            frame = compute_tree_frame(pos_j, anc_pos, bootstrap_frame)

            tokens.extend([
                ('action', ADD),
                ('from', from_id),
                ('atom', atom_type),
                ('r', r_bin),
                ('hp0', hp0),
                ('hp1', hp1),
                ('hp2', hp2),
            ])

        atom_table[step] = AtomEntry(
            pos=pos_j.copy(),
            parent_id=from_id,
            frame=frame.copy(),
            v1=(pos_j - positions[from_orig]).copy() if from_orig >= 0 else np.zeros(3),
            v2=np.zeros(3),
            atomic_num=atom_type,
            original_idx=atom_idx,
            is_bootstrap=is_bs,
        )
        idx_map[atom_idx] = step + 1
        placed.add(atom_idx)

        # Emit LINK tokens for newly complete edges
        newly_ready = []
        still_pending = []
        for edge in pending_links:
            i, j = edge
            if i in placed and j in placed:
                newly_ready.append(edge)
            else:
                still_pending.append(edge)
        pending_links = still_pending

        newly_ready.sort(key=lambda e: (idx_map[e[0]] + idx_map[e[1]], idx_map[e[0]]))

        for (i, j) in newly_ready:
            lid_from = idx_map[i]
            lid_to = idx_map[j]
            d_vec_link = positions[j] - positions[i]
            link_frame = atom_table[lid_from - 1].frame
            r_bin_l, hp0_l, hp1_l, hp2_l = encode_direction(d_vec_link, link_frame)

            tokens.extend([
                ('action', LINK),
                ('from', lid_from),
                ('to', lid_to),
                ('r', r_bin_l),
                ('hp0', hp0_l),
                ('hp1', hp1_l),
                ('hp2', hp2_l),
            ])
            n_links += 1

    tokens.append(('action', END))

    return TokenizedMolecule(
        tokens=tokens,
        atom_table=atom_table,
        idx_map=idx_map,
        n_atoms=len(atom_order),
        n_links=n_links,
        n_tokens=len(tokens),
        chain_length=n_chain - 1,
        theta_bins=(theta_c, theta_f),
        bootstrap_triple=(atomL, atomC, atomR),
    )


# ============================================================
# Roundtrip test
# ============================================================

def test_roundtrip(mol, positions, verbose=False):
    """Test that tokenize -> reconstruct recovers positions."""
    result = tokenize_molecule(mol, positions)
    if result is None:
        return None

    arr = tokens_to_array(result.tokens)
    tokens_back = array_to_tokens(arr)
    atoms, links = reconstruct_from_tokens(tokens_back)

    n = mol.GetNumAtoms()
    if len(atoms) != n:
        if verbose:
            print(f"  Atom count mismatch: {len(atoms)} vs {n}")
        return None

    orig_coords = positions.copy()
    recon_coords = np.array([a.pos for a in atoms])

    orig_coords -= orig_coords.mean(axis=0)
    recon_coords -= recon_coords.mean(axis=0)

    diffs = np.linalg.norm(orig_coords - recon_coords, axis=1)
    rmsd = np.sqrt(np.mean(diffs ** 2))

    if verbose:
        print(f"  n_atoms={n}, seq_len={result.n_tokens}, "
              f"links={result.n_links}, RMSD={rmsd:.4f}")

    return rmsd


if __name__ == "__main__":
    import pickle

    print("Loading QM9 data...")
    _qm9_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "freeorder", "qm9_mols_cache_v3b_noh.pkl")
    with open(_qm9_cache, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} molecules")

    random.seed(42)
    sample = random.sample(range(len(data)), 500)

    rmsds = []
    failures = 0
    for count, idx in enumerate(sample):
        mol, pos, smi = data[idx]
        rmsd = test_roundtrip(mol, pos, verbose=(count < 5))
        if rmsd is None:
            failures += 1
        else:
            rmsds.append(rmsd)

    rmsds = np.array(rmsds)
    print(f"\nRoundtrip test: {len(rmsds)} success, {failures} failures")
    print(f"RMSD: mean={rmsds.mean():.4f}, median={np.median(rmsds):.4f}, "
          f"max={rmsds.max():.4f}, P99={np.percentile(rmsds, 99):.4f}")
