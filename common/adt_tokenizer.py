"""
ADT v2 Core Tokenizer — Bootstrap Canonical Encoding (v4: tree-local frame)

SE(3)-invariant tokenization with tree-local frames:
  - Bootstrap triple (atomL, atomC, atomR) found via bidirectional chain growth
  - Internal coordinates (r, θ) for bootstrap atoms → rotation-invariant
  - Tree-local frame for regular atoms: built from ancestor chain edges
      u1 = parent→self, u2 = grandparent→parent
      If parallel: shift up — u1←u2, u2←(great-grandparent→grandparent)
      Root fallback: bootstrap frame axes
  - DFS child order shuffled for augmentation (spanning tree diversity)
  - Visited guard prevents duplicate atoms in fused-ring DFS

v4 changes from v3:
  - Frame construction uses tree-local ancestors instead of sequential DFS state
  - DFS children shuffled (spanning tree augmentation) with visited guard
  - Fixed duplicate-atom bug in fused-ring DFS traversal
  - Removed prev_u1/prev_u2 sequential state entirely

Actions: ADD_INIT(0), ADD_CHAIN(1), ADD_ANGLE(2), ADD(3), LINK(4), END(5)
Token:   [action, from, atom/to, r, hp0/θc, hp1/θf, hp2]

Invariance proof:
  - INIT: atom_type only → scalar → invariant
  - CHAIN: r = distance → invariant
  - ANGLE: θ = bond angle → invariant
  - REGULAR: frame built from tree edges (relative vectors) → invariant
"""

import numpy as np
import sys as _sys
import math as _math
import healpy as hp
import networkx as nx
from rdkit import Chem
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict, deque
import random

# ============================================================
# Action constants
# ============================================================

ADD_INIT  = 0   # Root atom: [action, 0, z, 0, 0, 0, 0]
ADD_CHAIN = 1   # Chain atom: [action, from, z, r, 0, 0, 0]
ADD_ANGLE = 2   # Angle atom: [action, from, z, r, θc, θf, 0]
ADD       = 3   # Regular atom: [action, from, z, r, hp0, hp1, hp2]
LINK      = 4   # Ring closure: [action, from, to, r, hp0, hp1, hp2]
END       = 5   # End of sequence

ACTION_NAMES = {
    ADD_INIT: "INIT", ADD_CHAIN: "CHAIN", ADD_ANGLE: "ANGLE",
    ADD: "ADD", LINK: "LINK", END: "END",
}

# ADT_HLINK=1 to add HLINK (Hidden Link) action for non-bonded proximity
import os as _os_h
_HLINK_ENABLED = _os_h.environ.get("ADT_HLINK", "0") == "1"
if _HLINK_ENABLED:
    HLINK = 6   # Hidden link (non-bonded proximity)
    ACTION_NAMES[HLINK] = "HLINK"
    N_ACTIONS = 7
else:
    N_ACTIONS = 6

# ============================================================
# Distance discretization
# ============================================================

R_MIN = 0.80   # Å
R_MAX = 2.50   # Å
import os as _os2
R_BINS = int(_os2.environ.get("ADT_R_BINS", "200"))
# Logarithmic mesh: r_i = R_MIN * (R_MAX/R_MIN)^(i/(R_BINS-1))
_R_LOG_RATIO = np.log(R_MAX / R_MIN)


def r_to_bin(r: float) -> int:
    if r <= R_MIN:
        return 0
    if r >= R_MAX:
        return R_BINS - 1
    b = np.log(r / R_MIN) / _R_LOG_RATIO * (R_BINS - 1)
    return int(np.clip(round(b), 0, R_BINS - 1))


def bin_to_r(b: int) -> float:
    return R_MIN * np.exp(b / (R_BINS - 1) * _R_LOG_RATIO)


# ============================================================
# Angle discretization (for ANGLE phase)
# ============================================================

THETA_BINS = 192  # 12 × 16, resolution ≈ 0.94°


def theta_to_bins(theta_rad: float) -> tuple[int, int]:
    """Bond angle (radians) → (θ_coarse 0..11, θ_fine 0..15)."""
    theta_deg = np.degrees(theta_rad)
    theta_bin = int(np.clip(round(theta_deg / 180.0 * (THETA_BINS - 1)), 0, THETA_BINS - 1))
    return theta_bin // 16, theta_bin % 16


def bins_to_theta(tc: int, tf: int) -> float:
    """(θ_coarse, θ_fine) → angle in radians."""
    theta_bin = tc * 16 + tf
    theta_deg = theta_bin / (THETA_BINS - 1) * 180.0
    return np.radians(theta_deg)


# ============================================================
# HEALPix encoding
# ============================================================

# Direction is decomposed into nhp0 x nhp1 x nhp2 bins.
# NSIDE is derived: npix = nhp0 * nhp1 * nhp2 = 12 * NSIDE^2.
NHP0 = int(_os2.environ.get("ADT_NHP0", "12"))
NHP1 = int(_os2.environ.get("ADT_NHP1", "16"))
NHP2 = int(_os2.environ.get("ADT_NHP2", "16"))

_NPIX = NHP0 * NHP1 * NHP2
_NSIDE_SQ = _NPIX // 12
if _NPIX != 12 * _NSIDE_SQ:
    print(f"FATAL: NHP0*NHP1*NHP2 = {_NPIX} is not divisible by 12", file=_sys.stderr)
    _sys.exit(1)
HEALPIX_NSIDE = int(round(_math.sqrt(_NSIDE_SQ)))
if HEALPIX_NSIDE * HEALPIX_NSIDE != _NSIDE_SQ:
    print(f"FATAL: NSIDE^2 = {_NSIDE_SQ} is not a perfect square", file=_sys.stderr)
    _sys.exit(1)
if HEALPIX_NSIDE & (HEALPIX_NSIDE - 1) != 0:
    print(f"FATAL: NSIDE = {HEALPIX_NSIDE} is not a power of 2", file=_sys.stderr)
    _sys.exit(1)
HEALPIX_NPIX = _NPIX
_HP12 = NHP1 * NHP2  # sub-pixel count per coarse pixel

NULL_POINTER = 0
PARALLEL_THRESHOLD = 1e-6


def vec_to_healpix(d_local: np.ndarray) -> tuple[int, int, int]:
    d = d_local / np.linalg.norm(d_local)
    pixel = hp.vec2pix(HEALPIX_NSIDE, d[0], d[1], d[2], nest=True)
    return pixel // _HP12, (pixel // NHP2) % NHP1, pixel % NHP2


def healpix_to_vec(hp0: int, hp1: int, hp2: int) -> np.ndarray:
    pixel = hp0 * _HP12 + hp1 * NHP2 + hp2
    dx, dy, dz = hp.pix2vec(HEALPIX_NSIDE, pixel, nest=True)
    return np.array([dx, dy, dz])


def healpix_pixel(hp0: int, hp1: int, hp2: int) -> int:
    return hp0 * _HP12 + hp1 * NHP2 + hp2


# ============================================================
# Frame construction
# ============================================================

def is_parallel(a: np.ndarray, b: np.ndarray, threshold: float = PARALLEL_THRESHOLD) -> bool:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return True
    cos = abs(np.dot(a / na, b / nb))
    return cos > 1.0 - threshold


def arbitrary_perpendicular(z: np.ndarray) -> np.ndarray:
    z = z / np.linalg.norm(z)
    if abs(z[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])
    x = v - np.dot(v, z) * z
    return x / np.linalg.norm(x)


def build_frame(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """
    Build orthonormal frame from two non-parallel vectors.
    Returns (3, 3) matrix [x; y; z] (rows).
    z = normalize(u2), x = GS(u1, z), y = z × x.
    """
    z = u2 / np.linalg.norm(u2)
    x = u1 - np.dot(u1, z) * z
    xnorm = np.linalg.norm(x)
    if xnorm < 1e-10:
        x = arbitrary_perpendicular(z)
    else:
        x = x / xnorm
    y = np.cross(z, x)
    return np.stack([x, y, z])


# ============================================================
# Tree-local frame construction (v4)
# ============================================================

def compute_tree_frame(pos_self, ancestor_positions, bootstrap_frame):
    """
    Build tree-local frame from ancestor chain edges.

    Args:
        pos_self: position of current atom
        ancestor_positions: [parent_pos, grandparent_pos, great_grandparent_pos, ...]
        bootstrap_frame: fallback when ancestor chain is exhausted

    Normal case:
        u1 = pos_self - parent_pos          (parent→self)
        u2 = parent_pos - grandparent_pos   (grandparent→parent)
        frame = build_frame(u1, u2)

    If u1 ∥ u2 (collinear A-P-G):
        u1 ← u2  (= grandparent→parent)
        u2 ← grandparent_pos - great_grandparent_pos
        ... repeat until non-parallel pair found or root reached

    Root fallback: use bootstrap_frame axes as u2.
    """
    if not ancestor_positions:
        return bootstrap_frame.copy()

    u1 = pos_self - ancestor_positions[0]

    for k in range(len(ancestor_positions) - 1):
        u2 = ancestor_positions[k] - ancestor_positions[k + 1]
        if not is_parallel(u1, u2):
            return build_frame(u1, u2)
        u1 = u2  # shift up: adopt this edge as u1, look further for u2

    # All ancestors collinear or only one ancestor: fallback to bootstrap frame
    bs_z = bootstrap_frame[2]  # z-axis of bootstrap
    if not is_parallel(u1, bs_z):
        return build_frame(u1, bs_z)
    bs_x = bootstrap_frame[0]  # x-axis of bootstrap
    if not is_parallel(u1, bs_x):
        return build_frame(u1, bs_x)
    # Absolute fallback (should never happen for real molecules)
    return build_frame(u1, np.array([0.0, 0.0, 1.0]))


def _ancestor_positions_orig(atom_idx, parent_map, positions):
    """
    Collect ancestor positions [parent, grandparent, ...] using original
    atom indices and parent_map. Used during encoding (tokenize_molecule).
    """
    chain = []
    current = atom_idx
    while True:
        p = parent_map.get(current, -1)
        if p < 0:
            break
        chain.append(positions[p])
        current = p
    return chain


def _ancestor_positions_recon(from_id, atom_table):
    """
    Collect ancestor positions [parent, grandparent, ...] starting from
    from_id (1-indexed pointer). Used during decoding (reconstruct_from_tokens).
    """
    chain = []
    current_id = from_id  # 1-indexed
    while current_id > 0:
        step = current_id - 1
        chain.append(atom_table[step].pos)
        current_id = atom_table[step].parent_id
    return chain


def encode_direction(d_vec: np.ndarray, frame: np.ndarray) -> tuple[int, int, int, int]:
    r = np.linalg.norm(d_vec)
    d_local = frame @ (d_vec / r)
    hp0, hp1, hp2 = vec_to_healpix(d_local)
    return r_to_bin(r), hp0, hp1, hp2


def decode_direction(r_bin: int, hp0: int, hp1: int, hp2: int,
                     frame: np.ndarray) -> tuple[float, np.ndarray]:
    r = bin_to_r(r_bin)
    d_local = healpix_to_vec(hp0, hp1, hp2)
    d_world = frame.T @ d_local
    return r, d_world


# ============================================================
# Atom table entry
# ============================================================

@dataclass
class AtomEntry:
    pos: np.ndarray
    parent_id: int              # pointer ID (0=NULL)
    frame: np.ndarray           # (3, 3) local frame
    v1: np.ndarray              # arrival direction (kept for compatibility)
    v2: np.ndarray              # reference direction (kept for compatibility)
    atomic_num: int = 0
    original_idx: int = -1
    is_bootstrap: bool = False


# ============================================================
# Bootstrap triple discovery
# ============================================================

COLLINEAR_COS = np.cos(np.radians(5.0))  # ≈ 0.9962


def find_bootstrap_triple(mol, positions, start_idx=None):
    """
    Grow chain bidirectionally from start. Find first non-collinear neighbor.

    Returns:
        atomL: int — Lead atom (DFS root, placed by ADD_INIT)
        chain: list[int] — [atomL, ..., atomC] in DFS order
        atomR: int or None — Reference atom (non-collinear, placed by ADD_ANGLE)

    Naming convention (independent of atom indices):
        atomL = Lead      — placed by ADD_INIT, origin
        atomC = Chain end — placed by ADD_CHAIN, angle vertex
        atomR = Reference — placed by ADD_ANGLE, defines the plane
    """
    adj = defaultdict(list)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    if start_idx is None:
        start_idx = random.randint(0, mol.GetNumAtoms() - 1)

    chain = deque([start_idx])
    in_chain = {start_idx}
    right_stuck = False
    left_stuck = False

    while not (right_stuck and left_stuck):
        for side in ['right', 'left']:
            if side == 'right' and right_stuck:
                continue
            if side == 'left' and left_stuck:
                continue

            end = chain[-1] if side == 'right' else chain[0]
            extended = False

            for nb in adj[end]:
                if nb in in_chain:
                    continue

                if len(chain) < 2:
                    # First extension: unconditionally accept
                    if side == 'right':
                        chain.append(nb)
                    else:
                        chain.appendleft(nb)
                    in_chain.add(nb)
                    extended = True
                    break

                # Collinearity check
                chain_dir = positions[chain[-1]] - positions[chain[0]]
                cdn = np.linalg.norm(chain_dir)
                if cdn < 1e-10:
                    if side == 'right':
                        chain.append(nb)
                    else:
                        chain.appendleft(nb)
                    in_chain.add(nb)
                    extended = True
                    break
                chain_dir /= cdn

                new_dir = positions[nb] - positions[end]
                ndn = np.linalg.norm(new_dir)
                if ndn < 1e-10:
                    continue
                new_dir /= ndn

                if abs(np.dot(chain_dir, new_dir)) > COLLINEAR_COS:
                    # Collinear → extend chain
                    if side == 'right':
                        chain.append(nb)
                    else:
                        chain.appendleft(nb)
                    in_chain.add(nb)
                    extended = True
                    break
                else:
                    # Non-collinear → atomR found!
                    atomR = nb
                    if side == 'right':
                        return chain[0], list(chain), atomR
                    else:
                        rev = list(reversed(chain))
                        return rev[0], rev, atomR

            if not extended:
                if side == 'right':
                    right_stuck = True
                else:
                    left_stuck = True

    # Fully linear molecule
    return chain[0], list(chain), None


# ============================================================
# DFS decomposition with bootstrap priority
# ============================================================

def bootstrap_decompose(mol, chain, atomR):
    """
    DFS order: chain[0](=atomL) → ... → chain[-1](=atomC) → atomR → subtrees.
    Children are shuffled for augmentation (spanning tree diversity for fused rings).
    Visited guard prevents duplicate atoms when sibling subtrees merge.
    Returns atom_order, parent_map, link_edges.
    """
    adj = defaultdict(set)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].add(j)
        adj[j].add(i)

    visited = set(chain) | {atomR}
    atom_order = list(chain) + [atomR]
    parent_map = {chain[0]: -1}
    for k in range(1, len(chain)):
        parent_map[chain[k]] = chain[k - 1]
    parent_map[atomR] = chain[-1]

    def dfs(node):
        children = [nb for nb in adj[node] if nb not in visited]
        random.shuffle(children)  # augmentation: spanning tree diversity
        for nb in children:
            if nb in visited:   # guard: sibling's subtree may have visited nb
                continue
            visited.add(nb)
            atom_order.append(nb)
            parent_map[nb] = node
            dfs(nb)

    # DFS from atomR first, then backtrack through chain
    dfs(atomR)
    for atom in reversed(chain):
        dfs(atom)

    # Link edges (non-tree)
    tree_set = set()
    for c, p in parent_map.items():
        if p >= 0:
            tree_set.add((min(c, p), max(c, p)))
    all_edges = set((min(i, j), max(i, j)) for i in adj for j in adj[i] if i < j)
    link_edges = all_edges - tree_set

    return atom_order, parent_map, list(link_edges)


# ============================================================
# Main tokenization
# ============================================================

@dataclass
class TokenizedMolecule:
    tokens: list = field(default_factory=list)
    atom_table: dict = field(default_factory=dict)
    idx_map: dict = field(default_factory=dict)
    n_atoms: int = 0
    n_links: int = 0
    n_tokens: int = 0
    # Bootstrap statistics
    chain_length: int = 0            # number of ADD_CHAIN tokens (1 = normal)
    theta_bins: tuple = None         # (theta_coarse, theta_fine)
    bootstrap_triple: tuple = None   # (atomL, atomC, atomR) original indices


def tokenize_molecule(mol: Chem.Mol, positions: np.ndarray,
                      root: Optional[int] = None,
                      seed: Optional[int] = None) -> Optional[TokenizedMolecule]:
    """
    SE(3)-invariant tokenization using bootstrap canonical encoding
    with tree-local frames.

    Returns None for linear molecules (no valid bootstrap triple).
    """
    positions = positions.copy()
    n_atoms = mol.GetNumAtoms()

    if n_atoms < 3:
        return None  # Too small for bootstrap

    # --- Find bootstrap triple ---
    start = root if root is not None else (
        random.Random(seed).randint(0, n_atoms - 1) if seed is not None
        else random.randint(0, n_atoms - 1)
    )
    atomL, chain, atomR = find_bootstrap_triple(mol, positions, start)
    # atomL = Lead (root), chain[-1] = atomC (chain end), atomR = Reference
    atomC = chain[-1]

    if atomR is None:
        return None  # Linear molecule — discard

    n_chain = len(chain)  # includes atomL
    n_bootstrap = n_chain + 1  # chain + atomR
    bootstrap_set = set(chain) | {atomR}

    # --- DFS order with bootstrap priority ---
    atom_order, parent_map, link_edges = bootstrap_decompose(mol, chain, atomR)
    assert atom_order[:n_chain] == chain
    assert atom_order[n_chain] == atomR

    # --- Compute bootstrap geometry (rotation-invariant quantities) ---
    # Bond angle at atomC: angle(atomL-side, atomC, atomR)
    v_LC = positions[atomL] - positions[atomC]
    v_RC = positions[atomR] - positions[atomC]
    cos_theta = np.dot(v_LC / np.linalg.norm(v_LC), v_RC / np.linalg.norm(v_RC))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    theta_c, theta_f = theta_to_bins(theta)

    # --- Compute bootstrap frame (from actual positions) ---
    v_chain = positions[atomC] - positions[atomL]
    v_depart = positions[atomR] - positions[atomC]
    bootstrap_frame = build_frame(v_depart, v_chain)
    # z = chain direction (atomL→atomC), x = departure component ⊥ chain

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
            # === INIT: root atom ===
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
            # === CHAIN: linear continuation ===
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
            # === ANGLE: first non-collinear atom ===
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
            # === REGULAR: HEALPix encoding with tree-local frame ===
            # Encode direction using parent's frame
            from_frame = atom_table[from_id - 1].frame
            d_vec = pos_j - positions[from_orig]
            r_bin, hp0, hp1, hp2 = encode_direction(d_vec, from_frame)

            # Build tree-local frame for this atom (for its children to use)
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

        # Store atom entry
        atom_table[step] = AtomEntry(
            pos=pos_j.copy(),
            parent_id=from_id,
            frame=frame.copy(),
            v1=(pos_j - positions[from_orig]).copy() if from_orig >= 0 else np.zeros(3),
            v2=np.zeros(3),  # not used in tree-local mode
            atomic_num=atom_type,
            original_idx=atom_idx,
            is_bootstrap=is_bs,
        )
        idx_map[atom_idx] = step + 1  # pointer ID (1-indexed)
        placed.add(atom_idx)

        # --- Emit LINK tokens for newly complete edges ---
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

    # --- END ---
    tokens.append(('action', END))

    return TokenizedMolecule(
        tokens=tokens,
        atom_table=atom_table,
        idx_map=idx_map,
        n_atoms=len(atom_order),
        n_links=n_links,
        n_tokens=len(tokens),
        chain_length=n_chain - 1,        # number of ADD_CHAIN tokens
        theta_bins=(theta_c, theta_f),
        bootstrap_triple=(atomL, atomC, atomR),
    )


# ============================================================
# Reconstruction (decode tokens → 3D coordinates)
# ============================================================

def reconstruct_from_tokens(tokens: list) -> tuple[list[AtomEntry], list[tuple]]:
    """
    Reconstruct 3D coordinates from bootstrap-encoded token sequence.
    Uses tree-local frames (DFS-order-independent).
    """
    atom_table = {}
    links = []
    n_atoms = 0

    # Bootstrap state
    chain_positions = []  # positions during INIT/CHAIN phase
    bootstrap_frame = None

    i = 0
    while i < len(tokens):
        slot_name, action = tokens[i]
        assert slot_name == 'action'

        if action == END:
            break

        if action == ADD_INIT:
            _, from_id = tokens[i + 1]
            _, atom_type = tokens[i + 2]
            i += 7

            pos = np.array([0.0, 0.0, 0.0])
            chain_positions.append(pos)

            atom_table[n_atoms] = AtomEntry(
                pos=pos, parent_id=0,
                frame=np.eye(3),  # temporary, overwritten at ANGLE
                v1=np.zeros(3), v2=np.zeros(3),
                atomic_num=atom_type, is_bootstrap=True,
            )
            n_atoms += 1

        elif action == ADD_CHAIN:
            _, from_id = tokens[i + 1]
            _, atom_type = tokens[i + 2]
            _, r_bin = tokens[i + 3]
            i += 7

            r = bin_to_r(r_bin)
            parent_pos = atom_table[from_id - 1].pos

            # Chain direction: always z-axis in canonical placement
            if len(chain_positions) >= 2:
                chain_dir = chain_positions[-1] - chain_positions[0]
                cdn = np.linalg.norm(chain_dir)
                if cdn > 1e-10:
                    chain_dir = chain_dir / cdn
                else:
                    chain_dir = np.array([0.0, 0.0, 1.0])
            else:
                chain_dir = np.array([0.0, 0.0, 1.0])

            pos = parent_pos + r * chain_dir
            chain_positions.append(pos)

            atom_table[n_atoms] = AtomEntry(
                pos=pos, parent_id=from_id,
                frame=np.eye(3),  # temporary
                v1=chain_dir * r, v2=np.zeros(3),
                atomic_num=atom_type, is_bootstrap=True,
            )
            n_atoms += 1

        elif action == ADD_ANGLE:
            _, from_id = tokens[i + 1]
            _, atom_type = tokens[i + 2]
            _, r_bin = tokens[i + 3]
            _, theta_c = tokens[i + 4]
            _, theta_f = tokens[i + 5]
            i += 7

            r = bin_to_r(r_bin)
            theta = bins_to_theta(theta_c, theta_f)
            parent_pos = atom_table[from_id - 1].pos

            # Reconstruct ANGLE position
            if len(chain_positions) >= 2:
                chain_dir = chain_positions[-1] - chain_positions[0]
                cdn = np.linalg.norm(chain_dir)
                if cdn > 1e-10:
                    chain_dir = chain_dir / cdn
                else:
                    chain_dir = np.array([0.0, 0.0, 1.0])
            else:
                chain_dir = np.array([0.0, 0.0, 1.0])

            perp = arbitrary_perpendicular(chain_dir)
            # d makes angle θ with reverse-chain direction (-chain_dir)
            # So d = -cos(θ)*chain_dir + sin(θ)*perp
            d = -np.cos(theta) * chain_dir + np.sin(theta) * perp
            pos = parent_pos + r * d

            # --- Build bootstrap frame ---
            v_chain = chain_dir  # already normalized
            v_depart = d         # already unit (since cos²+sin²=1)
            bootstrap_frame = build_frame(v_depart, v_chain)
            # z = chain_dir, x = component of v_depart ⊥ chain_dir

            # Assign bootstrap frame to ALL bootstrap atoms
            for k in range(n_atoms):
                atom_table[k].frame = bootstrap_frame.copy()
                atom_table[k].is_bootstrap = True

            atom_table[n_atoms] = AtomEntry(
                pos=pos, parent_id=from_id,
                frame=bootstrap_frame.copy(),
                v1=(pos - parent_pos).copy(),
                v2=np.zeros(3),
                atomic_num=atom_type, is_bootstrap=True,
            )
            n_atoms += 1

        elif action == ADD:
            _, from_id = tokens[i + 1]
            _, atom_type = tokens[i + 2]
            _, r_bin = tokens[i + 3]
            _, hp0 = tokens[i + 4]
            _, hp1 = tokens[i + 5]
            _, hp2 = tokens[i + 6]
            i += 7

            # Decode position using parent's frame
            from_frame = atom_table[from_id - 1].frame
            parent_pos = atom_table[from_id - 1].pos
            r, d_world = decode_direction(r_bin, hp0, hp1, hp2, from_frame)
            pos = parent_pos + r * d_world

            # Build tree-local frame for this atom
            anc_pos = _ancestor_positions_recon(from_id, atom_table)
            frame = compute_tree_frame(pos, anc_pos, bootstrap_frame)

            atom_table[n_atoms] = AtomEntry(
                pos=pos, parent_id=from_id,
                frame=frame.copy(),
                v1=(pos - parent_pos).copy(),
                v2=np.zeros(3),
                atomic_num=atom_type,
            )
            n_atoms += 1

        elif action == LINK:
            _, from_id = tokens[i + 1]
            _, to_id = tokens[i + 2]
            _, r_bin = tokens[i + 3]
            _, hp0 = tokens[i + 4]
            _, hp1 = tokens[i + 5]
            _, hp2 = tokens[i + 6]
            i += 7
            links.append((from_id, to_id, r_bin, hp0, hp1, hp2))

        else:
            break

    atoms = [atom_table[k] for k in sorted(atom_table.keys())]
    return atoms, links


# ============================================================
# Token ↔ integer array conversion
# ============================================================

SLOT_NAMES = ['action', 'from', 'atom', 'r', 'hp0', 'hp1', 'hp2']


def tokens_to_array(tokens: list) -> np.ndarray:
    return np.array([v for (_, v) in tokens], dtype=np.int32)


def array_to_tokens(arr: np.ndarray) -> list:
    tokens = []
    i = 0
    while i < len(arr):
        action = arr[i]
        if action == END:
            tokens.append(('action', END))
            i += 1
            break
        if i + 7 > len(arr):
            break
        slot_names = SLOT_NAMES.copy()
        if action == LINK:
            slot_names[2] = 'to'
        for j in range(7):
            tokens.append((slot_names[j], int(arr[i + j])))
        i += 7
    return tokens


# ============================================================
# Utility
# ============================================================

def mol_from_smiles(smiles: str) -> tuple[Chem.Mol, np.ndarray]:
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    positions = np.array([conf.GetAtomPosition(i)
                          for i in range(mol.GetNumAtoms())])
    return mol, positions
