"""
Collision detection for generated molecules.

Uses curated per-pair thresholds where available, falls back to
RDKit periodic table (covalent / van der Waals radii) for unknown pairs.
"""

import warnings
from rdkit.Chem import GetPeriodicTable

_pt = GetPeriodicTable()

COLLISION_R = 0.9

# Curated shortest known bond lengths (Angstrom)
SHORTEST_BOND = {
    (1, 1):   0.74,
    (1, 6):   1.09,
    (1, 7):   1.01,
    (1, 8):   0.96,
    (1, 9):   0.92,
    (6, 6):   1.20,   # C≡C triple
    (6, 7):   1.16,   # C≡N triple
    (6, 8):   1.23,   # C=O double
    (6, 9):   1.35,   # C-F single
    (6, 15):  1.84,   # C-P
    (6, 16):  1.61,   # C-S
    (6, 17):  1.74,   # C-Cl
    (6, 35):  1.91,   # C-Br
    (6, 53):  2.10,   # C-I
    (7, 7):   1.10,   # N≡N triple
    (7, 8):   1.21,   # N=O double
    (7, 9):   1.36,
    (8, 8):   1.21,
    (8, 9):   1.42,
    (8, 15):  1.48,   # O-P
    (8, 16):  1.43,   # O-S
    (9, 9):   1.42,
    (9, 15):  1.54,   # F-P
    (15, 15): 2.21,   # P-P
    (16, 16): 2.05,   # S-S
}

# Curated non-bonded minimum distances (absolute floor)
NONBOND_MIN = {
    (1, 1): 1.0, (1, 6): 1.2, (1, 7): 1.2, (1, 8): 1.2, (1, 9): 1.2,
    (6, 6): 1.5, (6, 7): 1.5, (6, 8): 1.5, (6, 9): 1.5,
    (6, 15): 1.5, (6, 16): 1.5, (6, 17): 1.5, (6, 35): 1.6,
    (7, 7): 1.5, (7, 8): 1.5, (7, 9): 1.5,
    (8, 8): 1.5, (8, 9): 1.5, (8, 16): 1.5,
    (9, 9): 1.5,
    (15, 15): 1.8, (16, 16): 1.8, (17, 17): 1.8,
}

# Factors for fallback computation
_BONDED_FACTOR = 0.75      # cov_sum * COLLISION_R * this
_NONBOND_VDW_FACTOR = 0.45  # vdw_sum * this


def _get_radius(z, kind):
    """Get radius with validation. kind='cov' or 'vdw'."""
    assert isinstance(z, int) and z > 0, f"invalid atomic number: {z}"
    try:
        if kind == 'cov':
            r = _pt.GetRcovalent(z)
        else:
            r = _pt.GetRvdw(z)
    except Exception as e:
        raise ValueError(f"RDKit has no {kind} radius for Z={z}: {e}")
    assert r > 0, (
        f"{kind} radius for Z={z} is {r} (non-positive). "
        f"Element may not be supported."
    )
    return r


def get_collision_threshold(z_i, z_j, collision_R=COLLISION_R, is_bonded=True):
    """Pair-specific collision threshold.

    Uses curated dict if available, otherwise falls back to RDKit
    periodic table radii. Asserts on invalid inputs.
    """
    assert isinstance(z_i, int) and z_i > 0, f"z_i={z_i} invalid"
    assert isinstance(z_j, int) and z_j > 0, f"z_j={z_j} invalid"

    key = (min(z_i, z_j), max(z_i, z_j))

    if is_bonded:
        curated = SHORTEST_BOND.get(key)
        if curated is not None:
            thr = curated * collision_R
        else:
            r_cov = _get_radius(z_i, 'cov') + _get_radius(z_j, 'cov')
            thr = r_cov * collision_R * _BONDED_FACTOR
            warnings.warn(
                f"collision: no SHORTEST_BOND for {key}, "
                f"fallback cov={r_cov:.2f} → thr={thr:.3f}",
                stacklevel=2,
            )
    else:
        curated = NONBOND_MIN.get(key)
        if curated is not None:
            thr = curated
        else:
            r_vdw = _get_radius(z_i, 'vdw') + _get_radius(z_j, 'vdw')
            thr = r_vdw * _NONBOND_VDW_FACTOR
            warnings.warn(
                f"collision: no NONBOND_MIN for {key}, "
                f"fallback vdw={r_vdw:.2f} → thr={thr:.3f}",
                stacklevel=2,
            )

    assert thr > 0, f"collision threshold non-positive: {thr} for {key}"
    assert thr < 5.0, f"collision threshold suspiciously large: {thr} for {key}"

    return thr


def check_collisions(coords, atomic_nums, bonds):
    """Check all pairwise distances against thresholds.

    Args:
        coords: list/array of (x,y,z) per atom
        atomic_nums: list of int per atom
        bonds: set of (i,j) pairs (0-indexed)

    Returns:
        (has_collision: bool, details: list of (i, j, dist, thr))
    """
    import numpy as np
    n = len(coords)
    assert n == len(atomic_nums), f"coords/atomic_nums length mismatch: {n} vs {len(atomic_nums)}"

    bond_set = set()
    for (a, b) in bonds:
        bond_set.add((a, b))
        bond_set.add((b, a))

    details = []
    for a in range(n):
        for b in range(a + 1, n):
            d = float(np.linalg.norm(np.array(coords[a]) - np.array(coords[b])))
            is_bonded = (a, b) in bond_set
            thr = get_collision_threshold(
                int(atomic_nums[a]), int(atomic_nums[b]),
                is_bonded=is_bonded)
            if d < thr:
                details.append((a, b, d, thr))

    return len(details) > 0, details


if __name__ == "__main__":
    # Self-test: verify thresholds for common pairs
    pairs = [
        (6, 6, True, "C-C bonded"),
        (6, 6, False, "C-C nonbond"),
        (6, 7, True, "C-N bonded"),
        (6, 8, True, "C-O bonded"),
        (6, 17, True, "C-Cl bonded"),
        (6, 17, False, "C-Cl nonbond"),
        (6, 35, True, "C-Br bonded"),
        (16, 16, True, "S-S bonded"),
        (26, 7, True, "Fe-N bonded (fallback)"),
        (26, 7, False, "Fe-N nonbond (fallback)"),
    ]
    for z_i, z_j, bonded, label in pairs:
        thr = get_collision_threshold(z_i, z_j, is_bonded=bonded)
        src = "curated" if (min(z_i,z_j), max(z_i,z_j)) in (SHORTEST_BOND if bonded else NONBOND_MIN) else "fallback"
        print(f"  {label:25s}: thr={thr:.3f} ({src})")
