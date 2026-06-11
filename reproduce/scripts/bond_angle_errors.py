"""Compute bond-length and bond-angle errors between pre/post xTB structures.

Uses existing retained xyz files (~20 per scaffold) from XVR pipeline.
Filters to topology-preserving (XVR-positive) samples.
"""
import os as _os_adt
_HERE_adt = _os_adt.path.dirname(_os_adt.path.abspath(__file__))
ADT_ROOT = _os_adt.environ.get("ADT_ROOT", _os_adt.path.abspath(_os_adt.path.join(_HERE_adt, "..", "..")))
DRUGS_DATA = _os_adt.path.join(ADT_ROOT, "Drugs", "data", "freeorder_v26")
QM9_DATA = _os_adt.path.join(ADT_ROOT, "QM9", "data", "freeorder")

import os, json, statistics
import numpy as np
from pathlib import Path

ELEMENTS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53}

# Heavy-atom max bond distance threshold per pair (in Å)
# Use generous values to detect bonds; we'll only include pairs that are bonded in BOTH pre and post
def bond_threshold(z1, z2):
    # Generic: covalent_radius(z1) + covalent_radius(z2) + 0.4
    r_cov = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39}
    return r_cov.get(z1, 1.0) + r_cov.get(z2, 1.0) + 0.4

def parse_xyz(path):
    """Return (heavy_z, heavy_xyz) — heavy atoms only."""
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0])
    atoms_z = []
    atoms_xyz = []
    for i in range(2, 2 + n):
        parts = lines[i].split()
        if not parts:
            continue
        sym = parts[0]
        if sym == "H":
            continue  # skip hydrogens
        z = ELEMENTS.get(sym, 0)
        if z == 0:
            continue
        x, y, zc = float(parts[1]), float(parts[2]), float(parts[3])
        atoms_z.append(z)
        atoms_xyz.append([x, y, zc])
    return np.array(atoms_z), np.array(atoms_xyz)

def get_bonds(z, xyz):
    """Return list of (i, j) bond pairs based on distance threshold."""
    n = len(z)
    bonds = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(xyz[i] - xyz[j])
            if d < bond_threshold(z[i], z[j]):
                bonds.append((i, j, d))
    return bonds

def compute_errors(pre_z, pre_xyz, post_z, post_xyz):
    """Return (bond_errors, angle_errors). Filter to bonds that exist in BOTH pre and post."""
    if len(pre_z) != len(post_z) or not all(pre_z == post_z):
        return [], []
    pre_bonds = {(i, j): d for i, j, d in get_bonds(pre_z, pre_xyz)}
    post_bonds = {(i, j): d for i, j, d in get_bonds(post_z, post_xyz)}
    common = set(pre_bonds.keys()) & set(post_bonds.keys())
    bond_errors = []
    for (i, j) in common:
        bond_errors.append(post_bonds[(i, j)] - pre_bonds[(i, j)])

    # Angles: A-B-C with A-B and B-C both bonded
    adj = {}
    for (i, j) in common:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)

    angle_errors = []
    for b in adj:
        neighbors = sorted(adj[b])
        for k in range(len(neighbors)):
            for l in range(k+1, len(neighbors)):
                a, c = neighbors[k], neighbors[l]
                # angle at b
                v1_pre = pre_xyz[a] - pre_xyz[b]
                v2_pre = pre_xyz[c] - pre_xyz[b]
                v1_post = post_xyz[a] - post_xyz[b]
                v2_post = post_xyz[c] - post_xyz[b]
                cos_pre = np.dot(v1_pre, v2_pre) / (np.linalg.norm(v1_pre) * np.linalg.norm(v2_pre))
                cos_post = np.dot(v1_post, v2_post) / (np.linalg.norm(v1_post) * np.linalg.norm(v2_post))
                cos_pre = np.clip(cos_pre, -1, 1)
                cos_post = np.clip(cos_post, -1, 1)
                angle_pre = np.degrees(np.arccos(cos_pre))
                angle_post = np.degrees(np.arccos(cos_post))
                angle_errors.append(angle_post - angle_pre)
    return bond_errors, angle_errors

def process_scaffold_dir(base_dir, scaffold):
    """Return all (bond_errors, angle_errors) for topology-preserving samples."""
    work = Path(base_dir) / scaffold / "xtb_work"
    res_path = Path(base_dir) / scaffold / "xtb_results.json"
    if not res_path.exists() or not work.exists():
        return [], [], 0
    with open(res_path) as f:
        results = json.load(f)
    # Map idx -> result entry
    idx_to_res = {r["idx"]: r for r in results}
    all_bond_errs = []
    all_angle_errs = []
    n_used = 0
    # Find retained pre files
    for pre_path in sorted(work.glob("mol_*.xyz")):
        name = pre_path.name
        if "xtbopt" in name or "xtblast" in name:
            continue
        # parse idx from "mol_N.xyz"
        try:
            idx = int(name[4:-4])
        except ValueError:
            continue
        post_path = work / f"mol_{idx}.xtbopt.xyz"
        if not post_path.exists():
            continue
        # Filter: only topology-preserving samples
        r = idx_to_res.get(idx)
        if not r or not r.get("ok") or not r.get("same"):
            continue
        try:
            pre_z, pre_xyz = parse_xyz(pre_path)
            post_z, post_xyz = parse_xyz(post_path)
        except Exception:
            continue
        be, ae = compute_errors(pre_z, pre_xyz, post_z, post_xyz)
        all_bond_errs.extend(be)
        all_angle_errs.extend(ae)
        n_used += 1
    return all_bond_errs, all_angle_errs, n_used

def main():
    SCAFFOLDS = ["benzene", "pyridine", "pyrimidine", "pyrazine", "furan", "thiophene", "cyclohexane"]
    for label, base in [("30-atom (v26s)", "" + DRUGS_DATA + "/v26s_scaffolds_n10k"),
                        ("50-atom (v26g)", "" + DRUGS_DATA + "/v26g_scaffolds_n10k")]:
        print(f"\n=== {label} ===")
        all_be, all_ae = [], []
        total_mols = 0
        for s in SCAFFOLDS:
            be, ae, n = process_scaffold_dir(base, s)
            print(f"  {s:<12} mols={n:>3} bonds={len(be):>5} angles={len(ae):>5}")
            all_be.extend(be)
            all_ae.extend(ae)
            total_mols += n
        if all_be:
            be_abs = [abs(x) for x in all_be]
            print(f"  TOTAL: {total_mols} mols, {len(all_be)} bonds, {len(all_ae)} angles")
            print(f"  bond-length |error|: median={statistics.median(be_abs)*1000:.1f} mÅ, p90={sorted(be_abs)[int(0.9*len(be_abs))]*1000:.1f} mÅ, max={max(be_abs)*1000:.1f} mÅ")
            ae_abs = [abs(x) for x in all_ae]
            print(f"  bond-angle |error|: median={statistics.median(ae_abs):.2f}°, p90={sorted(ae_abs)[int(0.9*len(ae_abs))]:.2f}°, max={max(ae_abs):.2f}°")

if __name__ == "__main__":
    main()
