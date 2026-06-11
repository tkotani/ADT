#!/usr/bin/env python3
"""
uniq_validation2.py — Species accumulation analysis from .pt files.

Loads generated molecules, validates (3D + stable),
subsamples at various K, plots Known/Novel accumulation curves,
and fits 2-component (QM9) + 3-param (Novel) models.

Usage:
    python uniq_validation2.py gen_all.pt
    python uniq_validation2.py gen_*.pt --n_points 30
    python uniq_validation2.py gen_*.pt --alpha 0.135
"""

import argparse
import sys
import os
import time
import numpy as np
import torch
from datetime import datetime
from collections import Counter

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

# ============================================================
# Constants
# ============================================================
Z_SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
STANDARD_VALENCE = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1,
                     'S': 2, 'P': 3, 'Cl': 1, 'Br': 1, 'I': 1}
RELAXED_VALENCE = {'N': {3, 4}, 'S': {2, 4}}  # additional allowed valences


# ============================================================
# Core functions (from util_validation.py)
# ============================================================

def min_pairwise_dist(positions):
    pos = np.asarray(positions, dtype=np.float64)
    n = len(pos)
    if n < 2:
        return 999.0
    diff = pos[:, None, :] - pos[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, 999.0)
    return float(dists.min())


# Reference bond lengths in Angstrom (EDM style)
# Reference single-bond lengths in Angstrom (extended; covers C/N/O/F/P/S/Cl/Br/I)
_BONDS1 = {
    6:  {6: 1.535, 7: 1.469, 8: 1.440, 9: 1.382, 15: 1.847, 16: 1.815, 17: 1.790, 35: 1.954, 53: 2.139},
    7:  {6: 1.469, 7: 1.411, 8: 1.463, 9: 1.410, 15: 1.730, 16: 1.680, 17: 1.750, 35: 1.940, 53: 2.150},
    8:  {6: 1.440, 7: 1.463, 8: 1.480, 9: 1.420, 15: 1.620, 16: 1.580, 17: 1.700, 35: 1.870, 53: 2.080},
    9:  {6: 1.382, 7: 1.410, 8: 1.420, 9: 1.420, 15: 1.560, 16: 1.640, 17: 1.660},
    15: {6: 1.847, 7: 1.730, 8: 1.620, 9: 1.560, 15: 2.214, 16: 2.090, 17: 2.040, 35: 2.220},
    16: {6: 1.815, 7: 1.680, 8: 1.580, 9: 1.640, 15: 2.090, 16: 2.050, 17: 2.070, 35: 2.240},
    17: {6: 1.790, 7: 1.750, 8: 1.700, 9: 1.660, 15: 2.040, 16: 2.070, 17: 1.990},
    35: {6: 1.954, 7: 1.940, 8: 1.870, 15: 2.220, 16: 2.240, 35: 2.281},
    53: {6: 2.139, 7: 2.150, 8: 2.080, 53: 2.666},
}
_BONDS2 = {
    6: {6: 1.339, 7: 1.279, 8: 1.229},
    7: {6: 1.279, 7: 1.240, 8: 1.210},
    8: {6: 1.229, 7: 1.210, 8: 1.210},
}
_BONDS3 = {
    6: {6: 1.212, 7: 1.158, 8: 1.128},
    7: {6: 1.158, 7: 1.098},
}
_MARGIN1 = 0.10
_MARGIN2 = 0.05
_MARGIN3 = 0.03


def _get_bond_order_by_distance(z1, z2, dist):
    """EDM-style distance-based bond order."""
    if z1 in _BONDS1 and z2 in _BONDS1[z1]:
        if dist < _BONDS1[z1][z2] + _MARGIN1:
            if z1 in _BONDS3 and z2 in _BONDS3.get(z1, {}):
                if dist < _BONDS3[z1][z2] + _MARGIN3:
                    return 3
            if z1 in _BONDS2 and z2 in _BONDS2.get(z1, {}):
                if dist < _BONDS2[z1][z2] + _MARGIN2:
                    return 2
            return 1
    return 0


def _detect_nohydrogen(atomic_nums):
    """Detect if molecule was generated without hydrogens."""
    return all(z != 1 for z in atomic_nums)




def _try_aromatize_rings(rw, pos, atomic_nums):
    """Post-process: detect rings where all bond distances are aromatic-like,
    set those bonds to AROMATIC. No SanitizeMol here - let caller handle it."""
    from rdkit.Chem import rdmolops
    AROMATIC_ATOMS = {6, 7, 8, 16}  # C, N, O, S
    
    try:
        rdmolops.FastFindRings(rw)
        ri = rw.GetRingInfo()
        
        for ring in ri.AtomRings():
            ring_size = len(ring)
            if ring_size not in (5, 6):
                continue
            
            # All atoms must be aromatic-capable
            if not all(atomic_nums[idx] in AROMATIC_ATOMS for idx in ring):
                continue
            
            # Check all ring bonds exist and have aromatic-like distance
            ring_bonds = []
            ok = True
            for k in range(ring_size):
                i, j = ring[k], ring[(k+1) % ring_size]
                bond = rw.GetBondBetweenAtoms(i, j)
                if bond is None:
                    ok = False
                    break
                dist = float(np.linalg.norm(pos[i] - pos[j]))
                if dist > 1.48 or dist < 1.25:
                    ok = False
                    break
                ring_bonds.append((i, j))
            
            if not ok or len(ring_bonds) != ring_size:
                continue
            
            # Set all ring bonds to AROMATIC
            for i, j in ring_bonds:
                bond = rw.GetBondBetweenAtoms(i, j)
                bond.SetBondType(Chem.BondType.AROMATIC)
                bond.SetIsAromatic(True)
            
            for idx in ring:
                rw.GetAtomWithIdx(idx).SetIsAromatic(True)
    except:
        pass


def validate_3D(atomic_nums, positions, charge=0, aromatize=None, emitted_bonds=None):
    if aromatize is None:
        aromatize = os.environ.get("AROMATIZE_RINGS", "0") == "1"
    n = len(atomic_nums)
    info = {'n_atoms': n}
    if n < 1:
        info['valid'] = False
        return None, None, info

    is_noh = _detect_nohydrogen(atomic_nums)

    try:
        if is_noh:
            # Heavy-atom-only: distance-based bond orders
            rw = Chem.RWMol()
            for z in atomic_nums:
                rw.AddAtom(Chem.Atom(int(z)))

            # Determine connectivity + bond orders
            pos = np.asarray(positions, dtype=np.float64)
            bond_type_map = {1: Chem.BondType.SINGLE,
                             2: Chem.BondType.DOUBLE,
                             3: Chem.BondType.TRIPLE}
            if emitted_bonds is not None:
                # emitted_bonds are 1-indexed pointer IDs (0=NULL); convert to 0-indexed atom ids
                EMITTED_DIST_CUTOFF = 2.5  # filter spurious long-range LINKs
                seen = set()
                for a_raw, b_raw in emitted_bonds:
                    a0 = int(a_raw) - 1
                    b0 = int(b_raw) - 1
                    if a0 < 0 or b0 < 0: continue
                    if a0 >= n or b0 >= n: continue
                    if a0 == b0: continue
                    key = (min(a0, b0), max(a0, b0))
                    if key in seen: continue
                    seen.add(key)
                    dist = float(np.linalg.norm(pos[a0] - pos[b0]))
                    if dist > EMITTED_DIST_CUTOFF:
                        continue
                    order = _get_bond_order_by_distance(atomic_nums[a0], atomic_nums[b0], dist)
                    if order == 0:
                        order = 1
                    rw.AddBond(a0, b0, bond_type_map[order])
            else:
                # Distance-based bond detection (fallback when no emitted bonds)
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.linalg.norm(pos[i] - pos[j])
                        order = _get_bond_order_by_distance(
                            atomic_nums[i], atomic_nums[j], dist)
                        if order > 0:
                            rw.AddBond(i, j, bond_type_map[order])

            conf = Chem.Conformer(n)
            for j in range(n):
                conf.SetAtomPosition(j, [float(x) for x in positions[j]])
            rw.AddConformer(conf, assignId=True)

            # Try to aromatize rings with aromatic-like distances
            if aromatize:
                _try_aromatize_rings(rw, pos, atomic_nums)
            
            mol = rw.GetMol()
            try:
                Chem.SanitizeMol(mol)
            except:
                # If aromatization broke something, try without
                pass
            smiles = Chem.MolToSmiles(mol)
            info['valid'] = True
            info['smiles'] = smiles

            # Stability: check GetTotalValence (includes implicit H)
            n_stable = 0
            n_stable_relaxed = 0
            n_atoms = mol.GetNumAtoms()
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                tv = atom.GetTotalValence()
                if sym in STANDARD_VALENCE and tv == STANDARD_VALENCE[sym]:
                    n_stable += 1
                    n_stable_relaxed += 1
                elif sym in RELAXED_VALENCE and tv in RELAXED_VALENCE[sym]:
                    n_stable_relaxed += 1
            info['mol_stable'] = (n_stable == n_atoms)
            info['mol_stable_n4s4'] = (n_stable_relaxed == n_atoms)
            return mol, smiles, info

        else:
            # With hydrogens: original DetermineBonds approach
            rw = Chem.RWMol()
            for z in atomic_nums:
                rw.AddAtom(Chem.Atom(int(z)))
            conf = Chem.Conformer(n)
            for j in range(n):
                conf.SetAtomPosition(j, [float(x) for x in positions[j]])
            rw.AddConformer(conf, assignId=True)
            rdDetermineBonds.DetermineConnectivity(rw)
            rdDetermineBonds.DetermineBondOrders(rw, charge=charge)
            mol = rw.GetMol()
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(Chem.RemoveAllHs(mol))
            info['valid'] = True
            info['smiles'] = smiles
            # Atom stability
            n_stable = 0
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                if sym in STANDARD_VALENCE:
                    bond_sum = sum(b.GetBondTypeAsDouble()
                                  for b in atom.GetBonds())
                    if abs(bond_sum - STANDARD_VALENCE[sym]) < 0.1:
                        n_stable += 1
            info['mol_stable'] = (n_stable == n)
            return mol, smiles, info

    except Exception as e:
        info['valid'] = False
        info['error'] = str(e)[:100]
        return None, None, info


def load_molecules_from_pt(pt_paths):
    molecules = []
    for pt_path in pt_paths:
        print(f"  Loading {pt_path}...")
        data = torch.load(pt_path, weights_only=False)
        infos = data.get('infos', [])
        coords_list = data.get('coords', [])
        atoms_list = data.get('atoms', [])
        smiles_list = data.get('smiles', [])
        loaded = 0
        if infos:
            sample = infos[0]
            has_pos = 'positions' in sample or 'pos' in sample
            has_z = 'atomic_nums' in sample or 'z' in sample
            if has_pos and has_z:
                for info in infos:
                    pos = info.get('positions', info.get('pos'))
                    z = info.get('atomic_nums', info.get('z'))
                    smi = info.get('smiles')
                    if pos is not None and z is not None:
                        molecules.append((list(z), np.asarray(pos, dtype=np.float64), smi))
                        loaded += 1
            elif 'tokens_raw' in sample:
                try:
                    from adt_tokenizer import reconstruct_from_tokens, SLOT_NAMES
                    for i, info in enumerate(infos):
                        tr = info.get('tokens_raw', [])
                        if not tr:
                            continue
                        try:
                            subtokens = [(SLOT_NAMES[j % len(SLOT_NAMES)], v)
                                         for j, v in enumerate(tr)]
                            recon_atoms, _ = reconstruct_from_tokens(subtokens)
                            if len(recon_atoms) < 1:
                                continue
                            z = [a.atomic_num for a in recon_atoms]
                            pos = np.array([a.pos for a in recon_atoms])
                            molecules.append((z, pos, info.get('smiles')))
                            loaded += 1
                        except Exception:
                            pass
                except ImportError:
                    print("    ERROR: adt_tokenizer not importable")
        elif coords_list and atoms_list:
            for i in range(min(len(coords_list), len(atoms_list))):
                pos = coords_list[i]
                z = atoms_list[i]
                if pos is not None and z is not None:
                    smi = smiles_list[i] if i < len(smiles_list) else None
                    molecules.append((list(z), np.asarray(pos, dtype=np.float64), smi))
                    loaded += 1
        print(f"    {loaded} molecules loaded")
    return molecules


def load_qm9_canonical(pyg_root='/tmp/qm9_pyg'):
    cache_path = 'qm9_canonical_cache.pt'
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=False)
        if isinstance(cached, dict):
            cached = cached.get('canonical', cached)
        result = set(cached) if not isinstance(cached, set) else cached
        print(f"  QM9 cache: {len(result)} SMILES")
        return result
    try:
        from torch_geometric.datasets import QM9 as QM9Dataset
        dataset = QM9Dataset(root=pyg_root)
        qm9_set = set()
        for data in dataset:
            try:
                z = data.z.numpy()
                pos = data.pos.numpy()
                rw = Chem.RWMol()
                for zi in z:
                    rw.AddAtom(Chem.Atom(int(zi)))
                conf = Chem.Conformer(len(z))
                for j in range(len(z)):
                    conf.SetAtomPosition(j, pos[j].tolist())
                rw.AddConformer(conf, assignId=True)
                rdDetermineBonds.DetermineConnectivity(rw)
                rdDetermineBonds.DetermineBondOrders(rw)
                mol = rw.GetMol()
                Chem.SanitizeMol(mol)
                qm9_set.add(Chem.MolToSmiles(Chem.RemoveAllHs(mol)))
            except Exception:
                pass
        torch.save(list(qm9_set), cache_path)
        print(f"  QM9 from PyG: {len(qm9_set)} SMILES")
        return qm9_set
    except ImportError:
        for name in ['qm9_tokens_v3.pt', 'qm9_tokens.pt']:
            if os.path.exists(name):
                d = torch.load(name, weights_only=False)
                qm9_set = set()
                for s in d.get('smiles', []):
                    try:
                        mol = Chem.MolFromSmiles(s)
                        if mol:
                            qm9_set.add(Chem.MolToSmiles(Chem.RemoveAllHs(mol)))
                    except Exception:
                        pass
                print(f"  QM9 from {name}: {len(qm9_set)} SMILES")
                return qm9_set
        print("  WARNING: No QM9 reference found")
        return set()


# ============================================================
# Subsampling & accumulation
# ============================================================

def compute_accumulation(stable_smiles_list, qm9_smiles, n_points=25):
    """Compute species accumulation at logarithmically spaced K values."""
    N = len(stable_smiles_list)
    if N == 0:
        return []

    # Log-spaced sample points
    K_values = np.unique(np.geomspace(100, N, n_points).astype(int))
    K_values = np.append(K_values, N)
    K_values = np.unique(np.clip(K_values, 1, N))

    results = []
    seen = set()
    known_set = set()
    novel_set = set()
    prev_K = 0

    for K in sorted(K_values):
        # Incrementally add SMILES from prev_K to K
        for smi in stable_smiles_list[prev_K:K]:
            if smi not in seen:
                seen.add(smi)
                if smi in qm9_smiles:
                    known_set.add(smi)
                else:
                    novel_set.add(smi)
        prev_K = K

        unique = len(seen)
        known = len(known_set)
        novel = len(novel_set)
        collisions = K - unique

        results.append({
            'K': int(K),
            'unique': unique,
            'known': known,
            'novel': novel,
            'collisions': collisions,
        })
        print(f"  K={K:>9,}  unique={unique:>9,}  "
              f"known={known:>7,}  novel={novel:>9,}  "
              f"dup={collisions:>7,}  nov/uniq={novel/max(unique,1)*100:.1f}%")

    return results


# ============================================================
# Fitting & plotting
# ============================================================

def fit_and_plot(results, n_qm9, alpha, output_prefix='accum'):
    from scipy.optimize import curve_fit
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    K_arr = np.array([r['K'] for r in results])
    known_arr = np.array([r['known'] for r in results])
    novel_arr = np.array([r['novel'] for r in results])
    unique_arr = np.array([r['unique'] for r in results])

    # ---- Fit QM9 (known): 2-component ----
    def model_qm9(K, N1, p):
        N2 = n_qm9 - N1
        return N1 * (1.0 - np.exp(-K * p)) + N2 * (1.0 - np.exp(-K * p * alpha))

    try:
        popt_q, _ = curve_fit(model_qm9, K_arr, known_arr,
                              p0=[30000, 1.5e-5],
                              bounds=([1, 1e-9], [n_qm9-1, 1e-2]))
        N1, p_q = popt_q
        N2 = n_qm9 - N1
        K1 = 1.0 / p_q
        K2 = 1.0 / (p_q * alpha)
        rmse_q = np.sqrt(np.mean((model_qm9(K_arr, *popt_q) - known_arr)**2))
        print(f"\n  QM9 fit (α={alpha}):")
        print(f"    N1={N1:,.0f} (fast), N2={N2:,.0f} (slow)")
        print(f"    K*_1={K1:,.0f}, K*_2={K2:,.0f}")
        print(f"    RMSE={rmse_q:.0f}")
        qm9_fit_ok = True
    except Exception as e:
        print(f"\n  QM9 fit failed: {e}")
        qm9_fit_ok = False

    # ---- Fit Novel: 3-param (saturating + linear) ----
    def model_novel(K, M_sat, q, c):
        return M_sat * (1.0 - np.exp(-K * q)) + c * K

    try:
        popt_n, _ = curve_fit(model_novel, K_arr, novel_arr,
                              p0=[50000, 5e-6, 0.3],
                              bounds=([0, 1e-9, 0], [1e8, 1e-2, 100]))
        M_sat, q_n, c_n = popt_n
        rmse_n = np.sqrt(np.mean((model_novel(K_arr, *popt_n) - novel_arr)**2))
        print(f"\n  Novel fit (3-param):")
        print(f"    M_sat={M_sat:,.0f}, K*={1/q_n:,.0f}, c={c_n:.4f}")
        print(f"    RMSE={rmse_n:.0f}")
        novel_fit_ok = True
    except Exception as e:
        print(f"\n  Novel fit failed: {e}")
        novel_fit_ok = False

    # ---- Novel: 1-exp reference ----
    def model_novel_1exp(K, N_eff, K_star):
        return N_eff * (1.0 - np.exp(-K / K_star))

    try:
        popt_n1, _ = curve_fit(model_novel_1exp, K_arr, novel_arr,
                               p0=[200000, 300000])
        rmse_n1 = np.sqrt(np.mean((model_novel_1exp(K_arr, *popt_n1) - novel_arr)**2))
        novel_1exp_ok = True
    except Exception:
        novel_1exp_ok = False

    # ---- Predictions ----
    print(f"\n  --- Predictions (α={alpha}) ---")
    print(f"  {'K':>9s}  {'Known':>8s}  {'%QM9':>6s}  {'Novel':>9s}  {'nov/uniq':>8s}")
    print(f"  {'-'*50}")
    K_max = K_arr[-1]
    for Kp in [K_max, 500_000, 1_000_000, 1_440_000, 2_000_000]:
        if qm9_fit_ok:
            kn = model_qm9(Kp, *popt_q)
        else:
            kn = float('nan')
        if novel_fit_ok:
            nv = model_novel(Kp, *popt_n)
        else:
            nv = float('nan')
        u = kn + nv
        print(f"  {Kp:9,}  {kn:8,.0f}  {kn/n_qm9*100:5.1f}%  {nv:9,.0f}  "
              f"{nv/max(u,1)*100:7.1f}%")

    # ---- Plot ----
    K_plot = np.linspace(1, max(K_max * 2, 2_000_000), 2000)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10),
                              gridspec_kw={'height_ratios': [3, 1.2]})

    ax = axes[0]

    # Data
    ax.plot(K_arr, known_arr, 'go', markersize=7, zorder=5, label='Data: Known (QM9)')
    ax.plot(K_arr, novel_arr, 'rs', markersize=7, zorder=5, label='Data: Novel (non-QM9)')

    # QM9 fit
    if qm9_fit_ok:
        y_q = model_qm9(K_plot, *popt_q)
        ax.plot(K_plot, y_q, 'g-', linewidth=2.5,
                label=f'Known (α={alpha}): $N_1$={N1:,.0f}, RMSE={rmse_q:.0f}')

    # Novel fits
    if novel_fit_ok:
        y_n = model_novel(K_plot, *popt_n)
        ax.plot(K_plot, y_n, 'r-', linewidth=2.5,
                label=f'Novel 3p: $M_{{sat}}$={M_sat:,.0f} + {c_n:.3f}$K$, RMSE={rmse_n:.0f}')

    if novel_1exp_ok:
        y_n1 = model_novel_1exp(K_plot, *popt_n1)
        ax.plot(K_plot, y_n1, 'b--', linewidth=2,
                label=f'Novel 1-exp: $N_{{eff}}$={popt_n1[0]:,.0f}, RMSE={rmse_n1:.0f}')

    ax.axhline(y=n_qm9, color='green', linestyle=':', alpha=0.4, linewidth=1.5,
               label=f'$N_{{QM9}} = {n_qm9:,}$')

    ymax = max(novel_arr[-1] * 2.5, n_qm9 * 1.5) if len(novel_arr) else n_qm9 * 2
    ax.set_ylabel('Molecules discovered', fontsize=14)
    ax.set_xlim(0, K_plot[-1])
    ax.set_ylim(0, ymax)
    ax.set_title(f'Species accumulation (stable-based, α={alpha})', fontsize=13)
    ax.legend(fontsize=9.5, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    # ---- Bottom: novel/unique ----
    ax3 = axes[1]
    ratio_data = novel_arr / np.maximum(unique_arr, 1) * 100
    ax3.plot(K_arr, ratio_data, 'mo', markersize=6, label='Data')

    if qm9_fit_ok and novel_fit_ok:
        y_u = model_qm9(K_plot, *popt_q) + model_novel(K_plot, *popt_n)
        y_nv = model_novel(K_plot, *popt_n)
        ratio_3p = np.where(y_u > 0, y_nv / y_u * 100, 50)
        ax3.plot(K_plot, ratio_3p, 'r-', linewidth=2, label='3-param model')

    if qm9_fit_ok and novel_1exp_ok:
        y_u1 = model_qm9(K_plot, *popt_q) + model_novel_1exp(K_plot, *popt_n1)
        y_nv1 = model_novel_1exp(K_plot, *popt_n1)
        ratio_1e = np.where(y_u1 > 0, y_nv1 / y_u1 * 100, 50)
        ax3.plot(K_plot, ratio_1e, 'b--', linewidth=2, label='1-exp model')

    ax3.set_xlabel('Stable samples', fontsize=14)
    ax3.set_ylabel('Novel / Unique  (%)', fontsize=14)
    ax3.set_xlim(0, K_plot[-1])
    ax3.set_ylim(max(ratio_data.min() - 5, 0), 100)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    ax3.tick_params(labelsize=12)

    plt.tight_layout()
    outfile = f'{output_prefix}.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\n  Saved {outfile}")

    # ---- Save data as .pt for reuse ----
    torch.save({
        'results': results,
        'n_qm9': n_qm9,
        'alpha': alpha,
        'qm9_fit': {'N1': float(N1), 'N2': float(N2), 'p': float(p_q),
                     'rmse': float(rmse_q)} if qm9_fit_ok else None,
        'novel_fit': {'M_sat': float(M_sat), 'q': float(q_n), 'c': float(c_n),
                      'rmse': float(rmse_n)} if novel_fit_ok else None,
    }, f'{output_prefix}_data.pt')
    print(f"  Saved {output_prefix}_data.pt")


# ============================================================
# Main
# ============================================================

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {' '.join(sys.argv)}")

    parser = argparse.ArgumentParser(
        description='Species accumulation analysis (stable-based)')
    parser.add_argument('inputs', nargs='+', help='Input .pt files')
    parser.add_argument('--pyg_root', type=str, default='/tmp/qm9_pyg')
    parser.add_argument('--collision_threshold', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=0.135,
                        help='Rate ratio for 2-component QM9 model')
    parser.add_argument('--n_points', type=int, default=25,
                        help='Number of subsample points')
    parser.add_argument('--output', type=str, default='accum',
                        help='Output prefix for plots and data')
    parser.add_argument('--max_molecules', type=int, default=0)
    args = parser.parse_args()

    col_thr = args.collision_threshold

    # ---- Load ----
    print("Loading molecules...")
    molecules = load_molecules_from_pt(args.inputs)
    n_loaded = len(molecules)
    print(f"  Total loaded: {n_loaded:,}")

    if n_loaded == 0:
        print("ERROR: No molecules loaded.")
        sys.exit(1)

    if args.max_molecules > 0:
        molecules = molecules[:args.max_molecules]
        n_loaded = len(molecules)

    # ---- QM9 ----
    print("Loading QM9 reference...")
    qm9_smiles = load_qm9_canonical(args.pyg_root)
    n_qm9 = len(qm9_smiles)

    # ---- Validate & collect stable SMILES in order ----
    print(f"\nValidating {n_loaded:,} molecules (collision_thr={col_thr} Å)...")
    stable_smiles_list = []   # ordered list, with duplicates
    n_clean = 0
    n_valid_3d = 0
    n_mol_stable = 0
    t0 = time.time()

    for idx, (z, pos, _smi) in enumerate(molecules):
        min_d = min_pairwise_dist(pos)
        if min_d < col_thr:
            continue
        n_clean += 1

        mol, smi, vinfo = validate_3D(z, pos)
        if not vinfo.get('valid', False):
            continue
        n_valid_3d += 1

        if vinfo.get('mol_stable', False):
            n_mol_stable += 1
            stable_smiles_list.append(smi)

        if (idx + 1) % 50000 == 0:
            elapsed = time.time() - t0
            print(f"  {idx+1:,}/{n_loaded:,}  clean={n_clean:,}  "
                  f"valid3D={n_valid_3d:,}  stable={n_mol_stable:,}  "
                  f"({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.1f}s ({n_loaded/max(elapsed,0.1):.0f} mol/s)")
    print(f"\n  loaded={n_loaded:,}  clean={n_clean:,}  "
          f"valid3D={n_valid_3d:,}  stable={n_mol_stable:,}")

    if n_mol_stable == 0:
        print("ERROR: No stable molecules found.")
        sys.exit(1)

    # ---- Accumulation ----
    print(f"\nComputing accumulation ({args.n_points} points)...")
    results = compute_accumulation(stable_smiles_list, qm9_smiles,
                                   n_points=args.n_points)

    # ---- SUMMARY (final point) ----
    r = results[-1]
    print(f"\n  SUMMARY: loaded={n_loaded:,} clean={n_clean:,} "
          f"valid3D={n_valid_3d:,} stable={n_mol_stable:,} "
          f"unique={r['unique']:,} novel={r['novel']:,} qm9={r['known']:,}")

    # ---- Frequency distribution ----
    print("\nFrequency distribution (how many times each molecule appears)...")
    from collections import Counter
    smi_counts = Counter(stable_smiles_list)

    # Build freq -> (qm9_count, novel_count)
    freq_qm9 = Counter()
    freq_novel = Counter()
    for smi, cnt in smi_counts.items():
        if smi in qm9_smiles:
            freq_qm9[cnt] += 1
        else:
            freq_novel[cnt] += 1

    all_freqs = sorted(set(list(freq_qm9.keys()) + list(freq_novel.keys())), reverse=True)
    max_freq = all_freqs[0] if all_freqs else 0

    print(f"\n  {'freq':>6s}  {'QM9':>8s}  {'Novel':>8s}  {'Total':>8s}  {'cum_QM9':>8s}  {'cum_Nov':>8s}  {'cum_Tot':>8s}")
    print(f"  {'-'*62}")

    cum_q, cum_n = 0, 0
    for f in all_freqs:
        nq = freq_qm9.get(f, 0)
        nn = freq_novel.get(f, 0)
        cum_q += nq
        cum_n += nn
        print(f"  {f:>6d}  {nq:>8,}  {nn:>8,}  {nq+nn:>8,}  {cum_q:>8,}  {cum_n:>8,}  {cum_q+cum_n:>8,}")

    # Summary stats
    total_unique = len(smi_counts)
    total_qm9 = sum(freq_qm9.values())
    total_novel = sum(freq_novel.values())
    mean_freq = np.mean(list(smi_counts.values()))
    median_freq = np.median(list(smi_counts.values()))

    # Top-N most frequent
    print(f"\n  Unique: {total_unique:,} (QM9={total_qm9:,}, Novel={total_novel:,})")
    print(f"  Mean freq: {mean_freq:.2f}, Median freq: {median_freq:.0f}, Max freq: {max_freq}")
    print(f"  Singletons (freq=1): QM9={freq_qm9.get(1,0):,}  Novel={freq_novel.get(1,0):,}  Total={freq_qm9.get(1,0)+freq_novel.get(1,0):,}")

    top_n = 20
    print(f"\n  Top {top_n} most frequent molecules:")
    print(f"  {'rank':>4s}  {'freq':>6s}  {'type':>5s}  SMILES")
    print(f"  {'-'*60}")
    for rank, (smi, cnt) in enumerate(smi_counts.most_common(top_n), 1):
        tag = "QM9" if smi in qm9_smiles else "Novel"
        print(f"  {rank:>4d}  {cnt:>6,}  {tag:>5s}  {smi}")

    # ---- Fit & Plot ----
    print("\nFitting models...")
    fit_and_plot(results, n_qm9, args.alpha, output_prefix=args.output)


if __name__ == '__main__':
    main()
