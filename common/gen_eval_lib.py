"""Shared gen_eval helpers for QM9 and Drugs ADT pipelines.

Provides:
  - extract_metrics(mol): InChIKey-skeleton + topology SMILES
  - try_write_sdf(mol, kek_file, arom_file): try kekulize, fall back to aromatic
  - xtb_relax(idx, sdf_str, xtb_workdir, xtb_bin): run xTB GFN2 on one mol
  - consistency_check(pre_inchi, pre_topo, mol_post): compare pre/post SMILES
"""
import os, json, subprocess
import numpy as np
import re
from rdkit import Chem


def extract_metrics(mol):
    """Compute (inchikey_skeleton, topology_smiles) for a sanitized mol.
    Returns ('', '') on failure."""
    inchi_skel, topo = '', ''
    try:
        inchi_skel = Chem.MolToInchiKey(mol).split('-')[0]
    except Exception:
        pass
    try:
        topo = Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        pass
    return inchi_skel, topo


def try_write_sdf(mol, sdf_kek_f, sdf_arom_f, stats):
    """Try kekulized SDF, fall back to aromatic. Updates stats counter.
    Returns (sdf_str, sdf_tag) where tag in {'kekulized', 'aromatic', 'failed'}."""
    try:
        sdf = Chem.MolToMolBlock(mol)
        if sdf_kek_f:
            sdf_kek_f.write(sdf); sdf_kek_f.write('\n$$$$\n'); sdf_kek_f.flush()
        if stats is not None:
            stats['sdf_kekulized'] = stats.get('sdf_kekulized', 0) + 1
        return sdf, 'kekulized'
    except Exception:
        pass
    try:
        sdf = Chem.MolToMolBlock(mol, kekulize=False)
        if sdf_arom_f:
            sdf_arom_f.write(sdf); sdf_arom_f.write('\n$$$$\n'); sdf_arom_f.flush()
        if stats is not None:
            stats['sdf_aromatic'] = stats.get('sdf_aromatic', 0) + 1
        return sdf, 'aromatic'
    except Exception:
        if stats is not None:
            stats['sdf_failed'] = stats.get('sdf_failed', 0) + 1
        return None, 'failed'


_SYMBOL_TO_Z = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'P':15, 'S':16, 'Cl':17, 'Br':35, 'I':53}


def parse_xtbopt_xyz(opt_xyz_path):
    """Parse xtbopt.xyz; return (heavy_coords, heavy_anums) or (None, None)."""
    try:
        with open(opt_xyz_path) as fo:
            lines = fo.readlines()
        n_at = int(lines[0].strip())
        heavy_coords, heavy_anums = [], []
        for j in range(2, n_at + 2):
            tok = lines[j].split()
            if tok[0] != 'H':
                heavy_coords.append([float(tok[1]), float(tok[2]), float(tok[3])])
                heavy_anums.append(_SYMBOL_TO_Z.get(tok[0], 0))
        return heavy_coords, heavy_anums
    except Exception:
        return None, None


def xtb_relax(idx, mol_block, init_heavy_coords, xtb_workdir, xtb_bin, charge=0, timeout=300):
    """Run xTB GFN2 optimization. Returns dict with ok, e_gain, rmsd, rmsd_heavy.
    init_heavy_coords: list of [x,y,z] for heavy atoms (for RMSD against optimized)."""
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False)
    if mol is None:
        return {'ok': False, 'reason': 'mol_parse_fail'}
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    mol_h = Chem.AddHs(mol, addCoords=True)

    xyz_path = os.path.join(xtb_workdir, f'mol_{idx}.xyz')
    with open(xyz_path, 'w') as f:
        f.write(f'{mol_h.GetNumAtoms()}\n\n')
        conf = mol_h.GetConformer()
        for a in mol_h.GetAtoms():
            p = conf.GetAtomPosition(a.GetIdx())
            f.write(f'{a.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}\n')

    namespace = f'mol_{idx}'
    try:
        result = subprocess.run(
            [xtb_bin, os.path.abspath(xyz_path), '--opt', '--charge', str(charge),
             '--namespace', namespace],
            capture_output=True, text=True, timeout=timeout, cwd=xtb_workdir)
        e_match = re.search(r'total energy gain.*?(-?[\d.]+)\s+kcal/mol', result.stdout)
        r_match = re.search(r'total RMSD.*?([\d.]+)\s', result.stdout)
        if not (e_match and r_match):
            return {'ok': False, 'reason': 'no_parse'}

        e_gain = float(e_match.group(1))
        rmsd_all = float(r_match.group(1))

        # Compute heavy-atom RMSD vs initial
        rmsd_heavy = None
        opt_xyz = os.path.join(xtb_workdir, f'{namespace}.xtbopt.xyz')
        opt_heavy_coords, opt_heavy_anums = parse_xtbopt_xyz(opt_xyz)
        if opt_heavy_coords and len(opt_heavy_coords) == len(init_heavy_coords):
            diff = np.array(opt_heavy_coords) - np.array(init_heavy_coords)
            rmsd_heavy = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

        return {
            'ok': True, 'e_gain': e_gain, 'rmsd': rmsd_all, 'rmsd_heavy': rmsd_heavy,
            'opt_xyz': opt_xyz, 'opt_heavy_coords': opt_heavy_coords,
            'opt_heavy_anums': opt_heavy_anums,
        }
    except Exception as ex:
        return {'ok': False, 'reason': str(ex)[:80]}


def consistency_check(inchi_pre, topo_pre, opt_heavy_anums, opt_heavy_coords, validate_3D_fn):
    """Re-perceive mol from xTB-optimized heavy atoms, compare to pre.
    Returns dict with same_inchi, same_topo, inchi_post, topo_post, same (=either)."""
    if not opt_heavy_coords or not opt_heavy_anums:
        return {'inchi_post': '', 'topo_post': '', 'same_inchi': False,
                'same_topo': False, 'same': False}
    mol_post, _, _ = validate_3D_fn(opt_heavy_anums, opt_heavy_coords)
    inchi_post, topo_post = '', ''
    if mol_post is not None:
        inchi_post, topo_post = extract_metrics(mol_post)
    same_inchi = (inchi_pre == inchi_post and inchi_pre != '')
    same_topo = (topo_pre == topo_post and topo_pre != '')
    return {
        'inchi_post': inchi_post, 'topo_post': topo_post,
        'same_inchi': same_inchi, 'same_topo': same_topo,
        'same': same_inchi or same_topo,
    }


def print_summary(stats, xtb_results, n_topo_same, n_inchi_same, n_either_same, total):
    """Print consistency summary. total = ok xtb count."""
    print(f"\n=== Consistency (xTB relaxed → re-perceived) ===")
    print(f"  Topology preserved: {n_topo_same}/{total} ({n_topo_same/max(total,1)*100:.1f}%)")
    print(f"  InChIKey preserved: {n_inchi_same}/{total} ({n_inchi_same/max(total,1)*100:.1f}%)")
    print(f"  Either preserved:   {n_either_same}/{total} ({n_either_same/max(total,1)*100:.1f}%)")
