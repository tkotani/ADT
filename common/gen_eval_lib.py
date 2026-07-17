"""Shared gen_eval helpers for QM9 and Drugs ADT pipelines.

Provides:
  - extract_metrics(mol): InChIKey-skeleton + topology SMILES
  - try_write_sdf(mol, kek_file, arom_file): try kekulize, fall back to aromatic
  - xtb_relax(idx, sdf_str, xtb_workdir, xtb_bin): run xTB GFN2 on one mol
  - consistency_check(pre_inchi, pre_topo, mol_post): compare pre/post SMILES
"""
import os, json, subprocess, glob
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


HARTREE2KCAL = 627.5094740631


def _xtb_energy_kcal(opt_xyz_path):
    """Parse the converged total energy (Hartree) from an xtbopt.xyz comment line -> kcal/mol (or None)."""
    try:
        with open(opt_xyz_path) as f:
            comment = f.readlines()[1]
        m = re.search(r'energy:\s*(-?[\d.]+)', comment)
        return float(m.group(1)) * HARTREE2KCAL if m else None
    except Exception:
        return None


def xtb_relax(idx, mol_block, init_heavy_coords, xtb_workdir, xtb_bin, charge=0, timeout=300):
    """Run xTB GFN2 optimization. Returns dict with ok, e_gain, e_full(kcal/mol), rmsd, rmsd_heavy.
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
            'ok': True, 'e_gain': e_gain, 'e_full': _xtb_energy_kcal(opt_xyz),
            'rmsd': rmsd_all, 'rmsd_heavy': rmsd_heavy,
            'opt_xyz': opt_xyz, 'opt_heavy_coords': opt_heavy_coords,
            'opt_heavy_anums': opt_heavy_anums,
        }
    except Exception as ex:
        return {'ok': False, 'reason': str(ex)[:80]}


def xtb_hrelax(idx, mol_block, n_heavy, xtb_workdir, xtb_bin, charge=0, timeout=300):
    """H-only xTB GFN2 optimization: freeze heavy atoms (1..n_heavy) via '$fix atoms:', relax only H.
    Returns (all-atom molblock with H-relaxed coords, E_hprerelax in kcal/mol); (None, None) on failure.
    Purpose: remove crude-H-placement contamination from the estrain reward so the subsequent full
    relaxation's energy gain reflects the HEAVY geometry alone (H already at its optimum)."""
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False)
    if mol is None:
        if os.environ.get("HPRE_DEBUG"):
            with open(os.path.join(xtb_workdir, "hpre_why.log"), "a") as _f:
                _f.write(f"idx={idx} PARSE_FAIL n_heavy={n_heavy} mb_head={mol_block.splitlines()[3][:60] if len(mol_block.splitlines())>3 else '?'}\n")
        return None, None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    n_all = mol.GetNumAtoms()
    if n_all <= n_heavy:                                             # no H present -> nothing to relax
        return mol_block, None
    conf = mol.GetConformer()
    xyz_path = os.path.join(xtb_workdir, f'hpre_{idx}.xyz')
    with open(xyz_path, 'w') as f:
        f.write(f'{n_all}\n\n')
        for a in mol.GetAtoms():
            p = conf.GetAtomPosition(a.GetIdx())
            f.write(f'{a.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}\n')
    ctrl_path = os.path.join(xtb_workdir, f'hpre_{idx}.inp')
    with open(ctrl_path, 'w') as f:                                  # freeze heavy (1..n_heavy), relax only H.
        f.write(f'$fix\n   atoms: 1-{n_heavy}\n$end\n$opt\n   engine=lbfgs\n   maxcycle=500\n$end\n')  # lbfgs REQUIRED: default rf/ANCOPT ignores $fix (internal coords). maxcycle: big molecules need more.
    ns = f'hpre_{idx}'
    for _pat in (f'{ns}.xtb*', f'{ns}.NOT_CONVERGED', f'{ns}.charges', f'{ns}.wbo'):
        for _stale in glob.glob(os.path.join(xtb_workdir, _pat)):     # purge xTB OUTPUTS only (NOT the .xyz
            try:                                                      # input we just wrote): the workdir is
                os.remove(_stale)                                     # reused and ns = hpre_<idx-in-batch>,
            except OSError:                                           # so a run that writes nothing would
                pass                                                  # otherwise inherit the PREVIOUS
                                                                      # molecule's geometry files.
    try:
        _r = subprocess.run(
            [xtb_bin, os.path.abspath(xyz_path), '--opt', '--input', os.path.abspath(ctrl_path),
             '--charge', str(charge), '--namespace', ns],
            capture_output=True, text=True, timeout=timeout, cwd=xtb_workdir)
        _dbg = os.environ.get("HPRE_DEBUG")
        opt_xyz = os.path.join(xtb_workdir, f'{ns}.xtbopt.xyz')
        if not os.path.exists(opt_xyz):
            # NOT_CONVERGED: xTB writes no xtbopt.xyz but DOES leave the last geometry. Dropping the
            # molecule here was counting a PRE-PROCESSING non-convergence as a chemistry failure -- and it
            # hits big molecules hardest (~30% at 38+ heavy atoms), i.e. it manufactured a large part of the
            # "size cliff". The H-only relax just needs H off their placement artefacts (heavy is frozen),
            # so the last LBFGS geometry is fine; the FULL relax that follows still decides XTP.
            last_xyz = os.path.join(xtb_workdir, f'{ns}.xtblast.xyz')
            if not os.path.exists(last_xyz):
                if _dbg:
                    with open(os.path.join(xtb_workdir, "hpre_fail.log"), "a") as _f:
                        _f.write(f"=== {ns} n_all={n_all} n_heavy={n_heavy} rc={_r.returncode}\n"
                                 f"--- stdout tail:\n" + "\n".join(_r.stdout.splitlines()[-12:]) + "\n"
                                 f"--- stderr tail:\n" + "\n".join(_r.stderr.splitlines()[-6:]) + "\n")
                return None, None
            opt_xyz = last_xyz
            if os.environ.get("HPRE_DEBUG"):
                with open(os.path.join(xtb_workdir, "hpre_why.log"), "a") as _f:
                    _f.write(f"idx={idx} USED_XTBLAST (not converged) n_all={n_all}\n")
        with open(opt_xyz) as f:
            lines = f.read().splitlines()
        if int(lines[0].split()[0]) != n_all:
            if os.environ.get("HPRE_DEBUG"):
                with open(os.path.join(xtb_workdir, "hpre_why.log"), "a") as _f:
                    _f.write(f"idx={idx} NATOM_MISMATCH xyz={lines[0].split()[0]} expected={n_all}\n")
            return None, None
        for i in range(n_all):                                       # write H-relaxed coords back (same order)
            parts = lines[2 + i].split()
            conf.SetAtomPosition(i, [float(parts[1]), float(parts[2]), float(parts[3])])
        return Chem.MolToMolBlock(mol), _xtb_energy_kcal(opt_xyz)
    except Exception as _e:
        if os.environ.get("HPRE_DEBUG"):
            with open(os.path.join(xtb_workdir, "hpre_exc.log"), "a") as _f:
                _f.write(f"{ns} n_all={n_all} n_heavy={n_heavy} EXC={type(_e).__name__}: {str(_e)[:120]}\n")
        return None, None


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
