"""Load (or build) GEOM-Drugs molecules for emb_offset training.

load_drugs_mols() is LOAD-OR-BUILD:
  * cache .pkl exists      -> load it (fast path; this is what training uses).
  * else                   -> build it from the raw GEOM-Drugs rdkit_folder. If the
                              raw folder is absent, DOWNLOAD it from the GEOM Harvard
                              Dataverse first (~7GB tar), extract, then build. So a
                              fresh machine can regenerate the dataset from scratch.

GEOM: Axelrod & Gomez-Bombarelli, "GEOM, energy-annotated molecular conformations",
doi:10.7910/DVN/JNGTDF. rdkit_folder.tar.gz -> rdkit_folder/drugs/<hash>.pickle, each
a dict with ['conformers'][0]['rd_mol'] (an RDKit mol carrying a 3D conformer).

Run directly to build the standard max30 cache (downloads GEOM if needed):
    python3 load_drugs.py
"""
import os, sys, pickle, tarfile, urllib.request
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

_HERE = os.path.dirname(os.path.abspath(__file__))                      # = Drugs/data/geom
GEOM_URL = "https://dataverse.harvard.edu/api/access/datafile/4327252"  # GEOM rdkit_folder.tar.gz (doi:10.7910/DVN/JNGTDF)
DEFAULT_GEOM_DIR = os.path.join(_HERE, "geom_drugs", "rdkit_folder", "drugs")
DEFAULT_CACHE = os.path.join(_HERE, "drugs_mols_v26_max30.pkl")


def download_geom(dest_root=None):
    """Download + extract the GEOM rdkit_folder; return the drugs/ dir of per-mol pickles."""
    dest_root = dest_root or os.path.join(_HERE, "geom_drugs")
    for cand in (os.path.join(dest_root, "rdkit_folder", "drugs"), os.path.join(dest_root, "drugs")):
        if os.path.isdir(cand):
            return cand
    os.makedirs(dest_root, exist_ok=True)
    tarpath = os.path.join(dest_root, "rdkit_folder.tar.gz")
    if not os.path.exists(tarpath):
        print(f"Downloading GEOM rdkit_folder (~7GB) from {GEOM_URL}\n  -> {tarpath}")
        urllib.request.urlretrieve(GEOM_URL, tarpath)
    print(f"Extracting {tarpath} (large) ...")
    with tarfile.open(tarpath, "r:gz") as t:
        t.extractall(dest_root)
    for cand in (os.path.join(dest_root, "rdkit_folder", "drugs"), os.path.join(dest_root, "drugs")):
        if os.path.isdir(cand):
            return cand
    raise RuntimeError(f"extracted GEOM but no drugs/ under {dest_root} -- check archive layout")


def load_drugs_mols(geom_dir=None, nohydrogen=True, max_atoms=30, require_benzene=False,
                    cache_path=DEFAULT_CACHE):
    """Load GEOM-Drugs, filter, return list of (mol, positions, smiles).
    Builds (downloading raw GEOM if needed) when the cache is absent."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached: {cache_path}")
        with open(cache_path, 'rb') as f:
            mols = pickle.load(f)
        print(f"  {len(mols)} molecules")
        return mols

    geom_dir = geom_dir or DEFAULT_GEOM_DIR
    if not os.path.isdir(geom_dir):
        print(f"raw GEOM not at {geom_dir} -> downloading from Harvard Dataverse")
        geom_dir = download_geom()

    files = sorted(os.listdir(geom_dir))
    print(f"Processing {len(files)} GEOM-Drugs files...")
    mols = []
    seen = set()

    for fi, fname in enumerate(files):
        try:
            with open(os.path.join(geom_dir, fname), 'rb') as f:
                data = pickle.load(f)
            confs = data.get('conformers', [])
            if not confs: continue
            mol = confs[0].get('rd_mol')
            if mol is None: continue

            if nohydrogen:
                mol_proc = Chem.RemoveAllHs(mol)
            else:
                mol_proc = mol

            smi = Chem.MolToSmiles(mol_proc)
            if smi in seen: continue
            seen.add(smi)

            na = mol_proc.GetNumAtoms()
            if na > max_atoms or na < 3: continue

            if require_benzene:
                ri = mol_proc.GetRingInfo()
                has_benz = False
                for ring in ri.AtomRings():
                    if len(ring) == 6:
                        if all(mol_proc.GetAtomWithIdx(a).GetIsAromatic() and
                               mol_proc.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in ring):
                            has_benz = True; break
                if not has_benz: continue

            # 3D coordinates (heavy atoms only when nohydrogen)
            conf = mol.GetConformer()
            all_pos = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            if nohydrogen:
                heavy_idx = [i for i in range(mol.GetNumAtoms())
                            if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
                pos = all_pos[heavy_idx]
            else:
                pos = all_pos

            mols.append((mol_proc, pos, smi))
        except Exception:
            continue

        if (fi + 1) % 50000 == 0:
            print(f"  {fi+1}/{len(files)}: {len(mols)} molecules")

    print(f"Total: {len(mols)} molecules (max_atoms={max_atoms}, benzene={require_benzene})")

    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(mols, f)
        print(f"Saved to {cache_path}")
    return mols


def find_benzene_atoms(mol):
    """Indices of atoms in benzene rings (aromatic 6-ring of C)."""
    ri = mol.GetRingInfo()
    benz_atoms = set()
    for ring in ri.AtomRings():
        if len(ring) == 6:
            if all(mol.GetAtomWithIdx(a).GetIsAromatic() and
                   mol.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in ring):
                benz_atoms.update(ring)
    return list(benz_atoms)


if __name__ == '__main__':
    # Build the standard max30 cache (downloads GEOM-Drugs if the raw folder is absent).
    mols = load_drugs_mols(max_atoms=30, require_benzene=False, cache_path=DEFAULT_CACHE)
    from collections import Counter
    sizes = Counter(m[0].GetNumAtoms() for m in mols)
    print("\nAtom-count distribution:")
    for s in sorted(sizes):
        print(f"  {s:3d}: {sizes[s]:6d}")
