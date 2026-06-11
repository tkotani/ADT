"""Load GEOM-Drugs molecules for v21 training."""
import os, pickle, numpy as np
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

def load_drugs_mols(geom_dir='/home/takao/ADT/Drugs/geom_drugs/drugs/',
                    nohydrogen=True, max_atoms=30, require_benzene=True,
                    cache_path='drugs_mols_v21.pkl'):
    """Load GEOM-Drugs, filter, return list of (mol, positions, smiles)."""
    if os.path.exists(cache_path):
        print(f"Loading cached: {cache_path}")
        with open(cache_path, 'rb') as f:
            mols = pickle.load(f)
        print(f"  {len(mols)} molecules")
        return mols

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

            # Get 3D coordinates
            conf = mol.GetConformer()
            all_pos = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            if nohydrogen:
                heavy_idx = [i for i in range(mol.GetNumAtoms())
                            if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
                pos = all_pos[heavy_idx]
            else:
                pos = all_pos

            mols.append((mol_proc, pos, smi))
        except:
            continue

        if (fi + 1) % 50000 == 0:
            print(f"  {fi+1}/{len(files)}: {len(mols)} molecules")

    print(f"Total: {len(mols)} molecules (max_atoms={max_atoms}, benzene={require_benzene})")

    with open(cache_path, 'wb') as f:
        pickle.dump(mols, f)
    print(f"Saved to {cache_path}")
    return mols


def find_benzene_atoms(mol):
    """Find indices of atoms in benzene rings (aromatic 6-ring of C)."""
    ri = mol.GetRingInfo()
    benz_atoms = set()
    for ring in ri.AtomRings():
        if len(ring) == 6:
            if all(mol.GetAtomWithIdx(a).GetIsAromatic() and
                   mol.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in ring):
                benz_atoms.update(ring)
    return list(benz_atoms)


if __name__ == '__main__':
    mols = load_drugs_mols()
    print(f"\nAtom count distribution:")
    from collections import Counter
    sizes = Counter(m[0].GetNumAtoms() for m in mols)
    for s in sorted(sizes.keys()):
        print(f"  {s:3d}: {sizes[s]:6d}")
