"""
adt_dataset.py — Bootstrap Canonical Encoding (v3)

Changes from v2:
  - No random_rotation (invariance is built into tokenization)
  - DynamicQM9Dataset retries on linear molecules or failures
  - Action types: 6 (ADD_INIT=0, ADD_CHAIN=1, ADD_ANGLE=2, ADD=3, LINK=4, END=5)
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Action constants
ADD_INIT  = 0
ADD_CHAIN = 1
ADD_ANGLE = 2
ADD       = 3
LINK      = 4
END       = 5
PAD_VALUE = -1
N_SLOTS = 7
N_FRAME_STEPS = 3  # INIT + CHAIN + ANGLE = frame (first 3 atoms)


# ============================================================
# Static dataset
# ============================================================

class QM9TokenDataset(Dataset):
    def __init__(self, pt_path, max_len=None, split='train',
                 n_val=2000, seed=42):
        data = torch.load(pt_path, weights_only=False)
        tokens_all = data['tokens']
        self.metadata = data.get('metadata', {})
        n_total = len(tokens_all)

        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_total)
        n_val = min(n_val, n_total)
        n_train = n_total - n_val

        if split == 'train':
            idx = indices[:n_train]
        elif split == 'val':
            idx = indices[n_train:n_train + n_val]
        else:  # 'all'
            idx = indices

        self.tokens = [tokens_all[i] for i in idx]
        self.smiles = [data['smiles'][i] for i in idx]

        if max_len is not None:
            keep = [i for i, t in enumerate(self.tokens) if len(t) <= max_len]
            self.tokens = [self.tokens[i] for i in keep]
            self.smiles = [self.smiles[i] for i in keep]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx], dtype=torch.long)


# ============================================================
# Dynamic dataset (tokenize on-the-fly with random root)
# ============================================================

def mol_from_pyg(z, pos):
    """Build RDKit Mol directly from atomic numbers and 3D positions.

    Uses rdDetermineBonds to infer connectivity from geometry.
    Guarantees atom order matches pos order (no SMILES reordering).
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    n = len(z)
    rw = Chem.RWMol()
    for zi in z:
        rw.AddAtom(Chem.Atom(int(zi)))
    conf = Chem.Conformer(n)
    for j in range(n):
        conf.SetAtomPosition(j, pos[j].tolist())
    rw.AddConformer(conf, assignId=True)
    rdDetermineBonds.DetermineConnectivity(rw)
    rdDetermineBonds.DetermineBondOrders(rw)
    return rw.GetMol()


def remove_hydrogens(mol, pos):
    """Remove hydrogen atoms from mol and pos array.

    Returns (mol_heavy, pos_heavy) with only heavy atoms,
    or (None, None) if fewer than 3 heavy atoms remain.
    """
    from rdkit import Chem
    heavy_idx = [i for i in range(mol.GetNumAtoms())
                 if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]
    if len(heavy_idx) < 3:
        return None, None

    # Build heavy-atom-only molecule using rdDetermineBonds
    pos_heavy = pos[heavy_idx]
    z_heavy = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in heavy_idx]

    rw = Chem.RWMol()
    for zi in z_heavy:
        rw.AddAtom(Chem.Atom(int(zi)))
    conf = Chem.Conformer(len(z_heavy))
    for j in range(len(z_heavy)):
        conf.SetAtomPosition(j, pos_heavy[j].tolist())
    rw.AddConformer(conf, assignId=True)

    from rdkit.Chem import rdDetermineBonds
    rdDetermineBonds.DetermineConnectivity(rw)
    try:
        rdDetermineBonds.DetermineBondOrders(rw)
    except Exception:
        pass  # bond orders are not critical for tokenization

    return rw.GetMol(), pos_heavy


def load_qm9_mols(root_dir='/tmp/qm9_pyg', nohydrogen=False):
    """Load QM9 from PyG -> list of (RDKit Mol, positions, smiles)."""
    import pickle
    suffix = '_noh' if nohydrogen else ''
    cache_path = f'qm9_mols_cache_v3b{suffix}.pkl'
    if os.path.exists(cache_path):
        print(f"Loading cached molecules from {cache_path}...")
        with open(cache_path, 'rb') as f:
            molecules = pickle.load(f)
        print(f"  Loaded {len(molecules)} molecules from cache")
        return molecules

    from torch_geometric.datasets import QM9 as QM9Dataset
    from rdkit import Chem
    from adt_tokenizer import find_bootstrap_triple

    print(f"Loading QM9 from PyG (root={root_dir})...")
    dataset = QM9Dataset(root=root_dir)

    molecules = []
    n_skip = 0
    n_bad_bond = 0
    n_linear = 0
    for i, data in enumerate(dataset):
        try:
            z = data.z.numpy()
            pos = data.pos.numpy()
            smi = data.smiles if hasattr(data, 'smiles') else f"mol_{i}"
            mol = mol_from_pyg(z, pos)

            # Sanity check: all bond distances < 2.0A
            bad = False
            for bond in mol.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                d = np.linalg.norm(pos[a] - pos[b])
                if d > 2.0:
                    bad = True
                    break
            if bad:
                n_bad_bond += 1
                continue

            # Pre-filter: reject linear molecules (no valid bootstrap triple)
            heavy = [j for j in range(mol.GetNumAtoms())
                     if mol.GetAtomWithIdx(j).GetAtomicNum() != 1]
            if not heavy:
                n_linear += 1
                continue
            tokenizable = False
            for root in heavy[:5]:
                _, _, atomR = find_bootstrap_triple(mol, pos, root)
                if atomR is not None:
                    tokenizable = True
                    break
            if not tokenizable:
                n_linear += 1
                continue

            if nohydrogen:
                mol_h, pos_h = remove_hydrogens(mol, pos)
                if mol_h is None:
                    n_skip += 1
                    continue
                mol, pos = mol_h, pos_h

            molecules.append((mol, pos, smi))
        except Exception:
            n_skip += 1

        if (i + 1) % 20000 == 0:
            print(f"  Processed {i+1}/{len(dataset)}...")

    print(f"  Loaded {len(molecules)}, skipped {n_skip}, "
          f"bad bonds {n_bad_bond}, linear {n_linear}"
          f"{', nohydrogen=True' if nohydrogen else ''}")
    with open(cache_path, 'wb') as f:
        pickle.dump(molecules, f)
    print(f"  Cached to {cache_path}")
    return molecules


class DynamicQM9Dataset(Dataset):
    """
    Every __getitem__ tokenizes with a random root.
    Linear molecules are retried with different roots.
    No rotation augmentation needed (SE(3)-invariant tokenization).
    """

    def __init__(self, molecules, split='train',
                 n_val=2000, seed=42):
        n_total = len(molecules)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_total)
        n_val = min(n_val, n_total)
        n_train = n_total - n_val

        if split == 'train':
            idx = indices[:n_train]
        elif split == 'val':
            idx = indices[n_train:n_train + n_val]
        else:  # 'all'
            idx = indices

        self.mols = [molecules[i][0] for i in idx]
        self.positions = [molecules[i][1] for i in idx]
        self.smiles = [molecules[i][2] for i in idx]

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        from adt_tokenizer import tokenize_molecule, tokens_to_array
        import random as _random
        pos = self.positions[idx].copy()
        mol = self.mols[idx]
        n = mol.GetNumAtoms()

        # Heavy atom indices (non-H) for root selection
        heavy = [j for j in range(n) if mol.GetAtomWithIdx(j).GetAtomicNum() != 1]
        if not heavy:
            return torch.tensor([END], dtype=torch.long)

        # Try random heavy-atom root, retry on failure
        max_retries = min(len(heavy), 10)
        for attempt in range(max_retries):
            try:
                root = _random.choice(heavy)
                result = tokenize_molecule(mol, pos, root=root)
                if result is not None and result.chain_length == 1:
                    return torch.tensor(
                        tokens_to_array(result.tokens), dtype=torch.long)
            except Exception:
                pass

        # All retries failed: try first heavy atom as fallback
        try:
            result = tokenize_molecule(mol, pos, root=heavy[0])
            if result is not None and result.chain_length == 1:
                return torch.tensor(
                    tokens_to_array(result.tokens), dtype=torch.long)
        except Exception:
            pass

        # Absolute fallback
        return torch.tensor([END], dtype=torch.long)


# ============================================================
# Collate function
# ============================================================

def collate_fn(batch):
    lengths = [len(v) for v in batch]
    max_len = max(lengths)
    B = len(batch)

    padded = torch.full((B, max_len), PAD_VALUE, dtype=torch.long)
    for i, v in enumerate(batch):
        padded[i, :len(v)] = v

    input_values = padded[:, :-1].clone()
    target_values = padded[:, 1:].clone()
    L = input_values.size(1)

    target_slots = torch.arange(1, L + 1).unsqueeze(0).expand(B, -1) % N_SLOTS
    input_slots = torch.arange(L).unsqueeze(0).expand(B, -1) % N_SLOTS

    # Action type for each position (from the step's action subtoken)
    action_types = torch.zeros(B, L, dtype=torch.long)
    for i in range(B):
        seq_len = min(lengths[i] - 1, L)
        for t in range(seq_len):
            step_start = t - (t % N_SLOTS)
            action_types[i, t] = padded[i, step_start].clamp(min=0)

    valid_mask = target_values != PAD_VALUE
    padding_mask = input_values == PAD_VALUE

    # Frame mask: mask out first 3 steps in target (no loss for frame prediction)
    # Target positions 0..(N_FRAME_STEPS*N_SLOTS - 2) predict frame tokens
    n_frame_target = N_FRAME_STEPS * N_SLOTS - 1  # = 20
    frame_mask = torch.ones(B, L, dtype=torch.bool)
    frame_mask[:, :min(n_frame_target, L)] = False

    input_values = input_values.clamp(min=0)
    target_clamped = target_values.clamp(min=0)

    return {
        'input_values': input_values,
        'target_values': target_clamped,
        'target_values_raw': target_values,
        'input_slots': input_slots,
        'target_slots': target_slots,
        'action_types': action_types,
        'valid_mask': valid_mask,
        'frame_mask': frame_mask,
        'padding_mask': padding_mask,
    }


def create_dataloader(pt_path, batch_size=128, split='train',
                      num_workers=4, max_len=None, **kwargs):
    dataset = QM9TokenDataset(pt_path, max_len=max_len, split=split)
    shuffle = (split == 'train')
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, **kwargs,
    )
    return loader, dataset


# ============================================================
# Frame sampler (for generation)
# ============================================================

class FrameSampler:
    """Sample molecular frames (first 3 atoms) from training data.

    A frame is the first N_FRAME_STEPS * N_SLOTS = 21 tokens of a sequence:
      Step 0 (INIT):  [ADD_INIT, 0, z0, 0, 0, 0, 0]
      Step 1 (CHAIN): [ADD_CHAIN, from, z1, r, 0, 0, 0]
      Step 2 (ANGLE): [ADD_ANGLE, from, z2, r, θc, θf, 0]
    """

    FRAME_LEN = N_FRAME_STEPS * N_SLOTS  # 21

    def __init__(self, molecules=None, pyg_root=None, n_extract=None):
        """
        Args:
            molecules: list of (mol, pos, smiles) from load_qm9_mols
            pyg_root: path to PyG QM9 data root (will call load_qm9_mols)
            n_extract: max molecules to tokenize (None = all)
        """
        import random as _random
        self.frames = []

        if molecules is None and pyg_root is not None:
            molecules = load_qm9_mols(pyg_root)

        if molecules is None:
            raise ValueError("Provide molecules or pyg_root")

        from adt_tokenizer import tokenize_molecule, tokens_to_array

        n_roots_per_mol = 5  # extract multiple frames per molecule
        targets = molecules if n_extract is None else molecules[:n_extract]
        n_ok = 0
        n_fail = 0
        n_has_h = 0  # frames containing H

        for mol, pos, smi in targets:
            n = mol.GetNumAtoms()
            heavy = [j for j in range(n)
                     if mol.GetAtomWithIdx(j).GetAtomicNum() != 1]
            if not heavy:
                n_fail += 1
                continue

            # Sample multiple roots for diversity
            roots = _random.sample(heavy, min(n_roots_per_mol, len(heavy)))
            for root in roots:
                try:
                    result = tokenize_molecule(mol, pos, root=root)
                    if result is not None:
                        arr = tokens_to_array(result.tokens)
                        if len(arr) >= self.FRAME_LEN:
                            frame = arr[:self.FRAME_LEN]
                            # Check if all 3 frame atoms are heavy
                            z0 = frame[2]                    # INIT atom
                            z1 = frame[N_SLOTS + 2]          # CHAIN atom
                            z2 = frame[2 * N_SLOTS + 2]      # ANGLE atom
                            all_heavy = (z0 != 1 and z1 != 1 and z2 != 1)
                            self.frames.append((frame, all_heavy))
                            n_ok += 1
                            if not all_heavy:
                                n_has_h += 1
                        else:
                            n_fail += 1
                    else:
                        n_fail += 1
                except Exception:
                    n_fail += 1

        # Prioritize all-heavy frames; keep H-containing as fallback
        heavy_frames = [f for f, h in self.frames if h]
        h_frames = [f for f, h in self.frames if not h]

        if heavy_frames:
            self.frames = heavy_frames
            print(f"FrameSampler: {len(heavy_frames)} all-heavy frames "
                  f"(discarded {n_has_h} with H), {n_fail} failed")
        else:
            # Fallback: use all frames including H-containing
            self.frames = [f for f, _ in self.frames]
            print(f"FrameSampler: {n_ok} frames (all contain H), {n_fail} failed")
            print(f"  WARNING: no all-heavy frames found")

    def sample(self):
        """Return a random frame as list of ints (length 21)."""
        import random as _random
        return list(_random.choice(self.frames))

    def save(self, path):
        """Save frames to file."""
        torch.save({'frames': self.frames}, path)
        print(f"FrameSampler: saved {len(self.frames)} frames to {path}")

    @classmethod
    def load(cls, path):
        """Load frames from file (compatible with build_frame_cache.py)."""
        obj = cls.__new__(cls)
        data = torch.load(path, weights_only=False)
        obj.frames = data['frames']
        stats = data.get('stats', {})
        print(f"FrameSampler: loaded {len(obj.frames)} frames from {path}")
        if stats:
            print(f"  (n_roots={data.get('n_roots', '?')}, "
                  f"seed={data.get('seed', '?')})")
        return obj
