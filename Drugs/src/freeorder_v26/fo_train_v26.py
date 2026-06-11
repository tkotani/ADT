"""
fo_train_v26.py — Hybrid absolute input / offset output pointer.

- Input embedding: emb_pointer[absolute_id] (same as v23, stable atom identity).
- Output head: predicts offset class in [0, max_offset-1] where
    offset = current_atom_count + 1 - target_id  (unified for ADD and LINK).
- Tokenizer: unchanged (v21 free-order, absolute from_id).

Advantages over v23: translation invariance at output → extrapolates beyond
  training max_atoms without the max_pointer bottleneck.
Advantages over v25: input still uses stable absolute atom identity, not a
  context-dependent offset, so the pointer embedding is well-defined.
"""

import argparse
import os
import sys
import time
import pickle
import random
import logging
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader

# Parent directory for shared modules
sys.path.insert(0, os.path.dirname(__file__))

# [common] ensure ADT/common is on path
import os as _os_adt, sys as _sys_adt
_common_dir = _os_adt.path.abspath(_os_adt.path.join(_os_adt.path.dirname(__file__), "../../../common"))
if _common_dir not in _sys_adt.path:
    _sys_adt.path.insert(0, _common_dir)
from adt_model import build_model, ADTv2Model, N_ACTIONS
from adt_model import ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD, LINK, END
from adt_tokenizer import (
    tokens_to_array, array_to_tokens, reconstruct_from_tokens,
    r_to_bin, bin_to_r, R_BINS,
    vec_to_healpix, healpix_to_vec, encode_direction, decode_direction,
    build_frame, arbitrary_perpendicular, compute_tree_frame,
    theta_to_bins, bins_to_theta,
    NULL_POINTER, AtomEntry,
    _ancestor_positions_recon,
)
from fo_tokenizer import tokenize_molecule
from load_drugs import load_drugs_mols

# ============================================================
# Constants
# ============================================================
N_SLOTS = 7
PAD_VALUE = -1
N_FRAME_STEPS = 3

# ============================================================
# Dataset
# ============================================================

class FreeOrderDrugsDataset(Dataset):
    """QM9 with on-the-fly free-order tokenization."""

    def __init__(self, molecules, split='train', n_val=5000, seed=42):
        n_total = len(molecules)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_total)
        n_val = min(n_val, n_total)
        n_train = n_total - n_val

        if split == 'train':
            idx = indices[:n_train]
        elif split == 'val':
            idx = indices[n_train:n_train + n_val]
        else:
            idx = indices

        self.mols = [molecules[i][0] for i in idx]
        self.positions = [molecules[i][1] for i in idx]
        self.smiles = [molecules[i][2] for i in idx]

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        pos = self.positions[idx].copy()
        mol = self.mols[idx]
        n = mol.GetNumAtoms()

        heavy = [j for j in range(n) if mol.GetAtomWithIdx(j).GetAtomicNum() != 1]
        if not heavy:
            return torch.tensor([END], dtype=torch.long)

        # Random heavy atom as root. Skip samples where the triple
        # (first 3 atoms) immediately closes into a 3-membered ring
        # (4th step = LINK). Training targets start AFTER the triple,
        # so a 3-ring triple would waste a sample on a pathological start.
        for attempt in range(15):
            try:
                root = random.choice(heavy)
                result = tokenize_molecule(mol, pos, root=root)
                if result is not None and result.chain_length == 1:
                    arr = tokens_to_array(result.tokens)
                    # Check step 3 action (first token after the triple).
                    # Triple is steps 0,1,2 (INIT, CHAIN, ANGLE); if step 3
                    # is LINK, the triple is itself a ring — skip and retry.
                    if len(arr) > 3 * N_SLOTS and int(arr[3 * N_SLOTS]) == LINK:
                        continue
                    return torch.tensor(arr, dtype=torch.long)
            except Exception:
                pass

        return torch.tensor([END], dtype=torch.long)


# ============================================================
# Collate (same as adt_dataset.py)
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

    action_types = torch.zeros(B, L, dtype=torch.long)
    for i in range(B):
        seq_len = min(lengths[i] - 1, L)
        for t in range(seq_len):
            step_start = t - (t % N_SLOTS)
            action_types[i, t] = padded[i, step_start].clamp(min=0)

    valid_mask = target_values != PAD_VALUE
    padding_mask = input_values == PAD_VALUE

    n_frame_target = N_FRAME_STEPS * N_SLOTS - 1
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


# ============================================================
# Loss wrapper (same as train.py)
# ============================================================

class ADTWrapper(nn.Module):
    def __init__(self, model, mask_frame=True):
        super().__init__()
        self.model = model
        self.mask_frame = mask_frame

    def forward(self, input_values, input_slots, action_types,
                padding_mask, target_values, target_slots, valid_mask,
                frame_mask=None):
        logits, h = self.model(input_values, input_slots, action_types, padding_mask)

        if self.mask_frame and frame_mask is not None:
            valid_mask = valid_mask & frame_mask

        loss, loss_dict = self.model.compute_loss(
            logits, target_values, target_slots, action_types, valid_mask)
        loss_dict = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v, device=loss.device))
                     for k, v in loss_dict.items()}

        return loss, loss_dict


# ============================================================
# Training / Evaluation
# ============================================================

def train_epoch(wrapper, loader, optimizer, device, scaler=None, amp_dtype=None):
    wrapper.train()
    total_loss = 0.0
    total_steps = 0
    slot_losses = {}
    use_amp = scaler is not None

    for batch in loader:
        kw = {k: v.to(device) for k, v in batch.items()
              if isinstance(v, torch.Tensor)}

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            loss, loss_dict = wrapper(
                kw['input_values'], kw['input_slots'], kw['action_types'],
                kw['padding_mask'], kw['target_values'], kw['target_slots'],
                kw['valid_mask'], kw.get('frame_mask'))
            loss = loss.mean()

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_steps += 1
        for k, v in loss_dict.items():
            slot_losses[k] = slot_losses.get(k, 0) + (
                v if isinstance(v, float) else v.mean().item())

    avg = total_loss / max(total_steps, 1)
    avg_slots = {k: v / max(total_steps, 1) for k, v in slot_losses.items()}
    return avg, avg_slots


@torch.no_grad()
def eval_epoch(wrapper, loader, device, amp_dtype=None):
    wrapper.eval()
    total_loss = 0.0
    total_steps = 0
    slot_losses = {}
    use_amp = amp_dtype is not None

    for batch in loader:
        kw = {k: v.to(device) for k, v in batch.items()
              if isinstance(v, torch.Tensor)}

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            loss, loss_dict = wrapper(
                kw['input_values'], kw['input_slots'], kw['action_types'],
                kw['padding_mask'], kw['target_values'], kw['target_slots'],
                kw['valid_mask'], kw.get('frame_mask'))
            loss = loss.mean()

        total_loss += loss.item()
        total_steps += 1
        for k, v in loss_dict.items():
            slot_losses[k] = slot_losses.get(k, 0) + (
                v if isinstance(v, float) else v.mean().item())

    avg = total_loss / max(total_steps, 1)
    avg_slots = {k: v / max(total_steps, 1) for k, v in slot_losses.items()}
    return avg, avg_slots


# ============================================================
# Generation
# ============================================================

ALLOWED_ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # match v21: C,N,O,F,P,S,Cl,Br,I
ALLOWED_VALENCES = {
    1: {1}, 5: {3}, 6: {4}, 7: {3}, 8: {2}, 9: {1},
    14: {4}, 15: {3, 5}, 16: {2, 4, 6}, 17: {1}, 35: {1}, 53: {1},
}
COLLISION_R = 0.9

# Shortest known bond length per atom pair (Angstrom)
# Collision if d < shortest × COLLISION_R for bonded pairs
SHORTEST_BOND = {
    (1, 1):   0.74,
    (1, 6):   1.09,
    (1, 7):   1.01,
    (1, 8):   0.96,
    (1, 9):   0.92,
    (6, 6):   1.20,   # C-C triple
    (6, 7):   1.16,   # C-N triple
    (6, 8):   1.23,   # C=O double
    (6, 9):   1.35,   # C-F single
    (7, 7):   1.10,   # N-N triple
    (7, 8):   1.21,   # N=O double
    (7, 9):   1.36,
    (8, 8):   1.21,
    (8, 9):   1.42,
    (9, 9):   1.42,
}

# Non-bonded minimum distances (absolute floor, no scaling)
NONBOND_MIN = {
    (1, 1): 1.0, (1, 6): 1.2, (1, 7): 1.2, (1, 8): 1.2, (1, 9): 1.2,
    (6, 6): 1.5, (6, 7): 1.5, (6, 8): 1.5, (6, 9): 1.5,
    (7, 7): 1.5, (7, 8): 1.5, (7, 9): 1.5,
    (8, 8): 1.5, (8, 9): 1.5, (9, 9): 1.5,
}

def get_collision_threshold(z_i, z_j, collision_R=0.9, is_bonded=True):
    """Pair-specific collision threshold (same as DFS version)."""
    key = (min(z_i, z_j), max(z_i, z_j))
    if is_bonded:
        shortest = SHORTEST_BOND.get(key, 0.9)
        return shortest * collision_R
    else:
        return NONBOND_MIN.get(key, 1.0)


def generate_one(model, device, frame_sampler=None, max_steps=200, temperature=1.0):
    """Generate one molecule using free-order model.

    Uses DFS-style generation (the model is free to choose from in any order).
    """
    model.eval()

    if frame_sampler is not None:
        # Resample until we get a non-linear frame (step2 == ADD_ANGLE)
        for _try in range(100):
            frame_tokens = frame_sampler.sample()
            if frame_tokens[2 * N_SLOTS] == ADD_ANGLE:
                break
        else:
            # Fallback if all tries failed
            frame_tokens = [
                ADD_INIT, 0, 6, 0, 0, 0, 0,
                ADD_CHAIN, 1, 6, 50, 0, 0, 0,
                ADD_ANGLE, 2, 6, 50, 6, 8, 0,
            ]
    else:
        # Fallback: simple C-C-C frame
        frame_tokens = [
            ADD_INIT, 0, 6, 0, 0, 0, 0,
            ADD_CHAIN, 1, 6, 50, 0, 0, 0,
            ADD_ANGLE, 2, 6, 50, 6, 8, 0,
        ]

    tokens = list(frame_tokens)

    # Reconstruct bootstrap atoms from full frame (handles any N_FRAME_STEPS
    # including 7-step benzene: 6 C ADD + 1 LINK ring closure).
    # Using reconstruct_from_tokens so all atoms in the frame (not just the
    # first N_FRAME_STEPS=3) end up in atom_table — critical for offset
    # pointer generation, which uses n_atoms to decode offset → from_id.
    frame_arr = np.asarray(tokens, dtype=np.int64)
    frame_tuples = array_to_tokens(frame_arr)
    atoms_list, links_list = reconstruct_from_tokens(list(frame_tuples) + [('action', END)])
    atom_table = {i: ae for i, ae in enumerate(atoms_list)}
    n_atoms = len(atoms_list)
    # ANGLE step establishes the bootstrap frame; copy from atom[2] if exists.
    bootstrap_frame = atoms_list[2].frame.copy() if n_atoms >= 3 else np.eye(3)

    bonds = set()
    # Tree bonds (parent_id != 0)
    for i, ae in enumerate(atoms_list):
        pid = getattr(ae, 'parent_id', 0)
        if pid and pid > 0:
            bonds.add((min(pid, i + 1), max(pid, i + 1)))
    # LINK bonds from frame (e.g., benzene ring closure)
    for lk in links_list:
        a, b = int(lk[0]), int(lk[1])
        if a > 0 and b > 0:
            bonds.add((min(a, b), max(a, b)))

    with torch.no_grad():
        for step in range(max_steps):
            arr = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            L = arr.shape[1]
            slots = (torch.arange(L, device=device) % N_SLOTS).unsqueeze(0)
            act_types = torch.zeros(1, L, dtype=torch.long, device=device)
            for i in range(L):
                s = i // N_SLOTS
                ai = s * N_SLOTS
                if ai < L:
                    act_types[0, i] = arr[0, ai]

            logits, _ = model(arr, slots, act_types)

            # Slot 0: action
            action_logits = logits[0][0, -1] / temperature
            action_logits[ADD_INIT] = -float('inf')
            action_logits[ADD_CHAIN] = -float('inf')
            action_logits[ADD_ANGLE] = -float('inf')
            probs = F.softmax(action_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()

            if action == END:
                tokens.append(END)
                break

            tokens.append(action)

            # Slot 1: from
            arr2 = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            L2 = arr2.shape[1]
            slots2 = (torch.arange(L2, device=device) % N_SLOTS).unsqueeze(0)
            act_types2 = torch.zeros(1, L2, dtype=torch.long, device=device)
            for i in range(L2):
                s = i // N_SLOTS
                ai = s * N_SLOTS
                if ai < L2:
                    act_types2[0, i] = arr2[0, ai]
            logits2, _ = model(arr2, slots2, act_types2)

            from_logits = logits2[1][0, -1] / temperature
            if getattr(model, 'output_pointer_mode', 'absolute') == 'offset':
                # class c in [0, max_offset-1] -> offset = c+1 -> from_id = n_atoms - c
                max_c = min(n_atoms, from_logits.shape[-1]) - 1
                if max_c < 0:
                    from_id = 1
                    tokens.append(from_id)
                else:
                    masked = from_logits.clone()
                    if max_c + 1 < from_logits.shape[-1]:
                        masked[max_c + 1:] = -float('inf')
                    probs_from = F.softmax(masked[:max_c + 1], dim=-1)
                    c = torch.multinomial(probs_from, 1).item()
                    from_id = n_atoms - c
                    tokens.append(from_id)
            else:
                from_logits[n_atoms + 1:] = -float('inf')
                from_logits[0] = -float('inf')
                probs_from = F.softmax(from_logits[:n_atoms + 1], dim=-1)
                from_id = torch.multinomial(probs_from, 1).item()
                tokens.append(from_id)

            # Slots 2-6: predict one at a time
            for slot in range(2, 7):
                arr_s = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                Ls = arr_s.shape[1]
                slots_s = (torch.arange(Ls, device=device) % N_SLOTS).unsqueeze(0)
                act_s = torch.zeros(1, Ls, dtype=torch.long, device=device)
                for i in range(Ls):
                    s = i // N_SLOTS
                    ai = s * N_SLOTS
                    if ai < Ls:
                        act_s[0, i] = arr_s[0, ai]
                logits_s, _ = model(arr_s, slots_s, act_s)

                if slot == 2:
                    if action == ADD:
                        pred_logits = logits_s['atom'][0, -1] / temperature
                        mask_t = torch.full_like(pred_logits, -float('inf'))
                        for a in ALLOWED_ATOMS:
                            mask_t[a] = 0
                        pred_logits = pred_logits + mask_t
                    else:  # LINK
                        pred_logits = logits_s['to'][0, -1] / temperature
                        if getattr(model, 'output_pointer_mode', 'absolute') == 'offset':
                            max_c = min(n_atoms, pred_logits.shape[-1]) - 1
                            if max_c >= 0 and max_c + 1 < pred_logits.shape[-1]:
                                pred_logits[max_c + 1:] = -float('inf')
                            # also prevent offset=0 classes would be invalid; class 0 = offset 1 is OK
                        else:
                            pred_logits[n_atoms + 1:] = -float('inf')
                            pred_logits[0] = -float('inf')
                elif slot == 3:
                    pred_logits = logits_s[3][0, -1] / temperature
                elif slot == 4:
                    pred_logits = logits_s[4][0, -1] / temperature
                elif slot == 5:
                    pred_logits = logits_s[5][0, -1] / temperature
                elif slot == 6:
                    pred_logits = logits_s[6][0, -1] / temperature

                probs_slot = F.softmax(pred_logits, dim=-1)
                val = torch.multinomial(probs_slot, 1).item()
                # For LINK slot-2 in offset mode, convert class -> absolute to_id
                if slot == 2 and action == LINK and \
                        getattr(model, 'output_pointer_mode', 'absolute') == 'offset':
                    val = n_atoms - val  # = (n_atoms + 1) - (val + 1)
                tokens.append(val)

            # Reconstruct atom position from completed step
            step_base = len(tokens) - N_SLOTS
            r_bin = tokens[step_base + 3]
            hp0 = tokens[step_base + 4]
            hp1 = tokens[step_base + 5]
            hp2 = tokens[step_base + 6]
            atom_val = tokens[step_base + 2]

            if action == ADD:
                from_frame = atom_table[from_id - 1].frame
                parent_pos = atom_table[from_id - 1].pos
                r, d_world = decode_direction(r_bin, hp0, hp1, hp2, from_frame)
                pos = parent_pos + r * d_world

                anc_pos = _ancestor_positions_recon(from_id, atom_table)
                frame = compute_tree_frame(pos, anc_pos, bootstrap_frame)

                atom_table[n_atoms] = AtomEntry(
                    pos=pos, parent_id=from_id, frame=frame.copy(),
                    v1=(pos - parent_pos), v2=np.zeros(3),
                    atomic_num=atom_val,
                )
                bonds.add((min(from_id, n_atoms + 1), max(from_id, n_atoms + 1)))
                n_atoms += 1

            elif action == LINK:
                to_id = tokens[step_base + 2]
                bonds.add((min(from_id, to_id), max(from_id, to_id)))

    return atom_table, bonds, n_atoms


def evaluate_generation(model, device, frame_sampler=None, n_mols=1000, temperature=1.0):
    """Generate n_mols molecules and compute metrics. Collision check LAST."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdmolops
    RDLogger.logger().setLevel(RDLogger.ERROR)
    from collision_check import check_collisions

    model.eval()
    counts = {
        'total': 0, 'too_small': 0, 'gen_fail': 0,
        'build_fail': 0, 'disconnected': 0,
        'unstable': 0, 'mol_stable': 0,
        'collision': 0, 'mol_stable_clean': 0,
    }

    for i in range(n_mols):
        try:
            atom_table, mol_bonds, n_atoms = generate_one(
                model, device, frame_sampler=frame_sampler, temperature=temperature)
        except Exception:
            counts['gen_fail'] += 1
            counts['total'] += 1
            continue

        counts['total'] += 1
        if n_atoms < 2:
            counts['too_small'] += 1
            continue

        coords = [atom_table[k].pos for k in range(n_atoms)]
        atomic_nums = [int(atom_table[k].atomic_num) for k in range(n_atoms)]

        try:
            rw = Chem.RWMol()
            id_map = {}
            for k in range(n_atoms):
                z = atom_table[k].atomic_num
                ai = rw.AddAtom(Chem.Atom(z))
                id_map[k + 1] = ai
            for (e1, e2) in mol_bonds:
                if e1 in id_map and e2 in id_map:
                    rw.AddBond(id_map[e1], id_map[e2], Chem.BondType.SINGLE)
            mol = rw.GetMol()
            Chem.SanitizeMol(mol)
        except Exception:
            counts['build_fail'] += 1
            continue

        if len(rdmolops.GetMolFrags(mol)) > 1:
            counts['disconnected'] += 1
            continue

        try:
            molh = Chem.AddHs(mol)
            ok = True
            for atom in molh.GetAtoms():
                z = atom.GetAtomicNum()
                val = int(round(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())))
                if val not in ALLOWED_VALENCES.get(z, set()):
                    ok = False
                    break
        except Exception:
            ok = False

        if not ok:
            counts['unstable'] += 1
            continue

        counts['mol_stable'] += 1

        perceived_bonds = set()
        for bond in mol.GetBonds():
            perceived_bonds.add((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        has_coll, _ = check_collisions(coords, atomic_nums, perceived_bonds)
        if has_coll:
            counts['collision'] += 1
        else:
            counts['mol_stable_clean'] += 1

    N = counts['total']
    for k in ['total','gen_fail','too_small','build_fail','disconnected',
              'unstable','mol_stable','collision','mol_stable_clean']:
        v = counts[k]
        print(f"  {k:<20s} {v:5d}  ({100*v/max(N,1):.1f}%)")

    return counts


# ============================================================
# Main
# ============================================================

def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=2560)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path (best.pt or epoch_XXX.pt)')
    parser.add_argument('--pretrain', type=str, default=None, help='Load only model weights (no optimizer/scheduler/epoch). Use for fine-tuning from a different dataset.')
    parser.add_argument('--save_dir', type=str, default='checkpoints_v21')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--eval_after', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--eval_n', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--amp', type=str, default='bf16', choices=['bf16', 'fp16', 'off'])
    parser.add_argument('--nohydrogen', action='store_true')
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--frame_cache', type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_atoms', type=int, default=30, help='Max heavy atoms per molecule (data filter)')
    parser.add_argument('--max_pointer', type=int, default=None,
                        help='Input pointer embedding size (emb_pointer table width). '
                             'Must cover max heavy atom index in dataset. In offset output mode '
                             'this only sizes the INPUT embedding; the output classifier uses '
                             '--max_offset instead.')
    parser.add_argument('--require_benzene', type=int, default=1, choices=[0,1], help='1=benzene-only dataset, 0=all')
    parser.add_argument('--data_cache', type=str, default='drugs_mols_v21.pkl')
    parser.add_argument('--max_offset', type=int, default=32,
                        help='[v26 primary] Output pointer softmax size (number of '
                             'offset classes). offset = current_atom_count + 1 - target_id. '
                             'Replaces max_pointer as the OUTPUT-side capacity limit: unlike '
                             'max_pointer (which scales with max_atoms), max_offset is bounded '
                             'by chemical locality (99% of real offsets <= 12 for Drugs), so '
                             'max_offset=32 covers all data while staying independent of '
                             'max_atoms. This is the architectural claim of v26.')
    args = parser.parse_args()
    if args.max_pointer is None:
        args.max_pointer = args.max_atoms
    assert args.max_atoms <= args.max_pointer, \
        f'max_atoms ({args.max_atoms}) must be <= max_pointer ({args.max_pointer})'

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    log_path = os.path.join('logs', f'fo_train_{datetime.now():%Y%m%d_%H%M%S}.log')
    logging.basicConfig(
        level=logging.INFO, format='%(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    log = logging.getLogger()
    log.info(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {' '.join(sys.argv)}")

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    use_ddp = local_rank >= 0
    if use_ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device if args.device != 'auto' else
                             ('cuda' if torch.cuda.is_available() else 'cpu'))
    log.info(f"Device: {device}")

    # AMP
    use_amp = (args.amp != 'off') and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if args.amp == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16)) if use_amp else None
    log.info(f"AMP: {args.amp}")

    # Data
    molecules = load_drugs_mols(max_atoms=args.max_atoms, require_benzene=bool(args.require_benzene), cache_path=args.data_cache)
    # Scan actual max heavy-atom count (first-epoch one-off; negligible cost)
    actual_max_heavy = max(
        sum(1 for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() != 1)
        for mol, _, _ in molecules) if molecules else 0
    log.info(f"actual max heavy-atom count in dataset: {actual_max_heavy} (args.max_atoms={args.max_atoms})")
    if actual_max_heavy > args.max_atoms:
        log.warning(f"WARNING: actual max {actual_max_heavy} > args.max_atoms {args.max_atoms} — dataset filter upstream may be lax")

    train_ds = FreeOrderDrugsDataset(molecules, split='train')
    val_ds = FreeOrderDrugsDataset(molecules, split='val')
    pw = args.num_workers > 0
    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=pw)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=pw)
    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Frame sampler for generation
    frame_sampler = None
    if args.frame_cache and os.path.exists(args.frame_cache):
        from adt_dataset import FrameSampler
        frame_sampler = FrameSampler.load(args.frame_cache)

    # Model (same as DFS)
    # max_pointer must cover the largest heavy-atom index (args.max_atoms)
    config = {
        'd_model': args.d_model, 'n_heads': args.n_heads,
        'n_layers': args.n_layers, 'd_ff': args.d_ff,
        'dropout': args.dropout, 'nohydrogen': args.nohydrogen, 'n_r_bins': 200,
        'max_pointer': args.max_pointer,
        'output_pointer_mode': 'offset',
        'max_offset': args.max_offset,
    }
    model = build_model(config)
    model.to(device)
    log.info(f"Parameters: {model.count_parameters():,}")

    wrapper = ADTWrapper(model, mask_frame=True).to(device)
    if use_ddp:
        wrapper = DDP(wrapper, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # Scheduler: warmup + cosine
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
    base_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    if args.warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                                     total_iters=args.warmup_epochs)
        scheduler = SequentialLR(optimizer,
                                schedulers=[warmup_scheduler, base_scheduler],
                                milestones=[args.warmup_epochs])
    else:
        scheduler = base_scheduler

    best_val = float('inf')
    start_epoch = 1

    # Resume from checkpoint (full state) or load pretrained weights only
    if args.resume and os.path.exists(args.resume):
        log.info(f"Resuming from {args.resume}")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        ck_mp = ck.get('config', {}).get('max_pointer', 30)
        assert ck_mp == args.max_pointer, \
            f'resume ckpt max_pointer={ck_mp} != args.max_pointer={args.max_pointer}'
        target = wrapper.module if use_ddp else wrapper
        target.model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        if scaler is not None and "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("val_loss", best_val)
        log.info(f"Resumed at epoch {start_epoch}, best_val={best_val:.4f}")
    elif args.pretrain and os.path.exists(args.pretrain):
        log.info(f"Loading pretrained weights from {args.pretrain} (model only; fresh optimizer/scheduler)")
        ck = torch.load(args.pretrain, map_location=device, weights_only=False)
        ck_mp = ck.get('config', {}).get('max_pointer', 30)
        target = wrapper.module if use_ddp else wrapper
        pre_epoch = ck.get("epoch", "?")
        pre_val = ck.get("val_loss", "?")
        if ck_mp != args.max_pointer:
            log.info(f"Resizing pointer embedding: {ck_mp} -> {args.max_pointer}")
            sd = ck["model"]
            resize_keys = [k for k in sd.keys()
                           if sd[k].shape[0] == ck_mp and sd[k].shape[0] != args.max_pointer]
            for key in resize_keys:
                old_w = sd[key]
                if old_w.dim() == 2:
                    new_w = torch.zeros(args.max_pointer, old_w.shape[1], dtype=old_w.dtype)
                elif old_w.dim() == 1:
                    new_w = torch.zeros(args.max_pointer, dtype=old_w.dtype)
                else:
                    continue
                n_copy = min(ck_mp, args.max_pointer)
                new_w[:n_copy] = old_w[:n_copy]
                if old_w.dim() == 2:
                    torch.nn.init.normal_(new_w[n_copy:], std=0.02)
                sd[key] = new_w
                log.info(f"  {key}: {tuple(old_w.shape)} -> {tuple(new_w.shape)}")
            target.model.load_state_dict(sd)
        else:
            target.model.load_state_dict(ck["model"])
        log.info(f"Pretrained weights loaded (source epoch={pre_epoch}, val_loss={pre_val}). Starting fresh at epoch 1.")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        if use_ddp: train_sampler.set_epoch(epoch)
        train_loss, train_slots = train_epoch(
            wrapper, train_loader, optimizer, device,
            scaler=scaler, amp_dtype=amp_dtype)
        val_loss, val_slots = eval_epoch(
            wrapper, val_loader, device, amp_dtype=amp_dtype)
        scheduler.step()
        dt = time.time() - t0

        slot_str = ' '.join(f"{k}={v:.3f}" for k, v in train_slots.items())
        lr_now = scheduler.get_last_lr()[0]
        now = datetime.now().strftime('%H:%M:%S')
        log.info(f"E{epoch:03d} [{now}] | train={train_loss:.4f} val={val_loss:.4f} | "
                 f"{slot_str} | lr={lr_now:.1e} | {dt:.1f}s")

        save_dict = {
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'config': config, 'val_loss': val_loss, 'train_loss': train_loss,
        }
        if use_amp and scaler is not None:
            save_dict['scaler'] = scaler.state_dict()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(save_dict, os.path.join(args.save_dir, 'best.pt'))

        if epoch % args.save_every == 0:
            torch.save(save_dict, os.path.join(args.save_dir, f'epoch_{epoch:03d}.pt'))

        # Auto-eval
        if epoch >= args.eval_after and epoch % args.eval_every == 0:
            log.info(f"  Evaluating {args.eval_n} molecules...")
            t_eval = time.time()
            metrics = evaluate_generation(
                model, device, frame_sampler=frame_sampler, n_mols=args.eval_n)
            dt_eval = time.time() - t_eval
            log.info(f"  EVAL E{epoch:03d}: "
                     f"clean={metrics['n_clean']}/{metrics['n_total']} ({metrics['clean_rate']:.1%}), "
                     f"stable={metrics['n_stable']}/{metrics['n_total']} ({metrics['stable_rate']:.1%}) "
                     f"[{dt_eval:.0f}s]")

    log.info(f"\nDone. Best val loss: {best_val:.4f}")


    if use_ddp: dist.destroy_process_group()


if __name__ == '__main__':
    main()
