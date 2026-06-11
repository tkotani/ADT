"""
fo_train.py — Training for free-order tokenized QM9 molecules

Uses the same model as DFS (adt_model.py, 7-slot format),
but with free-order tokenization for data augmentation.

Usage:
  python3 fo_train.py --epochs 600 --batch_size 256 --d_model 512 --n_layers 12 --n_heads 4 --d_ff 2560 --dropout 0.2 --amp bf16 --warmup_epochs 5 --nohydrogen --dynamic

Auto-eval: generates 1000 molecules every 50 epochs after E100.
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
from torch.utils.data import Dataset, DataLoader

# Parent directory for shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# [common] ensure ADT/common is on path
import os as _os_adt, sys as _sys_adt
_common_dir = _os_adt.path.abspath(_os_adt.path.join(_os_adt.path.dirname(__file__), "../../../common"))
if _common_dir not in _sys_adt.path:
    _sys_adt.path.insert(0, _common_dir)
from adt_model import build_model, ADTv2Model, N_ACTIONS
from adt_model import ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD, LINK, END
from adt_tokenizer import (
    tokens_to_array, array_to_tokens, reconstruct_from_tokens,
    r_to_bin, bin_to_r, R_BINS, NHP0, NHP1, NHP2,
    vec_to_healpix, healpix_to_vec, encode_direction, decode_direction,
    build_frame, arbitrary_perpendicular, compute_tree_frame,
    theta_to_bins, bins_to_theta,
    NULL_POINTER, AtomEntry,
    _ancestor_positions_recon,
)
from fo_tokenizer import tokenize_molecule

# ============================================================
# Constants
# ============================================================
N_SLOTS = 7
PAD_VALUE = -1
N_FRAME_STEPS = 3

# ============================================================
# Dataset
# ============================================================

class FreeOrderQM9Dataset(Dataset):
    """QM9 with on-the-fly free-order tokenization."""

    def __init__(self, molecules, split='train', n_val=2000, seed=42):
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

        max_retries = min(len(heavy), 10)
        for attempt in range(max_retries):
            try:
                root = random.choice(heavy)
                result = tokenize_molecule(mol, pos, root=root)
                if result is not None and result.chain_length == 1:
                    return torch.tensor(
                        tokens_to_array(result.tokens), dtype=torch.long)
            except Exception:
                pass

        try:
            result = tokenize_molecule(mol, pos, root=heavy[0])
            if result is not None and result.chain_length == 1:
                return torch.tensor(
                    tokens_to_array(result.tokens), dtype=torch.long)
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

ALLOWED_ATOMS = [6, 7, 8, 9]
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

    # Reconstruct bootstrap atoms from frame
    atom_table = {}
    n_atoms = 0
    bootstrap_frame = None

    # Parse frame
    for step_idx in range(N_FRAME_STEPS):
        base = step_idx * N_SLOTS
        action = tokens[base]
        from_id = tokens[base + 1]
        atom_val = tokens[base + 2]
        r_val = tokens[base + 3]
        hp0_val = tokens[base + 4]
        hp1_val = tokens[base + 5]
        hp2_val = tokens[base + 6]

        if action == ADD_INIT:
            pos = np.array([0.0, 0.0, 0.0])
            atom_table[n_atoms] = AtomEntry(
                pos=pos, parent_id=0, frame=np.eye(3),
                v1=np.zeros(3), v2=np.zeros(3),
                atomic_num=atom_val, is_bootstrap=True,
            )
            n_atoms += 1

        elif action == ADD_CHAIN:
            r = bin_to_r(r_val)
            chain_dir = np.array([0.0, 0.0, 1.0])
            if n_atoms >= 2:
                cd = atom_table[n_atoms - 1].pos - atom_table[0].pos
                cdn = np.linalg.norm(cd)
                if cdn > 1e-10:
                    chain_dir = cd / cdn
            pos = atom_table[from_id - 1].pos + r * chain_dir
            atom_table[n_atoms] = AtomEntry(
                pos=pos, parent_id=from_id, frame=np.eye(3),
                v1=chain_dir * r, v2=np.zeros(3),
                atomic_num=atom_val, is_bootstrap=True,
            )
            n_atoms += 1

        elif action == ADD_ANGLE:
            r = bin_to_r(r_val)
            theta = bins_to_theta(hp0_val, hp1_val)
            chain_dir = np.array([0.0, 0.0, 1.0])
            if n_atoms >= 2:
                cd = atom_table[n_atoms - 1].pos - atom_table[0].pos
                cdn = np.linalg.norm(cd)
                if cdn > 1e-10:
                    chain_dir = cd / cdn
            perp = arbitrary_perpendicular(chain_dir)
            d = -np.cos(theta) * chain_dir + np.sin(theta) * perp
            pos = atom_table[from_id - 1].pos + r * d
            bootstrap_frame = build_frame(d, chain_dir)
            for k in range(n_atoms):
                atom_table[k].frame = bootstrap_frame.copy()
            atom_table[n_atoms] = AtomEntry(
                pos=pos, parent_id=from_id, frame=bootstrap_frame.copy(),
                v1=d * r, v2=np.zeros(3),
                atomic_num=atom_val, is_bootstrap=True,
            )
            n_atoms += 1

    bonds = set()
    # Add bootstrap bonds (CHAIN: 1-2, ANGLE: 2-3)
    for bi in range(1, n_atoms):
        bonds.add((bi, bi + 1))

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
                if slot == 2 and action == LINK and \
                        getattr(model, 'output_pointer_mode', 'absolute') == 'offset':
                    val = n_atoms - val
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
    """Generate n_mols molecules and compute metrics."""
    from rdkit import Chem, RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)

    model.eval()
    n_clean = 0
    n_stable = 0
    n_total = 0

    for i in range(n_mols):
        try:
            atom_table, mol_bonds, n_atoms = generate_one(
                model, device, frame_sampler=frame_sampler, temperature=temperature)
        except:
            n_total += 1
            continue

        n_total += 1
        if n_atoms < 2:
            continue

        # Check collisions (pair-specific thresholds, same as DFS version)
        coords = np.array([atom_table[k].pos for k in range(n_atoms)])
        atomic_nums = [int(atom_table[k].atomic_num) for k in range(n_atoms)]
        bond_set = set()
        for (e1, e2) in mol_bonds:
            bond_set.add((e1 - 1, e2 - 1))
            bond_set.add((e2 - 1, e1 - 1))
        has_collision = False
        for a in range(n_atoms):
            for b in range(a + 1, n_atoms):
                d = np.linalg.norm(coords[a] - coords[b])
                is_bonded = (a, b) in bond_set
                thr = get_collision_threshold(
                    atomic_nums[a], atomic_nums[b], COLLISION_R,
                    is_bonded=is_bonded)
                if d < thr:
                    has_collision = True
                    break
            if has_collision:
                break

        if not has_collision:
            n_clean += 1

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
                molh = Chem.AddHs(mol)
                ok = True
                for atom in molh.GetAtoms():
                    z = atom.GetAtomicNum()
                    val = int(round(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())))
                    if val not in ALLOWED_VALENCES.get(z, set()):
                        ok = False
                        break
                if ok:
                    n_stable += 1
            except:
                pass

    return {
        "n_total": n_total,
        "n_clean": n_clean,
        "n_stable": n_stable,
        "clean_rate": n_clean / n_total if n_total > 0 else 0,
        "stable_rate": n_stable / n_total if n_total > 0 else 0,
    }


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
    parser.add_argument('--save_dir', type=str, default='checkpoints_fo')
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
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max_offset', type=int, default=32,
                        help='v26: output offset size (class = offset - 1 in [0..max_offset-1])')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    log_path = os.path.join('logs', f'fo_train_{datetime.now():%Y%m%d_%H%M%S}.log')
    logging.basicConfig(
        level=logging.INFO, format='%(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    log = logging.getLogger()
    log.info(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {' '.join(sys.argv)}")

    device = torch.device(args.device if args.device != 'auto' else
                         ('cuda' if torch.cuda.is_available() else 'cpu'))
    log.info(f"Device: {device}")

    # AMP
    use_amp = (args.amp != 'off') and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if args.amp == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16)) if use_amp else None
    log.info(f"AMP: {args.amp}")

    # Data
    from adt_dataset import load_qm9_mols
    molecules = load_qm9_mols(nohydrogen=args.nohydrogen)

    train_ds = FreeOrderQM9Dataset(molecules, split='train')
    val_ds = FreeOrderQM9Dataset(molecules, split='val')
    pw = args.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=pw)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=pw)
    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Frame sampler for generation
    frame_sampler = None
    if args.frame_cache and os.path.exists(args.frame_cache):
        from adt_dataset import FrameSampler
        frame_sampler = FrameSampler.load(args.frame_cache)

    # Model (same as DFS)
    config = {
        'd_model': args.d_model, 'n_heads': args.n_heads,
        'n_layers': args.n_layers, 'd_ff': args.d_ff,
        'dropout': args.dropout, 'nohydrogen': args.nohydrogen, 'n_r_bins': R_BINS, 'n_hp0': NHP0, 'n_hp1': NHP1, 'n_hp2': NHP2,
        'output_pointer_mode': 'offset',
        'max_offset': args.max_offset,
        'max_pointer': 30,
    }
    model = build_model(config)
    model.to(device)
    log.info(f"Parameters: {model.count_parameters():,}")

    wrapper = ADTWrapper(model, mask_frame=True).to(device)
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

    # Resume (full: model + optimizer + scheduler + scaler + epoch + best_val)
    if args.resume and os.path.exists(args.resume):
        log.info(f"Resuming from {args.resume}")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        wrapper.model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        if scaler is not None and "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("val_loss", best_val)
        log.info(f"Resumed at epoch {start_epoch}, best_val={best_val:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

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


if __name__ == '__main__':
    main()
