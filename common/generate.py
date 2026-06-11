"""Unified molecule generation function for ADT Drugs pipeline.

Single source of truth for generate_one_benz. Used by:
  - gen_eval.py (serial eval)
  - gen_browse_parallel.py (parallel eval)
  - fo_train_v22.py (training-time eval via evaluate_generation)
  - gen_browse_data.py (browse data generation)

Collision check: flat COLLISION_R threshold for all atom pairs.
"""
import numpy as np
import random
import torch
import torch.nn.functional as F

from adt_model import ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD, LINK, END
from adt_tokenizer import (
    bin_to_r, decode_direction, encode_direction, build_frame,
    compute_tree_frame, AtomEntry, _ancestor_positions_recon, R_MAX,
    bins_to_theta, arbitrary_perpendicular,
)

N_SLOTS = 7
COLLISION_R = 0.9
ALLOWED_ATOMS = {6, 7, 8, 9, 15, 16, 17, 35, 53}


def generate_one_benz(model, device, frames, temperature=1.0,
                      max_steps=200, return_tokens=False):
    """Generate one molecule starting from a sampled frame.

    Args:
        model: ADT model (eval mode)
        device: torch device
        frames: list of frame token arrays (from frame_cache)
        temperature: sampling temperature
        max_steps: max autoregressive steps after frame
        return_tokens: if True, also return (tokens, n_frame_atoms)

    Returns:
        (atom_table, bonds, n_atoms) or
        (atom_table, bonds, n_atoms, tokens, n_frame_atoms) if return_tokens
    """
    model.eval()

    # Sample frame
    frame_tokens = list(frames[random.randint(0, len(frames) - 1)])
    tokens = list(frame_tokens)

    # Reconstruct atoms from frame tokens
    atom_table = {}
    n_atoms = 0
    bootstrap_frame = None
    bonds = set()

    n_steps_frame = len(frame_tokens) // N_SLOTS
    for step_idx in range(n_steps_frame):
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
                v1=np.zeros(3), v2=np.zeros(3), atomic_num=atom_val)
            n_atoms += 1

        elif action == ADD_CHAIN:
            r = bin_to_r(r_val)
            chain_dir = np.array([0.0, 0.0, 1.0])
            pos = atom_table[from_id - 1].pos + r * chain_dir
            atom_table[n_atoms] = AtomEntry(
                pos=pos, parent_id=from_id, frame=np.eye(3),
                v1=chain_dir * r, v2=np.zeros(3), atomic_num=atom_val)
            bonds.add((min(from_id, n_atoms + 1), max(from_id, n_atoms + 1)))
            n_atoms += 1

        elif action == ADD_ANGLE:
            r = bin_to_r(r_val)
            theta = bins_to_theta(hp0_val, hp1_val)
            parent_pos = atom_table[from_id - 1].pos
            cd = atom_table[1].pos - atom_table[0].pos
            cdn = np.linalg.norm(cd)
            if cdn > 1e-10:
                chain_dir = cd / cdn
            else:
                chain_dir = np.array([0.0, 0.0, 1.0])
            perp = arbitrary_perpendicular(chain_dir)
            d = -np.cos(theta) * chain_dir + np.sin(theta) * perp
            pos = parent_pos + r * d
            bootstrap_frame = build_frame(d, chain_dir)
            for k in range(n_atoms):
                atom_table[k].frame = bootstrap_frame.copy()
            atom_table[n_atoms] = AtomEntry(
                pos=pos, parent_id=from_id, frame=bootstrap_frame.copy(),
                v1=d * r, v2=np.zeros(3), atomic_num=atom_val)
            bonds.add((min(from_id, n_atoms + 1), max(from_id, n_atoms + 1)))
            n_atoms += 1

        elif action == ADD:
            from_frame = atom_table[from_id - 1].frame
            parent_pos = atom_table[from_id - 1].pos
            r, d_world = decode_direction(r_val, hp0_val, hp1_val, hp2_val, from_frame)
            pos = parent_pos + r * d_world
            anc_pos = _ancestor_positions_recon(from_id, atom_table)
            frame = compute_tree_frame(pos, anc_pos, bootstrap_frame)
            atom_table[n_atoms] = AtomEntry(
                pos=pos, parent_id=from_id, frame=frame.copy(),
                v1=(pos - parent_pos), v2=np.zeros(3), atomic_num=atom_val)
            bonds.add((min(from_id, n_atoms + 1), max(from_id, n_atoms + 1)))
            n_atoms += 1

        elif action == LINK:
            to_id = tokens[base + 2]
            bonds.add((min(from_id, to_id), max(from_id, to_id)))

    n_frame_atoms = n_atoms

    # Continue autoregressive generation
    with torch.no_grad():
        for step in range(max_steps):
            arr = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            L = arr.shape[1]
            slots = (torch.arange(L, device=device) % N_SLOTS).unsqueeze(0)
            act_types = torch.zeros(1, L, dtype=torch.long, device=device)
            for i in range(L):
                sb = (i // N_SLOTS) * N_SLOTS
                if sb < L:
                    act_types[0, i] = arr[0, sb]
            logits, _ = model(arr, slots, act_types)

            al = logits[0][0, -1] / temperature
            al[ADD_INIT] = -float('inf')
            al[ADD_CHAIN] = -float('inf')
            al[ADD_ANGLE] = -float('inf')
            action = torch.multinomial(F.softmax(al, dim=-1), 1).item()

            if action == END:
                tokens.append(END)
                break
            tokens.append(action)

            for slot in range(1, N_SLOTS):
                arr = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                L = arr.shape[1]
                slots = (torch.arange(L, device=device) % N_SLOTS).unsqueeze(0)
                act_types = torch.zeros(1, L, dtype=torch.long, device=device)
                for i in range(L):
                    sb = (i // N_SLOTS) * N_SLOTS
                    if sb < L:
                        act_types[0, i] = arr[0, sb]
                logits, _ = model(arr, slots, act_types)

                if slot == 1:
                    pl = logits[1][0, -1] / temperature
                    pl[n_atoms + 1:] = -float('inf')
                    pl[0] = -float('inf')
                    val = torch.multinomial(F.softmax(pl[:n_atoms + 1], dim=-1), 1).item()
                elif slot == 2:
                    if action == ADD:
                        pl = logits['atom'][0, -1] / temperature
                        mask_t = torch.full_like(pl, -float('inf'))
                        for a in ALLOWED_ATOMS:
                            mask_t[a] = 0
                        pl = pl + mask_t
                    else:
                        pl = logits['to'][0, -1] / temperature
                        pl[n_atoms + 1:] = -float('inf')
                        pl[0] = -float('inf')
                    val = torch.multinomial(F.softmax(pl, dim=-1), 1).item()
                else:
                    pl = logits[slot][0, -1] / temperature
                    val = torch.multinomial(F.softmax(pl, dim=-1), 1).item()
                tokens.append(val)

            # Decode the new step
            sb = len(tokens) - N_SLOTS
            from_id = tokens[sb + 1]
            atom_val = tokens[sb + 2]
            r_val = tokens[sb + 3]
            hp0_val = tokens[sb + 4]
            hp1_val = tokens[sb + 5]
            hp2_val = tokens[sb + 6]

            if action == ADD:
                if from_id < 1 or from_id > n_atoms:
                    continue
                from_frame = atom_table[from_id - 1].frame
                parent_pos = atom_table[from_id - 1].pos
                r, d_world = decode_direction(r_val, hp0_val, hp1_val, hp2_val, from_frame)
                pos = parent_pos + r * d_world
                anc_pos = _ancestor_positions_recon(from_id, atom_table)
                frame = compute_tree_frame(pos, anc_pos, bootstrap_frame)
                atom_table[n_atoms] = AtomEntry(
                    pos=pos, parent_id=from_id, frame=frame.copy(),
                    v1=(pos - parent_pos), v2=np.zeros(3), atomic_num=atom_val)
                bonds.add((min(from_id, n_atoms + 1), max(from_id, n_atoms + 1)))
                n_atoms += 1

            elif action == LINK:
                to_id = tokens[sb + 2]
                bonds.add((min(from_id, to_id), max(from_id, to_id)))

    if return_tokens:
        return atom_table, bonds, n_atoms, tokens, n_frame_atoms
    return atom_table, bonds, n_atoms
