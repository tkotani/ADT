"""ikt_common.py — shared helpers for the IKT stack (token packing, atom indexing, canonical frame).

The IKT pipeline:
  adt_model.py     -- ADT builds the LOCAL structure chain (autoregressive: graph + local geometry).
  ikt_corrector.py -- looks at the WHOLE molecule (bidirectional) and CORRECTS it globally
                      (categorical torsion/bend softmax) toward a valid, xTB-stable structure.
  ikt_conformer.py -- GENERATES DIVERSE conformers of that structure (multimodal: a fresh draw each
                      time) and, via the energy head, also finds the lowest-energy one.

These three functions are the only surviving pieces of the retired coordinate-displacement corrector
(the old "predict a 3-vector delta per atom" IKTModel + head_dv + kabsch loss). That regression head
could not beat "predict zero" on held-out molecules (measured 2026-07-12/13) and was replaced by the
categorical torsion corrector, so its model/loss code is GONE. Only these token/geometry helpers, which
the live corrector and conformer import, remain.
"""
import torch
from adt_model import ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD, END

N_SLOTS = 7
PAD_VALUE = -100
ATOM_ACTIONS = (ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD)   # actions that create an atom (LINK/END do not)


def build_inputs(tokens_list, device):
    """Pack raw token sequences (flat ints, 7 slots per step) into model inputs.

    Returns input_values (B,L), input_slots (B,L), action_types (B,L), padding_mask (B,L).
    Unlike log_p.compute_log_p_batch we do NOT shift by one: the IKT reads the COMPLETE
    sequence (it is a post-hoc corrector, not a next-token predictor).
    """
    B = len(tokens_list)
    L = max(len(t) for t in tokens_list)
    L = ((L + N_SLOTS - 1) // N_SLOTS) * N_SLOTS          # keep whole 7-slot steps
    vals = torch.full((B, L), PAD_VALUE, dtype=torch.long, device=device)
    for i, tk in enumerate(tokens_list):
        vals[i, :len(tk)] = torch.tensor(tk, dtype=torch.long, device=device)
    padding_mask = (vals == PAD_VALUE)
    safe = vals.clone()
    safe[padding_mask] = 0
    slots = (torch.arange(L, device=device) % N_SLOTS).unsqueeze(0).expand(B, -1)
    s_start = (torch.arange(L, device=device) // N_SLOTS) * N_SLOTS
    action_types = safe.gather(1, s_start.unsqueeze(0).expand(B, -1))
    return safe, slots, action_types, padding_mask


def atom_token_index(tokens):
    """Token positions of the slot-0 (action) token of each ATOM-creating step, in atom order.

    The rollout emits 7 tokens per step; atoms are created by ADD_INIT/ADD_CHAIN/ADD_ANGLE/ADD
    in the same order as atom_table, so the k-th entry here is atom k.
    """
    idx = []
    for s in range(0, len(tokens) // N_SLOTS * N_SLOTS, N_SLOTS):
        a = tokens[s]
        if a in ATOM_ACTIONS:
            idx.append(s)
        elif a == END:
            break
    return idx


def canonical_coords(pos, mask):
    """Center at the centroid, rotate into the inertia principal frame, fix the axis signs.

    Returns (Y, V, sgn):  Y = sgn * (X @ V)  -- rotation-invariant coordinates (the INPUT), and the
    (V, sgn) needed to map a vector BACK to world:  world = V @ (sgn * d).
    One molecule-level frame is smooth and still equivariant (it rotates with the molecule).
    """
    m = mask.unsqueeze(-1).to(pos.dtype)
    n = m.sum(1).clamp(min=1.0)
    X = (pos - (pos * m).sum(1, keepdim=True) / n.unsqueeze(1)) * m
    C = torch.einsum("bki,bkj->bij", X, X) / n.unsqueeze(-1)          # covariance (B,3,3)
    _w, V = torch.linalg.eigh(C.double())                             # ascending eigenvalues
    V = V.to(X.dtype)                                                 # columns = principal axes
    Y = torch.einsum("bki,bij->bkj", X, V)                            # project onto the axes
    sgn = torch.sign((Y ** 3 * m).sum(1, keepdim=True))               # fix axis signs by skewness
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    return Y * sgn * m, V, sgn
