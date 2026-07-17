"""ikt_model.py — IKT (Inverse-Kinematics Transformer), the "power amp" of ADT+IKT.

Same body as ADT (embeddings + TransformerEncoder over the 7-slot token stream) but:
  * BIDIRECTIONAL (no causal mask): every atom sees every other atom + every LINK token,
    so ring-closure / long-range constraints reach the correction (the whole point).
  * Output head: for each ATOM-creating step, the slot-0 (action) token's latent -> a continuous
    delta v (3 components, in that atom's LOCAL frame) via an MLP. Zero-initialised final layer
    => delta v == 0 exactly at init => ADT+IKT == ADT (near-identity start).
  * LINK steps are embedded as context but emit no delta v (they declare a bond, not an atom).

Global synthesis is EULER (each atom independent, no chain re-composition):
    global(k) = ADT_global(k) + frame(k).T @ dv(k)
(adt_tokenizer: world = frame.T @ local, so frame.T maps a local vector to world.)

Module names for the body are byte-identical to ADTv2Model, so an ADT ckpt warm-starts the IKT body
with load_state_dict(..., strict=False): every embedding/transformer weight transfers, the ADT output
heads are dropped, head_dv stays fresh (zero-init).
"""

import torch
import torch.nn as nn

from adt_model import ADTv2Model, ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD, LINK, END, N_ACTIONS

N_SLOTS = 7
PAD_VALUE = -100
ATOM_ACTIONS = (ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD)   # actions that create an atom (LINK/END do not)


class IKTModel(nn.Module):
    """Bidirectional corrector over the ADT token stream. Predicts per-atom local delta v."""

    def __init__(self, config=None):
        super().__init__()
        c = dict(config or {})
        # body = an ADT with the same config; we use ONLY embed_input + transformer + ln_final.
        # (Keeping the ADT class verbatim is what makes the warm-start byte-compatible.)
        self.body = ADTv2Model(c)
        d = self.body.d_model
        # ★ geometry input: the ADT's ACTUAL global coordinates, canonicalised into the molecule's
        # inertia frame. Without this the corrector would have to re-integrate the tree from relative
        # (rb, HEALPix) tokens to know where each atom ended up -- i.e. redo the very accumulation that
        # breaks. Canonical coords are rotation-invariant, and dv is emitted in each atom's LOCAL frame,
        # so the whole map stays equivariant.
        self.emb_xyz = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.head_dv = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3),
        )
        nn.init.zeros_(self.head_dv[-1].weight)      # near-identity: dv == 0 at init
        nn.init.zeros_(self.head_dv[-1].bias)

    def warm_start_from_adt(self, adt_state_dict):
        """Load ADT weights into the body (embeddings + transformer). Heads that do not exist here
        are ignored; head_dv is untouched (stays zero-init)."""
        sd = {f"body.{k}": v for k, v in adt_state_dict.items()}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        n_body = sum(1 for k in sd if k.startswith("body."))
        return n_body, [m for m in missing if not m.startswith("head_dv")], unexpected

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def canonical_coords(pos, mask):
        """Center at the centroid, rotate into the inertia principal frame, fix the axis signs.

        Returns (Y, V, sgn):  Y = sgn * (X @ V)  -- rotation-invariant coordinates (the INPUT), and the
        (V, sgn) needed to map a vector BACK to world:  world = V @ (sgn * d).

        The correction is emitted in THIS frame, not in each atom's local tree frame: the tree frames
        point in wildly different directions per atom (they are built from the ancestor chain), so the
        same smooth world-space displacement becomes a high-frequency function of the atom index when
        expressed in them -- a needlessly hard regression target. One molecule-level frame is smooth and
        still equivariant (it rotates with the molecule)."""
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

    def forward(self, tokens_list, atom_idx_list, device, dv_clip=None, pos=None, amask=None):
        """tokens_list: B token sequences. atom_idx_list: per molecule, the token position of each atom.
        pos (B,K,3): the ADT's global coordinates -> injected as an INPUT embedding at each atom's action
        token, i.e. BEFORE the attention, so every atom can see every other atom's geometry (that is the
        whole point of the corrector: a tail stabbing into the body is a RELATION between atoms).

        Returns dv (B,Kmax,3) in each atom's local frame + a validity mask.
        """
        vals, slots, acts, pad = self.build_inputs(tokens_list, device)
        x = self.body.embed_input(vals, slots, acts)

        B = len(tokens_list)
        Kmax = max((len(ix) for ix in atom_idx_list), default=0)
        Kmax = max(Kmax, 1)
        gather_idx = torch.zeros(B, Kmax, dtype=torch.long, device=device)
        mask = torch.zeros(B, Kmax, dtype=torch.bool, device=device)
        for b, ix in enumerate(atom_idx_list):
            if ix:
                gather_idx[b, :len(ix)] = torch.tensor(ix, dtype=torch.long, device=device)
                mask[b, :len(ix)] = True

        V = sgn = None
        if pos is not None:                       # geometry INTO the attention (canonical = rotation-invariant)
            m = (amask[:, :Kmax] if amask is not None else mask)
            xyz, V, sgn = self.canonical_coords(pos[:, :Kmax], m)
            g = self.emb_xyz(xyz) * m.unsqueeze(-1)                       # (B,Kmax,d)
            x = x.scatter_add(1, gather_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]), g)

        h = self.body.transformer(x, mask=None, src_key_padding_mask=pad)   # BIDIRECTIONAL
        h = self.body.ln_final(h)
        h_atom = h.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))   # (B,Kmax,d)
        dv = self.head_dv(h_atom)                                                     # (B,Kmax,3)
        if dv_clip:                                                                   # keep it a CORRECTION
            dv = dv_clip * torch.tanh(dv / dv_clip)
        dv = dv * mask.unsqueeze(-1)
        return dv, mask, V, sgn


def apply_dv_canon(pos, dv, V, sgn):
    """Euler synthesis in the molecule's canonical frame: world = ADT_world + V @ (sgn * d)."""
    return pos + torch.einsum("bij,bkj->bki", V, dv * sgn)


def apply_dv(pos, frames, dv):
    """Euler synthesis: world(k) = ADT_world(k) + frame(k).T @ dv(k).

    pos    : (B,K,3)   ADT global coords (const)
    frames : (B,K,3,3) ADT local frames (const; world = frame.T @ local)
    dv     : (B,K,3)   IKT local correction (grad)
    """
    return pos + torch.einsum("bkji,bkj->bki", frames, dv)


def kabsch_loss(pred, target, mask, loss_type="mse", huber_delta=0.5, mol_w=None):
    """Deviation after optimal superposition (per molecule), with pluggable per-atom loss and
    per-molecule weighting.

    pred (B,K,3) has grad; target (B,K,3) is a constant (the xTB realizable geometry); mask (B,K).
    The optimal rotation is DETACHED (it is the minimising alignment; detaching avoids SVD-backward
    instability while still pointing downhill for the prediction).

      loss_type "mse"   : mean squared deviation per atom          (L2; dominated by the worst atom)
      loss_type "huber" : Huber on the per-atom distance, delta A  (robust to a few wild atoms)
      mol_w  (B,)       : per-molecule weight (normalised to mean 1 by the caller), e.g. emphasise
                          big-displacement / large molecules ("teach the cliff harder").

    Returns (loss, per-molecule RMSD).
    """
    m = mask.unsqueeze(-1).float()
    n = m.sum(dim=1).clamp(min=1.0)                        # (B,1)
    p_c = (pred * m).sum(dim=1, keepdim=True) / n.unsqueeze(1)
    t_c = (target * m).sum(dim=1, keepdim=True) / n.unsqueeze(1)
    P = (pred - p_c) * m
    T = (target - t_c) * m

    with torch.no_grad():
        H = torch.einsum("bki,bkj->bij", P, T)             # (B,3,3)
        U, S, Vh = torch.linalg.svd(H.double())
        d = torch.sign(torch.linalg.det(torch.einsum("bij,bjk->bik", Vh.transpose(1, 2), U.transpose(1, 2))))
        D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
        R = torch.einsum("bij,bjk,bkl->bil", Vh.transpose(1, 2), D, U.transpose(1, 2)).to(pred.dtype)

    P_rot = torch.einsum("bij,bkj->bki", R, P)             # rotate prediction onto target
    sq = ((P_rot - T) ** 2).sum(-1)                        # (B,K) squared distance per atom
    msd = (sq * mask.float()).sum(dim=1) / n.squeeze(-1)   # (B,) mean squared dev -> RMSD^2
    rmsd = msd.detach().clamp(min=0).sqrt()

    if loss_type == "huber":
        dist = (sq.clamp(min=1e-12)).sqrt()                                  # (B,K) per-atom distance
        h = torch.where(dist <= huber_delta, 0.5 * dist ** 2,
                        huber_delta * (dist - 0.5 * huber_delta))
        per_mol = (h * mask.float()).sum(dim=1) / n.squeeze(-1)
    else:
        per_mol = msd

    if mol_w is not None:
        per_mol = per_mol * mol_w
    return per_mol.mean(), rmsd


# backward-compatible alias (uniform MSE)
def kabsch_mse(pred, target, mask):
    return kabsch_loss(pred, target, mask, loss_type="mse")
