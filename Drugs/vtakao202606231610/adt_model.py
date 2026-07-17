"""
adt_model.py — ADT, architecture version vtakao202606231610 (emb_offset / counter-less).

DEDICATED emb_offset build. Differs from the legacy absolute model (~/ADT/common/adt_model.py) ONLY in
the parent-pointer INPUT pathway:

  legacy (absolute):  slot1 / slot2-LINK store the ABSOLUTE parent index (from_id = na - offset) and are
                      embedded via emb_pointer(from_id), a fixed-size input table that CAPPED the atom
                      count (gather OOB once the atom index ran past the table).
  this   (offset):    slot1 / slot2-LINK store the 1-indexed OFFSET directly and are embedded via
                      emb_offset(offset-1), an Embedding(max_offset, d). NO emb_pointer, NO input
                      atom-count cap -> translation-invariant -> UNBOUNDED atom count.

The OUTPUT heads head_from/head_to already predicted a relative-offset class (max_offset wide) in the
legacy v26 model, so they are UNCHANGED here. Because the token stream now stores the offset, the loss
reads the target offset DIRECTLY (class = offset-1) instead of re-deriving it from an absolute index via
a running atom counter (this avoids the offset-of-offset double-conversion trap).

Module names are byte-for-byte identical to the legacy model EXCEPT emb_pointer -> emb_offset, so a legacy
ckpt (e.g. E142) loads into this model with strict=False: every weight transfers, emb_pointer is dropped,
emb_offset is fresh-initialised (warm-start / FT path). Default config also trains from scratch.

7 tokens per step: [action] [from] [atom/to] [r] [hp0] [hp1] [hp2].
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


_OFFSET_CLAMP_WARNED = 0


def _warn_offset_clamp(where, n_over, max_offset):
    """Loud (rate-limited) warning: an offset target exceeds max_offset and would be SILENTLY clamped
    to a WRONG parent. This must be prevented upstream (dataset free-order retry+skip); if it fires,
    a molecule slipped through the filter (or an unguarded caller fed an overflow)."""
    global _OFFSET_CLAMP_WARNED
    _OFFSET_CLAMP_WARNED += 1
    if _OFFSET_CLAMP_WARNED <= 10 or _OFFSET_CLAMP_WARNED % 500 == 0:
        import warnings
        warnings.warn(f"[compute_loss/{where}] {n_over} offset target(s) > max_offset={max_offset} -> "
                      f"silently clamped to a WRONG parent; filter upstream. [{_OFFSET_CLAMP_WARNED} events]")


# Action constants (must match adt_tokenizer)
ADD_INIT  = 0
ADD_CHAIN = 1
ADD_ANGLE = 2
ADD       = 3
LINK      = 4
END       = 5
# ADT_HLINK=1 to enable 7-action tokens (HLINK token for non-bonded proximity)
_HLINK_ENABLED = os.environ.get("ADT_HLINK", "0") == "1"
if _HLINK_ENABLED:
    HLINK = 6
    N_ACTIONS = 7
else:
    N_ACTIONS = 6


def gaussian_soft_label(true_bin, n_bins, sigma, device='cpu'):
    bins = torch.arange(n_bins, dtype=torch.float, device=device)
    true_bin = true_bin.float().unsqueeze(-1)
    target = torch.exp(-(bins - true_bin) ** 2 / (2 * sigma ** 2))
    return target / target.sum(dim=-1, keepdim=True)


class ADTv2Model(nn.Module):
    """
    ADT (emb_offset). 7 tokens per step:
      [action] [from] [atom/to] [r] [hp0/θc] [hp1/θf] [hp2]
       slot 0   slot 1  slot 2   slot 3  slot 4   slot 5   slot 6

    Active slots per action type:
      ADD_INIT  (0): slot 0, 2                    (atom_type only)
      ADD_CHAIN (1): slot 0, 1, 2, 3              (from, atom, r)
      ADD_ANGLE (2): slot 0, 1, 2, 3, 4, 5        (from, atom, r, θ_c, θ_f)
      ADD       (3): slot 0, 1, 2, 3, 4, 5, 6     (from, atom, r, hp0-2)
      LINK      (4): slot 0, 1, 2, 3, 4, 5, 6     (from, to, r, hp0-2)
      END       (5): slot 0 only

    Parent pointers (slot1 = from, slot2 when LINK = to) carry the 1-indexed OFFSET and are embedded via
    emb_offset(offset-1) over [0..max_offset-1]. No positional encoding (translation symmetry).
    """

    def __init__(self, config=None):
        super().__init__()
        c = config or {}
        self.d_model = c.get('d_model', 256)
        self.n_heads = c.get('n_heads', 8)
        self.n_layers = c.get('n_layers', 8)
        self.d_ff = c.get('d_ff', 1024)
        self.dropout = c.get('dropout', 0.1)
        # offset = parent distance in [1..max_offset]; bounds the pointer span (NOT the atom count).
        self.max_offset = c.get('max_offset', 32)
        # input_pointer_mode is always 'offset' for this dedicated build; recorded for ckpt self-description.
        self.input_pointer_mode = 'offset'
        self.output_pointer_mode = 'offset'
        self.n_atom_types = c.get('n_atom_types', 119)
        self.n_r_bins = c.get('n_r_bins', 100)
        self.n_hp0 = c.get("n_hp0", 12)   # also θ_coarse bins
        self.n_hp1 = c.get("n_hp1", 16)   # also θ_fine bins
        self.n_hp2 = c.get("n_hp2", 16)
        self.r_sigma = c.get('r_sigma', 1.5)

        d = self.d_model

        # --- Input Embeddings --- (emb_offset replaces the legacy emb_pointer; all other names identical)
        self.emb_action = nn.Embedding(N_ACTIONS, d)
        self.emb_offset = nn.Embedding(self.max_offset, d)   # parent OFFSET (1-indexed -> index offset-1)
        self.emb_atom = nn.Embedding(self.n_atom_types, d)
        self.emb_r = nn.Embedding(self.n_r_bins, d)
        self.emb_hp0 = nn.Embedding(self.n_hp0, d)   # shared: hp0 / θ_coarse
        self.emb_hp1 = nn.Embedding(self.n_hp1, d)   # shared: hp1 / θ_fine
        self.emb_hp2 = nn.Embedding(self.n_hp2, d)
        self.emb_slot = nn.Embedding(7, d)

        # --- Transformer ---
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=self.n_heads, dim_feedforward=self.d_ff,
            dropout=self.dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.n_layers)
        self.ln_final = nn.LayerNorm(d)

        # --- Output Heads --- (head_from/head_to emit an offset class, width max_offset)
        self.head_action = nn.Linear(d, N_ACTIONS)
        self.head_from = nn.Linear(d, self.max_offset)
        self.head_to = nn.Linear(d, self.max_offset)
        self.head_atom = nn.Linear(d, self.n_atom_types)
        self.head_r = nn.Linear(d, self.n_r_bins)
        self.head_hp0 = nn.Linear(d, self.n_hp0)
        self.head_hp1 = nn.Linear(d, self.n_hp1)
        self.head_hp2 = nn.Linear(d, self.n_hp2)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _emb_offset_vals(self, vals):
        """Embed a stored 1-indexed offset: index = offset-1, clamped to [0, max_offset-1].
        INIT/NULL stored as 0 -> (0-1) clamps to 0 (a loss-masked slot, so the collision is harmless)."""
        return self.emb_offset((vals - 1).clamp(0, self.max_offset - 1))

    def embed_input(self, input_values, input_slots, action_types):
        B, L = input_values.shape
        d = self.d_model
        device = input_values.device

        emb = torch.zeros(B, L, d, device=device)

        for slot in range(7):
            mask = (input_slots == slot)
            if not mask.any():
                continue

            vals = input_values[mask]

            if slot == 0:
                emb[mask] = self.emb_action(vals.clamp(0, N_ACTIONS - 1))
            elif slot == 1:
                # from-pointer: stored OFFSET -> emb_offset
                emb[mask] = self._emb_offset_vals(vals)
            elif slot == 2:
                # atom_type for ADD_INIT/ADD_CHAIN/ADD_ANGLE/ADD (actions 0-3); to-pointer for LINK (action 4)
                acts = action_types[mask]
                is_atom = (acts != LINK)
                if is_atom.any():
                    emb_flat = emb[mask].clone()
                    emb_flat[is_atom] = self.emb_atom(
                        vals[is_atom].clamp(0, self.n_atom_types - 1))
                    is_ptr = ~is_atom
                    if is_ptr.any():
                        emb_flat[is_ptr] = self._emb_offset_vals(vals[is_ptr])
                    emb[mask] = emb_flat
                else:
                    emb[mask] = self._emb_offset_vals(vals)
            elif slot == 3:
                emb[mask] = self.emb_r(vals.clamp(0, self.n_r_bins - 1))
            elif slot == 4:
                emb[mask] = self.emb_hp0(vals.clamp(0, self.n_hp0 - 1))
            elif slot == 5:
                emb[mask] = self.emb_hp1(vals.clamp(0, self.n_hp1 - 1))
            elif slot == 6:
                emb[mask] = self.emb_hp2(vals.clamp(0, self.n_hp2 - 1))

        slot_emb = self.emb_slot(input_slots)
        return emb + slot_emb

    def forward(self, input_values, input_slots, action_types,
                padding_mask=None):
        B, L = input_values.shape
        device = input_values.device

        x = self.embed_input(input_values, input_slots, action_types)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=device)

        h = self.transformer(
            x, mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        h = self.ln_final(h)

        logits = {
            0: self.head_action(h),
            1: self.head_from(h),
            'atom': self.head_atom(h),
            'to': self.head_to(h),
            3: self.head_r(h),
            4: self.head_hp0(h),
            5: self.head_hp1(h),
            6: self.head_hp2(h),
        }
        return logits, h

    def _build_slot_mask(self, action_types, target_slots, valid_mask):
        """
        Build per-position mask: True = this slot should contribute to loss.

        Active slots by action type:
          INIT:   0, 2
          CHAIN:  0, 1, 2, 3
          ANGLE:  0, 1, 2, 3, 4, 5
          ADD:    0, 1, 2, 3, 4, 5, 6
          LINK:   0, 1, 2, 3, 4, 5, 6
          END:    0
        """
        B, L = action_types.shape
        device = action_types.device

        acts = action_types  # (B, L)  action of the step containing position t
        slots = target_slots  # (B, L)

        slot_active = valid_mask.clone()

        # INIT: mask slots 1, 3, 4, 5, 6
        is_init = (acts == ADD_INIT)
        slot_active &= ~(is_init & ((slots == 1) | (slots == 3) |
                                     (slots == 4) | (slots == 5) | (slots == 6)))

        # CHAIN: mask slots 4, 5, 6
        is_chain = (acts == ADD_CHAIN)
        slot_active &= ~(is_chain & ((slots == 4) | (slots == 5) | (slots == 6)))

        # ANGLE: mask slot 6
        is_angle = (acts == ADD_ANGLE)
        slot_active &= ~(is_angle & (slots == 6))

        # END: mask slots 1-6
        is_end = (acts == END)
        slot_active &= ~(is_end & (slots != 0))

        return slot_active

    def compute_loss(self, logits, target_values, target_slots,
                     action_types, valid_mask):
        B, L = target_values.shape
        device = target_values.device
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)

        # Phase-aware slot mask
        slot_active = self._build_slot_mask(action_types, target_slots, valid_mask)

        for slot in range(7):
            slot_mask = (target_slots == slot) & slot_active
            if not slot_mask.any():
                name = ['action', 'from', 'atom_to', 'r', 'hp0', 'hp1', 'hp2'][slot]
                loss_dict[name] = 0.0
                continue

            targets = target_values[slot_mask]

            if slot == 0:  # action
                preds = logits[0][slot_mask]
                loss = F.cross_entropy(preds, targets.clamp(0, N_ACTIONS - 1))
                loss_dict['action'] = loss.detach()

            elif slot == 1:  # from-pointer: target is the stored OFFSET; class = offset-1
                preds = logits[1][slot_mask]
                _over = targets > self.max_offset
                if _over.any():
                    _warn_offset_clamp("from", int(_over.sum()), self.max_offset)
                tgt = (targets - 1).clamp(0, self.max_offset - 1)
                loss = F.cross_entropy(preds, tgt)
                loss_dict['from'] = loss.detach()

            elif slot == 2:  # atom (ADD) OR to-pointer (LINK)
                acts = action_types[slot_mask]
                is_atom = (acts != LINK)
                loss = torch.tensor(0.0, device=device)
                n = 0

                if is_atom.any():
                    preds_atom = logits['atom'][slot_mask][is_atom]
                    tgt_atom = targets[is_atom].clamp(0, self.n_atom_types - 1)
                    loss = loss + F.cross_entropy(preds_atom, tgt_atom) * is_atom.sum()
                    n += is_atom.sum()

                is_link = (acts == LINK)
                if is_link.any():
                    preds_to = logits['to'][slot_mask][is_link]
                    # to-pointer target is the stored OFFSET; class = offset-1
                    _lt = targets[is_link]
                    _over = _lt > self.max_offset
                    if _over.any():
                        _warn_offset_clamp("to(LINK)", int(_over.sum()), self.max_offset)
                    tgt_to = (_lt - 1).clamp(0, self.max_offset - 1)
                    loss = loss + F.cross_entropy(preds_to, tgt_to) * is_link.sum()
                    n += is_link.sum()

                if n > 0:
                    loss = loss / n
                loss_dict['atom_to'] = loss.detach()

            elif slot == 3:  # r (soft label)
                preds = logits[3][slot_mask]
                log_probs = F.log_softmax(preds, dim=-1)
                soft_targets = gaussian_soft_label(
                    targets.clamp(0, self.n_r_bins - 1),
                    self.n_r_bins, self.r_sigma, device=device)
                loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')
                loss_dict['r'] = loss.item()

            elif slot == 4:  # hp0 / θ_coarse (12 classes)
                preds = logits[4][slot_mask]
                loss = F.cross_entropy(preds, targets.clamp(0, self.n_hp0 - 1))
                loss_dict['hp0'] = loss.item()

            elif slot == 5:  # hp1 / θ_fine (16 classes)
                preds = logits[5][slot_mask]
                loss = F.cross_entropy(preds, targets.clamp(0, self.n_hp1 - 1))
                loss_dict['hp1'] = loss.item()

            elif slot == 6:  # hp2 (16 classes)
                preds = logits[6][slot_mask]
                loss = F.cross_entropy(preds, targets.clamp(0, self.n_hp2 - 1))
                loss_dict['hp2'] = loss.item()

            total_loss = total_loss + loss

        return total_loss, loss_dict

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config=None):
    default_config = {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 8,
        'd_ff': 1024,
        'dropout': 0.1,
        'max_offset': 32,
        'n_atom_types': 119,
        'n_r_bins': 100,
        'r_sigma': 1.5,
    }
    if config:
        default_config.update(config)
    return ADTv2Model(default_config)
