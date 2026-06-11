"""
adt_model.py — Bootstrap Canonical Encoding (v3)

Changes from v2:
  - Action vocab: 6 (ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD, LINK, END)
  - Phase-aware loss mask: each action type specifies which slots are active
  - slot 2 routing: atom_type for actions 0-3, pointer for action 4 (LINK)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# Action constants (must match adt_tokenizer_v3.py)
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
    ADT v2 with bootstrap canonical encoding.

    7 tokens per step:
      [action] [from] [atom/to] [r] [hp0/θc] [hp1/θf] [hp2]
       slot 0   slot 1  slot 2   slot 3  slot 4   slot 5   slot 6

    Active slots per action type:
      ADD_INIT  (0): slot 0, 2                    (atom_type only)
      ADD_CHAIN (1): slot 0, 1, 2, 3              (from, atom, r)
      ADD_ANGLE (2): slot 0, 1, 2, 3, 4, 5        (from, atom, r, θ_c, θ_f)
      ADD       (3): slot 0, 1, 2, 3, 4, 5, 6     (from, atom, r, hp0-2)
      LINK      (4): slot 0, 1, 2, 3, 4, 5, 6     (from, to, r, hp0-2)
      END       (5): slot 0 only
    """

    def __init__(self, config=None):
        super().__init__()
        c = config or {}
        self.d_model = c.get('d_model', 256)
        self.n_heads = c.get('n_heads', 8)
        self.n_layers = c.get('n_layers', 8)
        self.d_ff = c.get('d_ff', 1024)
        self.dropout = c.get('dropout', 0.1)
        self.max_pointer = c.get('max_pointer', 30)
        self.pointer_mode = c.get('pointer_mode', 'embedding')  # 'embedding' or 'attention'
        # v26: output_pointer_mode='offset' makes head_from/head_to predict relative offset
        #      (offset = current_atom_count + 1 - target) over [1..max_offset].
        #      Input embedding still uses absolute from_id (stable atom identity).
        self.output_pointer_mode = c.get('output_pointer_mode', 'absolute')  # 'absolute' or 'offset'
        self.max_offset = c.get('max_offset', 32)
        self.n_atom_types = c.get('n_atom_types', 119)
        self.n_r_bins = c.get('n_r_bins', 100)
        self.n_hp0 = c.get("n_hp0", 12)   # also θ_coarse bins
        self.n_hp1 = c.get("n_hp1", 16)   # also θ_fine bins
        self.n_hp2 = c.get("n_hp2", 16)
        self.r_sigma = c.get('r_sigma', 1.5)

        d = self.d_model

        # --- Input Embeddings ---
        self.emb_action = nn.Embedding(N_ACTIONS, d)
        self.emb_pointer = nn.Embedding(self.max_pointer, d)
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

        # --- Output Heads ---
        self.head_action = nn.Linear(d, N_ACTIONS)
        if self.pointer_mode == 'attention':
            self.pointer_query = nn.Linear(d, d, bias=False)
            self.pointer_key = nn.Linear(d, d, bias=False)
        else:
            out_dim = self.max_offset if self.output_pointer_mode == 'offset' else self.max_pointer
            self.head_from = nn.Linear(d, out_dim)
            self.head_to = nn.Linear(d, out_dim)
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
                emb[mask] = self.emb_pointer(vals.clamp(0, self.max_pointer - 1))
            elif slot == 2:
                # atom_type for ADD_INIT/ADD_CHAIN/ADD_ANGLE/ADD (actions 0-3)
                # pointer for LINK (action 4)
                acts = action_types[mask]
                is_atom = (acts != LINK)
                if is_atom.any():
                    emb_flat = emb[mask].clone()
                    emb_flat[is_atom] = self.emb_atom(
                        vals[is_atom].clamp(0, self.n_atom_types - 1))
                    is_ptr = ~is_atom
                    if is_ptr.any():
                        emb_flat[is_ptr] = self.emb_pointer(
                            vals[is_ptr].clamp(0, self.max_pointer - 1))
                    emb[mask] = emb_flat
                else:
                    emb[mask] = self.emb_pointer(
                        vals.clamp(0, self.max_pointer - 1))
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

        if self.pointer_mode == 'attention':
            pointer_logits = self._compute_pointer_attention(
                h, input_slots, action_types, padding_mask)
            logits = {
                0: self.head_action(h),
                1: pointer_logits,
                'atom': self.head_atom(h),
                'to': pointer_logits,
                3: self.head_r(h),
                4: self.head_hp0(h),
                5: self.head_hp1(h),
                6: self.head_hp2(h),
            }
        else:
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

    def _compute_pointer_attention(self, h, input_slots, action_types, padding_mask):
        """Content-based pointer via attention over placed atom hidden states."""
        B, L, d = h.shape
        device = h.device

        # Atom endpoints: slot 6 of ADD-type steps (0=INIT,1=CHAIN,2=ANGLE,3=ADD)
        is_slot6 = (input_slots == 6)
        is_add = (action_types <= ADD)
        atom_end = is_slot6 & is_add  # (B, L)
        if padding_mask is not None:
            atom_end = atom_end & ~padding_mask

        # Collect atom keys per batch (padded to max_atoms)
        max_atoms = max(atom_end.sum(dim=1).max().item(), 1)
        atom_keys = h.new_zeros(B, max_atoms, d)
        atom_positions = h.new_full((B, max_atoms), L, dtype=torch.long)
        atom_valid = h.new_zeros(B, max_atoms, dtype=torch.bool)

        for b in range(B):
            pos = atom_end[b].nonzero(as_tuple=True)[0]
            n = min(len(pos), max_atoms)
            atom_keys[b, :n] = h[b, pos[:n]]
            atom_positions[b, :n] = pos[:n]
            atom_valid[b, :n] = True

        # Query and key projections
        Q = self.pointer_query(h)          # (B, L, d)
        K = self.pointer_key(atom_keys)    # (B, max_atoms, d)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (d ** 0.5)  # (B, L, max_atoms)

        # Causal mask: only attend to atoms placed before current position
        all_pos = torch.arange(L, device=device).view(1, L, 1)
        causal = atom_positions.unsqueeze(1) < all_pos  # (B, L, max_atoms)
        valid = atom_valid.unsqueeze(1).expand(B, L, max_atoms)
        mask = causal & valid

        scores = scores.masked_fill(~mask, -float('inf'))
        # Guard: positions with no valid atoms get zero logits (masked in loss)
        all_inf = mask.sum(dim=2) == 0  # (B, L)
        if all_inf.any():
            scores[all_inf] = 0.0
        return scores

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

        # action_types[i,t] = action of the step containing position t
        acts = action_types  # (B, L)
        slots = target_slots  # (B, L)

        # Start with valid_mask
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

        # For offset output mode: compute per-position ref_count = atom_count_before_step + 1
        # so that target_offset = ref_count - target (works for both ADD and LINK).
        use_offset = (self.output_pointer_mode == 'offset')
        if use_offset:
            # Per-step action: take from every N_SLOTS-th position (each step's slot 0).
            # action_types is filled per-position with the containing step's action.
            step_start_idx = torch.arange(0, L, 7, device=device)  # N_SLOTS=7
            per_step_action = action_types[:, step_start_idx]  # (B, n_steps)
            is_atom_step = (per_step_action <= ADD).long()  # (B, n_steps)
            # atom_count_before_step[s] = cumsum up to s-1
            cs = is_atom_step.cumsum(dim=1)
            atom_count_before_step = torch.cat(
                [torch.zeros(B, 1, dtype=torch.long, device=device), cs[:, :-1]], dim=1)
            # Per-position step idx
            step_idx = torch.arange(L, device=device) // 7  # (L,)
            step_idx = step_idx.clamp(max=atom_count_before_step.shape[1] - 1)
            # Broadcast to (B, L)
            atom_count_before_pos = atom_count_before_step.gather(
                1, step_idx.unsqueeze(0).expand(B, -1))
            ref_count = atom_count_before_pos + 1  # (B, L)

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

            elif slot == 1:  # from
                preds = logits[1][slot_mask]
                if self.pointer_mode == 'attention':
                    tgt = (targets - 1).clamp(0, preds.shape[-1] - 1)
                elif use_offset:
                    # offset = ref_count - target, class = offset - 1
                    ref = ref_count[slot_mask]
                    tgt = (ref - targets - 1).clamp(0, self.max_offset - 1)
                else:
                    tgt = targets.clamp(0, self.max_pointer - 1)
                loss = F.cross_entropy(preds, tgt)
                loss_dict['from'] = loss.detach()

            elif slot == 2:  # atom/to
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
                    if self.pointer_mode == 'attention':
                        tgt_to = (targets[is_link] - 1).clamp(0, preds_to.shape[-1] - 1)
                    elif use_offset:
                        ref = ref_count[slot_mask][is_link]
                        tgt_to = (ref - targets[is_link] - 1).clamp(0, self.max_offset - 1)
                    else:
                        tgt_to = targets[is_link].clamp(0, self.max_pointer - 1)
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
        'max_pointer': 30,
        'pointer_mode': 'embedding',
        'output_pointer_mode': 'absolute',
        'max_offset': 32,
        'n_atom_types': 119,
        'n_r_bins': 100,
        'r_sigma': 1.5,
    }
    if config:
        default_config.update(config)
    return ADTv2Model(default_config)
