"""Batched log-probability computation for sampled trajectories.

Mirrors model.compute_loss slot routing but computes per-sample log_p (sum
over generated portion). Used for REINFORCE loss and KL anchor.
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/ADT/common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # version dir: train (emb_offset)

import torch
import torch.nn.functional as F
from adt_tokenizer import ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD, LINK, END, N_ACTIONS
from train import N_SLOTS

PAD_VALUE = -100  # ignored slot


def build_batch(tokens_list, n_frame_tokens_list, device):
    """Pack list of token sequences into padded batch tensor.

    Returns:
      tokens_padded   : (B, Lmax)  PAD_VALUE in padding
      lengths         : (B,)        true len of each sequence
      frame_lens      : (B,)        length of frame portion (= no grad target)
      grad_mask       : (B, Lmax)   True where gradient should be computed
                                    (= post-frame, pre-padding, target position)
    """
    B = len(tokens_list)
    lengths = [len(t) for t in tokens_list]
    Lmax = max(lengths)

    tokens_padded = torch.full((B, Lmax), PAD_VALUE, dtype=torch.long, device=device)
    grad_mask = torch.zeros(B, Lmax, dtype=torch.bool, device=device)
    for i, (tk, nf) in enumerate(zip(tokens_list, n_frame_tokens_list)):
        L = len(tk)
        tokens_padded[i, :L] = torch.tensor(tk, dtype=torch.long, device=device)
        # gradient is computed for target positions in [nf, L-1].
        # We predict token at pos t using context [0..t-1], so target slot at t.
        grad_mask[i, nf:L] = True
    return tokens_padded, torch.tensor(lengths, device=device), \
           torch.tensor(n_frame_tokens_list, device=device), grad_mask


def censored_size_kl(p_end, k_at, p_size_target, eps=1e-3):
    """Size loss L_size = KL(P_tgt || P_hat_theta) via the CENSORED virtual size distribution.

    KLRLmethod.md §4/§7/§13. The model's true size marginal needs hazards beyond the
    sampled END (truncation). We sidestep it with right-censoring (Kaplan-Meier): aggregate
    the realized per-decision END hazards into an at-risk-averaged hazard hhat[k], form its
    survival distribution over the OBSERVED sizes [0, n_obs), and lump everything beyond into
    a single censored bucket. Forward-KL (mass-covering -> holds mean+sigma+shape) against the
    deterministic target size pmf.

    Args:
      p_end        : (M,) differentiable END hazard at each slot-0 decision = P(END | k atoms).
      k_at         : (M,) long, atom-count k at that decision (= molecule size if END fires here).
      p_size_target: (Kmax,) target size pmf indexed by ABSOLUTE atom count (sums to ~1). const.
    Returns scalar L_size (>=0), differentiable through p_end. k_at and the at-risk counts are
    sample-derived constants (no grad) = risk-set-fixed Kaplan-Meier."""
    device = p_end.device
    Kmax = p_size_target.shape[0]
    n_obs = min(int(k_at.max().item()) + 1, Kmax)      # resolve bins [0, n_obs); rest = censored
    k = k_at.clamp(0, n_obs - 1)
    # at-risk aggregate hazard hhat[j] = mean p_end over decisions with k==j  (eq 5)
    sum_h = torch.zeros(n_obs, device=device).index_add_(0, k, p_end)
    cnt = torch.zeros(n_obs, device=device).index_add_(0, k, torch.ones_like(p_end))
    hhat = sum_h / cnt.clamp_min(1.0)                  # empty bins -> 0 (no at-risk decision there)
    # survival -> resolved bins + censored bucket  (eq 6; telescoping => sum(P_hat)+bucket = 1)
    surv = torch.cumprod((1.0 - hhat).clamp(eps, 1.0), dim=0)        # surv[j] = prod_{i<=j}(1-hhat[i])
    surv_prev = torch.cat([torch.ones(1, device=device), surv[:-1]])
    P_hat = hhat * surv_prev                           # (n_obs,) model resolved bins
    bucket = surv[-1]                                  # model censored mass (sizes >= n_obs)
    # deterministic target split at the same n_obs
    P_tgt = p_size_target[:n_obs]
    tgt_bucket = p_size_target[n_obs:].sum()
    # forward KL(P_tgt || P_hat) over {resolved bins} + {censored bucket}  (eq 7)
    L = (P_tgt * (torch.log(P_tgt + eps) - torch.log(P_hat + eps))).sum() \
        + tgt_bucket * (torch.log(tgt_bucket + eps) - torch.log(bucket + eps))
    return L


def compute_log_p_batch(model, tokens_padded, grad_mask, p_size_target=None, f_target=None, elem_idx=None, size_eps=1e-3, comp_eta=0.1, return_haz=False):
    """Compute per-sample log_p over the gradient-marked positions.

    Args:
      tokens_padded : (B, L)  full token sequence, PAD_VALUE for padding
      grad_mask     : (B, L)  True where this position's target is to be
                              included in log_p sum

    Returns:
      log_p_per_sample : (B,) float tensor with grad
    """
    B, L = tokens_padded.shape
    device = tokens_padded.device

    # Build input_values = tokens_padded[:, :-1], target_values = tokens_padded[:, 1:]
    input_values = tokens_padded[:, :-1].clone()
    target_values = tokens_padded[:, 1:].clone()
    target_grad_mask = grad_mask[:, 1:].clone()

    # PAD positions in input: replace -100 with safe value (will be masked out)
    input_safe = input_values.clone()
    input_safe[input_values == PAD_VALUE] = 0
    # Padding mask for transformer: True where padding
    padding_mask = (input_values == PAD_VALUE)

    # Slots: 0..6 cycling
    input_slots = (torch.arange(L - 1, device=device) % N_SLOTS).unsqueeze(0).expand(B, -1)
    target_slots = (torch.arange(1, L, device=device) % N_SLOTS).unsqueeze(0).expand(B, -1)

    # action_types: for each input position i, the action at step containing i.
    # The action is at position s_start = (i // N_SLOTS) * N_SLOTS.
    step_idx = torch.arange(L - 1, device=device) // N_SLOTS  # (L-1,)
    s_start = step_idx * N_SLOTS  # (L-1,)
    # action_types[b, i] = input_safe[b, s_start[i]]
    action_types = input_safe.gather(1, s_start.unsqueeze(0).expand(B, -1))

    # Forward
    logits, _ = model(input_safe, input_slots, action_types, padding_mask)

    # Now compute log_p per position per slot, gather at target.
    # logits structure: dict-like with keys 0,1,3,4,5,6 and 'atom','to'.
    # For slot 2: choose 'atom' if action != LINK, else 'to'.

    use_offset = (getattr(model, 'output_pointer_mode', 'absolute') == 'offset')
    use_in_offset = (getattr(model, 'input_pointer_mode', 'absolute') == 'offset')  # v2: token stores OFFSET directly
    max_offset = getattr(model, 'max_offset', 50)
    n_atom_types = getattr(model, 'n_atom_types', 119)
    n_r_bins = getattr(model, 'n_r_bins', 200)

    # Per-position ref_count for offset mode
    if use_offset:
        # is_atom_step at each step start = action <= ADD (= 0,1,2,3)
        # We need this on the input_values to know how many atoms have been placed.
        # step_count uses positions s*N_SLOTS to get the action.
        n_steps = (L - 1 + N_SLOTS - 1) // N_SLOTS
        per_step_idx = torch.arange(0, L - 1, N_SLOTS, device=device)  # (n_steps,)
        if per_step_idx.shape[0] == 0:
            ref_count = torch.zeros(B, L - 1, dtype=torch.long, device=device)
        else:
            per_step_action = input_safe[:, per_step_idx]  # (B, n_steps)
            is_atom_step = (per_step_action <= ADD).long()
            cs = is_atom_step.cumsum(dim=1)
            atom_count_before_step = torch.cat(
                [torch.zeros(B, 1, dtype=torch.long, device=device), cs[:, :-1]], dim=1)
            step_pos_idx = torch.arange(L - 1, device=device) // N_SLOTS
            step_pos_idx = step_pos_idx.clamp(max=atom_count_before_step.shape[1] - 1)
            atom_count_before_pos = atom_count_before_step.gather(
                1, step_pos_idx.unsqueeze(0).expand(B, -1))
            ref_count = atom_count_before_pos + 1  # (B, L-1)

    log_p_total = torch.zeros(B, device=device)

    # Get per-slot log_p contributions
    for slot in range(N_SLOTS):
        slot_mask = (target_slots == slot) & target_grad_mask
        if not slot_mask.any():
            continue

        flat_mask = slot_mask.view(-1)
        flat_targets = target_values.view(-1)[flat_mask]
        flat_action_types = action_types.view(-1)[flat_mask]
        # Indices of selected positions in (B, L-1) flat
        bi = torch.arange(B, device=device).unsqueeze(1).expand(B, L - 1).reshape(-1)[flat_mask]

        if slot == 0:  # action
            preds = logits[0].reshape(-1, N_ACTIONS)[flat_mask]
            tgt = flat_targets.clamp(0, N_ACTIONS - 1)
            log_softmax = F.log_softmax(preds, dim=-1)
            log_p_at_t = log_softmax.gather(1, tgt.unsqueeze(1)).squeeze(1)

        elif slot == 1:  # from
            preds = logits[1].reshape(-1, logits[1].shape[-1])[flat_mask]
            if use_in_offset:
                tgt = (flat_targets - 1).clamp(0, max_offset - 1)   # v2: target IS the stored offset (class=offset-1)
            elif use_offset:
                ref = ref_count.view(-1)[flat_mask]
                tgt = (ref - flat_targets - 1).clamp(0, max_offset - 1)
            else:
                raise RuntimeError("emb_offset build expects offset pointer mode (no absolute/emb_pointer path)")
            log_softmax = F.log_softmax(preds, dim=-1)
            log_p_at_t = log_softmax.gather(1, tgt.unsqueeze(1)).squeeze(1)

        elif slot == 2:  # atom OR to
            is_atom = (flat_action_types != LINK)
            log_p_at_t = torch.zeros_like(flat_targets, dtype=torch.float)
            if is_atom.any():
                preds_atom_full = logits['atom'].reshape(-1, n_atom_types)
                preds_atom = preds_atom_full[flat_mask][is_atom]
                tgt_atom = flat_targets[is_atom].clamp(0, n_atom_types - 1)
                ls = F.log_softmax(preds_atom, dim=-1)
                log_p_at_t[is_atom] = ls.gather(1, tgt_atom.unsqueeze(1)).squeeze(1)
            is_link = (flat_action_types == LINK)
            if is_link.any():
                preds_to_full = logits['to'].reshape(-1, logits['to'].shape[-1])
                preds_to = preds_to_full[flat_mask][is_link]
                if use_in_offset:
                    tgt_to = (flat_targets[is_link] - 1).clamp(0, max_offset - 1)   # v2: target IS the stored offset
                elif use_offset:
                    ref = ref_count.view(-1)[flat_mask][is_link]
                    tgt_to = (ref - flat_targets[is_link] - 1).clamp(0, max_offset - 1)
                else:
                    raise RuntimeError("emb_offset build expects offset pointer mode (no absolute/emb_pointer path)")
                ls = F.log_softmax(preds_to, dim=-1)
                log_p_at_t[is_link] = ls.gather(1, tgt_to.unsqueeze(1)).squeeze(1)

        elif slot == 3:
            preds = logits[3].reshape(-1, n_r_bins)[flat_mask]
            tgt = flat_targets.clamp(0, n_r_bins - 1)
            log_softmax = F.log_softmax(preds, dim=-1)
            log_p_at_t = log_softmax.gather(1, tgt.unsqueeze(1)).squeeze(1)

        else:  # 4, 5, 6
            n_classes = logits[slot].shape[-1]
            preds = logits[slot].reshape(-1, n_classes)[flat_mask]
            tgt = flat_targets.clamp(0, n_classes - 1)
            log_softmax = F.log_softmax(preds, dim=-1)
            log_p_at_t = log_softmax.gather(1, tgt.unsqueeze(1)).squeeze(1)

        # Scatter add to per-sample log_p
        log_p_total.index_add_(0, bi, log_p_at_t)

    if return_haz:
        # diagnostic: return the per-decision END hazard p_end + its atom-count k (no grad) so a
        # caller can accumulate hhat[k] over many batches and build P_hat_theta(n) for comparison.
        fm0 = ((target_slots == 0) & target_grad_mask).view(-1)
        a_logits = logits[0].reshape(-1, N_ACTIONS)[fm0]
        pe = F.softmax(a_logits, dim=-1)[:, END].detach()
        if use_offset:
            ka = (ref_count - 1).view(-1)[fm0].detach()
        else:
            ka = (torch.arange(L - 1, device=device) // N_SLOTS).unsqueeze(0).expand(B, -1).reshape(-1)[fm0]
        return pe, ka
    if p_size_target is None:
        return log_p_total
    # --- MERGED distribution-KL from the SAME forward (no 2nd forward = half the grad-activation memory).
    #     L_size = KL(P_tgt || P_hat_theta) via censored virtual size dist (KLRLmethod §4/§7);
    #     L_spec = KL(f_tgt || f_bar_theta) expected element fraction -> GEOM (§5). ---
    _eps = 1e-6
    # L_size: gather differentiable per-decision END hazard + its atom-count, then censored forward-KL
    L_size = torch.tensor(0.0, device=device)
    fm0 = ((target_slots == 0) & target_grad_mask).view(-1)
    if fm0.any():
        a_logits = logits[0].reshape(-1, N_ACTIONS)[fm0]
        p_end = F.softmax(a_logits, dim=-1)[:, END].clamp(_eps, 1.0 - _eps)
        if use_offset:
            k_at = (ref_count - 1).view(-1)[fm0]
        else:
            k_at = (torch.arange(L - 1, device=device) // N_SLOTS).unsqueeze(0).expand(B, -1).reshape(-1)[fm0]
        k_at = k_at.clamp(0, p_size_target.shape[0] - 1)
        L_size = censored_size_kl(p_end, k_at, p_size_target, eps=size_eps)
    # L_spec = element composition loss, RELATIVE floor eps_e = comp_eta*f_tgt (§5 eq 8').
    # ★2026-07-03 BUGFIX (Br sacrifice): the OLD forward-KL Σ f_tgt(e)·log(f_tgt/f_model) weighted each
    # element's log-ratio by f_tgt(e) → majors (C/N/O) dominated, rare elements (Br 0.19%) were nearly
    # ignored → estrain crushed Br (0.19→0.06) undetected (KLsp stayed low; the f_tgt(e) coefficient
    # re-imposed the abundance bias that eps_e was meant to remove). NEW = EQUAL-WEIGHT squared
    # log-ratio (mean over elements): scale-invariant per-element relative error, Br counts as much as
    # C. ≥0, =0 iff exact match (no sign-cancellation → KLsp usable as a composition-health metric).
    L_spec = torch.tensor(0.0, device=device)
    fm2 = ((target_slots == 2) & target_grad_mask & (action_types != LINK)).view(-1)
    if fm2.any():
        at_logits = logits['atom'].reshape(-1, logits['atom'].shape[-1])[fm2]
        p_atom = F.softmax(at_logits, dim=-1)
        f_model = p_atom[:, elem_idx].mean(dim=0)               # mean over ADD steps (softmax > 0)
        f_model = f_model / f_model.sum()
        eps_e = comp_eta * f_target
        _lr = torch.log(f_target + eps_e) - torch.log(f_model + eps_e)   # per-element log-ratio (signed)
        L_spec = (_lr * _lr).mean()                                      # equal-weight, mean over E (§5 eq 8')
    return log_p_total, L_size, L_spec


def compute_dist_kl_batch(model, tokens_padded, grad_mask, h_target, f_target, elem_idx, eps=1e-6):
    """DIRECT differentiable distribution-matching losses (pathwise = low variance, vs REINFORCE).
      L_size: per-step END hazard P_END(k) matched to the TARGET hazard h_target[k] -- bakes the
              size distribution into the END logits (k = #atoms placed before this slot-0 step;
              if END fires here the molecule has size k). = the crutch's hazard-match, but as a LOSS.
      L_spec: expected element fraction f_model (mean atom-type softmax over ADD steps) matched to
              GEOM f_target via KL -- bakes composition into the atom-type logits.
    Args:
      h_target : (Kmax+1,) target END-hazard per atom-count k (0 below min size).
      f_target : (E,) GEOM fractions over the atomic numbers in elem_idx (sums to 1).
      elem_idx : (E,) long, atomic-number indices into the atom-type softmax.
    Returns (L_size, L_spec) scalar tensors with grad (0.0 if no positions)."""
    B, L = tokens_padded.shape
    device = tokens_padded.device
    input_values = tokens_padded[:, :-1].clone()
    target_values = tokens_padded[:, 1:].clone()
    target_grad_mask = grad_mask[:, 1:].clone()
    input_safe = input_values.clone(); input_safe[input_values == PAD_VALUE] = 0
    padding_mask = (input_values == PAD_VALUE)
    input_slots = (torch.arange(L - 1, device=device) % N_SLOTS).unsqueeze(0).expand(B, -1)
    target_slots = (torch.arange(1, L, device=device) % N_SLOTS).unsqueeze(0).expand(B, -1)
    s_start = (torch.arange(L - 1, device=device) // N_SLOTS) * N_SLOTS
    action_types = input_safe.gather(1, s_start.unsqueeze(0).expand(B, -1))
    # #atoms placed before each position (offset-mode bookkeeping, same as compute_log_p_batch)
    per_step_idx = torch.arange(0, L - 1, N_SLOTS, device=device)
    per_step_action = input_safe[:, per_step_idx]
    is_atom_step = (per_step_action <= ADD).long()
    cs = is_atom_step.cumsum(dim=1)
    atom_count_before_step = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=device), cs[:, :-1]], dim=1)
    step_pos_idx = (torch.arange(L - 1, device=device) // N_SLOTS).clamp(max=atom_count_before_step.shape[1] - 1)
    atom_count_before_pos = atom_count_before_step.gather(1, step_pos_idx.unsqueeze(0).expand(B, -1))  # (B, L-1)

    logits, _ = model(input_safe, input_slots, action_types, padding_mask)

    # --- L_size: END hazard at slot-0 grad positions ---
    L_size = torch.tensor(0.0, device=device)
    fm0 = ((target_slots == 0) & target_grad_mask).view(-1)
    if fm0.any():
        a_logits = logits[0].reshape(-1, N_ACTIONS)[fm0]
        p_end = F.softmax(a_logits, dim=-1)[:, END].clamp(eps, 1.0 - eps)
        k_at = atom_count_before_pos.view(-1)[fm0].clamp(0, h_target.shape[0] - 1)
        h_tgt = h_target[k_at].clamp(eps, 1.0 - eps)
        L_size = F.binary_cross_entropy(p_end, h_tgt)

    # --- L_spec: expected element fraction at slot-2 ADD grad positions ---
    L_spec = torch.tensor(0.0, device=device)
    fm2 = ((target_slots == 2) & target_grad_mask & (action_types != LINK)).view(-1)
    if fm2.any():
        at_logits = logits['atom'].reshape(-1, logits['atom'].shape[-1])[fm2]
        p_atom = F.softmax(at_logits, dim=-1)
        f_model = p_atom[:, elem_idx].mean(dim=0) + eps
        f_model = f_model / f_model.sum()
        L_spec = (f_target * (torch.log(f_target + eps) - torch.log(f_model))).sum()  # KL(f_target || f_model)
    return L_size, L_spec
