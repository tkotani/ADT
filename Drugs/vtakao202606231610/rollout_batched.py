"""Batched token-returning rollout for RL (speed).

rollout_batch() generates B molecules IN LOCKSTEP with ONE model forward per slot
(batch-B) instead of B separate batch-1 rollouts (rollout_one_with_tokens). The
sampling + masking rules are byte-for-byte the same as the serial version, so the
sample DISTRIBUTION is identical (the RNG stream differs, so individual molecules
differ, but size/validity/atom-type statistics match within sampling noise).

Why: the serial rollout calls the model ~B*7*n_atoms times at batch-1, which leaves
the GPU bubbly (kernel-launch / Python-dispatch bound). Batching the B molecules
turns that into ~7*n_atoms forwards at batch-B => GPU saturated, ~order-of-magnitude
fewer launches. (KV-cache, an orthogonal O(L^2)->O(L) win, is a separate change.)

Verify equivalence with verify_rollout_batched.py before switching RL over.
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/ADT/common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # version dir: train (emb_offset), kv_cache

import numpy as np
import torch
import torch.nn.functional as F

from adt_tokenizer import (
    ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD, LINK, END,
    AtomEntry, reconstruct_from_tokens, array_to_tokens,
    decode_direction, compute_tree_frame, _ancestor_positions_recon,
)
from train import N_SLOTS
import relative_pointer  # common: absolute<->offset, for the emb_offset (v2) rollout token stream

ALLOWED_ATOMS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53}

# First-triple 'from' (slot-1 parent pointers), made explicit. The bootstrap frame is a 3-atom chain
# INIT -> CHAIN -> ANGLE; atom i's parent is the atom before it, encoded ABSOLUTELY as `from = i`
# (atom 0 = INIT has no real parent -> 0 is a placeholder). Named so the 0/1/2 read as deliberate.
TRIPLE0_FROM, TRIPLE1_FROM, TRIPLE2_FROM = 0, 1, 2

_DEFAULT_FRAME = [
    ADD_INIT,  TRIPLE0_FROM, 6, 0,  0, 0, 0,
    ADD_CHAIN, TRIPLE1_FROM, 6, 50, 0, 0, 0,
    ADD_ANGLE, TRIPLE2_FROM, 6, 50, 6, 8, 0,
]


def _sample_frame(frame_sampler):
    """Identical to the serial frame-inject (require a valid 3-atom ANGLE frame)."""
    if frame_sampler is not None:
        for _try in range(100):
            ft = frame_sampler.sample()
            if ft[2 * N_SLOTS] == ADD_ANGLE:
                break
        else:
            ft = list(_DEFAULT_FRAME)
    else:
        ft = list(_DEFAULT_FRAME)
    if hasattr(ft, "tolist"):
        ft = list(ft.tolist())
    else:
        ft = [int(x) for x in ft]
    return ft


def _bootstrap_from_frame(tokens):
    """Identical to the serial bootstrap: atoms/bonds/bootstrap_frame from the frame."""
    frame_arr = np.asarray(tokens, dtype=np.int64)
    frame_tuples = array_to_tokens(frame_arr)
    atoms_list, links_list = reconstruct_from_tokens(list(frame_tuples) + [('action', END)])
    atom_table = {i: ae for i, ae in enumerate(atoms_list)}
    n_atoms = len(atoms_list)
    bootstrap_frame = atoms_list[2].frame.copy() if n_atoms >= 3 else np.eye(3)
    bonds = set()
    for i, ae in enumerate(atoms_list):
        pid = getattr(ae, 'parent_id', 0)
        if pid and pid > 0:
            bonds.add((min(pid, i + 1), max(pid, i + 1)))
    for lk in links_list:
        a, b = int(lk[0]), int(lk[1])
        if a > 0 and b > 0:
            bonds.add((min(a, b), max(a, b)))
    return atom_table, bonds, n_atoms, bootstrap_frame


# --- generation-time valence mask (LINK_VALENCE_MASK=1) -------------------------------------------
# Measured 2026-07-12: 3.2% of 40-44-atom molecules DECLARE an over-coordinated atom (Cl/Br/F/I with two
# bonds, O with three). Those molecules almost never survive relaxation (XTP 5.6%) and account for ~11%
# of all failures at 40+. Each atom gets exactly ONE bond from its parent, so a second bond on a halogen
# can only come from (a) a LINK aimed at it, or (b) an ADD that chose it as the parent. Masking both at
# sampling time removes the defect with no retraining -- it is a decoding constraint, not a model change.
MAXDEG = {6: 4, 7: 4, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1}
VALENCE_MASK = os.environ.get("LINK_VALENCE_MASK", "0") == "1"


def _degrees(S):
    """degree of every existing atom (1-indexed), from the bond set built so far"""
    deg = {}
    for (a, b) in S['bonds']:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return deg


def _saturated(S, atom_id):
    """True if atom_id (1-indexed) already carries as many bonds as its element allows"""
    at = S['atom_table'].get(atom_id - 1)
    if at is None:
        return False
    return _degrees(S).get(atom_id, 0) >= MAXDEG.get(int(at.atomic_num), 4)


def _reconstruct_atom(S):
    """Identical to the serial atom reconstruction (mutates state dict S in place).
    Reads the last N_SLOTS tokens of S['tokens']; updates atom_table/bonds/n_atoms."""
    tokens = S['tokens']; action = S['action']; from_id = S['from_id']
    atom_table = S['atom_table']; bonds = S['bonds']; n_atoms = S['n_atoms']
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
        frame = compute_tree_frame(pos, anc_pos, S['bootstrap'])
        atom_table[n_atoms] = AtomEntry(
            pos=pos, parent_id=from_id, frame=frame.copy(),
            v1=(pos - parent_pos), v2=np.zeros(3), atomic_num=atom_val,
        )
        bonds.add((min(from_id, n_atoms + 1), max(from_id, n_atoms + 1)))
        S['n_atoms'] = n_atoms + 1
    elif action == LINK:
        # tokens[step_base+2] is the to-pointer: an OFFSET under emb_offset input, else absolute.
        _v = tokens[step_base + 2]
        to_id = (n_atoms - (_v - 1)) if S.get('use_in_offset') else _v
        bonds.add((min(from_id, to_id), max(from_id, to_id)))


def rollout_batch(model, frame_sampler, device, B, max_steps=200, temperature=1.0, end_bias=0.0, size_ceiling=None):
    """Generate B molecules in lockstep. Returns a list of B tuples identical in shape
    to rollout_one_with_tokens: (tokens, n_frame_tokens, atom_table, bonds, n_atoms, done).
    """
    model.eval()
    use_offset = getattr(model, 'output_pointer_mode', 'absolute') == 'offset'
    # ---- molecule SIZE ceiling for the rollout (END is forced once n_atoms reaches this) ----
    # emb_offset arch (vtakao202606231610) has NO emb_pointer -> NO absolute-atom-count limit.
    # The ONLY architectural constraint is max_offset (parent OFFSET span, enforced by the tokenizer
    # rejecting free-order orderings needing offset > max_offset); it bounds parent DISTANCE, not the
    # atom count (free-order references recent atoms, so offset stays < max_offset even for large mols).
    # => atom count is bounded ONLY by size_ceiling (=--size_nmax) at rollout time.
    # (Legacy absolute model capped atoms at emb_pointer.num_embeddings; that table + its input clamp
    #  are GONE here -- train.py: dead code removed 2026-06-14. The old "_ptr_rows-4 = 60" was a PHANTOM:
    #  getattr(model,'emb_pointer') is None on this arch -> silently fell back to 64 -> a fake 60 ceiling.
    #  FIXED 2026-07-06: cap = size_ceiling only; no phantom min-with-64.)
    _has_emb_pointer = getattr(model, 'emb_pointer', None) is not None            # legacy absolute model only
    if size_ceiling:
        max_atoms_cap = size_ceiling                                             # emb_offset: --size_nmax IS the cap
    elif _has_emb_pointer:
        max_atoms_cap = model.emb_pointer.num_embeddings - 4                      # legacy absolute model: real emb_pointer cap
    else:
        max_atoms_cap = 10 ** 9                                                   # emb_offset + no ceiling: unbounded (bounded by max_steps)

    # --- per-molecule state ---
    St = []
    for b in range(B):
        ft = _sample_frame(frame_sampler)
        atom_table, bonds, n_atoms, bootstrap = _bootstrap_from_frame(ft)
        St.append(dict(tokens=list(ft), n_frame=len(ft), atom_table=atom_table,
                       bonds=bonds, n_atoms=n_atoms, bootstrap=bootstrap,
                       done=False, active=True, action=None, from_id=1))

    def _forward(active):
        """One batched forward over the given active-row indices. Returns (logits_dict,
        list-of-last-real-position-per-row)."""
        seqs = [St[b]['tokens'] for b in active]
        maxL = max(len(s) for s in seqs)
        nb = len(seqs)
        arr = torch.zeros(nb, maxL, dtype=torch.long, device=device)
        slots = torch.zeros(nb, maxL, dtype=torch.long, device=device)
        acts = torch.zeros(nb, maxL, dtype=torch.long, device=device)
        pad = torch.ones(nb, maxL, dtype=torch.bool, device=device)
        rng = torch.arange(maxL, device=device)
        for i, s in enumerate(seqs):
            L = len(s)
            t = torch.tensor(s, dtype=torch.long, device=device)
            arr[i, :L] = t
            slots[i, :L] = rng[:L] % N_SLOTS
            grp = (rng[:L] // N_SLOTS) * N_SLOTS       # start index of each atom-group
            acts[i, :L] = t[grp]
            pad[i, :L] = False
        with torch.no_grad():
            logits, _ = model(arr, slots, acts, padding_mask=pad)
        lastpos = [len(s) - 1 for s in seqs]
        return logits, lastpos

    with torch.no_grad():
        for _step in range(max_steps):
            # halt molecules at the atom cap (runaway: done stays False)
            for b in range(B):
                if St[b]['active'] and St[b]['n_atoms'] >= max_atoms_cap:
                    St[b]['active'] = False
            active = [b for b in range(B) if St[b]['active']]
            if not active:
                break

            # ---- slot 0: action ----
            logits, lastpos = _forward(active)
            still = []
            for i, b in enumerate(active):
                al = (logits[0][i, lastpos[i]] / temperature).clone()
                al[ADD_INIT] = -float('inf')
                al[ADD_CHAIN] = -float('inf')
                al[ADD_ANGLE] = -float('inf')
                al[END] -= end_bias          # END-suppression: shrink stop-prob -> longer molecules
                action = torch.multinomial(F.softmax(al, dim=-1), 1).item()
                if action == END:
                    St[b]['tokens'].append(END)
                    St[b]['done'] = True
                    St[b]['active'] = False
                else:
                    St[b]['tokens'].append(action)
                    St[b]['action'] = action
                    still.append(b)
            if not still:
                continue

            # ---- slot 1: from ----
            logits, lastpos = _forward(still)
            for i, b in enumerate(still):
                from_logits = logits[1][i, lastpos[i]] / temperature   # head over RELATIVE parent-offset
                na = St[b]['n_atoms']                                   # atoms placed so far
                if use_offset:
                    # ★ Parent is chosen by RELATIVE OFFSET (translation-invariant): offset o in [0..na-1],
                    #   parent = na - o  (o=0: most-recent atom, o=na-1: first atom). The model only ever
                    #   predicts o; the stored `from_id` is merely its absolute form (na - o), kept losslessly.
                    max_off = min(na, from_logits.shape[-1]) - 1       # o cannot exceed atoms placed / head width
                    if max_off < 0:                                    # na == 0 -> FIRST atom (INIT): no parent;
                        from_id = 1                                    #   from_id=1 is a FIXED placeholder (uniform 7-slot)
                    else:
                        masked = from_logits.clone()
                        if max_off + 1 < from_logits.shape[-1]:
                            masked[max_off + 1:] = -float('inf')
                        offset = torch.multinomial(F.softmax(masked[:max_off + 1], dim=-1), 1).item()
                        from_id = na - offset                          # absolute parent index = na - offset (derived)
                else:
                    from_logits = from_logits.clone()
                    from_logits[na + 1:] = -float('inf')
                    from_logits[0] = -float('inf')
                    from_id = torch.multinomial(F.softmax(from_logits[:na + 1], dim=-1), 1).item()
                St[b]['from_id'] = from_id
                St[b]['tokens'].append(from_id)

            # ---- slots 2-6 ----
            for slot in range(2, 7):
                logits, lastpos = _forward(still)
                for i, b in enumerate(still):
                    action = St[b]['action']; na = St[b]['n_atoms']
                    if slot == 2:
                        if action == ADD:
                            pl = (logits['atom'][i, lastpos[i]] / temperature).clone()
                            m = torch.full_like(pl, -float('inf'))
                            for a in ALLOWED_ATOMS:
                                m[a] = 0.0
                            pl = pl + m
                        else:  # LINK
                            pl = (logits['to'][i, lastpos[i]] / temperature).clone()
                            if use_offset:                            # LINK target uses the SAME relative offset as slot-1
                                max_off = min(na, pl.shape[-1]) - 1
                                if max_off >= 0 and max_off + 1 < pl.shape[-1]:
                                    pl[max_off + 1:] = -float('inf')
                            else:
                                pl[na + 1:] = -float('inf')
                                pl[0] = -float('inf')
                    else:
                        pl = logits[slot][i, lastpos[i]] / temperature
                    val = torch.multinomial(F.softmax(pl, dim=-1), 1).item()
                    if slot == 2 and action == LINK and use_offset:   # LINK: sampled value is a relative offset,
                        val = na - val                                #   convert to absolute to_id (na - offset, like slot-1)
                    St[b]['tokens'].append(val)

            # ---- reconstruct each still-active molecule's new atom ----
            for b in still:
                _reconstruct_atom(St[b])

    return [(St[b]['tokens'], St[b]['n_frame'], St[b]['atom_table'],
             St[b]['bonds'], St[b]['n_atoms'], St[b]['done']) for b in range(B)]


def rollout_batch_kv(model, frame_sampler, device, B, max_steps=200, temperature=1.0, end_bias=0.0, end_bias_arr=None, size_ceiling=None):
    """Same as rollout_batch but with a KV-cache (kv_cache.kv_init/kv_step): each new
    token is processed incrementally (O(L)) instead of re-running the full sequence
    (O(L^2)). All B molecules advance in lockstep; ended rows are fed a dummy token to
    keep the per-row caches batch-aligned (their output is ignored). Masking/sampling
    rules identical to rollout_batch => distribution-equivalent (the cache adds only
    ~fp-level numerical drift). Returns the same list of B tuples."""
    from kv_cache import kv_init, kv_step
    model.eval()
    use_offset = getattr(model, 'output_pointer_mode', 'absolute') == 'offset'
    use_in_offset = getattr(model, 'input_pointer_mode', 'absolute') == 'offset'  # v2 emb_offset INPUT
    # ---- molecule SIZE ceiling for the rollout (END is forced once n_atoms reaches this) ----
    # emb_offset arch (vtakao202606231610) has NO emb_pointer -> NO absolute-atom-count limit.
    # The ONLY architectural constraint is max_offset (parent OFFSET span, enforced by the tokenizer
    # rejecting free-order orderings needing offset > max_offset); it bounds parent DISTANCE, not the
    # atom count (free-order references recent atoms, so offset stays < max_offset even for large mols).
    # => atom count is bounded ONLY by size_ceiling (=--size_nmax) at rollout time.
    # (Legacy absolute model capped atoms at emb_pointer.num_embeddings; that table + its input clamp
    #  are GONE here -- train.py: dead code removed 2026-06-14. The old "_ptr_rows-4 = 60" was a PHANTOM:
    #  getattr(model,'emb_pointer') is None on this arch -> silently fell back to 64 -> a fake 60 ceiling.
    #  FIXED 2026-07-06: cap = size_ceiling only; no phantom min-with-64.)
    _has_emb_pointer = getattr(model, 'emb_pointer', None) is not None            # legacy absolute model only
    if size_ceiling:
        max_atoms_cap = size_ceiling                                             # emb_offset: --size_nmax IS the cap
    elif _has_emb_pointer:
        max_atoms_cap = model.emb_pointer.num_embeddings - 4                      # legacy absolute model: real emb_pointer cap
    else:
        max_atoms_cap = 10 ** 9                                                   # emb_offset + no ceiling: unbounded (bounded by max_steps)

    St = []
    frames = []
    for b in range(B):
        ft = _sample_frame(frame_sampler)                                   # ABSOLUTE frame
        atom_table, bonds, n_atoms, bootstrap = _bootstrap_from_frame(ft)   # geometry uses absolute ft
        # the model-input token stream must carry OFFSETS under emb_offset; geometry stays absolute
        ft_in = ([int(x) for x in relative_pointer.absolute_to_relative(np.asarray(ft, dtype=np.int64))]
                 if use_in_offset else ft)
        St.append(dict(tokens=list(ft_in), n_frame=len(ft_in), atom_table=atom_table,
                       bonds=bonds, n_atoms=n_atoms, bootstrap=bootstrap, use_in_offset=use_in_offset,
                       done=False, active=True, action=END, from_id=1))
        frames.append(ft_in)
    Lp = len(frames[0])

    def _heads(h1, key):
        """h1: (B,1,d) -> (B,vocab) logits for the requested head."""
        if key == 0:      return model.head_action(h1)[:, 0]
        if key == 1:      return model.head_from(h1)[:, 0]
        if key == 'atom': return model.head_atom(h1)[:, 0]
        if key == 'to':   return model.head_to(h1)[:, 0]
        return {3: model.head_r, 4: model.head_hp0, 5: model.head_hp1,
                6: model.head_hp2}[key](h1)[:, 0]

    with torch.no_grad():
        vals = torch.tensor(frames, dtype=torch.long, device=device)            # (B,Lp)
        rng = torch.arange(Lp, device=device)
        slots = (rng % N_SLOTS).unsqueeze(0).expand(B, Lp)
        acts = vals[:, (rng // N_SLOTS) * N_SLOTS]                               # (B,Lp)
        emb = model.embed_input(vals, slots, acts)
        caches, h_pr = kv_init(model, emb)
        h = h_pr[:, -1:]                                                         # (B,1,d) -> next-token logits
        pos = Lp

        def _advance(next_tok):
            """embed next_tok (B,) at the current pos with each row's action, kv_step."""
            nonlocal h, pos
            v1 = next_tok.view(B, 1)
            s1 = torch.full((B, 1), pos % N_SLOTS, dtype=torch.long, device=device)
            a1 = torch.tensor([St[b]['action'] for b in range(B)],
                              dtype=torch.long, device=device).view(B, 1)
            e1 = model.embed_input(v1, s1, a1)
            h, _ = kv_step(model, e1, caches)
            pos += 1

        for _step in range(max_steps):
            for b in range(B):
                if St[b]['active'] and St[b]['n_atoms'] >= max_atoms_cap:
                    St[b]['active'] = False
            if not any(St[b]['active'] for b in range(B)):
                break

            # ---- slot 0: action ----
            al = (_heads(h, 0) / temperature).clone()
            al[:, ADD_INIT] = -float('inf'); al[:, ADD_CHAIN] = -float('inf'); al[:, ADD_ANGLE] = -float('inf')
            if end_bias_arr is not None:    # per-size END-bias b[n] (hazard shaping): index by current size
                _bv = torch.tensor([end_bias_arr.get(St[b]['n_atoms'], 0.0) for b in range(B)], dtype=al.dtype, device=device)
                al[:, END] -= _bv
            elif end_bias != 0.0:
                al[:, END] -= end_bias          # scalar END-suppression: shrink stop-prob -> longer molecules
            nt = torch.zeros(B, dtype=torch.long, device=device)
            for b in range(B):
                if not St[b]['active']:
                    continue
                a = torch.multinomial(F.softmax(al[b], dim=-1), 1).item()
                if a == END:
                    St[b]['tokens'].append(END); St[b]['done'] = True; St[b]['active'] = False
                    nt[b] = END
                else:
                    St[b]['tokens'].append(a); St[b]['action'] = a; nt[b] = a
            _advance(nt)
            still = [b for b in range(B) if St[b]['active']]
            if not still:
                continue

            # ---- slot 1: from (RELATIVE parent-offset; full explanation in rollout_batch above) ----
            from_logits = _heads(h, 1) / temperature
            nt = torch.zeros(B, dtype=torch.long, device=device)
            for b in range(B):
                if not St[b]['active']:
                    continue
                fb = from_logits[b]; na = St[b]['n_atoms']
                if use_offset:
                    # offset o in [0..na-1] -> parent = na - o (translation-invariant); from_id = its absolute form
                    max_off = min(na, fb.shape[-1]) - 1
                    if max_off < 0:                  # na == 0 -> FIRST atom (INIT): from_id=1 fixed placeholder
                        from_id = 1; stored = 1
                    else:
                        masked = fb.clone()
                        if max_off + 1 < fb.shape[-1]:
                            masked[max_off + 1:] = -float('inf')
                        if VALENCE_MASK:                     # parent (ADD) / source (LINK) must have a free slot
                            for _o in range(max_off + 1):
                                if _saturated(St[b], na - _o):
                                    masked[_o] = -float('inf')
                            if not torch.isfinite(masked[:max_off + 1]).any():
                                masked = fb.clone()          # everything saturated -> fall back (never deadlock)
                                if max_off + 1 < fb.shape[-1]:
                                    masked[max_off + 1:] = -float('inf')
                        offset = torch.multinomial(F.softmax(masked[:max_off + 1], dim=-1), 1).item()
                        from_id = na - offset
                        stored = (offset + 1) if use_in_offset else from_id   # v2: token stream stores OFFSET (=class+1)
                else:
                    fb = fb.clone(); fb[na + 1:] = -float('inf'); fb[0] = -float('inf')
                    from_id = torch.multinomial(F.softmax(fb[:na + 1], dim=-1), 1).item()
                    stored = from_id
                St[b]['from_id'] = from_id; St[b]['tokens'].append(stored); nt[b] = stored
            _advance(nt)

            # ---- slots 2-6 ----
            for slot in range(2, 7):
                key = ('atom' if slot == 2 else slot)
                nt = torch.zeros(B, dtype=torch.long, device=device)
                # slot 2 splits by action (ADD->atom, LINK->to); fetch both heads when needed
                lg_atom = _heads(h, 'atom') if slot == 2 else None
                lg_to = _heads(h, 'to') if slot == 2 else None
                lg_oth = _heads(h, slot) if slot != 2 else None
                for b in range(B):
                    if not St[b]['active']:
                        continue
                    action = St[b]['action']; na = St[b]['n_atoms']
                    if slot == 2:
                        if action == ADD:
                            pl = (lg_atom[b] / temperature).clone()
                            mm = torch.full_like(pl, -float('inf'))
                            for a in ALLOWED_ATOMS:
                                mm[a] = 0.0
                            pl = pl + mm
                        else:
                            pl = (lg_to[b] / temperature).clone()
                            if use_offset:                            # LINK target uses the SAME relative offset as slot-1
                                max_off = min(na, pl.shape[-1]) - 1
                                if max_off >= 0 and max_off + 1 < pl.shape[-1]:
                                    pl[max_off + 1:] = -float('inf')
                                if VALENCE_MASK and max_off >= 0:     # the LINK target must have a free slot too
                                    _fid = St[b]['from_id']
                                    for _o in range(max_off + 1):
                                        _tid = na - _o
                                        if _tid == _fid or _saturated(St[b], _tid) or \
                                           (min(_fid, _tid), max(_fid, _tid)) in St[b]['bonds']:
                                            pl[_o] = -float('inf')
                                    if not torch.isfinite(pl[:max_off + 1]).any():
                                        pl = (lg_to[b] / temperature).clone()   # never deadlock
                                        if max_off + 1 < pl.shape[-1]:
                                            pl[max_off + 1:] = -float('inf')
                            else:
                                pl[na + 1:] = -float('inf'); pl[0] = -float('inf')
                    else:
                        pl = lg_oth[b] / temperature
                    val = torch.multinomial(F.softmax(pl, dim=-1), 1).item()
                    if slot == 2 and action == LINK and use_offset:   # LINK: sampled value is a relative offset class
                        # store OFFSET (class+1) for emb_offset input; _reconstruct_atom recovers abs to_id = na-class
                        val = (val + 1) if use_in_offset else (na - val)
                    St[b]['tokens'].append(val); nt[b] = val
                _advance(nt)

            for b in still:
                _reconstruct_atom(St[b])

    return [(St[b]['tokens'], St[b]['n_frame'], St[b]['atom_table'],
             St[b]['bonds'], St[b]['n_atoms'], St[b]['done']) for b in range(B)]
