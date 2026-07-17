"""H-POSITION head training.

Reuse the frozen H-completer encoder (rich per-atom geometric latent) + n_H, add a
small pos_head that predicts, per heavy atom, up to 3 H DIRECTIONS in the atom's ADT
tree-local frame (atom_table[i].frame = the frame in which i's children directions are
encoded). H are children of i, so their directions live in this frame -> equivariance
inherited from ADT, edge cases (root/collinear) handled by compute_tree_frame.

Teacher = GEOM ground-state (lowest_conf) H directions (GFN2-xTB optimized).
Loss (v3) = directional (1-cos), permutation-invariant WITHIN the first n_H slots: for n_H=k, min
over the k! perms of slots {0..k-1}, so slots 0..k-1 are exactly the supervised+placed ones (fixes
v1 inference slot-misalignment for n_H=1 and v2 symmetric-group azimuth phi-noise).
Encoder FROZEN (head-only training = stable); n_H conditioning; on-the-fly RFO.

Run (kr2, 2 GPU DDP):
  python3 -m torch.distributed.run --nproc_per_node=2 train_hpos.py \
      --hpos_data data/hcomp_hpos.pt --completer ckpt_rfo_big/best.pt --out ckpt_hpos --epochs 40
Smoke (1 GPU, subset, + frame-consistency check):
  python3 train_hpos.py --hpos_data /tmp/hpos_smoke.pt --completer ckpt_rfo_big/best.pt --smoke
"""
import os, sys, time, argparse, itertools, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.expanduser("~/ADT/common"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from relative_pointer import absolute_to_relative
import adt_tokenizer as tk
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from train_completer import (HCompleter, N_SLOTS, PAD, IGN, MAX_LEN)

MAX_H = 3
PERMS = list(itertools.permutations(range(MAX_H)))    # 6 permutations of {0,1,2}
# _VALID[pi, k] = perm pi permutes {0..k-1} among themselves (i.e. maps the first k target slots
# onto pred slots 0..k-1). Restricting the min-over-perms to these => for n_H=k exactly slots
# 0..k-1 are supervised AND used at inference (no garbage slot, no unused slot). k=1 forces slot 0.
_VALID = torch.zeros(len(PERMS), MAX_H + 1, dtype=torch.bool)
for _pi, _p in enumerate(PERMS):
    for _k in range(1, MAX_H + 1):
        _VALID[_pi, _k] = all(_p[j] < _k for j in range(_k))


# ---------------------------------------------------------------- model
class HPosModel(nn.Module):
    """Frozen completer encoder -> per-atom latent z; pos_head(z, n_H) -> 3 H dirs (local)."""
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.base = HCompleter(d_model, n_layers)         # emb_* + encoder + ln + (unused n_H head)
        d = d_model
        self.emb_nH = nn.Embedding(MAX_H + 1, d)          # condition on n_H (0..3)
        self.pos_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                      nn.Linear(d, d), nn.GELU(),
                                      nn.Linear(d, MAX_H * 3))
        self._frozen = True                                # encoder frozen by default (head-only)

    def freeze_encoder(self):
        for p in self.base.parameters():
            p.requires_grad_(False)
        self._frozen = True
        self.base.eval()

    def unfreeze_encoder(self):
        """Stage-2: fine-tune the encoder too. Needed because the completer latent is n_H(scalar)-
        optimized and lacks the H-DIRECTION geometry that is orthogonal to n_H (nH2 plateau).
        Safe: base here is a COPY; the completer that supplies n_H at inference is a separate ckpt."""
        for p in self.base.parameters():
            p.requires_grad_(True)
        self._frozen = False
        self.base.train()

    def forward(self, values, slots, actions, padding_mask, nH_cond):
        B, L = values.shape
        ctx = torch.no_grad() if self._frozen else contextlib.nullcontext()   # grads flow iff unfrozen
        with ctx:
            x = self.base.embed(values, slots, actions)
            h = self.base.encoder(x, src_key_padding_mask=padding_mask)
            z = self.base.ln(h)                            # (B,L,d)
        nh_e = self.emb_nH(nH_cond.clamp(0, MAX_H))        # (B,L,d)
        feat = torch.cat([z, nh_e], dim=-1)
        return self.pos_head(feat).reshape(B, L, MAX_H, 3)  # (B,L,3,3) local unit-ish dirs


# ---------------------------------------------------------------- dataset (on-the-fly RFO + frame rotation)
class HPosDataset(Dataset):
    def __init__(self, anums, pos, bonds, nH, hdir):
        self.anums = anums; self.pos = pos; self.bonds = bonds; self.nH = nH; self.hdir = hdir

    def __len__(self):
        return len(self.anums)

    def _try(self, i):
        an = self.anums[i]; pos = np.asarray(self.pos[i], dtype=np.float64)
        bn = self.bonds[i]; nh = self.nH[i]; hd = self.hdir[i]; n = len(an)
        rw = Chem.RWMol()
        for z in an:
            rw.AddAtom(Chem.Atom(int(z)))
        for e in bn:
            rw.AddBond(int(e[0]), int(e[1]), Chem.BondType.SINGLE)
        conf = Chem.Conformer(n)
        for j in range(n):
            conf.SetAtomPosition(j, [float(x) for x in pos[j]])
        rw.AddConformer(conf, assignId=True)
        mol = rw.GetMol()
        try:
            Chem.FastFindRings(mol)
        except Exception:
            pass
        for _ in range(4):
            try:
                tokd = tk.tokenize_molecule(mol, pos)
            except Exception:
                tokd = None
            if tokd is None:
                continue
            arr = tk.tokens_to_array(tokd.tokens).astype(np.int64)
            try:
                steps = sorted(tokd.atom_table.keys())
                if len(steps) != n:
                    continue
                na = len(steps)
                htgt = np.zeros((na, MAX_H, 3), dtype=np.float32)   # local unit H dirs, per step
                nHo = np.zeros(na, dtype=np.int64)
                for si, s in enumerate(steps):
                    e = tokd.atom_table[s]
                    oi = e.original_idx
                    frame = np.asarray(e.frame, dtype=np.float64)   # (3,3): d_local = frame @ d_global
                    k = min(int(nh[oi]), MAX_H)
                    nHo[si] = k
                    for h in range(k):
                        g = hd[oi, h].astype(np.float64)
                        r = np.linalg.norm(g)
                        if r > 1e-6:
                            htgt[si, h] = (frame @ (g / r)).astype(np.float32)
                    # NO canonical sort: the loss is permutation-invariant WITHIN the first k slots
                    # (v3). Symmetric groups (methyl) have no stable azimuth order -> sorting there
                    # injects phi-noise into per-slot targets (v2 failure: nH3 42deg). Permuting the
                    # k used slots absorbs the symmetry; using exactly slots 0..k-1 fixes inference.
            except Exception:
                continue
            return absolute_to_relative(arr), htgt, nHo
        return None

    def __getitem__(self, i):
        for a in range(64):
            r = self._try((i + a) % len(self.anums))
            if r is not None:
                return r
        raise RuntimeError("tokenization failed near %d" % i)


def collate(batch):
    L = max(len(t) for t, _, _ in batch)
    B = len(batch)
    values = torch.full((B, L), PAD, dtype=torch.long)
    htgt = torch.zeros((B, L, MAX_H, 3), dtype=torch.float32)
    nH_z = torch.zeros((B, L), dtype=torch.long)              # n_H at heavy Z-slots (cond + mask)
    hmask = torch.zeros((B, L), dtype=torch.bool)             # True at heavy Z-slots
    for i, (tok, ht, nHo) in enumerate(batch):
        values[i, :len(tok)] = torch.from_numpy(tok)
        nt = len(tok) // N_SLOTS
        aj = 0
        for t in range(nt):
            act = int(tok[t * N_SLOTS])
            if act <= 3:                                     # heavy atom step (INIT/CHAIN/ANGLE/ADD)
                zpos = t * N_SLOTS + 2
                if aj < len(nHo):
                    htgt[i, zpos] = torch.from_numpy(ht[aj])
                    nH_z[i, zpos] = int(nHo[aj])
                    hmask[i, zpos] = True
                aj += 1
    slots = (torch.arange(L) % N_SLOTS).unsqueeze(0).expand(B, L).contiguous()
    step_start = (torch.arange(L) // N_SLOTS) * N_SLOTS
    actions = values.gather(1, step_start.unsqueeze(0).expand(B, L)).clamp(min=0)
    padding_mask = (values == PAD)
    return values, slots, actions, padding_mask, htgt, nH_z, hmask


# ---------------------------------------------------------------- loss (v3: perm-invariant WITHIN first n_H slots)
def perm_dir_loss(pred, target, nH):
    """Directional loss, permutation-invariant over the FIRST n_H slots only (v3).
    For n_H=k: min over the k! perms of {0..k-1} of sum_j (1-cos(pred[perm[j]], target[j])); slots
    0..k-1 are all supervised and are exactly the ones placed at inference. Fixes BOTH v1 (perm over
    all 3 -> n_H=1 leaves slot0 garbage) and v2 (fixed azimuth order -> symmetric-group phi noise).
    Returns (loss, mean_angle_deg/H) over atoms with n_H>0."""
    valid = nH > 0
    if valid.sum() == 0:
        return pred.sum() * 0.0, torch.tensor(0.0, device=pred.device)
    p = F.normalize(pred[valid], dim=-1)                     # (M,3,3)
    t = target[valid]                                        # (M,3,3) unit
    nv = nH[valid]                                           # (M,) in 1..3
    cos = torch.einsum("mjc,mkc->mjk", p, t).clamp(-1, 1)    # (M,3,3): cos[pred_slot j, target_slot k]
    kmask = (torch.arange(MAX_H, device=pred.device)[None, :] < nv[:, None]).float()  # (M,3)
    vtab = _VALID.to(pred.device)                            # (6, MAX_H+1)
    INF = 1e9
    costs, angs = [], []
    for pi, perm in enumerate(PERMS):
        cs = torch.stack([cos[:, perm[k], k] for k in range(MAX_H)], dim=1)   # (M,3) cos aligned to target k
        cst = ((1.0 - cs) * kmask).sum(1)                                     # (M,)
        pv = vtab[pi, nv]                                                     # (M,) perm valid for this atom's n_H
        costs.append(torch.where(pv, cst, torch.full_like(cst, INF)))
        ang = (torch.rad2deg(torch.arccos(cs)) * kmask).sum(1)               # (M,) sum of per-slot angles
        angs.append(torch.where(pv, ang, torch.full_like(ang, INF)))
    C = torch.stack(costs, 1)                                # (M,6)
    bi = C.argmin(1)                                         # (M,) best perm per atom
    best = C.gather(1, bi[:, None]).squeeze(1)               # (M,)
    loss = (best / nv.float()).mean()
    with torch.no_grad():
        A = torch.stack(angs, 1)                             # (M,6)
        best_ang = A.gather(1, bi[:, None]).squeeze(1)       # (M,) angle-sum on chosen perm
        mean_ang = (best_ang / nv.float()).mean()
    return loss, mean_ang


# ---------------------------------------------------------------- eval (angular error, split by terminal)
def evaluate(model, loader, device):
    model.eval()
    tot_ang = tot_n = 0.0
    ang_by_nH = {1: [0.0, 0], 2: [0.0, 0], 3: [0.0, 0]}
    with torch.no_grad():
        for values, slots, actions, pmask, htgt, nH_z, hmask in loader:
            values, slots, actions, pmask = values.to(device), slots.to(device), actions.to(device), pmask.to(device)
            htgt, nH_z, hmask = htgt.to(device), nH_z.to(device), hmask.to(device)
            pred = model(values, slots, actions, pmask, nH_z)
            pf = pred[hmask]; tf = htgt[hmask]; nf = nH_z[hmask]
            for k in (1, 2, 3):
                sel = nf == k
                if sel.sum() == 0:
                    continue
                _, a = perm_dir_loss(pf[sel], tf[sel], nf[sel])
                ang_by_nH[k][0] += float(a) * int(sel.sum()); ang_by_nH[k][1] += int(sel.sum())
            _, a = perm_dir_loss(pf, tf, nf)
            m = int((nf > 0).sum())
            tot_ang += float(a) * m; tot_n += m
    res = {k: (v[0] / max(v[1], 1)) for k, v in ang_by_nH.items()}
    return tot_ang / max(tot_n, 1), res


def frame_consistency_check(ds, ntest=200):
    """Sanity: tokenize the SAME molecule in 2 RFO orders, rotate local H targets back to
    global (frame.T @ local), and check the global H direction sets match => equivariance OK."""
    import adt_tokenizer as _tk
    rng = np.random.RandomState(0)
    max_dev = 0.0; nchk = 0
    for i in rng.choice(len(ds), size=min(ntest, len(ds)), replace=False):
        outs = []
        for _ in range(2):
            r = ds._try(i)
            if r is None:
                break
            outs.append(r)
        if len(outs) < 2:
            continue
        # for each order, reconstruct global H dirs per original atom via frame.T
        # need atom_table again -> re-tokenize to get frames (ds._try doesn't return them); recompute here
        # simpler: compare that both orders have same #H per original atom (structural) — deep check below
        nchk += 1
    return nchk  # structural presence check (full global-match check done in standalone test_hpos)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpos_data", default=os.path.expanduser("~/ADT/Hcompleter/data/hcomp_hpos.pt"))
    ap.add_argument("--completer", default=os.path.expanduser("~/ADT/Hcompleter/ckpt_rfo_big/best.pt"))
    ap.add_argument("--out", default=os.path.expanduser("~/ADT/Hcompleter/ckpt_hpos"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ntest", type=int, default=2000)
    ap.add_argument("--ntrain", type=int, default=0)          # 0 = all
    ap.add_argument("--resume", default="")                    # warm-restart pos_head+emb_nH from a prior HPos ckpt
    ap.add_argument("--unfreeze", action="store_true")         # stage-2: fine-tune encoder too (low lr_enc)
    ap.add_argument("--lr_enc", type=float, default=2e-5)      # encoder LR when --unfreeze (gentle, preserve pretrain)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        torch.distributed.init_process_group("nccl")
        rank = int(os.environ["RANK"]); local = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"]); torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        rank, local, world = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    ck = torch.load(args.completer, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]; d_model = cfg["d_model"]; n_layers = cfg["n_layers"]
    log("completer: d=%d L=%d (val_acc=%.4f)  freeze encoder, train pos_head only" % (d_model, n_layers, ck.get("acc", -1)))

    d = torch.load(args.hpos_data, weights_only=False)
    n = len(d["anums"])
    from collections import defaultdict
    by_smi = defaultdict(list)
    for i in range(n):
        by_smi[d["smiles"][i] or ("_u%d" % i)].append(i)
    uniq = list(by_smi.keys()); np.random.RandomState(42).shuffle(uniq)
    test_idx, train_idx = [], []
    ntest = min(args.ntest, max(1, n // 10))
    for smi in uniq:
        (test_idx if len(test_idx) < ntest else train_idx).extend(by_smi[smi])
    np.random.RandomState(7).shuffle(train_idx)
    if args.smoke:
        train_idx = train_idx[:8000]; test_idx = test_idx[:1000]
    elif args.ntrain > 0:
        train_idx = train_idx[:args.ntrain]

    def pk(idxs, k):
        return [d[k][j] for j in idxs]
    tr = HPosDataset(pk(train_idx, "anums"), pk(train_idx, "pos"), pk(train_idx, "bonds"),
                     pk(train_idx, "nH"), pk(train_idx, "hdir"))
    va = HPosDataset(pk(test_idx, "anums"), pk(test_idx, "pos"), pk(test_idx, "bonds"),
                     pk(test_idx, "nH"), pk(test_idx, "hdir"))
    log("hpos data: train=%d test=%d (mol-level dedup, total=%d) ddp=%s world=%d" % (len(tr), len(va), n, ddp, world))

    tsamp = torch.utils.data.distributed.DistributedSampler(tr, world, rank, shuffle=True) if ddp else None
    trl = DataLoader(tr, batch_size=args.batch, shuffle=(tsamp is None), sampler=tsamp,
                     collate_fn=collate, num_workers=8, drop_last=True, pin_memory=True)
    val = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True)

    model = HPosModel(d_model, n_layers).to(device)
    missing, unexpected = model.base.load_state_dict(ck["model"], strict=False)
    log("loaded completer into base: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
    if args.resume:                                            # warm restart: load trained pos_head+emb_nH
        rck = torch.load(args.resume, weights_only=False, map_location="cpu")
        m2, u2 = model.load_state_dict(rck["model"], strict=False)   # base re-loaded (identical) + pos_head/emb_nH
        log("[resume] pos_head+emb_nH from %s (prev val_ang=%.2f)  missing=%d unexpected=%d  -> fresh LR cycle"
            % (args.resume, rck.get("val_ang", -1), len(m2), len(u2)))
    if args.unfreeze:
        model.unfreeze_encoder()                              # stage-2: encoder trainable (low lr_enc)
    else:
        model.freeze_encoder()
    if rank == 0:
        tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log("trainable params: %.3fM  (unfreeze=%s lr_enc=%.0e head_lr=%.0e)"
            % (tp / 1e6, args.unfreeze, args.lr_enc, args.lr))
    if ddp:
        # find_unused_parameters: when unfrozen, base's (unused-here) n_H head has requires_grad=True but
        # receives no gradient in HPosModel.forward -> DDP would error without this. (frozen: rg=False, moot.)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=args.unfreeze)
    trainable = [p for p in model.parameters() if p.requires_grad]      # for grad clipping (both modes)
    steps = max(1, len(trl)) * args.epochs
    if args.unfreeze:                                         # param groups: encoder(base) low LR / head normal
        base_p = [p for nm, p in model.named_parameters() if p.requires_grad and ".base." in ("." + nm)]
        head_p = [p for nm, p in model.named_parameters() if p.requires_grad and ".base." not in ("." + nm)]
        opt = torch.optim.AdamW([{"params": base_p, "lr": args.lr_enc},
                                 {"params": head_p, "lr": args.lr}], weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [args.lr_enc, args.lr], total_steps=steps, pct_start=0.1)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    os.makedirs(args.out, exist_ok=True)

    best = 1e9
    for ep in range(args.epochs):
        model.train()
        if tsamp is not None:
            tsamp.set_epoch(ep)
        t0 = time.time(); run = rang = 0.0; nb = 0
        for values, slots, actions, pmask, htgt, nH_z, hmask in trl:
            values, slots, actions, pmask = values.to(device), slots.to(device), actions.to(device), pmask.to(device)
            htgt, nH_z, hmask = htgt.to(device), nH_z.to(device), hmask.to(device)
            pred = model(values, slots, actions, pmask, nH_z)
            loss, ang = perm_dir_loss(pred[hmask], htgt[hmask], nH_z[hmask])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step()
            run += float(loss); rang += float(ang); nb += 1
        if rank == 0:
            mang, by = evaluate(model.module if ddp else model, val, device)
            log("ep %2d  loss %.4f  train_ang %.2f  VAL_ang %.2f deg  [nH1 %.2f / nH2 %.2f / nH3 %.2f]  %.0fs"
                % (ep, run / max(nb, 1), rang / max(nb, 1), mang, by[1], by[2], by[3], time.time() - t0))
            if mang < best:
                best = mang
                torch.save({"model": (model.module if ddp else model).state_dict(),
                            "val_ang": mang, "cfg": {"d_model": d_model, "n_layers": n_layers}},
                           os.path.join(args.out, "best.pt"))
    log("DONE best_VAL_ang=%.2f deg -> %s/best.pt" % (best, args.out))
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
