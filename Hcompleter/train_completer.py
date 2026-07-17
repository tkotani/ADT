"""H-completer training.

A BIDIRECTIONAL transformer encoder over ADT's 7-slot heavy-atom tokens that
predicts, per heavy atom, the number of attached hydrogens n_H in {0,1,2,3}.
= a learned valence/aromaticity perceiver (replaces RDKit's distance-based
perception, which fails on aromatics / charged N).

Input tokens: [action, from(offset), atom(Z)/to(offset), r, hp0, hp1, hp2] x steps + END.
Pointers are converted absolute->relative offset (relative_pointer) to match ADT.
Target: n_H per atom, placed at the atom's action-slot (step-start) position.

Reuses ADT's per-slot-summed embedding design (own weights, N_R=200 to match the
tokenizer default used to build the dataset). Bidirectional (no causal mask) so
each atom sees its whole bonding environment before committing valence.

Run (kr2, 2 GPU DDP):
  torchrun --nproc_per_node=2 train_completer.py --data data/hcomp_train.pt --epochs 30
Smoke (1 GPU, subset):
  python3 train_completer.py --data data/hcomp_train.pt --smoke
"""
import os, sys, time, argparse, math
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

# ---- token vocab (matches adt_tokenizer defaults used to build the dataset) ----
N_ACTIONS = 6           # INIT,CHAIN,ANGLE,ADD,LINK,END
LINK = 4
MAX_OFFSET = 40         # relative pointer offset embedding width (clamp beyond)
N_ATOM = 119            # atomic number 0..118
N_R = 200               # R_BINS default in adt_tokenizer
N_HP0, N_HP1, N_HP2 = 12, 16, 16
N_SLOTS = 7
N_H_CLASSES = 4         # n_H in {0,1,2,3}
MAX_LEN = 288           # >= max tok_len (260)
PAD = -1
IGN = -100


class HCompleter(nn.Module):
    def __init__(self, d_model=256, n_layers=6, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        d = d_model
        self.d = d
        self.emb_action = nn.Embedding(N_ACTIONS, d)
        self.emb_offset = nn.Embedding(MAX_OFFSET, d)
        self.emb_atom = nn.Embedding(N_ATOM, d)
        self.emb_r = nn.Embedding(N_R, d)
        self.emb_hp0 = nn.Embedding(N_HP0, d)
        self.emb_hp1 = nn.Embedding(N_HP1, d)
        self.emb_hp2 = nn.Embedding(N_HP2, d)
        self.emb_slot = nn.Embedding(N_SLOTS, d)
        self.emb_pos = nn.Embedding(MAX_LEN, d)          # bidirectional needs positional signal
        layer = nn.TransformerEncoderLayer(d, n_heads, d_ff, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_H_CLASSES))

    def embed(self, values, slots, actions):
        B, L = values.shape
        v = values.clamp(min=0)
        emb = torch.zeros(B, L, self.d, device=values.device, dtype=self.emb_slot.weight.dtype)
        for slot in range(N_SLOTS):
            m = (slots == slot)
            if not m.any():
                continue
            vv = v[m]
            if slot == 0:
                emb[m] = self.emb_action(vv.clamp(0, N_ACTIONS - 1))
            elif slot == 1:
                emb[m] = self.emb_offset((vv - 1).clamp(0, MAX_OFFSET - 1))
            elif slot == 2:
                act = actions[m]
                is_link = (act == LINK)
                e = torch.zeros(vv.shape[0], self.d, device=values.device, dtype=emb.dtype)
                if (~is_link).any():
                    e[~is_link] = self.emb_atom(vv[~is_link].clamp(0, N_ATOM - 1))
                if is_link.any():
                    e[is_link] = self.emb_offset((vv[is_link] - 1).clamp(0, MAX_OFFSET - 1))
                emb[m] = e
            elif slot == 3:
                emb[m] = self.emb_r(vv.clamp(0, N_R - 1))
            elif slot == 4:
                emb[m] = self.emb_hp0(vv.clamp(0, N_HP0 - 1))
            elif slot == 5:
                emb[m] = self.emb_hp1(vv.clamp(0, N_HP1 - 1))
            elif slot == 6:
                emb[m] = self.emb_hp2(vv.clamp(0, N_HP2 - 1))
        pos = torch.arange(L, device=values.device).clamp(max=MAX_LEN - 1)
        return emb + self.emb_slot(slots) + self.emb_pos(pos).unsqueeze(0)

    def forward(self, values, slots, actions, padding_mask):
        x = self.embed(values, slots, actions)
        h = self.encoder(x, src_key_padding_mask=padding_mask)   # bidirectional
        return self.head(self.ln(h))                             # (B, L, 4)


class HCompDataset(Dataset):
    def __init__(self, tok, nH):
        self.tok = tok
        self.nH = nH

    def __len__(self):
        return len(self.tok)

    def __getitem__(self, i):
        tok = self.tok[i].astype(np.int64)
        off = absolute_to_relative(tok)          # absolute pointers -> relative offsets
        return off, self.nH[i].astype(np.int64)


class StructDataset(Dataset):
    """On-the-fly RFO: reconstruct the heavy mol and tokenize with a FRESH random
    order every __getitem__, so each epoch sees a new ordering => the model must
    learn order-INVARIANT n_H (n_H is an atom property, not a sequence property)."""
    def __init__(self, anums, pos, bonds, nH):
        self.anums = anums; self.pos = pos; self.bonds = bonds; self.nH = nH

    def __len__(self):
        return len(self.anums)

    def _try(self, i):
        an = self.anums[i]; pos = np.asarray(self.pos[i], dtype=np.float64)
        bn = self.bonds[i]; nh = self.nH[i]; n = len(an)
        rw = Chem.RWMol()
        for z in an:
            rw.AddAtom(Chem.Atom(int(z)))
        for e in bn:
            rw.AddBond(int(e[0]), int(e[1]), Chem.BondType.SINGLE)   # connectivity only
        conf = Chem.Conformer(n)
        for j in range(n):
            conf.SetAtomPosition(j, [float(x) for x in pos[j]])
        rw.AddConformer(conf, assignId=True)
        mol = rw.GetMol()
        try:
            Chem.FastFindRings(mol)          # ring info for LINK detection (no full sanitize)
        except Exception:
            pass
        for _ in range(4):                   # a few fresh random orders until tokenizer succeeds
            try:
                tokd = tk.tokenize_molecule(mol, pos)
            except Exception:
                tokd = None
            if tokd is None:
                continue
            arr = tk.tokens_to_array(tokd.tokens).astype(np.int64)
            try:
                tgt = np.array([nh[tokd.atom_table[s].original_idx]
                                for s in sorted(tokd.atom_table.keys())], dtype=np.int64)
            except Exception:
                continue
            if len(tgt) == n:
                return absolute_to_relative(arr), tgt
        return None

    def __getitem__(self, i):
        for a in range(64):
            r = self._try((i + a) % len(self.anums))
            if r is not None:
                return r
        raise RuntimeError("tokenization failed repeatedly near index %d" % i)


def _winit(wid):
    """Seed python-random / numpy per worker so on-the-fly RFO orders differ across
    workers (and across epochs, since non-persistent workers reseed each epoch)."""
    import random as _r
    s = (torch.initial_seed() + wid) % (2 ** 31)
    _r.seed(s); np.random.seed(s)


def collate(batch):
    lens = [len(t) for t, _ in batch]
    L = max(lens)
    B = len(batch)
    values = torch.full((B, L), PAD, dtype=torch.long)
    target = torch.full((B, L), IGN, dtype=torch.long)
    for i, (tok, nH) in enumerate(batch):
        values[i, :len(tok)] = torch.from_numpy(tok)
        nt = len(tok) // N_SLOTS
        aj = 0
        for t in range(nt):
            act = int(tok[t * N_SLOTS])
            if act <= 3:                          # INIT/CHAIN/ANGLE/ADD = a heavy atom
                if aj < len(nH):
                    target[i, t * N_SLOTS + 2] = int(nH[aj])   # anchor at Z-slot
                aj += 1
    slots = (torch.arange(L) % N_SLOTS).unsqueeze(0).expand(B, L).contiguous()
    step_start = (torch.arange(L) // N_SLOTS) * N_SLOTS
    actions = values.gather(1, step_start.unsqueeze(0).expand(B, L)).clamp(min=0)
    padding_mask = (values == PAD)
    return values, slots, actions, target, padding_mask


def evaluate(model, loader, device):
    model.eval()
    cc = torch.zeros(N_H_CLASSES, dtype=torch.long)      # per-class correct
    ct = torch.zeros(N_H_CLASSES, dtype=torch.long)      # per-class total
    tot = corr = 0
    with torch.no_grad():
        for values, slots, actions, target, pmask in loader:
            values, slots, actions, target, pmask = (values.to(device), slots.to(device),
                                                     actions.to(device), target.to(device), pmask.to(device))
            logits = model(values, slots, actions, pmask)
            m = target != IGN
            pred = logits.argmax(-1)
            corr += int((pred[m] == target[m]).sum()); tot += int(m.sum())
            for c in range(N_H_CLASSES):
                mc = m & (target == c)
                ct[c] += int(mc.sum()); cc[c] += int((pred[mc] == c).sum())
    acc = corr / max(tot, 1)
    pca = [(int(cc[c]) / max(int(ct[c]), 1)) for c in range(N_H_CLASSES)]
    return acc, pca, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/ADT/Hcompleter/data/hcomp_train.pt"))
    ap.add_argument("--out", default=os.path.expanduser("~/ADT/Hcompleter/ckpt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--struct_data", default="")     # on-the-fly RFO structure data
    ap.add_argument("--ntrain", type=int, default=100000)
    ap.add_argument("--ntest", type=int, default=4000)
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

    if args.struct_data:
        d = torch.load(args.struct_data, weights_only=False)
        n = len(d["anums"])
        from collections import defaultdict
        by_smi = defaultdict(list)
        for i in range(n):
            by_smi[d["smiles"][i] or ("_uniq%d" % i)].append(i)   # dedup key (empty->unique)
        uniq = list(by_smi.keys())
        np.random.RandomState(42).shuffle(uniq)
        test_idx, train_idx = [], []
        for smi in uniq:                       # whole SMILES-group -> test or train (no leak)
            (test_idx if len(test_idx) < args.ntest else train_idx).extend(by_smi[smi])
        np.random.RandomState(7).shuffle(train_idx)
        if args.smoke:
            train_idx = train_idx[:20000]
        elif args.ntrain > 0:
            train_idx = train_idx[:args.ntrain]
        def pk(idxs, k):
            return [d[k][j] for j in idxs]
        tr = StructDataset(pk(train_idx, "anums"), pk(train_idx, "pos"), pk(train_idx, "bonds"), pk(train_idx, "nH"))
        va = StructDataset(pk(test_idx, "anums"), pk(test_idx, "pos"), pk(test_idx, "bonds"), pk(test_idx, "nH"))
        log("STRUCT on-the-fly RFO: train=%d test=%d  (molecule-level SMILES-dedup, total=%d)  ddp=%s world=%d"
            % (len(tr), len(va), n, ddp, world))
    else:
        d = torch.load(args.data, weights_only=False)
        tok, nH = d["tok"], d["nH"]
        n = len(tok)
        idx = np.random.RandomState(42).permutation(n)
        if args.smoke:
            idx = idx[:40000]
        nval = int(len(idx) * args.val_frac)
        vidx, tidx = idx[:nval], idx[nval:]
        tr = HCompDataset([tok[i] for i in tidx], [nH[i] for i in tidx])
        va = HCompDataset([tok[i] for i in vidx], [nH[i] for i in vidx])
        log("data: train=%d val=%d  (n=%d)  ddp=%s world=%d" % (len(tr), len(va), n, ddp, world))

    tsamp = torch.utils.data.distributed.DistributedSampler(tr, world, rank, shuffle=True) if ddp else None
    nw = 8 if args.struct_data else 4
    trl = DataLoader(tr, batch_size=args.batch, shuffle=(tsamp is None), sampler=tsamp,
                     collate_fn=collate, num_workers=nw, drop_last=True, pin_memory=True,
                     worker_init_fn=_winit)
    val = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate,
                     num_workers=4, pin_memory=True, worker_init_fn=_winit)

    model = HCompleter(args.d_model, args.n_layers).to(device)
    if rank == 0:
        np_ = sum(p.numel() for p in model.parameters())
        log("model params: %.2fM  d=%d layers=%d" % (np_ / 1e6, args.d_model, args.n_layers))
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(trl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    os.makedirs(args.out, exist_ok=True)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        if tsamp is not None:
            tsamp.set_epoch(ep)
        t0 = time.time(); run = 0.0; nb = 0
        for values, slots, actions, target, pmask in trl:
            values, slots, actions, target, pmask = (values.to(device), slots.to(device),
                                                     actions.to(device), target.to(device), pmask.to(device))
            logits = model(values, slots, actions, pmask)
            loss = F.cross_entropy(logits.reshape(-1, N_H_CLASSES), target.reshape(-1), ignore_index=IGN)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += float(loss); nb += 1
        if rank == 0:
            acc, pca, ntok = evaluate(model.module if ddp else model, val, device)
            log("ep %2d  loss %.4f  val_acc %.4f  per-class[0/1/2/3] %.3f/%.3f/%.3f/%.3f  n=%d  %.0fs"
                % (ep, run / max(nb, 1), acc, pca[0], pca[1], pca[2], pca[3], ntok, time.time() - t0))
            if acc > best:
                best = acc
                torch.save({"model": (model.module if ddp else model).state_dict(),
                            "acc": acc, "cfg": vars(args)}, os.path.join(args.out, "best.pt"))
    log("DONE best_val_acc=%.4f -> %s/best.pt" % (best, args.out))
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
