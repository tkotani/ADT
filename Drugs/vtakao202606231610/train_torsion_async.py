#!/usr/bin/env python3
"""train_torsion_async.py — the same on-the-fly torsion-IKT loop, but xTB no longer blocks the GPU.

WHY. In the synchronous version one iteration is

      generate (GPU, seconds)  ->  xTB (CPU, 150-200 s)  ->  4-6 gradient steps (GPU, ~2 s)

so the GPU is idle ~95% of the wall clock and the corrector sees only a handful of gradient steps per
xTB batch. After 20 iterations it had taken ~100 updates -- not even a warm-up for an 86M model, which
is why "is it learning?" was unanswerable.

WHAT CHANGES. A PRODUCER thread does generate -> xTB -> teachers -> (periodically) the paired
evaluation, and pushes teachers into the replay buffer. The MAIN thread trains continuously from that
buffer and never waits for xTB. Same xTB budget, same data, but the GPU runs flat out: gradient steps
per hour go up by 1-2 orders of magnitude.

Nothing about the science changes: same teachers (the torsion/bend change the clamp->unclamp relaxation
performed), same rescuer-mode evaluation (only molecules xTB rejected are touched, so "broken" is
impossible), same harvesting of a rescue into a teacher, same metrics.jsonl schema.
"""
import os, sys, json, time, argparse, threading, copy, math
from collections import deque

sys.path.insert(0, os.path.expanduser("~/ADT/common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AROMATIZE_RINGS", "1")

import numpy as np
import torch

from adt_model import build_model, ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD, END
from adt_dataset import FrameSampler
from rollout_batched import rollout_batch_kv
from reward_xtb import xvr_reward_batch
from ikt_model import IKTModel
from torsion_ikt import TorsionIKT, forward_kinematics, selfmis

from train_torsion_online import (dihedral, tree_from_atoms, torsions, bendangles,
                                  geom_score, band)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="the PAPER ADT (frozen)")
    p.add_argument("--frame_cache", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hours", type=float, default=48.0)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_sched", default="const", choices=["const", "cosine"],
                   help="const = what we have been running: lr 3e-4 for every one of the 2,700 updates, no "
                        "warm-up, no decay. That is a big stride all the way to the end, and a plausible "
                        "reason the rescue rate rose fast and then stopped improving. cosine decays to "
                        "--lr_min over --lr_horizon updates.")
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--lr_horizon", type=int, default=20000)
    p.add_argument("--lr_warmup", type=int, default=0, help="linear warm-up updates")
    p.add_argument("--wd", type=float, default=0.0, help="AdamW weight decay (we have been running 0)")
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--dump_teachers", default="",
                   help="append every teacher to this .jsonl. A corpus on disk lets the Adam/schedule "
                        "sweep run OFFLINE, with no xTB at all -- which is the only way to iterate on the "
                        "optimiser when xTB is 98% of the wall clock.")
    p.add_argument("--n_bins", type=int, default=240)
    p.add_argument("--max_deg", type=float, default=180.0)
    p.add_argument("--sigma_bins", type=float, default=2.0)
    p.add_argument("--bend_bins", type=int, default=0)
    p.add_argument("--bend_deg", type=float, default=10.0)
    p.add_argument("--w_bend", type=float, default=1.0)
    p.add_argument("--end_bias", type=float, default=2.0)
    p.add_argument("--size_nmax", type=int, default=0, help="0 = from the ckpt")
    p.add_argument("--max_steps_per_mol", type=int, default=0, help="0 = from the ckpt")
    p.add_argument("--eval_every", type=int, default=4, help="paired evaluation every N produced batches")
    p.add_argument("--xtb_workers", type=int, default=28)
    p.add_argument("--xtb_bin", default=os.path.expanduser("~/xtb/bin/xtb"))
    p.add_argument("--replay", type=int, default=6000)
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--producers", type=int, default=1,
                   help="xTB producer threads. 1 already saturates the CPU (xvr_reward_batch has its own "
                        "worker pool); 2 only helps if generation is a visible fraction of the batch time.")
    p.add_argument("--n_cand", type=int, default=48,
                   help="rotamer proposals per failed molecule. The screen that ranks them (ring closure + "
                        "steric) needs no xTB, so proposing many is nearly free -- only the top --xtb_cand "
                        "ever cost an xTB run.")
    p.add_argument("--cand_temp", type=float, default=1.0)
    p.add_argument("--xtb_cand", type=int, default=6,
                   help="how many screened rotamers are actually verified with xTB. XTP is an EXISTENCE "
                        "question, so any number of attempts is legitimate.")
    p.add_argument("--pool", type=int, default=4000,
                   help="hard pool: molecules no rotamer has solved yet. They are re-attacked in later "
                        "batches, when the corrector is better -- and a solution then is exactly the "
                        "teacher we could never get before.")
    p.add_argument("--retry", type=int, default=12, help="hard-pool molecules re-attacked per batch")
    p.add_argument("--resume", default="",
                   help="warm-start the corrector from a previous torsion_ikt.pt. The head is then already "
                        "peaked, so --warmup_teachers can be 0: it may propose rotamers from batch 1.")
    p.add_argument("--warmup_teachers", type=int, default=400,
                   help="do not let the corrector PROPOSE rotamers until it has seen this many teachers. "
                        "At init its head is uniform over the whole circle, so a sample is a uniform random "
                        "torsion on EVERY atom -- the molecule is scrambled, all n_cand proposals are "
                        "garbage, and the xTB spent verifying them is wasted (measured: 1/17 rescued vs "
                        "50-60% once trained). Until then, only the free relaxation teachers are collected.")
    p.add_argument("--max_tries", type=int, default=6, help="give up on a molecule after this many attacks")
    p.add_argument("--skip_truncated", type=int, default=1)
    p.add_argument("--min_na", type=int, default=35,
                   help="do not spend xTB below this: <=34 atoms already pass at 94%, so there is nothing "
                        "to rescue there. The filter runs BEFORE xTB, so it is a pure saving.")
    p.add_argument("--harvest", type=int, default=1)
    p.add_argument("--log_every", type=int, default=200, help="gradient steps between log lines")
    p.add_argument("--reuse", type=float, default=40.0,
                   help="GOVERNOR: how many times, on average, a teacher molecule may be revisited. Freed "
                        "from xTB the trainer runs ~10 updates/s while the producer supplies ~0.2 "
                        "teachers/s -- left alone it would revisit every teacher ~900 times per batch and "
                        "simply memorise them. The trainer waits whenever it is ahead of this budget. "
                        "0 = no governor.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wd = os.path.join(args.out_dir, "xtb_work"); os.makedirs(wd, exist_ok=True)
    met = open(os.path.join(args.out_dir, "metrics.jsonl"), "a", buffering=1)
    device = "cuda"
    torch.manual_seed(0); np.random.seed(0)
    os.environ["BANK_STRUCT"] = "1"

    ckpt = torch.load(args.ckpt, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    policy = build_model(cfg).to(device); policy.load_state_dict(ckpt["model"]); policy.eval()
    for q in policy.parameters():
        q.requires_grad_(False)
    gen = ckpt.get("gen") or {}
    if not args.size_nmax:
        args.size_nmax = int(gen.get("size_nmax", 56))
    if not args.max_steps_per_mol:
        args.max_steps_per_mol = int(gen.get("max_steps_per_mol", 60))
    print(f"[async] generation params taken FROM THE CKPT: size_nmax={args.size_nmax} "
          f"max_steps_per_mol={args.max_steps_per_mol} max_offset={gen.get('max_offset')}", flush=True)
    fs = FrameSampler.load(args.frame_cache)

    ikt = TorsionIKT(cfg, n_bins=args.n_bins, max_deg=args.max_deg,
                     bend_bins=args.bend_bins, bend_deg=args.bend_deg).to(device)
    ikt.warm_start_from_adt(ckpt["model"])
    if args.resume:
        _rc = torch.load(args.resume, weights_only=False, map_location=device)
        ikt.load_state_dict(_rc["ikt"])
        print(f"[async] RESUMED from {args.resume} (updates={_rc.get('updates')}) -- the head is already "
              f"peaked, so no warm-up is needed", flush=True)
    opt = torch.optim.AdamW(ikt.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, args.beta2))
    print(f"[async] AdamW lr={args.lr} sched={args.lr_sched} (min {args.lr_min}, horizon {args.lr_horizon}, "
          f"warmup {args.lr_warmup})  wd={args.wd}  beta2={args.beta2}", flush=True)
    dumpf = open(args.dump_teachers, "a", buffering=1) if args.dump_teachers else None

    def set_lr(u):
        if args.lr_warmup and u < args.lr_warmup:
            lr = args.lr * (u + 1) / args.lr_warmup
        elif args.lr_sched == "cosine":
            f = min(1.0, u / max(args.lr_horizon, 1))
            lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * f))
        else:
            lr = args.lr
        for gp in opt.param_groups:
            gp["lr"] = lr
        return lr
    print(f"[async] torsion-IKT: {args.n_bins} bins over +-{args.max_deg:.0f} deg "
          f"({2*args.max_deg/args.n_bins:.2f} deg each), params={sum(q.numel() for q in ikt.parameters())/1e6:.1f}M",
          flush=True)
    if ikt.bend:
        print(f"[async] BEND head ON: {args.bend_bins} bins over +-{args.bend_deg:.0f} deg "
              f"({2*args.bend_deg/args.bend_bins:.2f} deg each), w_bend={args.w_bend}", flush=True)
    print(f"[async] {args.producers} xTB producer thread(s) x {args.xtb_workers} workers; the trainer never "
          f"waits for xTB", flush=True)

    replay = deque(maxlen=args.replay)
    hard = deque(maxlen=args.pool)    # molecules nobody has solved yet -> re-attacked as the IKT improves
    plock = threading.Lock()          # hard pool
    rlock = threading.Lock()          # replay
    glock = threading.Lock()          # every GPU use of `policy` / `ikt` (train step and producer forward)
    stop = threading.Event()
    t0 = time.time()
    state = {"batches": 0, "teachers": 0, "harvested": 0, "updates": 0, "ce": float("nan"),
             "ceb": float("nan"), "xtb_s": 0.0, "ce_fresh": float("nan"), "waited": 0.0,
             "pool_solved": 0}
    slock = threading.Lock()

    def ce_of(items, want_bend=True):
        """cross-entropy of `items` under the current model. Called by the producer on the teachers it has
        just made, BEFORE they enter the replay buffer -> a genuine holdout loss: data the model has never
        trained on. Train CE falling while this does not = memorisation, and we would see it immediately."""
        with glock, torch.no_grad():
            logits, blogits, _ = ikt([c["tokens"] for c in items], [c["idx"] for c in items],
                                     device, want_bend=want_bend)
            ce = 0.0; n = 0
            for b, c in enumerate(items):
                msk = torch.tensor(c["mask"], device=device)
                if not msk.any():
                    continue
                tgt = torch.tensor(c["dtheta"], device=device)[msk]
                lg = logits[b, :c["na"]][msk]
                soft = ikt.soft_labels(tgt, sigma_bins=args.sigma_bins)
                ce += float(-(soft * torch.log_softmax(lg, -1)).sum()); n += int(msk.sum())
        return ce / max(n, 1)

    # ---------------- PRODUCER: generate -> xTB -> teachers (and, periodically, the paired eval) --------
    def produce():
        while not stop.is_set():
            try:
                mols, tokens_list, idxs, metas = [], [], [], []
                with glock, torch.no_grad():
                    for tokens, n_frame, atoms, bonds, na, done in rollout_batch_kv(
                            policy, fs, device, args.batch, max_steps=args.max_steps_per_mol,
                            temperature=1.0, end_bias=args.end_bias, end_bias_arr=None,
                            size_ceiling=args.size_nmax):
                        if na < args.min_na:
                            continue
                        if args.skip_truncated and na >= args.size_nmax:
                            continue                 # ceiling => truncated mid-molecule, not a molecule
                        ix = IKTModel.atom_token_index(tokens)
                        if len(ix) < na:
                            continue
                        parent, child_of, b0 = tree_from_atoms(atoms, bonds, na)
                        mols.append((atoms, bonds, na)); tokens_list.append(tokens); idxs.append(ix[:na])
                        metas.append({"na": na,
                                      "pos": np.array([atoms[k].pos for k in range(na)], float),
                                      "anum": np.array([atoms[k].atomic_num for k in range(na)], int),
                                      "parent": parent, "child_of": child_of, "b0": b0,
                                      "order": list(range(na))})
                if not mols:
                    continue

                tx = time.time()
                rinfo = xvr_reward_batch([(a, b, n) for a, b, n in mols], args.xtb_bin, wd,
                                         max_workers=args.xtb_workers)
                xtb_s = time.time() - tx
                raw_pass = [bool(r.get("same_topo")) for r in rinfo]

                # teachers: what the relaxation did, in the model's own output space
                fresh = []
                for i, r in enumerate(rinfo):
                    rh = r.get("realizable_heavy")
                    m = metas[i]
                    if not raw_pass[i] or rh is None or len(rh) != m["na"]:
                        continue
                    t_adt, ok0 = torsions(m["pos"], m["parent"], m["child_of"])
                    t_rel, ok1 = torsions(np.asarray(rh, float), m["parent"], m["child_of"])
                    ok = ok0 & ok1
                    if not ok.any():
                        continue
                    d = np.arctan2(np.sin(t_rel - t_adt), np.cos(t_rel - t_adt))
                    ent = {"tokens": tokens_list[i], "idx": idxs[i], "na": m["na"],
                           "dtheta": d.astype(np.float32), "mask": ok}
                    if ikt.bend:
                        b_adt, okb0 = bendangles(m["pos"], m["parent"])
                        b_rel, okb1 = bendangles(np.asarray(rh, float), m["parent"])
                        ent["dbend"] = (b_rel - b_adt).astype(np.float32)
                        ent["bmask"] = okb0 & okb1
                    fresh.append(ent)
                n_new = len(fresh)

                # HOLDOUT: score the new teachers BEFORE they are ever trained on
                ce_fresh = ce_of(fresh[:24]) if fresh else float("nan")
                with rlock:
                    for ent in fresh:
                        replay.append(ent)
                if dumpf is not None:
                    for ent in fresh:
                        dumpf.write(json.dumps({
                            "tokens": list(map(int, ent["tokens"])), "idx": list(map(int, ent["idx"])),
                            "na": int(ent["na"]), "src": "relax",
                            "dtheta": [round(float(x), 5) for x in ent["dtheta"]],
                            "mask": [bool(x) for x in ent["mask"]],
                            **({"dbend": [round(float(x), 5) for x in ent["dbend"]],
                                "bmask": [bool(x) for x in ent["bmask"]]} if "dbend" in ent else {})}) + "\n")

                with slock:
                    state["batches"] += 1; state["teachers"] += n_new; state["xtb_s"] += xtb_s
                    if not np.isnan(ce_fresh):
                        state["ce_fresh"] = ce_fresh
                    bidx = state["batches"]
                    ce_now, ceb_now, upd = state["ce"], state["ceb"], state["updates"]

                rec = {"step": bidx, "n": len(mols), "n_teacher": n_new, "replay": len(replay),
                       "updates": upd,
                       "ce": None if np.isnan(ce_now) else ce_now,
                       "ceb": None if np.isnan(ceb_now) else ceb_now,
                       "ce_fresh": None if np.isnan(ce_fresh) else round(ce_fresh, 4),
                       "xtp_raw": {b: [0, 0] for b in ("<=34", "35-39", "40+")},
                       "xtb_s": round(xtb_s, 1), "t": round(time.time() - t0, 1)}
                for i, m in enumerate(metas):
                    rec["xtp_raw"][band(m["na"])][0] += 1
                    rec["xtp_raw"][band(m["na"])][1] += int(raw_pass[i])

                # ---- SOLVER: manufacture "this rotation worked" examples ----------------------------
                # Every molecule xTB rejected is attacked with many rotamer proposals, screened xTB-free,
                # and the survivors are verified with xTB. EVERY solution becomes a teacher. This -- not the
                # relaxation-derived teacher -- is the signal we actually want: a teacher for a molecule that
                # FAILED. Molecules nobody solved go into a hard pool and are re-attacked in later batches,
                # by which time the corrector has improved.
                targets = [{"tokens": tokens_list[i], "idx": idxs[i], "m": metas[i], "atoms": mols[i][0],
                            "bonds": mols[i][1], "na": mols[i][2], "tries": 0, "fresh": True}
                           for i in range(len(metas)) if not raw_pass[i]]
                with plock:
                    retry = [hard.popleft() for _ in range(min(args.retry, len(hard)))]
                for it in retry:
                    it["fresh"] = False
                items = targets + retry

                n_harv = 0; solved_fresh = set(); n_pool_solved = 0
                with slock:
                    warm = state["teachers"] + state["harvested"] >= args.warmup_teachers
                if items and not warm:
                    with plock:                       # keep them: they are attacked once the head is peaked
                        for it in targets:
                            hard.append(it)
                    items = []
                if items:
                    # 1. one forward pass, then EVERYTHING on the CPU: the Rodrigues chain is a per-atom
                    #    python loop, so running n_cand of them under the GPU lock would stall the trainer.
                    with glock, torch.no_grad():
                        lg, bl, _ = ikt([it["tokens"] for it in items], [it["idx"] for it in items],
                                        device, want_bend=True)
                        lg = lg.detach().cpu()
                        bl = bl.detach().cpu() if bl is not None else None

                    for j, it in enumerate(items):
                        m = it["m"]; na = it["na"]
                        pos = torch.tensor(m["pos"], dtype=torch.float32)
                        cands = []
                        with torch.no_grad():
                            for ci in range(max(1, args.n_cand)):
                                mode = "argmax" if ci == 0 else "sample"
                                dth = ikt.predict(lg[j:j+1], mode=mode, temperature=args.cand_temp)[0]
                                dbn = (ikt.predict_bend(bl[j:j+1], mode=mode, temperature=args.cand_temp)[0]
                                       if bl is not None else None)
                                new = forward_kinematics(
                                    pos, m["parent"], m["order"], dth[:na],
                                    dbend=(dbn[:na] if dbn is not None else None)).numpy()
                                cands.append((geom_score(m["anum"], new, m["b0"], na), new,
                                              dth[:na].numpy().astype(np.float32),
                                              (dbn[:na].numpy().astype(np.float32)
                                               if dbn is not None else None)))
                        cands.sort(key=lambda t: t[0])          # the xTB-FREE screen is what makes n_cand
                        it["cands"] = cands[:max(1, args.xtb_cand)]   # large affordable: only the top few
                                                                      # ever cost an xTB run

                    # 2. verify the screened rotamers with xTB -- ALL of them in ONE batch.
                    #    Doing it as `xtb_cand` sequential rounds looks cheaper (early exit once a molecule
                    #    is solved) but it is far slower in wall clock: round 1 has only ~20 jobs for 56
                    #    workers, round 2 fewer still, so the pool sits idle and the rounds serialise. One
                    #    flat batch of (failure x rotamer) fills every worker: same number of xTB runs,
                    #    ~3x less wall clock. Wall clock is what limits the teacher supply.
                    jobs = []
                    for it in items:
                        for ci, (_, newc, dthc, dbnc) in enumerate(it["cands"]):
                            at = copy.deepcopy(it["atoms"])
                            for k in range(it["na"]):
                                at[k].pos = newc[k].astype(np.float64)
                            jobs.append((it, ci, at, dthc, dbnc))
                    if jobs:
                        rin2 = xvr_reward_batch([(at, it["bonds"], it["na"]) for it, _, at, _, _ in jobs],
                                                args.xtb_bin, wd, max_workers=args.xtb_workers)
                        for (it, ci, at, dthc, dbnc), r2 in zip(jobs, rin2):
                            if not bool(r2.get("same_topo")):
                                continue
                            if it.get("solved") and it.get("rank", 99) <= ci:
                                continue                      # keep the best-screened rotamer that worked
                            it["solved"] = True; it["rank"] = ci
                            it["dth_used"] = dthc; it["dbn_used"] = dbnc

                    # 3. harvest every solution; recycle the rest
                    for it in items:
                        if it.get("solved"):
                            ent = {"tokens": it["tokens"], "idx": it["idx"], "na": it["na"],
                                   "dtheta": it["dth_used"], "mask": np.ones(it["na"], bool)}
                            if ikt.bend and it.get("dbn_used") is not None:
                                ent["dbend"] = it["dbn_used"]
                                ent["bmask"] = np.ones(it["na"], bool)
                            with rlock:
                                replay.append(ent)
                            if dumpf is not None:
                                dumpf.write(json.dumps({
                                    "tokens": list(map(int, ent["tokens"])), "idx": list(map(int, ent["idx"])),
                                    "na": int(ent["na"]), "src": "solved",
                                    "dtheta": [round(float(x), 5) for x in ent["dtheta"]],
                                    "mask": [bool(x) for x in ent["mask"]],
                                    **({"dbend": [round(float(x), 5) for x in ent["dbend"]],
                                        "bmask": [bool(x) for x in ent["bmask"]]} if "dbend" in ent else {})}) + "\n")
                            n_harv += 1
                            if it["fresh"]:
                                solved_fresh.add(id(it["m"]))
                            else:
                                n_pool_solved += 1
                        else:
                            it["tries"] += 1
                            if it["tries"] < args.max_tries:
                                for key in ("cands", "dth_used", "dbn_used"):
                                    it.pop(key, None)
                                with plock:
                                    hard.append(it)          # a smarter corrector gets another go at it
                    with slock:
                        state["harvested"] += n_harv
                        state["pool_solved"] += n_pool_solved

                ikt_pass = [raw_pass[i] or (id(metas[i]) in solved_fresh) for i in range(len(metas))]
                ev = {b: {"n": 0, "raw": 0, "ikt": 0, "rescued": 0, "broken": 0}
                      for b in ("<=34", "35-39", "40+")}
                for i, m in enumerate(metas):
                    e = ev[band(m["na"])]
                    e["n"] += 1; e["raw"] += int(raw_pass[i]); e["ikt"] += int(ikt_pass[i])
                    e["rescued"] += int(ikt_pass[i] and not raw_pass[i])
                rec["eval"] = ev
                rec["harvested"] = n_harv
                rec["pool_solved"] = n_pool_solved
                rec["pool"] = len(hard)
                parts = []
                for k in ("35-39", "40+"):
                    e = ev[k]
                    if e["n"]:
                        nf = e["n"] - e["raw"]
                        parts.append(f"{k}: {100*e['raw']/e['n']:.0f}%->{100*e['ikt']/e['n']:.0f}% "
                                     f"({e['rescued']}/{nf})")
                if not warm:
                    with slock:
                        ntea_w = state["teachers"] + state["harvested"]
                    print(f"[warmup] batch {bidx:4d}  teachers {ntea_w}/{args.warmup_teachers}  "
                          f"(the head is still uniform: a proposal would scramble the molecule, so no xTB "
                          f"is spent on rotamers yet)  pool={len(hard)}  xtb={xtb_s:.0f}s", flush=True)
                    met.write(json.dumps(rec) + "\n")
                    continue
                ranks = [it["rank"] for it in items if it.get("solved")]
                print(f"[solve] batch {bidx:4d} upd {upd:6d}  " + "  ".join(parts)
                      + f"  teachers +{n_new}rlx +{n_harv}solved ({n_pool_solved} from pool)"
                        f"  rank0={sum(1 for r in ranks if r==0)}/{len(ranks)}"
                        f"  pool={len(hard)}  xtb={xtb_s:.0f}s  {time.time()-t0:.0f}s", flush=True)

                met.write(json.dumps(rec) + "\n")
            except Exception as e:
                print(f"[producer] {type(e).__name__}: {e}", flush=True)
                time.sleep(2)

    threads = [threading.Thread(target=produce, daemon=True) for _ in range(args.producers)]
    for th in threads:
        th.start()

    # ---------------- TRAINER: never waits for xTB ----------------
    t_last = time.time(); upd_last = 0
    while time.time() - t0 < args.hours * 3600:
        with rlock:
            n = len(replay)
        if n < 32:
            time.sleep(2); continue
        # GOVERNOR: never revisit a teacher more than `reuse` times on average. Without it the freed
        # trainer runs ~300x faster than the producer supplies data and just memorises the buffer.
        if args.reuse > 0:
            with slock:
                budget = args.reuse * (state["teachers"] + state["harvested"]) / max(args.train_bs, 1)
                ahead = state["updates"] >= budget
            if ahead:
                with slock:
                    state["waited"] += 0.5
                time.sleep(0.5); continue
        with rlock:
            items = [replay[i] for i in np.random.choice(n, min(args.train_bs, n), replace=False)]
        with glock:
            cur_lr = set_lr(state["updates"])
            logits, blogits, _ = ikt([c["tokens"] for c in items], [c["idx"] for c in items],
                                     device, want_bend=True)
            ce = torch.zeros((), device=device); nt = 0
            ceb = torch.zeros((), device=device); nb = 0
            for b, c in enumerate(items):
                msk = torch.tensor(c["mask"], device=device)
                if msk.any():
                    tgt = torch.tensor(c["dtheta"], device=device)[msk]
                    lg = logits[b, :c["na"]][msk]
                    soft = ikt.soft_labels(tgt, sigma_bins=args.sigma_bins)
                    ce = ce + -(soft * torch.log_softmax(lg, -1)).sum(); nt += int(msk.sum())
                if blogits is not None and c.get("bmask") is not None:
                    mb = torch.tensor(c["bmask"], device=device)
                    if mb.any():
                        tb = torch.tensor(c["dbend"], device=device)[mb]
                        lb = blogits[b, :c["na"]][mb]
                        sb = ikt.bend_soft_labels(tb, sigma_bins=args.sigma_bins)
                        ceb = ceb + -(sb * torch.log_softmax(lb, -1)).sum(); nb += int(mb.sum())
            ce = ce / max(nt, 1); ceb = ceb / max(nb, 1)
            loss = ce + (args.w_bend * ceb if nb else 0.0)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(ikt.parameters(), 1.0); opt.step()

        with slock:
            state["updates"] += 1; state["ce"] = float(ce)
            state["ceb"] = float(ceb) if nb else float("nan")
            upd = state["updates"]; nbatch = state["batches"]
            cef = state["ce_fresh"]; ntea = state["teachers"] + state["harvested"]

        if upd % args.log_every == 0:
            dt = time.time() - t_last
            rate = (upd - upd_last) / max(dt, 1e-6)
            el = time.time() - t0
            # CE     = training loss (on the replay buffer)
            # CEfresh= the SAME loss on teachers the model has never seen (the producer scores them before
            #          they enter the buffer). CE falling while CEfresh does not = memorisation.
            print(f"upd {upd:7d} ({rate:5.2f}/s)  lr={cur_lr:.1e}  CE={float(ce):.3f}  CEfresh={cef:.3f}"
                  + (f"  CEbend={float(ceb):.3f}" if nb else "")
                  + f"  replay={n}  batches={nbatch}  teachers={ntea}  "
                    f"reuse={upd*args.train_bs/max(ntea,1):.0f}x  {el:.0f}s", flush=True)
            t_last = time.time(); upd_last = upd
            torch.save({"ikt": ikt.state_dict(), "config": cfg, "updates": upd, "args": vars(args)},
                       os.path.join(args.out_dir, "torsion_ikt.pt"))

    stop.set()
    print("[async] done", flush=True)


if __name__ == "__main__":
    main()
