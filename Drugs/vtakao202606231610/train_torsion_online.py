#!/usr/bin/env python3
"""train_torsion_online.py — ADT -> xTB -> torsion-IKT, fully on the fly.

Every step:
  1. the FROZEN paper ADT generates a batch
  2. xTB runs the real XTP protocol on the RAW geometry (clamp -> unclamp)
        -> XTP_raw, and for the molecules that pass, the relaxed geometry
  3. teacher = the torsion change the relaxation performed, read in the model's own output space:
        theta(k) = dihedral(grandparent, parent, k, child)   about the (parent -> k) axis
        target   = wrap(theta_relaxed - theta_ADT), clipped to the +-max_deg window
     -> categorical cross-entropy (the correction is MULTIMODAL; an L2 head collapses to "do nothing",
        which is exactly what we measured)
  4. every --eval_every steps, the SAME molecules are re-run with the IKT correction applied
        -> XTP_ikt, paired with XTP_raw:  rescued (fail -> pass) and broken (pass -> fail), by size band

Everything lands in metrics.jsonl, so the time axis shows whether the corrector is actually improving
the thing we care about: XTP at 40+ atoms.
"""
import os, sys, json, time, argparse
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
from torsion_ikt import TorsionIKT, forward_kinematics, selfmis, geometry_loss

ATOM_ACTS = (ADD_INIT, ADD_CHAIN, ADD_ANGLE, ADD)


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return float(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


def tree_from_atoms(atoms, bonds, na):
    """parent / one reference child / declared graph, straight from the rollout state (exact)."""
    parent = np.full(na, -1, int)
    for k in range(na):
        p = int(getattr(atoms[k], "parent_id", 0)) - 1
        if 0 <= p < na and p != k:
            parent[k] = p
    b0 = sorted({(min(int(a) - 1, int(b) - 1), max(int(a) - 1, int(b) - 1)) for a, b in bonds
                 if 0 <= int(a) - 1 < na and 0 <= int(b) - 1 < na and int(a) != int(b)})
    child_of = {}
    for k in range(na):
        p = parent[k]
        if p >= 0 and p not in child_of:
            child_of[p] = k
    return parent, child_of, b0


def torsions(pos, parent, child_of):
    th = np.zeros(len(parent)); ok = np.zeros(len(parent), bool)
    for k in range(len(parent)):
        p = parent[k]
        if p < 0:
            continue
        g = parent[p]; c = child_of.get(k)
        if g < 0 or c is None:
            continue
        th[k] = dihedral(pos[g], pos[p], pos[k], pos[c]); ok[k] = True
    return th, ok


def bendangles(pos, parent):
    """angle between (grandparent->parent) and (parent->self). Rotating the bond about the normal of
    that plane by +dbend increases this angle -- the same sign convention forward_kinematics uses."""
    th = np.zeros(len(parent)); ok = np.zeros(len(parent), bool)
    for k in range(len(parent)):
        p = parent[k]
        if p < 0:
            continue
        g = parent[p]
        if g < 0:
            continue
        u = pos[p] - pos[g]; v = pos[k] - pos[p]
        nu = float(np.linalg.norm(u)); nv = float(np.linalg.norm(v))
        if nu < 1e-6 or nv < 1e-6:
            continue
        th[k] = float(np.arccos(np.clip(float(np.dot(u, v)) / (nu * nv), -1.0, 1.0))); ok[k] = True
    return th, ok


def geom_score(anum, pos, b0, na):
    """cheap, xTB-free score of a candidate conformer: how badly does it violate the DECLARED graph?
    (rings that never closed + atoms stabbed into each other). Lower is better."""
    miss, spur = selfmis(anum, pos, b0, na)
    return miss * 10.0 + spur * 1.0        # an unclosed ring is far worse than a marginal contact


def band(na):
    return "<=34" if na <= 34 else ("35-39" if na <= 39 else "40+")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="the PAPER ADT (frozen)")
    p.add_argument("--frame_cache", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_bins", type=int, default=60)
    p.add_argument("--max_deg", type=float, default=60.0)
    p.add_argument("--sigma_bins", type=float, default=2.0)
    p.add_argument("--bend_bins", type=int, default=0,
                   help="0 = torsion only. >0 adds a BEND head (the bond angle at the parent).")
    p.add_argument("--bend_deg", type=float, default=10.0,
                   help="half-window of the bend head, degrees. sp-hybridisation nearly fixes the "
                        "bond angle -> keep this SMALL (the relaxation only moves 1-3 by 0.027 A).")
    p.add_argument("--w_bend", type=float, default=1.0, help="weight of the bend CE")
    p.add_argument("--end_bias", type=float, default=1.6, help="push the size distribution up to reach 40+")
    p.add_argument("--size_nmax", type=int, default=0,
                   help="0 = take it from the ckpt's self-describing gen block (the generator was trained "
                        "with size_nmax=56, max_steps_per_mol=60, max_offset=50; inventing a smaller "
                        "ceiling truncates molecules mid-generation and creates a fake 'big molecule' band)")
    p.add_argument("--max_steps_per_mol", type=int, default=0, help="0 = from the ckpt")
    p.add_argument("--eval_every", type=int, default=10, help="paired ADT vs ADT+IKT evaluation")
    p.add_argument("--xtb_workers", type=int, default=14)
    p.add_argument("--xtb_bin", default=os.path.expanduser("~/xtb/bin/xtb"))
    p.add_argument("--replay", type=int, default=3000, help="teacher molecules kept for replay")
    p.add_argument("--inner", type=int, default=4, help="gradient steps per generated batch")
    p.add_argument("--n_cand", type=int, default=4,
                   help="correction CANDIDATES sampled from the per-atom torsion distribution. XTP is an "
                        "EXISTENCE question, so proposing several conformers and keeping the best one is "
                        "legitimate -- and this is exactly a conformer generator.")
    p.add_argument("--cand_temp", type=float, default=1.0)
    p.add_argument("--xtb_cand", type=int, default=1,
                   help="how many of the screened candidates are actually VERIFIED with xTB (only for the "
                        "molecules xTB rejected). XTP is an existence question, so trying several rotamers "
                        "is legitimate -- and this is exactly what a conformer search does.")
    p.add_argument("--skip_truncated", type=int, default=1,
                   help="drop the molecules that hit the size ceiling: the rollout stops them WITHOUT an "
                        "END token (rollout_batched.py:203), so they are cut mid-molecule -- an artefact, "
                        "not a big molecule. With a strong end_bias they otherwise dominate the 40+ band.")
    p.add_argument("--min_na", type=int, default=30,
                   help="skip the small molecules: below ~30 atoms the corrector has nothing to do "
                        "(XTP_raw is already 97-98%), so they only dilute the signal and burn xTB")
    p.add_argument("--w_geom", type=float, default=0.0,
                   help="add a DIFFERENTIABLE geometry loss (close the declared rings, un-stab contacts) on "
                        "the soft-expected torsion. It needs no teacher, so it also gives gradient on the "
                        "molecules that FAILED -- where the relaxation-derived teacher does not exist.")
    p.add_argument("--harvest", type=int, default=1,
                   help="a candidate that RESCUES a molecule xTB had rejected becomes a teacher. There is "
                        "no single right answer -- any torsion set that xTB accepts is one -- and this is "
                        "the only way to get a teacher for a FAILED molecule.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wd = os.path.join(args.out_dir, "xtb_work"); os.makedirs(wd, exist_ok=True)
    met = open(os.path.join(args.out_dir, "metrics.jsonl"), "a", buffering=1)
    device = "cuda"
    torch.manual_seed(0); np.random.seed(0)

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
    print(f"[online] generation params taken FROM THE CKPT: size_nmax={args.size_nmax} "
          f"max_steps_per_mol={args.max_steps_per_mol} max_offset={gen.get('max_offset')}", flush=True)
    fs = FrameSampler.load(args.frame_cache)

    ikt = TorsionIKT(cfg, n_bins=args.n_bins, max_deg=args.max_deg,
                     bend_bins=args.bend_bins, bend_deg=args.bend_deg).to(device)
    ikt.warm_start_from_adt(ckpt["model"])
    opt = torch.optim.AdamW(ikt.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"[online] ADT frozen = {args.ckpt}", flush=True)
    print(f"[online] torsion-IKT: {args.n_bins} bins over +-{args.max_deg:.0f} deg "
          f"({2*args.max_deg/args.n_bins:.1f} deg each), params={sum(q.numel() for q in ikt.parameters())/1e6:.1f}M",
          flush=True)
    if ikt.bend:
        print(f"[online] BEND head ON: {args.bend_bins} bins over +-{args.bend_deg:.0f} deg "
              f"({2*args.bend_deg/args.bend_bins:.2f} deg each), w_bend={args.w_bend}. "
              f"Bond LENGTHS stay exact; the only angle a bend moves is angle(grandparent,parent,self).",
              flush=True)
    print(f"[online] end_bias={args.end_bias} nmax={args.size_nmax} -> the stream must contain 40+ molecules",
          flush=True)

    replay = deque(maxlen=args.replay)
    ema = {}
    t0 = time.time()

    for step in range(args.steps):
        # ---- 1. generate ----
        mols, tokens_list, idxs, metas = [], [], [], []
        with torch.no_grad():
            for tokens, n_frame, atoms, bonds, na, done in rollout_batch_kv(
                    policy, fs, device, args.batch, max_steps=args.max_steps_per_mol, temperature=1.0,
                    end_bias=args.end_bias, end_bias_arr=None, size_ceiling=args.size_nmax):
                if na < args.min_na:
                    continue
                if args.skip_truncated and na >= args.size_nmax:
                    continue                       # hit the ceiling => truncated mid-generation, not a molecule
                ix = IKTModel.atom_token_index(tokens)
                if len(ix) < na:
                    continue
                parent, child_of, b0 = tree_from_atoms(atoms, bonds, na)
                pos = np.array([atoms[k].pos for k in range(na)], float)
                anum = np.array([atoms[k].atomic_num for k in range(na)], int)
                mols.append((atoms, bonds, na)); tokens_list.append(tokens); idxs.append(ix[:na])
                metas.append({"na": na, "pos": pos, "anum": anum, "parent": parent,
                              "child_of": child_of, "b0": b0, "order": list(range(na))})
        if not mols:
            continue

        # ---- 2. xTB on the RAW geometry: XTP_raw + the relaxed geometry of the passes ----
        os.environ["BANK_STRUCT"] = "1"
        rinfo = xvr_reward_batch([(a, b, n) for a, b, n in mols], args.xtb_bin, wd,
                                 max_workers=args.xtb_workers)
        raw_pass = [bool(r.get("same_topo")) for r in rinfo]

        # ---- 3. teachers: the torsion change the relaxation performed ----
        n_new = 0
        for i, r in enumerate(rinfo):
            rh = r.get("realizable_heavy")
            if not raw_pass[i] or rh is None or len(rh) != metas[i]["na"]:
                continue
            m = metas[i]
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
                ent["dbend"] = (b_rel - b_adt).astype(np.float32)     # small: sp fixes the angle
                ent["bmask"] = okb0 & okb1
            replay.append(ent)
            n_new += 1

        # ---- 4. train the corrector ----
        ce_val = float("nan"); geo_val = float("nan")
        if len(replay) >= 32:
            for _ in range(args.inner):
                items = [replay[i] for i in np.random.choice(len(replay), 16, replace=False)]
                logits, blogits, _ = ikt([c["tokens"] for c in items], [c["idx"] for c in items],
                                         device, want_bend=True)
                ce = torch.zeros((), device=device); n = 0
                ceb = torch.zeros((), device=device); nb = 0
                for b, c in enumerate(items):
                    msk = torch.tensor(c["mask"], device=device)
                    if msk.any():
                        tgt = torch.tensor(c["dtheta"], device=device)[msk]
                        lg = logits[b, :c["na"]][msk]
                        soft = ikt.soft_labels(tgt, sigma_bins=args.sigma_bins)
                        ce = ce + -(soft * torch.log_softmax(lg, -1)).sum(); n += int(msk.sum())
                    if blogits is not None and c.get("bmask") is not None:
                        mb = torch.tensor(c["bmask"], device=device)
                        if mb.any():
                            tb = torch.tensor(c["dbend"], device=device)[mb]
                            lb = blogits[b, :c["na"]][mb]
                            sb = ikt.bend_soft_labels(tb, sigma_bins=args.sigma_bins)
                            ceb = ceb + -(sb * torch.log_softmax(lb, -1)).sum(); nb += int(mb.sum())
                ce = ce / max(n, 1)
                ceb = ceb / max(nb, 1)
                loss = ce + (args.w_bend * ceb if nb else 0.0)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(ikt.parameters(), 1.0); opt.step()
                ce_val = float(ce)

        # differentiable geometry loss on the CURRENT batch (no teacher needed -> it also trains on the
        # molecules xTB rejected). Uses the soft expectation of the torsion distribution, so the gradient
        # flows through the softmax and the forward kinematics.
        if args.w_geom > 0:
            logits, _ = ikt(tokens_list, idxs, device)
            probs = torch.softmax(logits, dim=-1)
            theta_soft = (probs * ikt.bin_centers.to(device)).sum(-1)          # (B,K) differentiable
            gl = torch.zeros((), device=device); ng = 0
            for i, m in enumerate(metas):
                pos = torch.tensor(m["pos"], device=device, dtype=torch.float32)
                new = forward_kinematics(pos, m["parent"], m["order"], theta_soft[i, :m["na"]])
                b0 = set(m["b0"])
                tree = {(min(int(m["parent"][k]), k), max(int(m["parent"][k]), k))
                        for k in range(m["na"]) if m["parent"][k] >= 0}
                link = sorted(b0 - tree)
                adj = {x: set() for x in range(m["na"])}
                for (a, b) in b0:
                    adj[a].add(b); adj[b].add(a)
                excl = set(b0)
                for j in range(m["na"]):
                    nb = sorted(adj[j])
                    for u in range(len(nb)):
                        for v in range(u + 1, len(nb)):
                            excl.add((min(nb[u], nb[v]), max(nb[u], nb[v])))
                nbp = [(a, b) for a in range(m["na"]) for b in range(a + 1, m["na"]) if (a, b) not in excl]
                gl = gl + geometry_loss(new, m["anum"], sorted(tree), link, nbp,
                                        theta_soft[i, :m["na"]], lam_dth=0.02)
                ng += 1
            gl = args.w_geom * gl / max(ng, 1)
            opt.zero_grad(set_to_none=True); gl.backward()
            torch.nn.utils.clip_grad_norm_(ikt.parameters(), 1.0); opt.step()
            geo_val = float(gl)

        rec = {"step": step, "n": len(mols), "n_teacher": n_new, "replay": len(replay),
               "ce": None if np.isnan(ce_val) else ce_val,
               "geom": None if np.isnan(geo_val) else geo_val,
               "xtp_raw": {b: [0, 0] for b in ("<=34", "35-39", "40+")},
               "t": round(time.time() - t0, 1)}
        for i, m in enumerate(metas):
            rec["xtp_raw"][band(m["na"])][0] += 1
            rec["xtp_raw"][band(m["na"])][1] += int(raw_pass[i])

        # ---- 5. paired evaluation: the SAME molecules, with the IKT correction applied ----
        if step % args.eval_every == 0 and len(replay) >= 64:
            with torch.no_grad():
                logits, blogits, _ = ikt(tokens_list, idxs, device, want_bend=True)
            sm_before = sm_after = 0
            dth_mag = []
            for i, m in enumerate(metas):
                pos = torch.tensor(m["pos"], device=device, dtype=torch.float32)
                # --- propose n_cand conformers: the dominant mode + samples from the torsion distribution
                cands = []
                with torch.no_grad():
                    for ci in range(max(1, args.n_cand)):
                        mode = "argmax" if ci == 0 else "sample"
                        dth_i = ikt.predict(logits[i:i+1], mode=mode,
                                            temperature=args.cand_temp)[0]
                        dbn_i = None
                        if blogits is not None:
                            dbn_i = ikt.predict_bend(blogits[i:i+1], mode=mode,
                                                     temperature=args.cand_temp)[0]
                        new = forward_kinematics(
                            pos, m["parent"], m["order"], dth_i[:m["na"]].to(pos.dtype),
                            dbend=(dbn_i[:m["na"]].to(pos.dtype) if dbn_i is not None else None)
                        ).detach().cpu().numpy()
                        cands.append((geom_score(m["anum"], new, m["b0"], m["na"]), new,
                                      float(np.rad2deg(dth_i[:m["na"]].abs().mean().item())),
                                      dth_i[:m["na"]].detach().cpu().numpy().astype(np.float32),
                                      (dbn_i[:m["na"]].detach().cpu().numpy().astype(np.float32)
                                       if dbn_i is not None else None)))
                cands.sort(key=lambda t: t[0])
                _, new, mag, dth_used, dbn_used = cands[0]              # best by the xTB-free screen
                dth_mag.append(mag)
                m["dth_applied"] = dth_used
                m["dbn_applied"] = dbn_used
                m["cands"] = cands[:max(1, args.xtb_cand)]             # the rotamers we may hand to xTB
                a, b = selfmis(m["anum"], m["pos"], m["b0"], m["na"])
                c, d = selfmis(m["anum"], new, m["b0"], m["na"])
                sm_before += a + b; sm_after += c + d
                if not raw_pass[i]:                                    # only rescue what xTB rejected
                    atoms = mols[i][0]
                    for k in range(m["na"]):
                        atoms[k].pos = new[k].astype(np.float64)      # hand the corrected geometry to xTB
            # only the molecules xTB REJECTED are handed to the corrector (a rescuer, not a filter):
            # a molecule that already passed is never touched -> "broken" is impossible by construction.
            fail_idx = [i for i in range(len(metas)) if not raw_pass[i]]
            ikt_pass = list(raw_pass)
            for ci in range(max(1, args.xtb_cand)):                    # try the rotamers one after another
                todo = [i for i in fail_idx if not ikt_pass[i] and len(metas[i].get("cands", [])) > ci]
                if not todo:
                    break
                for i in todo:                                          # install candidate #ci
                    _, newc, _, dthc, dbnc = metas[i]["cands"][ci]
                    atoms = mols[i][0]
                    for k in range(metas[i]["na"]):
                        atoms[k].pos = newc[k].astype(np.float64)
                    metas[i]["dth_applied"] = dthc
                    metas[i]["dbn_applied"] = dbnc
                rin2 = xvr_reward_batch([(mols[i][0], mols[i][1], mols[i][2]) for i in todo],
                                        args.xtb_bin, wd, max_workers=args.xtb_workers)
                for j, i in enumerate(todo):
                    if bool(rin2[j].get("same_topo")):
                        ikt_pass[i] = True                              # this rotamer is realizable -> PASS
            n_harvest = 0
            if args.harvest:
                for i, m in enumerate(metas):
                    if ikt_pass[i] and not raw_pass[i] and m.get("dth_applied") is not None:
                        # xTB accepted a molecule it had rejected: the correction that did it IS a teacher
                        # (for a FAILED molecule -- which the relaxation-derived teacher can never provide)
                        msk = np.zeros(m["na"], bool)
                        for k in range(m["na"]):
                            p_ = m["parent"][k]
                            msk[k] = (p_ >= 0 and m["parent"][p_] >= 0 and k in m["child_of"].values())
                        msk = np.ones(m["na"], bool)          # every atom's torsion was part of the solution
                        ent = {"tokens": tokens_list[i], "idx": idxs[i], "na": m["na"],
                               "dtheta": m["dth_applied"], "mask": msk}
                        if ikt.bend and m.get("dbn_applied") is not None:
                            ent["dbend"] = m["dbn_applied"]
                            ent["bmask"] = np.ones(m["na"], bool)
                        replay.append(ent)
                        n_harvest += 1
            rec["harvested"] = n_harvest
            ev = {b: {"n": 0, "raw": 0, "ikt": 0, "rescued": 0, "broken": 0}
                  for b in ("<=34", "35-39", "40+")}
            for i, m in enumerate(metas):
                e = ev[band(m["na"])]
                e["n"] += 1; e["raw"] += int(raw_pass[i]); e["ikt"] += int(ikt_pass[i])
                e["rescued"] += int(ikt_pass[i] and not raw_pass[i])
                e["broken"] += int(raw_pass[i] and not ikt_pass[i])     # 0 by construction now
            rec["eval"] = ev
            rec["selfmis"] = [sm_before, sm_after]
            rec["dtheta_mean_deg"] = float(np.mean(dth_mag)) if dth_mag else 0.0
            rec["n_cand"] = args.n_cand
            for k, e in ev.items():
                if e["n"]:
                    nf = e["n"] - e["raw"]
                    a = f"{k}: {100*e['raw']/e['n']:.0f}%->{100*e['ikt']/e['n']:.0f}% "\
                        f"(rescued {e['rescued']}/{nf}, n={e['n']})"
                    ema[k] = a
            print(f"step {step:5d}  XTP raw->IKT   " + "   ".join(ema.get(k, "") for k in ("<=34", "35-39", "40+"))
                  + f"   selfmis {sm_before}->{sm_after}  |dth|={rec['dtheta_mean_deg']:.1f}deg  "
                    f"harvested={n_harvest}  CE={ce_val:.3f}  replay={len(replay)}  {time.time()-t0:.0f}s",
                  flush=True)
        else:
            tot = sum(v[0] for v in rec["xtp_raw"].values())
            ok = sum(v[1] for v in rec["xtp_raw"].values())
            print(f"step {step:5d}  XTP_raw {ok}/{tot}  teachers +{n_new} (replay {len(replay)})  "
                  f"CE={ce_val:.3f}  {time.time()-t0:.0f}s", flush=True)

        met.write(json.dumps(rec) + "\n")
        if step % 200 == 0 and step:
            torch.save({"ikt": ikt.state_dict(), "config": cfg, "step": step, "args": vars(args)},
                       os.path.join(args.out_dir, "torsion_ikt.pt"))


if __name__ == "__main__":
    main()
