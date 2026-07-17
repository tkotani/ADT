#!/usr/bin/env python3
"""ikt_eval.py — the honest test. FROZEN IKT, FRESH molecules, NO training, NO harvesting.

The training loop reports a rescue rate, but it is entangled with the loop: the model keeps learning,
a hard pool is re-attacked, and teachers are harvested from the very solutions being counted. This
script removes all of that:

  * the IKT checkpoint is loaded and FROZEN (eval mode, no optimiser, no replay)
  * every molecule is FRESH from the frozen paper ADT -- none of them has ever been seen
  * nothing is harvested; nothing is added to any buffer
  * the rescue rate is reported AS A FUNCTION OF THE xTB BUDGET (1, 2, 4, ... rotamers verified),
    because "XTP" is an existence question and the number of attempts is part of the claim
  * sizes are reported in fine bands (35-39 / 40-44 / 45-49 / 50+), because the whole point of a
    corrector is that it should NOT degrade with size

A molecule xTB already accepts is never touched, so "broken" is impossible by construction; the only
thing that can happen is a rescue.
"""
import os, sys, json, time, argparse, copy
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/ADT/common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AROMATIZE_RINGS", "1")

from adt_model import build_model
from adt_dataset import FrameSampler
from rollout_batched import rollout_batch_kv
from reward_xtb import xvr_reward_batch
from ikt_model import IKTModel
from torsion_ikt import TorsionIKT, forward_kinematics, selfmis
from train_torsion_online import tree_from_atoms, geom_score


def band(na):
    # extended past the old 50+ bucket: the whole point of this run is the region the generator has
    # never been allowed to enter (its size cap was 56), and where the corrector has no teachers.
    if na <= 34: return "<=34"
    if na <= 39: return "35-39"
    if na <= 44: return "40-44"
    if na <= 49: return "45-49"
    if na <= 55: return "50-55"
    if na <= 65: return "56-65 *"
    if na <= 75: return "66-75 *"
    return "76+   *"


BANDS = ("<=34", "35-39", "40-44", "45-49", "50+")


def wilson(k, n, z=1.96):
    if not n: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    import math
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0, c - h), 100 * min(1, c + h))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adt", required=True)
    p.add_argument("--ikt", required=True, help="the FROZEN corrector")
    p.add_argument("--frame_cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_mol", type=int, default=600)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--end_bias", type=float, default=2.4)
    p.add_argument("--min_na", type=int, default=35)
    p.add_argument("--n_cand", type=int, default=48)
    p.add_argument("--xtb_cand", type=int, default=6)
    p.add_argument("--cand_temp", type=float, default=1.2)
    p.add_argument("--xtb_workers", type=int, default=15)
    p.add_argument("--xtb_bin", default=os.path.expanduser("~/xtb/bin/xtb"))
    p.add_argument("--no_ikt", type=int, default=0,
                   help="1 = NULL CONTROL: use an UNTRAINED head (uniform over the whole circle) with the "
                        "identical candidate machinery. This is what separates 'the IKT learned something' "
                        "from 'rotamer search works'.")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = "cuda"
    wd = os.path.join(os.path.dirname(args.out) or ".", "xtb_work_eval")
    os.makedirs(wd, exist_ok=True)
    os.environ["BANK_STRUCT"] = "1"

    ck = torch.load(args.adt, weights_only=False, map_location=device)
    cfg = ck["config"]
    policy = build_model(cfg).to(device); policy.load_state_dict(ck["model"]); policy.eval()
    for q in policy.parameters():
        q.requires_grad_(False)
    gen = ck["gen"]
    nmax = int(args.size_ceiling or gen["size_nmax"])
    msteps = int(args.max_steps or gen["max_steps_per_mol"])
    if args.size_ceiling or args.max_steps:
        print(f"[eval] OVERRIDING the ckpt caps: size_ceiling={nmax} max_steps={msteps} "
              f"(ckpt says {gen}) -- the architecture has no length bound, only these do.", flush=True)
    fs = FrameSampler.load(args.frame_cache)

    ic = torch.load(args.ikt, weights_only=False, map_location=device)
    ia = ic["args"]
    ikt = TorsionIKT(cfg, n_bins=ia["n_bins"], max_deg=ia["max_deg"],
                     bend_bins=ia.get("bend_bins", 0), bend_deg=ia.get("bend_deg", 10.0)).to(device)
    if args.no_ikt:
        ikt.warm_start_from_adt(ck["model"])      # head stays zero-init = uniform = the null control
        tag = "NULL CONTROL (untrained head: uniform over +-180 deg)"
    else:
        ikt.load_state_dict(ic["ikt"])
        tag = f"FROZEN IKT  updates={ic.get('updates')}  {args.ikt}"
    ikt.eval()
    for q in ikt.parameters():
        q.requires_grad_(False)

    print(f"[eval] {tag}", flush=True)
    print(f"[eval] ADT frozen={args.adt}  size_nmax={nmax}  end_bias={args.end_bias}  min_na={args.min_na}",
          flush=True)
    print(f"[eval] {args.n_cand} rotamer proposals -> xTB-free screen -> top {args.xtb_cand} verified with xTB",
          flush=True)
    print(f"[eval] NO training, NO harvesting, every molecule FRESH.\n", flush=True)

    # rescue[b][k] = rescued with a budget of (k+1) xTB attempts
    stat = {b: {"n": 0, "raw": 0, "resc": [0] * args.xtb_cand} for b in BANDS}
    # PER-MOLECULE record. The paper will ask: the corrector bought you XTP -- did it buy you JUNK?
    # So keep, for every molecule: its size, whether ADT alone passed, the cheapest xTB budget at which
    # the corrector rescued it, the strain/heavy of the accepted structure, and its SMILES (for diversity).
    recs = []
    seen = 0
    t0 = time.time()
    while seen < args.n_mol:
        mols, tok, idx, metas = [], [], [], []
        with torch.no_grad():
            for tokens, nf, atoms, bonds, na, done in rollout_batch_kv(
                    policy, fs, device, args.batch, max_steps=msteps, temperature=1.0,
                    end_bias=args.end_bias, end_bias_arr=None, size_ceiling=nmax):
                if na < args.min_na or na >= nmax:
                    continue                       # ceiling => truncated mid-molecule, not a molecule
                ix = IKTModel.atom_token_index(tokens)
                if len(ix) < na:
                    continue
                parent, child_of, b0 = tree_from_atoms(atoms, bonds, na)
                mols.append((atoms, bonds, na)); tok.append(tokens); idx.append(ix[:na])
                metas.append({"na": na, "pos": np.array([atoms[k].pos for k in range(na)], float),
                              "anum": np.array([atoms[k].atomic_num for k in range(na)], int),
                              "parent": parent, "child_of": child_of, "b0": b0,
                              "order": list(range(na))})
        if not mols:
            continue

        rin = xvr_reward_batch([(a, b, n) for a, b, n in mols], args.xtb_bin, wd,
                               max_workers=args.xtb_workers)
        raw = [bool(r.get("same_topo")) for r in rin]
        for i, m in enumerate(metas):
            stat[band(m["na"])]["n"] += 1
            stat[band(m["na"])]["raw"] += int(raw[i])
            recs.append({"na": int(m["na"]), "raw": bool(raw[i]),
                         "rescued_at": None,
                         "strain_raw": (rin[i].get("strain_pa") if raw[i] else None),
                         "strain_ikt": None,
                         "rmsd_raw": (rin[i].get("rmsd_heavy") if raw[i] else None),
                         "rmsd_ikt": None,
                         "smi": rin[i].get("smi")})
        base = len(recs) - len(mols)         # index of this batch's first record
        seen += len(mols)

        fails = [i for i in range(len(metas)) if not raw[i]]
        if fails:
            with torch.no_grad():
                lg, bl, _ = ikt([tok[i] for i in fails], [idx[i] for i in fails], device, want_bend=True)
                lg = lg.cpu(); bl = bl.cpu() if bl is not None else None
            jobs = []
            for j, i in enumerate(fails):
                m = metas[i]; na = m["na"]
                pos = torch.tensor(m["pos"], dtype=torch.float32)
                cands = []
                with torch.no_grad():
                    for c in range(args.n_cand):
                        mode = "argmax" if c == 0 else "sample"
                        dth = ikt.predict(lg[j:j+1], mode=mode, temperature=args.cand_temp)[0]
                        dbn = (ikt.predict_bend(bl[j:j+1], mode=mode, temperature=args.cand_temp)[0]
                               if bl is not None else None)
                        new = forward_kinematics(pos, m["parent"], m["order"], dth[:na],
                                                 dbend=(dbn[:na] if dbn is not None else None)).numpy()
                        cands.append((geom_score(m["anum"], new, m["b0"], na), new))
                cands.sort(key=lambda t: t[0])
                for c, (_, new) in enumerate(cands[:args.xtb_cand]):
                    at = copy.deepcopy(mols[i][0])
                    for k in range(na):
                        at[k].pos = new[k].astype(np.float64)
                    jobs.append((i, c, at))
            rin2 = xvr_reward_batch([(at, mols[i][1], mols[i][2]) for i, _, at in jobs],
                                    args.xtb_bin, wd, max_workers=args.xtb_workers)
            best = {}
            for (i, c, _), r2 in zip(jobs, rin2):
                if bool(r2.get("same_topo")) and (i not in best or c < best[i][0]):
                    best[i] = (c, r2)              # the CHEAPEST budget at which this molecule is rescued
            for i, (c, r2) in best.items():
                b = band(metas[i]["na"])
                for k in range(c, args.xtb_cand):  # solved with c+1 attempts => also solved with more
                    stat[b]["resc"][k] += 1
                rc = recs[base + i]
                rc["rescued_at"] = int(c) + 1
                rc["strain_ikt"] = r2.get("strain_pa")
                rc["rmsd_ikt"] = r2.get("rmsd_heavy")
                if r2.get("smi"):
                    rc["smi"] = r2.get("smi")

        el = time.time() - t0
        print(f"  {seen}/{args.n_mol} molecules  {el/60:.0f} min", flush=True)
        json.dump({"stat": stat, "args": vars(args), "seen": seen, "recs": recs},
                  open(args.out, "w"))

    print(f"\n=== {tag} ===")
    print(f"{'band':>7} {'n':>5} {'XTP_raw':>8} " +
          "  ".join(f"{'+'+str(k+1)+'xtb':>12}" for k in range(args.xtb_cand)))
    for b in BANDS:
        s = stat[b]
        if not s["n"]:
            continue
        nf = s["n"] - s["raw"]
        cells = []
        for k in range(args.xtb_cand):
            eff = (s["raw"] + s["resc"][k]) / s["n"]
            cells.append(f"{100*eff:11.1f}%")
        print(f"{b:>7} {s['n']:5d} {100*s['raw']/s['n']:7.1f}% " + "  ".join(cells))
    print()
    print(f"{'band':>7} {'failures':>9} " + "  ".join(f"{'rescue@'+str(k+1):>18}" for k in range(args.xtb_cand)))
    for b in BANDS:
        s = stat[b]
        nf = s["n"] - s["raw"]
        if not nf:
            continue
        cells = []
        for k in range(args.xtb_cand):
            lo, hi = wilson(s["resc"][k], nf)
            cells.append(f"{100*s['resc'][k]/nf:5.0f}% [{lo:3.0f},{hi:3.0f}]")
        print(f"{b:>7} {nf:9d} " + "  ".join(f"{c:>18}" for c in cells))
    print(f"\nXTP_eff = XTP_raw UNION the rescues. A molecule xTB already accepted is never touched,")
    print(f"so nothing can be broken: the rescue is a lower bound on what the corrector is worth.")


if __name__ == "__main__":
    main()
