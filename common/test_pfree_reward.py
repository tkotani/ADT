"""Tier test for the perception-free reward (reward_pfree) on E240-generated molecules.
Verifies: completer loads, reward tiers (0.0/0.3 clashVR/0.6/estrain-shaped topo), clash_pass
rate, XVR, strain_pa. Run on kt1 with XVR_ESTRAIN_TAU/CLASHVR/COMPLETER_CKPT set."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("AROMATIZE_RINGS", "1")
sys.path.insert(0, os.path.expanduser("~/ADT/common"))
sys.path.insert(0, os.path.expanduser("~/ADT/Drugs/vtakao202606231610"))
import torch
import numpy as np
import train as T
from adt_dataset import FrameSampler
import reward_pfree
from collections import Counter

dev = "cuda" if torch.cuda.is_available() else "cpu"
E240 = os.environ.get("GEN_CKPT", os.path.expanduser("~/ADT/Drugs/vtakao202606231610/ckpts/scratch_epoch240.pt"))
print("gen ckpt = %s" % E240)
ck = torch.load(E240, map_location="cpu", weights_only=False)
model = T.build_model(ck.get("config", ck.get("cfg")))
model.load_state_dict(ck["model"]); model.to(dev).eval()
fs = FrameSampler.load(os.path.expanduser("~/ADT/Drugs/data/geom/scaffolds/frame_cache_bootstrap3.pt"))
XTB = os.path.expanduser("~/xtb/bin/xtb")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30

mols = []
for _ in range(N):
    try:
        at, bonds, na = T.generate_one(model, dev, frame_sampler=fs, temperature=1.0)
        mols.append((at, bonds, na))
    except Exception:
        continue
res = reward_pfree.pfree_reward_batch(mols, XTB, "/tmp/pfree_test_work", max_workers=16, collect_relax=True)

tiers = Counter()
for r in res:
    rw = r["reward"]
    if rw == 0.0:
        tiers["0.0 fail"] += 1
    elif abs(rw - 0.3) < 1e-6:
        tiers["0.3 clashVR"] += 1
    elif abs(rw - 0.6) < 1e-6:
        tiers["0.6 xtb-ok(topo変)"] += 1
    else:
        tiers["topo(0.6-1.0 estrain)"] += 1
print("N=%d  reward tiers: %s" % (len(res), dict(tiers)))
print("clash_pass rate = %.2f  XVR(topo) = %.2f"
      % (sum(1 for r in res if r.get("clash_pass")) / max(len(res), 1),
         sum(1 for r in res if r.get("same_topo")) / max(len(res), 1)))
from collections import defaultdict
by_placer = defaultdict(list)
for r in res:
    if r.get("strain_pa") is not None:
        by_placer[r.get("placer")].append(r["strain_pa"])
print("H_PLACER=%s  placer別 strain_pa median:" % os.environ.get("H_PLACER", "rdkit"))
for pl, v in by_placer.items():
    print("   %-6s n=%d  strain median=%.2f  (topo tier = 0.6+0.4*exp(-strain/3.5))" % (pl, len(v), np.median(v)))
sp = [r["strain_pa"] for r in res if r.get("strain_pa") is not None]
if sp:
    print("全体 strain_pa median=%.3f" % np.median(sp))
print("examples:")
for r in res[:6]:
    print("  reward=%.3f clash_pass=%s xtb_ok=%s topo=%s strain_pa=%s"
          % (r["reward"], r.get("clash_pass"), r.get("xtb_ok"), r.get("same_topo"),
             ("%.3f" % r["strain_pa"]) if r.get("strain_pa") is not None else None))
