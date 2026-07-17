"""funnel_stats.py — aggregate molrecord_v2 banks into the per-scaffold funnel + strain stats.

Reads records WITHOUT regeneration. Merges every bank in <bank_dir> into one pool; each record
self-describes provenance (gen_ckpt_hash) + scaffold, so banks from DIFFERENT machines can be dropped
into one dir and aggregated (no directory scheme needed).

Funnel (monotonic, each stage a subset of the previous):
    gen -> noclash -> MLnH -> mlhadd -> H-prerelax -> full -> XTP
  XTP = xTB Topology-Preserved (= 合格). Aggregate rate = XTPR.
  [+ rdkit_valid: SUPPLEMENTARY stricter check = RDKit perceives a neutral closed-shell molecule.
   合格=XTP; rdkit_valid is a conservative lower bound -- the small XTP-vs-rdkit_valid gap is mostly
   RDKit Kekule/valence perception limits on real molecules (perceivable as charged), not true
   invalidity. Set RDKIT_VALID=0 to skip it (faster).]

Usage:  python3 funnel_stats.py <bank_dir>
"""
import sys
import os
import glob
import numpy as np
import torch

bank_dir = sys.argv[1] if len(sys.argv) > 1 else "."
DO_RV = os.environ.get("RDKIT_VALID", "1") != "0"
if DO_RV:
    sys.path.insert(0, os.path.expanduser("~/ADT/common"))
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from util_validation import validate_3D

    def rdkit_valid(r):
        sr = r.get("struct_relaxed")
        if not isinstance(sr, dict):
            return False
        try:
            m, _, _ = validate_3D(list(sr["anums"]), np.asarray(sr["coords"], float), charge=0)
            if m is None:
                return False
            Chem.MolToSmiles(Chem.RemoveAllHs(m))
            return True
        except Exception:
            return False

banks = sorted(set(glob.glob(os.path.join(bank_dir, "*", "pfree_bank_*.pt")) +
                   glob.glob(os.path.join(bank_dir, "pfree_bank_*.pt"))))
pool, hashes = [], set()
for p in banks:
    b = torch.load(p, weights_only=False)
    pool.extend(b["mols"])
    hashes.add(b.get("gen_ckpt_hash", "?"))


def nested(R):
    nc = [r for r in R if r["connected"] and not r["clash"]]
    ml = [r for r in nc if r["mlnh_ok"]]
    mh = [r for r in ml if r["hplace_method"] == "mlhadd"]
    hp = [r for r in mh if r["hprerelax_ok"]]
    fu = [r for r in hp if r["full_ok"]]
    xv = [r for r in fu if r["xvr"]]                            # XTP = xTB Topology-Preserved (= 合格)
    rvn = sum(1 for r in xv if rdkit_valid(r)) if DO_RV else 0  # supplementary stricter (RDKit closed-shell)
    return len(R), len(nc), len(ml), len(mh), len(hp), len(fu), len(xv), rvn, xv


scafs = sorted(set(r["scaffold"] for r in pool))
print("molrecord funnel  (banks=%d  records=%d  gen_ckpt_hash=%s)"
      % (len(banks), len(pool), " | ".join(sorted(h[:20] for h in hashes))))
hdr = "%-16s %5s %5s %5s %6s %6s %5s %5s" % ("scaffold", "gen", "ncls", "MLnH", "mlhadd", "Hprerx", "full", "XTP")
if DO_RV:
    hdr += " %7s" % "rdkit_v"
hdr += " | %7s %8s" % ("strain", "dE")
print(hdr + "   (XTP=xTB Topology-Preserved=合格; rdkit_v=補助的な厳しめ下限)")
allx = []
allrv = 0
for s in scafs:
    R = [r for r in pool if r["scaffold"] == s]
    g, nc, ml, mh, hp, fu, xn, rvn, xv = nested(R)
    allx += xv
    allrv += rvn
    st = [r["strain_pa"] for r in xv if r["strain_pa"] is not None]
    dE = [r["strain_dE"] for r in xv if r["strain_dE"] is not None]
    row = "%-16s %5d %5d %5d %6d %6d %5d %5d" % (s, g, nc, ml, mh, hp, fu, xn)
    if DO_RV:
        row += " %7d" % rvn
    row += " | %7s %8s" % ("%.2f" % np.median(st) if st else "-", "%.1f" % np.median(dE) if dE else "-")
    print(row)
if allx:
    st = np.array([r["strain_pa"] for r in allx if r["strain_pa"] is not None])
    dE = np.array([r["strain_dE"] for r in allx if r["strain_dE"] is not None])
    sz = [r["n_heavy"] for r in allx]
    rvstr = ("  rdkit_valid=%d (%.2f%% of XTP)" % (allrv, 100 * allrv / max(len(allx), 1))) if DO_RV else ""
    print("=== total ===  records=%d  XTPR=%d (%.1f%%)%s"
          % (len(pool), len(allx), 100 * len(allx) / max(len(pool), 1), rvstr))
    print("               strain_pa median=%.3f mean=%.3f p90=%.3f  dE median=%.1f kcal/mol  size=%.1f±%.1f"
          % (np.median(st), st.mean(), np.percentile(st, 90), np.median(dE), np.mean(sz), np.std(sz)))
