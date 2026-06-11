"""Offline xTB pass over existing browse_data.jsonl entries that are missing xTB results.

Reads entries with xtb_ok=False (or missing), reconstructs SDF from coords_pre+anums+bonds,
runs xtb_relax + consistency_check, writes augmented entries to output JSONL.

Usage:
  python xtb_post_process.py <input.jsonl> <output.jsonl> [--workers 4] [--omp 4]
"""
import argparse, os, sys, json, time
import multiprocessing as mp
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

_COMMON = os.path.expanduser("~/ADT/common")
sys.path.insert(0, _COMMON)


def build_sdf(entry):
    """Re-run validate_3D on the stored coords+anums to get the canonical mol
    (same path as the live producer), then MolToMolBlock. This avoids the
    aromaticity/bond-order re-perception mismatch that manual RWMol
    construction would produce."""
    from util_validation import validate_3D
    anums = entry["anums"]
    coords = entry["coords_pre"]
    mol, smi, info = validate_3D(anums, coords)
    if mol is None or not info.get("valid"):
        return None
    try:
        return Chem.MolToMolBlock(mol)
    except Exception:
        try: return Chem.MolToMolBlock(mol, kekulize=False)
        except Exception: return None


def worker(task_q, result_q, xtb_workdir, xtb_bin, omp_threads):
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(omp_threads)
    os.environ["AROMATIZE_RINGS"] = os.environ.get("AROMATIZE_RINGS", "1")
    sys.path.insert(0, _COMMON)
    from gen_eval_lib import xtb_relax, consistency_check
    from util_validation import validate_3D

    while True:
        item = task_q.get()
        if item is None:
            break
        entry, idx_str = item
        sdf = build_sdf(entry)
        if sdf is None:
            entry.update({"xtb_ok": False, "xtb_reason": "sdf_build_failed"})
            result_q.put(entry)
            continue
        mh = Chem.MolFromMolBlock(sdf, sanitize=False)
        if mh is None:
            entry.update({"xtb_ok": False, "xtb_reason": "sdf_parse_failed"})
            result_q.put(entry)
            continue
        try: Chem.SanitizeMol(mh)
        except Exception: pass
        mh = Chem.AddHs(mh, addCoords=True)
        conf = mh.GetConformer()
        init_heavy = []
        for a in mh.GetAtoms():
            if a.GetAtomicNum() != 1:
                p = conf.GetAtomPosition(a.GetIdx())
                init_heavy.append([p.x, p.y, p.z])
        charge = sum(a.GetFormalCharge() for a in mh.GetAtoms())
        res = xtb_relax(idx_str, sdf, init_heavy, xtb_workdir, xtb_bin, charge=charge)
        if res.get("ok"):
            cc = consistency_check(entry.get("inchi_pre", ""), entry.get("topo_pre", ""),
                                   res.get("opt_heavy_anums"), res.get("opt_heavy_coords"),
                                   validate_3D)
            entry.update({
                "xtb_ok": True,
                "coords_post": res.get("opt_heavy_coords"),
                "rmsd_heavy": res.get("rmsd_heavy"),
                "e_gain": res["e_gain"],
                "inchi_post": cc["inchi_post"], "topo_post": cc["topo_post"],
                "same_topo": cc["same_topo"], "same_inchi": cc["same_inchi"],
                "same": cc["same"],
            })
            entry.pop("xtb_reason", None)
        else:
            entry.update({"xtb_ok": False, "xtb_reason": res.get("reason", "")})
        # cleanup
        for pat in [f"mol_{idx_str}.xyz", f"mol_{idx_str}.out",
                    f"mol_{idx_str}.xtbtopo.mol", f"mol_{idx_str}.xtbopt.xyz",
                    f"mol_{idx_str}.xtbopt.log", f"mol_{idx_str}.charges",
                    f"mol_{idx_str}.wbo", f"mol_{idx_str}.xtbrestart"]:
            p = os.path.join(xtb_workdir, pat)
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
        for fn in os.listdir(xtb_workdir):
            if fn.startswith(f".mol_{idx_str}"):
                try: os.remove(os.path.join(xtb_workdir, fn))
                except Exception: pass
        result_q.put(entry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--omp", type=int, default=4)
    ap.add_argument("--xtb_bin", default=os.environ.get("XTB_BIN", os.path.expanduser("~/miniconda3/bin/xtb")))
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    workdir = args.workdir or os.path.join(os.path.dirname(args.output), "xtb_work_post")
    os.makedirs(workdir, exist_ok=True)
    print(f"Reading {args.input}...", flush=True)
    entries = []
    skip = 0
    with open(args.input) as f:
        for i, line in enumerate(f):
            e = json.loads(line)
            if e.get("xtb_ok"):
                skip += 1
                continue
            entries.append((e, f"post_{i}"))
    print(f"  total lines: {i+1}, needing xTB: {len(entries)} (skipping {skip} already-done)", flush=True)

    mp.set_start_method("spawn", force=True)
    task_q = mp.Queue(maxsize=args.workers * 4)
    result_q = mp.Queue()
    workers = []
    for w in range(args.workers):
        p = mp.Process(target=worker, args=(task_q, result_q, workdir, args.xtb_bin, args.omp))
        p.start(); workers.append(p)

    # Feeder + collector
    out_f = open(args.output, "w")
    n_done = 0; n_ok = 0; n_topo = 0; n_inchi = 0
    t0 = time.time()

    # Prime the queue
    feed_iter = iter(entries)
    for _ in range(min(args.workers * 4, len(entries))):
        try: task_q.put(next(feed_iter))
        except StopIteration: break

    for _ in range(len(entries)):
        r = result_q.get()
        out_f.write(json.dumps(r, default=float) + "\n"); out_f.flush()
        n_done += 1
        if r.get("xtb_ok"):
            n_ok += 1
            if r.get("same_topo"): n_topo += 1
            if r.get("same_inchi"): n_inchi += 1
        # Feed next task
        try: task_q.put(next(feed_iter))
        except StopIteration: pass
        if n_done % 500 == 0:
            el = (time.time() - t0) / 60
            print(f"  [{n_done}/{len(entries)}] {el:.1f}min ok={n_ok} topo={n_topo} inchi={n_inchi}", flush=True)

    # Poison
    for _ in range(args.workers):
        task_q.put(None)
    for p in workers:
        p.join(timeout=30)
        if p.is_alive(): p.terminate()
    out_f.close()

    el = (time.time() - t0) / 60
    print(f"\n=== Done in {el:.1f}min ===", flush=True)
    print(f"  xTB ok      : {n_ok}/{n_done} ({n_ok/max(n_done,1)*100:.1f}%)", flush=True)
    print(f"  topo preserved : {n_topo}/{n_ok} ({n_topo/max(n_ok,1)*100:.1f}%)", flush=True)
    print(f"  inchi preserved: {n_inchi}/{n_ok} ({n_inchi/max(n_ok,1)*100:.1f}%)", flush=True)
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
