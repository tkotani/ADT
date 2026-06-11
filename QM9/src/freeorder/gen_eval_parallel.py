"""QM9 baseline parallel pipeline:
  2 GPU producers → mp.Queue → N xTB workers → result aggregator.
  Cumulative summary every SUMMARY_EVERY (default 10000) generated mols.

Env vars:
  GEN_N_TOTAL=100000        target total generations (both GPUs combined)
  N_XTB_WORKERS=6           xTB consumer workers
  OMP_NUM_THREADS_XTB=4     threads per xTB worker
  SUMMARY_EVERY=10000       summary cadence
  CKPT=...                  default checkpoints_fo_wide_200bin_repro/best.pt
  FRAME_CACHE=...           default ../frame_cache_200bin.pt
  QM9_CACHE=...             default ../qm9_mols_cache_v3b_noh.pkl
  OUTDIR=eval_parallel_NXXX
"""
import os, sys, time, json, queue, signal
import multiprocessing as mp
from collections import Counter

GEN_N_TOTAL = int(os.environ.get("GEN_N_TOTAL", "100000"))
N_XTB_WORKERS = int(os.environ.get("N_XTB_WORKERS", "6"))
OMP_XTB = os.environ.get("OMP_NUM_THREADS_XTB", "4")
SUMMARY_EVERY = int(os.environ.get("SUMMARY_EVERY", "10000"))
CKPT = os.environ.get("CKPT", "checkpoints_fo_wide_200bin_repro/best.pt")
FRAME_CACHE = os.environ.get("FRAME_CACHE", "../frame_cache_200bin.pt")
QM9_CACHE = os.environ.get("QM9_CACHE", "../qm9_mols_cache_v3b_noh.pkl")
OUTDIR = os.environ.get("OUTDIR", f"eval_parallel_N{GEN_N_TOTAL}")
XTB_BIN = os.environ.get("XTB_BIN", os.path.expanduser("~/miniconda3/bin/xtb"))

_HERE = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "common"))


def producer(gpu_id, mol_q, gen_counter, stable_counter, lock, stop_evt, n_target_per_gpu):
    """Run on GPU=gpu_id. Generates, validates, pushes mol_stable to queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["ADT_R_BINS"] = os.environ.get("ADT_R_BINS", "200")
    os.environ["AROMATIZE_RINGS"] = os.environ.get("AROMATIZE_RINGS", "0")
    sys.path.insert(0, _HERE)
    sys.path.insert(0, os.path.join(_HERE, ".."))
    sys.path.insert(0, _COMMON)

    import torch, numpy as np
    from itertools import combinations
    from fo_train import generate_one, COLLISION_R
    from adt_model import build_model
    from adt_dataset import FrameSampler
    from util_validation import validate_3D
    from gen_eval_lib import try_write_sdf
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdmolops, rdMolDescriptors, Descriptors
    RDLogger.logger().setLevel(RDLogger.ERROR)
    ALLOWED = {6, 7, 8, 9}

    device = torch.device("cuda")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    max_pointer_ckpt = ckpt["config"].get("max_pointer", 30)
    print(f"ckpt max_pointer = {max_pointer_ckpt}", flush=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    fs = FrameSampler.load(FRAME_CACHE)
    print(f"[GPU{gpu_id}] loaded epoch={ckpt['epoch']} val={ckpt['val_loss']:.4f}", flush=True)

    local_stats = Counter()
    n_local = 0
    while not stop_evt.is_set():
        if n_local >= n_target_per_gpu:
            break
        n_local += 1
        try:
            at, bonds, na = generate_one(model, device, frame_sampler=fs, temperature=1.0)
        except Exception:
            local_stats["gen_fail"] += 1
            with lock: gen_counter.value += 1
            continue
        if na < 2:
            local_stats["too_small"] += 1
            with lock: gen_counter.value += 1
            continue
        anums = [at[k].atomic_num for k in range(na)]
        coords = [list(at[k].pos) for k in range(na)]
        if any(a not in ALLOWED for a in anums):
            local_stats["bad_atom"] += 1
            with lock: gen_counter.value += 1
            continue
        has_coll = any(np.linalg.norm(np.array(coords[a])-np.array(coords[b])) < COLLISION_R
                       for a, b in combinations(range(na), 2))
        if has_coll:
            local_stats["collision"] += 1
            with lock: gen_counter.value += 1
            continue
        mol, smi, info = validate_3D(anums, coords)
        if not info.get("valid"):
            local_stats["invalid_3d"] += 1
            with lock: gen_counter.value += 1
            continue
        local_stats["valid_3d"] += 1
        if len(rdmolops.GetMolFrags(mol)) != 1:
            local_stats["disconnected"] += 1
            with lock: gen_counter.value += 1
            continue
        if not info.get("mol_stable"):
            local_stats["unstable"] += 1
            with lock: gen_counter.value += 1
            continue
        local_stats["mol_stable"] += 1

        try:
            inchi_pre = Chem.MolToInchiKey(mol).split('-')[0]
            topo_pre = Chem.MolToSmiles(mol, isomericSmiles=False)
        except Exception:
            inchi_pre = topo_pre = ""

        try:
            sdf_kek = Chem.MolToMolBlock(mol)
            sdf_tag = "kekulized"
        except Exception:
            try:
                sdf_kek = Chem.MolToMolBlock(mol, kekulize=False)
                sdf_tag = "aromatic"
            except Exception:
                local_stats["sdf_failed"] += 1
                with lock: gen_counter.value += 1
                continue

        # init heavy coords + charge from sdf
        try:
            mol_h = Chem.MolFromMolBlock(sdf_kek, sanitize=False)
            try: Chem.SanitizeMol(mol_h)
            except Exception: pass
            mol_h = Chem.AddHs(mol_h, addCoords=True)
            init_heavy = []
            conf = mol_h.GetConformer()
            for a in mol_h.GetAtoms():
                if a.GetAtomicNum() != 1:
                    p = conf.GetAtomPosition(a.GetIdx())
                    init_heavy.append([p.x, p.y, p.z])
            charge = sum(a.GetFormalCharge() for a in mol_h.GetAtoms())
        except Exception:
            local_stats["init_fail"] += 1
            with lock: gen_counter.value += 1
            continue

        with lock:
            gen_counter.value += 1
            stable_counter.value += 1

        item = {"gpu": gpu_id, "smi": smi, "n_atoms": na, "sdf_tag": sdf_tag,
                "sdf": sdf_kek, "init_heavy": init_heavy, "charge": charge,
                "inchi": inchi_pre, "topo": topo_pre}
        mol_q.put(item)

    # Send local stats sentinel
    mol_q.put({"__producer_done__": gpu_id, "stats": dict(local_stats)})
    print(f"[GPU{gpu_id}] producer done, generated {n_local}", flush=True)


def consumer(worker_id, mol_q, result_q, xtb_workdir, omp_threads):
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(omp_threads)
    sys.path.insert(0, _COMMON)
    from gen_eval_lib import xtb_relax, consistency_check
    from util_validation import validate_3D

    while True:
        item = mol_q.get()
        if item is None:
            break
        if item.get("__producer_done__") is not None:
            # forward to main
            result_q.put(item)
            continue
        idx = f"w{worker_id}_{int(time.time()*1000)%10**8}"
        res = xtb_relax(idx, item["sdf"], item["init_heavy"], xtb_workdir, XTB_BIN,
                        charge=item["charge"])
        if res.get("ok"):
            cc = consistency_check(item["inchi"], item["topo"],
                                   res.get("opt_heavy_anums"), res.get("opt_heavy_coords"),
                                   validate_3D)
            rec = {"smi": item["smi"], "gpu": item["gpu"], "n_atoms": item["n_atoms"],
                   "e_gain": res["e_gain"], "rmsd": res["rmsd"],
                   "rmsd_heavy": res.get("rmsd_heavy"),
                   "topo_post": cc["topo_post"], "inchi_post": cc["inchi_post"],
                   "same_topo": cc["same_topo"], "same_inchi": cc["same_inchi"],
                   "same": cc["same"], "ok": True}
        else:
            rec = {"smi": item["smi"], "gpu": item["gpu"], "n_atoms": item["n_atoms"],
                   "ok": False, "reason": res.get("reason", "")}
        # cleanup work files
        for pat in [f"{idx}.xyz", f"mol_{idx}.xyz", f"mol_{idx}.out",
                    f"mol_{idx}.xtbtopo.mol", f"mol_{idx}.xtbopt.xyz",
                    f"mol_{idx}.xtbopt.log", f"mol_{idx}.charges",
                    f"mol_{idx}.wbo", f"mol_{idx}.xtbrestart"]:
            p = os.path.join(xtb_workdir, pat)
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
        for fn in os.listdir(xtb_workdir):
            if fn.startswith(f".mol_{idx}") or fn.startswith(idx):
                try: os.remove(os.path.join(xtb_workdir, fn))
                except Exception: pass
        result_q.put(rec)


def emit_summary(all_results, gen_total, stable_total, t0, qm9_smiles, label):
    """Print cumulative summary."""
    import numpy as np
    ok = [r for r in all_results if r.get("ok")]
    n_ok = len(ok); n_xtb = len([r for r in all_results if not r.get("__producer_done__")])
    n_topo = sum(1 for r in ok if r.get("same_topo"))
    n_inchi = sum(1 for r in ok if r.get("same_inchi"))
    n_either = sum(1 for r in ok if r.get("same"))
    smiles_set = set(r["smi"] for r in all_results if r.get("smi"))
    novel = 0
    if qm9_smiles:
        novel = sum(1 for s in smiles_set if s not in qm9_smiles)
    elapsed = (time.time() - t0) / 60

    rmsd_h = [r["rmsd_heavy"] for r in ok if r.get("rmsd_heavy") is not None]
    e_gain = [r["e_gain"] for r in ok]

    print(f"\n========== {label} ==========")
    print(f"  elapsed             : {elapsed:.1f} min")
    print(f"  generated total     : {gen_total}")
    print(f"  mol_stable          : {stable_total} ({stable_total/max(gen_total,1)*100:.1f}%)")
    print(f"  unique SMILES       : {len(smiles_set)}")
    if qm9_smiles:
        print(f"  novel (not in QM9)  : {novel}/{len(smiles_set)} ({novel/max(len(smiles_set),1)*100:.1f}%)")
    print(f"  xTB processed       : {n_xtb}")
    print(f"  xTB converged       : {n_ok} ({n_ok/max(n_xtb,1)*100:.1f}%)")
    if e_gain:
        print(f"  E gain (kcal/mol)   : mean={np.mean(e_gain):.1f} median={np.median(e_gain):.1f}")
    if rmsd_h:
        print(f"  RMSD heavy (Å)      : mean={np.mean(rmsd_h):.3f} median={np.median(rmsd_h):.3f}")
        for thr in [0.1, 0.25, 0.5, 1.0]:
            nn = sum(1 for r in rmsd_h if r < thr)
            print(f"    <{thr:.2f}Å           : {nn} ({nn/len(rmsd_h)*100:.1f}%)")
    print(f"  topology preserved  : {n_topo}/{n_ok} ({n_topo/max(n_ok,1)*100:.1f}%)")
    print(f"  InChIKey preserved  : {n_inchi}/{n_ok} ({n_inchi/max(n_ok,1)*100:.1f}%)")
    print(f"  either preserved    : {n_either}/{n_ok} ({n_either/max(n_ok,1)*100:.1f}%)")
    sys.stdout.flush()


def main():
    mp.set_start_method("spawn", force=True)
    os.makedirs(OUTDIR, exist_ok=True)
    xtb_workdir = os.path.join(OUTDIR, "xtb_work")
    os.makedirs(xtb_workdir, exist_ok=True)

    # Load QM9 reference set in main (for novelty)
    qm9_smiles = set()
    if os.path.exists(QM9_CACHE):
        sys.path.insert(0, _COMMON)
        from rdkit import Chem, RDLogger
        RDLogger.logger().setLevel(RDLogger.ERROR)
        import pickle
        with open(QM9_CACHE, "rb") as f:
            qd = pickle.load(f)
        for mol, pos, smi in qd:
            try:
                m2 = Chem.MolFromSmiles(smi)
                if m2:
                    Chem.RemoveStereochemistry(m2)
                    qm9_smiles.add(Chem.MolToSmiles(m2))
            except Exception:
                pass
        print(f"QM9 reference loaded: {len(qm9_smiles)} unique SMILES", flush=True)

    mol_q = mp.Queue(maxsize=200)
    result_q = mp.Queue()
    gen_counter = mp.Value("i", 0)
    stable_counter = mp.Value("i", 0)
    lock = mp.Lock()
    stop_evt = mp.Event()

    n_per_gpu = (GEN_N_TOTAL + 1) // 2
    producers = []
    for gid in [0, 1]:
        p = mp.Process(target=producer,
                       args=(gid, mol_q, gen_counter, stable_counter, lock, stop_evt, n_per_gpu),
                       name=f"prod{gid}")
        p.start()
        producers.append(p)

    workers = []
    for w in range(N_XTB_WORKERS):
        c = mp.Process(target=consumer,
                       args=(w, mol_q, result_q, xtb_workdir, OMP_XTB),
                       name=f"xtb{w}")
        c.start()
        workers.append(c)

    print(f"Started {len(producers)} GPU producers and {len(workers)} xTB workers", flush=True)
    print(f"Target: {GEN_N_TOTAL} (per-gpu {n_per_gpu}), summary every {SUMMARY_EVERY}", flush=True)

    inc_path = os.path.join(OUTDIR, "results_incremental.jsonl")
    inc_f = open(inc_path, "a")

    t0 = time.time()
    all_results = []
    next_summary = SUMMARY_EVERY
    producer_done = 0

    def shutdown(signum, frame):
        print(f"\n[main] received signal {signum}, shutting down", flush=True)
        stop_evt.set()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while True:
        try:
            r = result_q.get(timeout=2.0)
        except queue.Empty:
            r = None
        if r is not None:
            if r.get("__producer_done__") is not None:
                producer_done += 1
                print(f"[main] producer GPU{r['__producer_done__']} done, "
                      f"stats={r.get('stats')}", flush=True)
                if producer_done >= len(producers):
                    # signal consumers to drain & exit after queue empties
                    pass
            else:
                all_results.append(r)
                inc_f.write(json.dumps(r) + "\n"); inc_f.flush()

        gt = gen_counter.value
        if gt >= next_summary:
            emit_summary(all_results, gt, stable_counter.value, t0, qm9_smiles,
                         label=f"SUMMARY @ generated={gt}")
            next_summary += SUMMARY_EVERY

        # termination: producers done AND queue drained AND all xTB processed
        if producer_done >= len(producers) and mol_q.empty() and result_q.empty():
            # wait briefly
            time.sleep(2)
            if mol_q.empty() and result_q.empty():
                break
        if stop_evt.is_set():
            break

    # send poison to consumers
    for _ in range(N_XTB_WORKERS):
        mol_q.put(None)
    for c in workers:
        c.join(timeout=30)
        if c.is_alive():
            c.terminate()
    for p in producers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    inc_f.close()
    emit_summary(all_results, gen_counter.value, stable_counter.value, t0, qm9_smiles,
                 label=f"FINAL @ generated={gen_counter.value}")

    with open(os.path.join(OUTDIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nDone. Output dir: {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
