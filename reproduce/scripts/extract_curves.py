"""Extract (epoch, train_loss, val_loss) from training logs."""
import os as _os_adt
_HERE_adt = _os_adt.path.dirname(_os_adt.path.abspath(__file__))
ADT_ROOT = _os_adt.environ.get("ADT_ROOT", _os_adt.path.abspath(_os_adt.path.join(_HERE_adt, "..", "..")))
DRUGS_DATA = _os_adt.path.join(ADT_ROOT, "Drugs", "data", "freeorder_v26")
QM9_DATA = _os_adt.path.join(ADT_ROOT, "QM9", "data", "freeorder")

import re, sys

LOGS = [
    ("QM9", "" + os.path.join(QM9_DATA, "logs") + "/fo_train_20260407_124042.log"),
    ("Drugs 30-atom", "" + DRUGS_DATA + "/logs/fo_train_20260420_155621.log"),
    ("Drugs 50-atom FT", "" + DRUGS_DATA + "/nohup_v26_max50_ft_v2.out"),
]

PAT = re.compile(r"^E(\d+)\s.*train=([\d.]+)\s+val=([\d.]+)")

for label, path in LOGS:
    print(f"\n=== {label} : {path} ===")
    epochs = {}
    try:
        with open(path) as f:
            for line in f:
                m = PAT.match(line)
                if m:
                    e = int(m.group(1))
                    t = float(m.group(2))
                    v = float(m.group(3))
                    # take last entry per epoch (DDP duplicates)
                    epochs[e] = (t, v)
    except FileNotFoundError:
        print(f"  not found")
        continue
    es = sorted(epochs.keys())
    print(f"  total epochs: {len(es)}, range {min(es)}..{max(es)}")
    print("  TikZ train coords (subsample every 5 epochs):")
    coords_train = []
    coords_val = []
    for e in es:
        t, v = epochs[e]
        coords_train.append(f"({e},{t:.4f})")
        coords_val.append(f"({e},{v:.4f})")
    # subsample
    sub = coords_train[::5]
    sub_v = coords_val[::5]
    if sub[-1] != coords_train[-1]:
        sub.append(coords_train[-1])
        sub_v.append(coords_val[-1])
    print("  train:", " ".join(sub))
    print("  val:  ", " ".join(sub_v))
