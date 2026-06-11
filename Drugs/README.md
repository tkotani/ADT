# GEOM-Drugs — scaffold-conditional generation (paper Table 4/5, Fig. 5/6, App. B–E)

Reproduces the scaffold-conditional results for the seven seed scaffolds
(benzene, pyridine, pyrimidine, pyrazine, furan, thiophene, cyclohexane) at two
size limits (30- and 50-atom-truncated GEOM-Drugs). A single trained model
handles all scaffolds; the scaffold is an inference-time conditioning choice.

All commands assume `ADT_ROOT` points at the repository root and the data has
been linked with `tools/setup_data.sh`.

## Checkpoints

| Checkpoint | Paper | epoch | md5 |
|---|---|---|---|
| `checkpoints_v26_scratch/E142.pt` | Table 4 (30-atom) | 142 (val 1.9797) | `451ffc3c…` |
| `checkpoints_v26_max50_ft_v2/best.pt` | Table 5 (50-atom) | 157 (ft from 30-atom) | `0a569723…` |

The 50-atom model is fine-tuned from the 30-atom checkpoint.

```bash
md5sum $ADT_ROOT/Drugs/data/freeorder_v26/checkpoints_v26_scratch/E142.pt   # 451ffc3c...
```

## Generation + evaluation

```bash
# 30-atom (Table 4): 7 scaffolds × N=10000
MAX=30 GEN_N=10000 Drugs/src/freeorder_v26/run_paper_drugs_eval.sh
# 50-atom (Table 5)
MAX=50 GEN_N=10000 Drugs/src/freeorder_v26/run_paper_drugs_eval.sh
# quick check: GEN_N=1000
```

Each scaffold is sampled and run through xTB; output goes to
`data/freeorder_v26/v26{s,g}_scaffolds_n.../<scaffold>/` (`gen_stats.json`,
`stats.json`, `mol_stable_kekulized.sdf`, `xtb_results.json`). The packaged
results (`drugs_scaffolds_n10k.tgz`) already contain the N=10000 outputs, so the
tables/figures can be recomputed without regenerating.

Recompute the paper tables and figures with the scripts in
[`../reproduce/`](../reproduce/README.md) — e.g.
`per_scaffold_stats.py` (Table 4/5, App. B), `per_scaffold_diversity.py`
(Table 8/9, App. C), `analyze_structfidelity.py` (Fig. 6).

## Training (regenerate the checkpoints)

```bash
MAX=30 Drugs/src/freeorder_v26/run_paper_drugs_train.sh   # 30-atom scratch, ~14 h on 2× RTX 4090 (DDP)
MAX=50 Drugs/src/freeorder_v26/run_paper_drugs_train.sh   # 50-atom fine-tune from 30-atom, ~3.5 h
```

The 30-atom run uses `--max_atoms 30 --max_offset 50`, batch 64, lr 2e-4. The
paper uses **E142** (the run’s `best.pt` is the later E156, which the paper does
not use); `--save_every 5` keeps nearby checkpoints so E142 can be selected. The
50-atom run fine-tunes (`--pretrain` the 30-atom checkpoint, lr 5e-5, ~5 epochs
to plateau) and the paper uses **E157**.

## Key configuration

| key | 30-atom (Table 4) | 50-atom (Table 5) |
|---|---|---|
| epoch | 142 | 157 (ft from 30-atom) |
| `max_offset` / `max_atoms` / `max_pointer` | 50 / 30 / 64 | 50 / 50 / 64 |
| pointer | offset (v26 hybrid) | offset |
| distance mesh | log, 200 bins | log, 200 bins |
| `AROMATIZE_RINGS` (eval) | 1 | 1 |
| md5 | `451ffc3c…` | `0a569723…` |

## Layout

```
Drugs/
├── src/freeorder_v26/    fo_train_v26.py, gen_eval_v26.py, build_virtual_seeds.py,
│                         run_paper_drugs_{train,eval}.sh
└── data/freeorder_v26/   E142.pt, checkpoints_v26_max50_ft_v2/best.pt (E157),
                          v26{s,g}_scaffolds_n10k/, scaffolds/, training caches, logs
```

Shared model/tokenizer/utilities are in [`../common/`](../common/); paper-figure
scripts are in [`../reproduce/`](../reproduce/README.md).
