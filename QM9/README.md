# QM9 — unconditional generation (paper Table 2/3, Fig. 7)

Reproduces the QM9 results: the comparison table, the detailed metrics, and the
training curve.

All commands assume `ADT_ROOT` points at the repository root and the data has
been linked with `tools/setup_data.sh`.

## Canonical evaluation output (N=10000, checkpoint E562)

Paper **Table 2/3** are exactly the output of this run, stored under
`data/freeorder/eval_v26_E562_N10000/` (PTBANK archive
`qm9_eval_E562_N10000.tgz`):

| Table 3 row | count | % | source field |
|---|---|---|---|
| mol\_stable / N | 8746/10000 | **87.5%** | `gen_stats.mol_stable` |
| unique / mol\_stable | 8508/8746 | 97.3% | `n_unique` |
| novel / unique | 3635/8508 | 42.7% | `run.log` |
| kekulizable | 8746/8746 | 100% | `n_kekulizable` |
| xTB converged | 8743/8746 | 100.0% | `xtb_ok` |
| topology preserved / N (= XVR) | 8566/10000 | **85.7%** | `topo_same` |
| RMSD heavy, median | 0.184 Å | | `xtb_results.json: rmsd_heavy` |
| Energy gain, median | −1.31 eV | | `xtb_results.json: e_gain` (kcal/mol ÷ 23.0605) |

Generation is stochastic (T = 1.0), so re-running shifts each rate by
≈ ±0.3 pt (±√(Np(1−p)) ≈ ±33 molecules); a run with mol\_stable ≈ 87 ± 1 %
and topology ≈ 86 ± 1 % reproduces the table. A fixed seed does **not** give a
bit-identical sample.

## Checkpoint

```bash
md5sum $ADT_ROOT/QM9/data/freeorder/checkpoints_v26_scratch/best.pt
# 17ec74f188b0dd4c216ed05dab9c85a2
python3 -c 'import torch; ck=torch.load("'"$ADT_ROOT"'/QM9/data/freeorder/checkpoints_v26_scratch/best.pt",map_location="cpu",weights_only=False); print(ck["epoch"], ck["val_loss"])'
# 562 2.7496   (config: max_offset=32, output_pointer_mode=offset, n_r_bins=200)
```

## Generation + evaluation

```bash
GEN_N=10000 QM9/src/freeorder/run_paper_qm9_eval.sh
```

Phase 1 samples N molecules and applies the `mol_stable` check; Phase 2 runs xTB
GFN2 relaxation and the topology-preservation (XVR) check. Output goes to
`data/freeorder/eval_v26_E562_N10000/` (`stats.json`, `gen_stats.json`,
`mol_stable_kekulized.sdf`, `xtb_results.json`, `run.log`). A quick N=1000 run
(`GEN_N=1000`, ~5 min) reproduces the rates within sampling noise.

Key environment in the runner: `ADT_R_BINS=200` (log mesh, matching training),
`AROMATIZE_RINGS=0` (QM9 has essentially no aromatic rings).

## Training (regenerate E562 from scratch)

```bash
QM9/src/freeorder/run_paper_qm9_train.sh                 # 600 epochs, ~16.5 h on one RTX 4090
EPOCHS=200 SAVE_DIR=/tmp/qm9_test QM9/src/freeorder/run_paper_qm9_train.sh
```

Model: `d_model=768, n_layers=12, n_heads=8, d_ff=3072, dropout=0.2`, bf16 AMP,
`--max_offset 32`, log-spaced 200-bin distance mesh. The 134k-molecule training
set is cached in `qm9_mols_cache_v3b_noh.pkl`. A re-trained checkpoint should
land near `epoch ≈ 562, val ≈ 2.75`; without an identical seed the sample is not
bit-identical, but val within ±0.005 and mol\_stable within ±2 pt is a match.

## Key configuration

| key | value |
|---|---|
| epoch | 562 (val 2.7496) |
| `max_offset` | 32 |
| pointer | offset (`max_pointer=30`) |
| distance mesh | log-spaced, 200 bins |
| `AROMATIZE_RINGS` (eval) | 0 |
| md5 (`best.pt`) | `17ec74f188b0dd4c216ed05dab9c85a2` |

## Layout

```
QM9/
├── src/freeorder/        fo_train_v26.py, gen_eval_v26.py, gen_eval_parallel.py,
│                         fo_tokenizer.py, run_paper_qm9_{train,eval}.sh
└── data/freeorder/       best.pt (E562), frame_cache_200bin.pt,
                          qm9_mols_cache_v3b_noh.pkl, eval_v26_E562_N10000/
```

Shared model/tokenizer/utilities are in [`../common/`](../common/).
