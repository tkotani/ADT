# `reproduce/` — regenerate every paper Table and Figure

These scripts recompute the numbers and plots in the paper from the evaluation
outputs (per-scaffold generation results and xTB results). They are the
authoritative source for the claim that *every number reproduces from this
package*.

Every script resolves paths through `ADT_ROOT` (the repository root) and reads
the data linked by [`tools/setup_data.sh`](../tools/setup_data.sh). Run from
anywhere:

```bash
ADT_ROOT=/path/to/ADT python3 reproduce/scripts/<script>.py
```

## Script → paper element

| Script | Paper element |
|---|---|
| `per_scaffold_stats.py`     | Table 4/5 and App. B — per-scaffold kekulized / unique / heavy-atom stats |
| `per_scaffold_novelty.py`   | App. B — novelty columns (novel / novel %) |
| `per_scaffold_diversity.py` | App. C, Table 8/9 — Bemis–Murcko scaffold uniqueness, Tanimoto, MW, logP (deterministic, seed 42) |
| `success_examples.py`       | App. D, Table 10/11 — XVR-positive example molecules (ΔE in eV, heavy-atom RMSD) |
| `failure_breakdown.py`      | App. E, Table 12/13 — Type-A failure-mode decomposition |
| `failure_examples.py`       | App. E, Table 14 — representative Type-A1 failure SMILES |
| `bond_angle_errors.py`      | Fig. 6 — per-bond / per-angle error medians |
| `bond_angle_hist.py`        | Fig. 6 — histograms (TikZ coordinates) |
| `analyze_structfidelity.py` | Fig. 6 — medians + histograms over the full structural-fidelity set (QM9 + Drugs 30/50) |
| `regen_structfidelity.py`   | (re)generate the pre/post-xTB coordinates that Fig. 6 is computed from |
| `geom_drugs_v26_dist.py`    | Fig. 5 — generated heavy-atom distribution |
| `v21_nolimit_hist.py`       | Fig. 5 — reference GEOM-Drugs distribution |
| `extract_curves.py`         | Fig. 7 / App. F — training curves |
| `heavy_atom_stats.py`, `heavy_atom_stats_v26g.py` | pooled heavy-atom statistics (helpers) |

## `logs/`

Symlinks to the three training logs (QM9, Drugs 30-atom, Drugs 50-atom) used for
the training-curve figure. `tools/setup_data.sh` recreates them from the data
manifest.
