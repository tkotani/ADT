# ADT — Atomic Design Transformer

Reproduction package for the paper
**"Atomic Design Transformer: xTB-Validated 3D Molecule Generation from Scaffolds"**
(Takao Kotani).

ADT is a fully-discrete autoregressive transformer that builds a 3D molecule
**one atom at a time**. SE(3) invariance comes entirely from the tokenization
— each new atom is encoded in a local coordinate frame of a previously placed
atom — so the network itself is a plain causal transformer. Generation can be
conditioned on a fixed seed scaffold, and the model decides on its own when to
stop.

- **Interactive 3D browser** (atom-by-atom growth, one page per scaffold):
  <https://tkotani.github.io/ADT/>
- **Paper:** arXiv (DOI/ID: *to be added on posting*)
- **License:** MIT (see [`LICENSE`](LICENSE)); the method is patent-pending
  (see [`PATENTS.md`](PATENTS.md)).

---

## Repository map

Everything is reachable from here.

| Path | Contents |
|---|---|
| [`common/`](common/) | Shared library: model, tokenizer, dataset, utilities (used by both QM9 and Drugs). |
| [`QM9/`](QM9/README.md) | QM9 unconditional generation — paper **Table 2/3**, **Fig. 7**. |
| [`Drugs/`](Drugs/README.md) | GEOM-Drugs scaffold-conditional generation — **Table 4/5**, **Fig. 5/6**, **App. B–E**. |
| [`reproduce/`](reproduce/README.md) | Scripts that regenerate **every paper Table and Figure** from the evaluation outputs. |
| [`docs/`](docs/README.md) | Self-contained interactive 3D browser (the GitHub Pages site). |
| [`tools/`](tools/) | [`data_manifest.tsv`](tools/data_manifest.tsv) + [`setup_data.sh`](tools/setup_data.sh): fetch/link the large data. |
| [`LICENSE`](LICENSE) · [`PATENTS.md`](PATENTS.md) · [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md) | Licensing. |

```
ADT/                         (~23 MB code + browser; tracked in git)
├── common/                  shared lib (model / tokenizer / dataset / util)
├── QM9/   {README, src/freeorder/, data/freeorder/}
├── Drugs/ {README, src/freeorder_v26/, data/freeorder_v26/}
├── reproduce/ {README, scripts/, logs/}    paper Table/Figure scripts
├── docs/                    interactive 3D browser (GitHub Pages, 7 scaffolds × 50 mol)
├── tools/ {data_manifest.tsv, setup_data.sh}
└── PTBANK/                  large data hub — NOT in git (see "Data" below)
```

---

## Data — checkpoints & caches (Zenodo)

Code lives in this repository; the large binaries live on **Zenodo**
(DOI: [10.5281/zenodo.20635986](https://doi.org/10.5281/zenodo.20635986)) and are linked into the tree by `tools/setup_data.sh`.
The three model checkpoints are:

| Checkpoint | Paper result | Notes |
|---|---|---|
| `qm9_v26_E562.pt`     | QM9 (Table 2/3)        | E562, `max_offset=32`, md5 `17ec74f1…` |
| `drugs30_v26_E142.pt` | Drugs 30-atom (Table 4) | E142, `max_offset=50`, md5 `451ffc3c…` |
| `drugs50_v26_E157.pt` | Drugs 50-atom (Table 5) | E157, fine-tuned from 30-atom, md5 `0a569723…` |

plus frame caches, training-data caches, training logs, and the packaged
generation outputs — the full list is [`tools/data_manifest.tsv`](tools/data_manifest.tsv)
(~4.2 GB total).

```bash
git clone git@github.com:tkotani/ADT.git
cd ADT
export ADT_ROOT=$PWD

mkdir -p $ADT_ROOT/PTBANK
# download the data files from https://doi.org/10.5281/zenodo.20635986 into $ADT_ROOT/PTBANK/

tools/setup_data.sh            # create symlinks data/ -> PTBANK/
tools/setup_data.sh --verify   # md5-check every linked file
```

---

## Reproducing the paper

After `setup_data.sh`, the included generation outputs already reproduce the
tables/figures; the scripts below recompute them. Every script resolves paths
through `ADT_ROOT` and the data manifest.

| Paper element | How to reproduce |
|---|---|
| **Table 2/3** (QM9 mol\_stable, XVR, novelty, RMSD, ΔE) | `GEN_N=10000 QM9/src/freeorder/run_paper_qm9_eval.sh` (or read the included `eval_v26_E562_N10000/`) |
| **Table 4/5** (Drugs 30/50, per-scaffold) | `reproduce/scripts/per_scaffold_stats.py` |
| **Table 6/7** (App. B, per-scaffold detail + novelty) | `per_scaffold_stats.py`, `per_scaffold_novelty.py` |
| **Table 8/9** (App. C, chemical diversity) | `per_scaffold_diversity.py` — BM scaffold / Tanimoto / MW / logP, deterministic (seed 42) |
| **Table 10/11** (App. D, example molecules) | `success_examples.py` |
| **Table 12–14** (App. E, failure modes) | `failure_breakdown.py`, `failure_examples.py` |
| **Fig. 5** (heavy-atom distribution) | `geom_drugs_v26_dist.py`, `v21_nolimit_hist.py` |
| **Fig. 6** (bond/angle structural error) | `analyze_structfidelity.py` (or `bond_angle_errors.py` / `bond_angle_hist.py`) |
| **Fig. 7 / App. F** (training curves) | `extract_curves.py` |

```bash
cd reproduce/scripts
ADT_ROOT=$ADT_ROOT python3 per_scaffold_diversity.py   # example
```

---

## Training (regenerate the checkpoints)

Only needed if you want to retrain rather than download the checkpoints.

| Checkpoint | Runner | GPUs | Wall time |
|---|---|---|---|
| QM9 E562 (Table 2/3) | `QM9/src/freeorder/run_paper_qm9_train.sh` | 1× RTX 4090 | ~16.5 h (600 ep) |
| Drugs 30-atom E142 (Table 4) | `MAX=30 Drugs/src/freeorder_v26/run_paper_drugs_train.sh` | 2× RTX 4090 (DDP) | ~14 h |
| Drugs 50-atom E157 (Table 5) | `MAX=50 Drugs/src/freeorder_v26/run_paper_drugs_train.sh` | 2× RTX 4090 (DDP) | ~3.5 h (ft from 30-atom) |

Override with `EPOCHS=`, `SAVE_DIR=`, `PRETRAIN=` (the last selects the
fine-tuning source checkpoint for the 50-atom run).

## Generation (sample molecules)

```bash
# QM9, unconditional -> data/freeorder/eval_v26_E562_N10000/
GEN_N=10000 QM9/src/freeorder/run_paper_qm9_eval.sh

# Drugs, scaffold-conditional (7 scaffolds × N) -> data/freeorder_v26/.../<scaffold>/
MAX=30 GEN_N=10000 Drugs/src/freeorder_v26/run_paper_drugs_eval.sh
```

Each runner performs Phase 1 (sampling + `mol_stable` check) and Phase 2 (xTB
GFN2 relaxation + topology-preservation = XVR). Outputs per scaffold:
`mol_stable_kekulized.sdf`, `mol_stable_smiles.txt`, `gen_stats.json`,
`xtb_results.json`.

---

## Key configuration (paper checkpoints)

| | QM9 | Drugs 30 (Table 4) | Drugs 50 (Table 5) |
|---|---|---|---|
| epoch | 562 (val 2.7496) | 142 (val 1.980) | 157 (ft from 30) |
| `max_offset` | 32 | 50 | 50 |
| `max_atoms` / `max_pointer` | 9 / 30 | 30 / 64 | 50 / 64 |
| `AROMATIZE_RINGS` (eval) | 0 | 1 | 1 |
| pointer mode | offset (v26 hybrid) | offset | offset |
| distance mesh | log, 200 bins | log, 200 bins | log, 200 bins |
| ckpt md5 | `17ec74f1…` | `451ffc3c…` | `0a569723…` |

All tokens are 7-slot `(action, offset, Z, r_b, h0, h1, h2)`; direction is
HEALPix `Nside=16` (3072 pixels); distance is a log-spaced 200-bin index.

---

## Interactive 3D browser (`docs/`)

A self-contained web viewer that animates ADT's atom-by-atom construction for
each scaffold (50 XVR-positive molecules per scaffold; 3Dmol.js inlined, fully
offline). This directory **is** the GitHub Pages site.

- View locally: `xdg-open docs/index.html`
- Online (after enabling Pages → `main` / `/docs`): <https://tkotani.github.io/ADT/>
- Regenerate / details: [`docs/README.md`](docs/README.md) (needs only `rdkit` + `numpy`).

---

## License & citation

The code is released under the **MIT License** ([`LICENSE`](LICENSE)): free to
use, modify, and redistribute (including inside a company), provided the
copyright notice is kept.

The underlying method is **patent-pending** (JP 2026-16495 / 2026-65995). MIT
grants copyright permissions only, not patent rights: **commercial use of the
patented invention** requires a separate patent license
([`PATENTS.md`](PATENTS.md)); non-commercial use (research, evaluation,
teaching, personal) does not.

The bundled `docs/3Dmol-min.js` is 3Dmol.js under BSD-3-Clause, a separate
license ([`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md)).

**Citation.** A BibTeX entry will be added once the arXiv ID is assigned. If
you use this work, please cite the repository and the paper *"Atomic Design
Transformer: xTB-Validated 3D Molecule Generation from Scaffolds"* (T. Kotani).
