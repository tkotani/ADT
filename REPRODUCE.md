# Reproducing the ADT paper

ADT is a data-free RLVR 3D-molecule generator: a discrete autoregressive
transformer, pretrained on GEOM-Drugs then refined against a GFN2-xTB reward
(**XTP** = xTB-topology-preservation) with no molecular data in the RL loop.

**Two ways in:**
- **From the checkpoints** (recommended) — download from Zenodo, jump to Stage 3.
- **From scratch** — Stage 1 → 2 → 3 → 4. (Stage 1b, the hydrogen models, cannot be
  retrained from this release: see the note there.)

## Setup

Python 3.12 with torch 2.x (CUDA for generation), `rdkit==2025.9.*`, numpy, networkx,
torch_geometric and healpy, plus a GFN2-xTB binary. The repository can be cloned anywhere:
scripts locate `common/` and the model directory relative to their own path.

```bash
pip install torch "rdkit==2025.9.*" numpy networkx torch_geometric healpy zenodo_get
export XTB_BIN=~/xtb/bin/xtb
export ADT=$PWD                                    # this repository
```

The RDKit version matters only for the RDKit-side columns (N_XTP^smiles, N^gen, novelty,
diversity): 2025.09 reproduces the paper's numbers exactly; RDKit 2026.03 moves them by a
few molecules per 10,000. The xTB-side columns do not depend on it.

## Assets (Zenodo)

All checkpoints and the scaffold frame caches are one Zenodo record.
The concept DOI always resolves to the latest version:

**<https://doi.org/10.5281/zenodo.20635985>**

```bash
pip install zenodo_get
mkdir -p ~/assets && cd ~/assets
zenodo_get 10.5281/zenodo.20635985          # fetches all six files
zenodo_get -m 10.5281/zenodo.20635985       # writes md5sums.txt
md5sum -c md5sums.txt
mkdir -p ~/assets/frames && tar xzf frame_caches.tar.gz -C ~/assets/frames   # -> frame_cache_*.pt
```

| Zenodo file | size | md5 (head) | used by |
|---|---|---|---|
| `epoch_240.pt` | 343 MB | `a793a6cad4b7` | pretrained generator (E240) — start of Stage 2 |
| `rlvr_E240direct.pt` | 347 MB | `522f724d9436` | RLVR generator — the paper's results; start of Stage 3 |
| `completer_best.pt` | 104 MB | `33e56a7ef9e5` | MLnH: hydrogen count (perception-free, no RDKit) |
| `mlhadd_v6prod_best.pt` | 107 MB | `765517732e31` | MLHplacer: hydrogen directions |
| `ikt_torsion_bend.pt` | 349 MB | `b86cbc74e0ef` | IKT corrector (§4.5, Fig. 7) |
| `frame_caches.tar.gz` | 0.3 MB | `637a732a1598` | `frame_cache_bootstrap3.pt` (Stage 2) + one cache per scaffold (Stage 3) |

The checkpoints have the optimizer state stripped: they are for inference and for
starting the next stage, not for resuming the original run.

The two hydrogen models are needed by every stage that touches the xTB reward
(Stages 2–5), because the reward is perception-free — hydrogens are placed by
these models, not by an RDKit valence model:

```bash
export COMPLETER_CKPT=~/assets/completer_best.pt    # MLnH
export MLHADD_CKPT=~/assets/mlhadd_v6prod_best.pt   # MLHplacer
```

## Stage 1 — supervised pretraining → E240   *(skip if you downloaded `epoch_240.pt`)*

The pretraining cache is **not** on Zenodo; build it from the public GEOM-Drugs
release with the loader in this repository:

```bash
# 1. download GEOM-Drugs (https://doi.org/10.7910/DVN/JNGTDF) and unpack it
# 2. build the <=30-heavy-atom cache (one conformer per unique SMILES)
python3 -c "import sys; sys.path.insert(0,'Drugs/data/geom'); \
  from load_drugs import load_drugs_mols; \
  load_drugs_mols(geom_dir='<path>/drugs/', max_atoms=30, cache_path='drugs_mols_max30.pkl')"
# 3. pretrain (2-GPU DDP, ~300 epochs; the paper uses the epoch-240 checkpoint)
CACHE=drugs_mols_max30.pkl SAVE=/out/pre bash Drugs/vtakao202606231610/run_offset.sh
```

## Stage 1b — the hydrogen models   *(use the released checkpoints)*

`Hcompleter/train_completer.py` (MLnH) and `Hcompleter/train_hpos.py` (MLHplacer)
are the training scripts, but **their training data is not part of this release**,
so the two models cannot be retrained from what is published here. Use
`completer_best.pt` and `mlhadd_v6prod_best.pt` from Zenodo.

## Stage 2 — data-free RLVR → `rlvr_E240direct.pt`   *(skip if you downloaded it)*

```bash
INIT_CKPT=~/assets/epoch_240.pt \
FRAME_CACHE=~/assets/frames/frame_cache_bootstrap3.pt \
OUT_DIR=/out/rlvr bash run_rlvr_baseline.sh
```

No molecular data enters this loop: the only supervision is the GFN2-xTB reward.
The batch XTP rate climbs from ~51% to ~98% by step ~10,000 (Fig. 8b).
`rlvr_E240direct.pt` is the **final** checkpoint of this run, `ckpt_step9999.pt`
(optimizer state stripped), not `best.pt`. The script sets the XTP reward protocol
(ML hydrogens, H-prerelax, clamp/unclamp, no partial credit) explicitly, because the
code defaults differ. Cost is set by xTB on the CPU, not by the GPU: three GFN2
relaxations per generated molecule, ~20 s per step, ~2.3 days for 10,000 steps on
one RTX 4090 with 16 CPU cores. RL sampling and parallel xTB are not bit-reproducible;
expect the same curve and statistically equivalent tables, not identical weights.

## Stage 3 — generation → molrecord banks

```bash
GEN_CKPT=~/assets/rlvr_E240direct.pt FRAME_DIR=~/assets/frames OUT=/out/bank N=10000 \
  bash Drugs/vtakao202606231610/run_gen_records.sh    # 7 scaffolds x 10,000 (+ triple)
```

Each molecule is frozen into a record with its provenance, so the funnel and the
strain statistics can be re-tabulated without regenerating anything.

## Stage 4 — evaluation → tables

```bash
python3 common/funnel_stats.py /out/bank      # xTB-side funnel + strain (quick check)

# GEOM-Drugs reference SMILES for the novelty column (keys of summary_drugs.json in the
# public GEOM release, https://doi.org/10.7910/DVN/JNGTDF, rdkit_folder)
python3 common/geom_smiles.py <path>/rdkit_folder/summary_drugs.json geom_drugs.smi
python3 common/paper_tables.py /out/bank --geom_smi geom_drugs.smi   # Table 2, Table 3, Fig. 6
# GEOM side of Table 3: GEOM sub-sampled to each row's N^gen (from paper_tables.py)
python3 common/geom_matched_diversity.py geom_drugs.smi benzene=<N^gen> pyridine=<N^gen> ...
```

`paper_tables.py` prints every column of Table 2 (funnel, size, RMSD, strain) and the model
side of Table 3 (N_eff^scaf, distinct Murcko scaffolds, IntDiv_1, MW, logP, QED), the
per-heavy-atom strain percentiles, and the Fig. 6 bond/angle-shift histograms.

A regeneration of the benzene row with `rlvr_E240direct.pt` (N = 10,000, RTX 5090, 16 xTB
workers, 55 min) gave N_XTP 9805 / N^gen 9582 (95.8%) / median ΔE 10.1 kcal/mol against the
paper's 9808 / 9562 (95.6%) / 10.2: sampling noise, not bit identity.

## Stage 5 — Figure 7 (IKT)

```bash
python3 Drugs/vtakao202606231610/ikt_eval_big.py \
  --ckpt ~/assets/rlvr_E240direct.pt --ikt ~/assets/ikt_torsion_bend.pt \
  --out persize_ikt.json
python3 Drugs/vtakao202606231610/plot_ikt_xtp_size.py \
  --recs persize_ikt.json --out ikt_xtp_size.pdf
```

Both curves come from the same molecules: ADT alone is the fraction that is XTP
on the first try, ADT+IKT adds the ones the corrector rescues within six xTB calls.

## Paper items

| Paper element | Stage |
|---|---|
| Table 2, Table 3 (funnel, diversity, properties) | Stage 3 → 4 (`paper_tables.py`) |
| Fig. 6 bond / angle errors | Stage 3 → 4 (`paper_tables.py`) |
| Fig. 8(a) pretraining curve | Stage 1 training log |
| Fig. 8(b) RLVR curve | Stage 2 training log |
| Fig. 7 (IKT) | Stage 5 |

To use an RDKit valence model for the hydrogens instead of the learned ones
(not what the paper measures), set `H_PLACER=rdkit`.
