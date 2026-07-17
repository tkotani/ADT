# Reproducing the ADT paper

ADT is a data-free RLVR 3D-molecule generator: a discrete autoregressive
transformer, pretrained on GEOM-Drugs then refined against a GFN2-xTB reward
(**XTP** = xTB-topology-preservation) with no molecular data in the RL loop.

**Two ways in:**
- **From the checkpoints** (recommended) — download from Zenodo, jump to Stage 3.
- **From scratch** — Stage 1 → 2 → 3 → 4. (Stage 1b, the hydrogen models, cannot be
  retrained from this release: see the note there.)

## Setup

Python (torch 2.x, rdkit, numpy, networkx, torch_geometric) and a GFN2-xTB binary.

```bash
export XTB_BIN=~/xtb/bin/xtb
export ADT=$PWD                                    # this repository
```

## Assets (Zenodo)

All checkpoints and the scaffold frame caches are one Zenodo record.
The concept DOI always resolves to the latest version:

**<https://doi.org/10.5281/zenodo.20635985>**

```bash
pip install zenodo_get
mkdir -p ~/assets && cd ~/assets
zenodo_get 10.5281/zenodo.20635985          # fetches all six files
md5sum -c md5sums.txt                       # zenodo_get writes this
tar xzf frame_caches.tar.gz -C ~/assets/frames   # -> frame_cache_*.pt
```

| Zenodo file | size | md5 (head) | used by |
|---|---|---|---|
| `epoch_240.pt` | 343 MB | `a793a6cad4b7` | pretrained generator (E240) — start of Stage 2 |
| `paper_rlvl.ckpt` | 343 MB | `3b16c61034c3` | RLVR generator — the paper's results; start of Stage 3 |
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

## Stage 2 — data-free RLVR → `paper_rlvl.ckpt`   *(skip if you downloaded it)*

```bash
INIT_CKPT=~/assets/epoch_240.pt \
FRAME_CACHE=~/assets/frames/frame_cache_bootstrap3.pt \
OUT_DIR=/out/rlvr bash run_rlvr_baseline.sh
```

No molecular data enters this loop: the only supervision is the GFN2-xTB reward.
The batch XTP rate climbs from ~50% to ~98% over ~9,500 steps (Fig. 8b). The
run keeps `best.pt` at the running-max batch XTP; that is what `paper_rlvl.ckpt`
is. Cost is set by xTB on the CPU, not by the GPU: three GFN2 relaxations per
generated molecule.

## Stage 3 — generation → molrecord banks

```bash
GEN_CKPT=~/assets/paper_rlvl.ckpt FRAME_DIR=~/assets/frames OUT=/out/bank N=10000 \
  bash Drugs/vtakao202606231610/run_gen_records.sh    # 7 scaffolds x 10,000 (+ triple)
```

Each molecule is frozen into a record with its provenance, so the funnel and the
strain statistics can be re-tabulated without regenerating anything.

## Stage 4 — evaluation → tables

```bash
python3 common/funnel_stats.py /out/bank      # funnel + strain -> Table 2, Table 3
```

## Stage 5 — Figure 7 (IKT)

```bash
python3 Drugs/vtakao202606231610/ikt_eval_big.py \
  --ckpt ~/assets/paper_rlvl.ckpt --ikt ~/assets/ikt_torsion_bend.pt \
  --out persize_ikt.json
python3 Drugs/vtakao202606231610/plot_ikt_xtp_size.py \
  --recs persize_ikt.json --out ikt_xtp_size.pdf
```

Both curves come from the same molecules: ADT alone is the fraction that is XTP
on the first try, ADT+IKT adds the ones the corrector rescues within six xTB calls.

## Paper items

| Paper element | Stage |
|---|---|
| Table 2, Table 3 (funnel, strain) | Stage 3 → 4 |
| bond / angle errors, per-scaffold stats | Stage 3 → 4 |
| Fig. 8(a) pretraining curve | Stage 1 training log |
| Fig. 8(b) RLVR curve | Stage 2 training log |
| Fig. 7 (IKT) | Stage 5 |

To use an RDKit valence model for the hydrogens instead of the learned ones
(not what the paper measures), set `H_PLACER=rdkit`.
