# ADT — Atomic Design Transformer

Code and reproduction package for the paper
**"Atomic Design Transformer: Scaffold-Conditioned 3D Molecule Generation via
xTB-Reward Reinforcement Learning"** (Takao Kotani).

ADT is a fully-discrete autoregressive transformer that builds a 3D molecule
**one atom at a time**. SE(3) invariance comes entirely from the tokenization —
each new atom is encoded in a local coordinate frame anchored on a previously
placed atom — so the network itself is a plain causal transformer. Generation can
be conditioned on a fixed seed scaffold, and the model decides on its own when to
stop.

The model is then improved **without any molecular data**: reinforcement learning
against a verifiable physical reward (**XTP** — a GFN2-xTB relaxation must
preserve the heavy-atom topology the model declared) lifts the end-to-end yield of
distinct valid molecules from 42.2% to 96.4%. The reward is *perception-free*: the
hydrogens are placed by two learned models, and the topology is read from the
relaxed coordinates by a distance rule — no SMILES, no valence table.

- **Interactive 3D browser** (atom-by-atom growth, one page per scaffold):
  <https://tkotani.github.io/ADT/>
- **Checkpoints & frame caches (Zenodo):** <https://doi.org/10.5281/zenodo.20635985>
- **How to reproduce the paper:** [`REPRODUCE.md`](REPRODUCE.md)
- **Paper:** arXiv:2607.15918 — <https://arxiv.org/abs/2607.15918>

---

## Quick start (from the released checkpoints)

```bash
git clone https://github.com/tkotani/ADT.git && cd ADT

pip install torch "rdkit==2025.9.*" numpy networkx torch_geometric healpy matplotlib zenodo_get
mkdir -p ~/assets && cd ~/assets && zenodo_get 10.5281/zenodo.20635985
mkdir -p ~/assets/frames && tar xzf frame_caches.tar.gz -C ~/assets/frames
cd -

export XTB_BIN=~/xtb/bin/xtb
export COMPLETER_CKPT=~/assets/completer_best.pt    # MLnH        (hydrogen count)
export MLHADD_CKPT=~/assets/mlhadd_v6prod_best.pt   # MLHplacer   (hydrogen directions)

# generate 10,000 molecules per scaffold, then tabulate the funnel
GEN_CKPT=~/assets/rlvr_E240direct.pt FRAME_DIR=~/assets/frames OUT=/out/bank N=10000 \
  bash Drugs/vtakao202606231610/run_gen_records.sh
python3 common/funnel_stats.py /out/bank                       # quick xTB-side funnel

# every generation number of the paper (Tables 2-4, Fig. 6, text), with the GEOM-Drugs
# SMILES list used in the paper (on Zenodo)
python3 common/paper_tables.py /out/bank --geom_smi ~/assets/geom_drugs_smiles.smi --json tables.json
```

A GFN2-xTB binary (6.7.1) is required; generation needs a CUDA GPU.

Full walkthrough — pretraining, data-free RLVR, generation, evaluation, Fig. 7 —
is in [`REPRODUCE.md`](REPRODUCE.md).

## Confirming the results with an AI coding agent

The package is written so that an AI coding agent (e.g. Claude Code with Claude Opus) can confirm the
paper's numbers on its own. On a Linux machine with a CUDA GPU, copy this prompt into the agent (the same
prompt is printed in the paper's Reproducibility section):

> Clone https://github.com/tkotani/ADT (release v2.1), download the Zenodo record
> https://doi.org/10.5281/zenodo.20635985, and follow REPRODUCE.md to regenerate the RLVR benzene row of
> Table 2 of arXiv:2607.15918 (N = 10,000 molecules with rlvr_E240direct.pt), tabulate it with
> common/paper_tables.py, and compare the result with that row.

One scaffold takes about an hour on one GPU with 16 CPU cores. GPU sampling is not bit-reproducible,
so expect agreement within sampling noise (a few tenths of a percent), not identical digits.

## Repository map

| Path | Contents |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | **Start here.** Stage-by-stage reproduction of the paper. |
| [`run_rlvr_baseline.sh`](run_rlvr_baseline.sh) | The data-free RLVR run (Stage 2), with the paper's recipe. |
| [`Drugs/vtakao202606231610/`](Drugs/vtakao202606231610/) | Architecture, pretraining, RLVR, generation, IKT, Fig. 7 plot. |
| [`common/`](common/) | Tokenizer, dataset, the perception-free xTB reward, evaluation (`paper_tables.py`: the paper's tables). |
| [`Hcompleter/`](Hcompleter/) | Training scripts for the two hydrogen models (MLnH, MLHplacer). |
| [`docs/`](docs/README.md) | The interactive 3D browser — this directory **is** the GitHub Pages site. |
| [`LICENSE`](LICENSE) · [`PATENTS.md`](PATENTS.md) · [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md) | Licensing. |

Code is tracked here; checkpoints, caches and generated molecules are not — they
live on Zenodo (see above).

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

**Citation.** If you use this work, please cite the paper *"Atomic Design
Transformer: Scaffold-Conditioned 3D Molecule Generation via xTB-Reward
Reinforcement Learning"* (T. Kotani, arXiv:2607.15918) and this repository:

```bibtex
@article{kotani2026adt,
  title         = {Atomic Design Transformer: Scaffold-Conditioned 3D Molecule Generation via xTB-Reward Reinforcement Learning},
  author        = {Kotani, Takao},
  year          = {2026},
  eprint        = {2607.15918},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2607.15918}
}
```
