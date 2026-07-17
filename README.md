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
distinct valid molecules from 42.2% to 96.7%. The reward is *perception-free*: the
hydrogens are placed by two learned models, and the topology is read from the
relaxed coordinates by a distance rule — no SMILES, no valence table.

- **Interactive 3D browser** (atom-by-atom growth, one page per scaffold):
  <https://tkotani.github.io/ADT/>
- **Checkpoints & frame caches (Zenodo):** <https://doi.org/10.5281/zenodo.20635985>
- **How to reproduce the paper:** [`REPRODUCE.md`](REPRODUCE.md)
- **Paper:** arXiv (ID *to be added on posting*)

---

## Quick start (from the released checkpoints)

```bash
git clone git@github.com:tkotani/ADT.git && cd ADT

pip install zenodo_get
mkdir -p ~/assets && cd ~/assets && zenodo_get 10.5281/zenodo.20635985
mkdir -p ~/assets/frames && tar xzf frame_caches.tar.gz -C ~/assets/frames
cd -

export XTB_BIN=~/xtb/bin/xtb
export COMPLETER_CKPT=~/assets/completer_best.pt    # MLnH        (hydrogen count)
export MLHADD_CKPT=~/assets/mlhadd_v6prod_best.pt   # MLHplacer   (hydrogen directions)

# generate 10,000 molecules per scaffold, then tabulate the funnel
GEN_CKPT=~/assets/paper_rlvl.ckpt FRAME_DIR=~/assets/frames OUT=/out/bank N=10000 \
  bash Drugs/vtakao202606231610/run_gen_records.sh
python3 common/funnel_stats.py /out/bank
```

Full walkthrough — pretraining, data-free RLVR, generation, evaluation, Fig. 7 —
is in [`REPRODUCE.md`](REPRODUCE.md).

## Repository map

| Path | Contents |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | **Start here.** Stage-by-stage reproduction of the paper. |
| [`run_rlvr_baseline.sh`](run_rlvr_baseline.sh) | The data-free RLVR run (Stage 2), with the paper's recipe. |
| [`Drugs/vtakao202606231610/`](Drugs/vtakao202606231610/) | Architecture, pretraining, RLVR, generation, IKT, Fig. 7 plot. |
| [`common/`](common/) | Tokenizer, dataset, the perception-free xTB reward, evaluation. |
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

**Citation.** A BibTeX entry will be added once the arXiv ID is assigned. If you
use this work, please cite the paper *"Atomic Design Transformer:
Scaffold-Conditioned 3D Molecule Generation via xTB-Reward Reinforcement
Learning"* (T. Kotani) and this repository.
