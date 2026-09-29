# Reproduce the benzene row of Table 2 (instructions for an AI coding agent)

This file is written to be handed to an AI coding agent (e.g. Claude Code): "Read README_reproduce_benzene.md
in https://github.com/tkotani/ADT and do what it says." It regenerates the RLVR benzene row of Table 2 of the
paper (arXiv:2607.15918) from the released checkpoint and compares it with the paper.

Requirements: Linux, one CUDA GPU (RTX 4090 or better recommended), about 16 CPU cores, and ~10 GB of
free disk (a CUDA build of torch is ~6 GB, the Zenodo assets ~1.3 GB, the xTB scratch ~0.8 GB).
Time: about 40 minutes of generation on one RTX 4090 with 16 workers, plus the Zenodo download.

## Steps

1. **Code.** `git clone https://github.com/tkotani/ADT && cd ADT && git checkout v2.1` (use branch `main`
   if the tag is missing), then `export ADT=$PWD`.
2. **Python environment.** Python 3.12 with
   `pip install torch "rdkit==2025.9.*" numpy networkx torch_geometric healpy matplotlib zenodo_get`
   (a CUDA build of torch). RDKit 2025.09 matters for the RDKit-side columns.
   If you install into a virtual environment without activating it, `export PYBIN=<venv>/bin/python`:
   the generation script calls `${PYBIN:-python3}`, and a bare `python3` would be the system interpreter,
   which has no torch.
3. **GFN2-xTB 6.7.1.** Download `xtb-6.7.1-linux-x86_64.tar.xz` from
   https://github.com/grimme-lab/xtb/releases/tag/v6.7.1, unpack it, and `export XTB_BIN=<path>/bin/xtb`.
   Check with `$XTB_BIN --version`.
4. **Checkpoints and frames (Zenodo).**
   ```bash
   mkdir -p ~/assets && cd ~/assets
   zenodo_get 10.5281/zenodo.20635985 && zenodo_get -m 10.5281/zenodo.20635985 && md5sum -c md5sums.txt
   mkdir -p frames && tar xzf frame_caches.tar.gz -C frames
   ```
   `md5sum -c` must print OK for every file.
5. **Generate 10,000 molecules from the benzene frames** (this is the long step; run it in the background
   and check the log):
   ```bash
   cd $ADT && mkdir -p ~/repro ~/repro/xtb_work
   GEN_CKPT=~/assets/rlvr_E240direct.pt COMPLETER_CKPT=~/assets/completer_best.pt \
   MLHADD_CKPT=~/assets/mlhadd_v6prod_best.pt FRAME_DIR=~/assets/frames OUT=~/repro/bank \
   XTB_BIN=$XTB_BIN PYBIN=${PYBIN:-python3} XTB_WORKDIR=~/repro/xtb_work \
   SCAFS=benzene_real N=10000 XTB_WORKERS=16 \
     bash Drugs/vtakao202606231610/run_gen_records.sh > ~/repro/gen.log 2>&1
   ```
   Try `N=20` first as a smoke test (a few minutes) before the full run.
   `XTB_BIN` and `PYBIN` are repeated here on purpose: this block must carry them even if you run it in a
   different shell from steps 2 and 3. Without `XTB_BIN` the code looks for `~/xtb/bin/xtb`. `XTB_WORKDIR`
   keeps the xTB scratch (~0.8 GB, deleted as it goes) out of `/tmp`; omit it if `/tmp` has the room.
6. **Tabulate** with the same script that made the paper's tables:
   ```bash
   python3 common/paper_tables.py ~/repro/bank --geom_smi ~/assets/geom_drugs_smiles.smi --json ~/repro/tables.json
   ```
7. **Compare** the benzene row with the paper and report a table of paper vs. reproduced values.

## Expected values (paper, Table 2, RLVR model, benzene, N = 10,000)

| N_Hprerx | N_XTP | N_XTP^smiles | N^gen (rate) | N^Novel | N_fullrx | size mean/std/med/min/max | RMSD (Å) | ΔE (kcal/mol) |
|---|---|---|---|---|---|---|---|---|
| 9961 | 9834 | 9636 | 9588 (95.9%) | 9544 | 9958 | 25.9/5.4/26/7/46 | 0.24 | 9.8 |

GPU sampling is not bit-reproducible, so the digits will differ. The result is confirmed when the counts
agree within sampling noise (about ±50 out of 10,000, i.e. a few tenths of a percent) and the medians
(size, RMSD, ΔE) agree to within a few percent. If a step fails, report the step and the error rather than
working around it.

The full pipeline (all scaffolds, the pretrained model, training) is in [`REPRODUCE.md`](REPRODUCE.md).
