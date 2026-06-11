#!/usr/bin/env bash
# Paper QM9 reproduction (adt.tex Tab 6/7)
# 検証済 N=1000 mol_stable=88.8% / topo=87.2% (paper 88.0% / 86.3%)
# ckpt: E562 v26 (max_offset=32, R_BINS=200, AROMATIZE_RINGS=0)
# md5: 17ec74f188b0dd4c216ed05dab9c85a2
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# ADT_ROOT = 3 levels up from runner: src/freeorder → src → QM9 → ADT_ROOT
ADT_ROOT="${ADT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

source ~/.venv/bin/activate
GEN_N="${GEN_N:-1000}"
DATA="$ADT_ROOT/QM9/data/freeorder"
OUTDIR="${OUTDIR:-$DATA/eval_v26_E562_N${GEN_N}}"
mkdir -p "$OUTDIR"
cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES=0 \
GEN_N="$GEN_N" ADT_R_BINS=200 AROMATIZE_RINGS=0 \
CKPT="$DATA/checkpoints_v26_scratch/best.pt" \
FRAME_CACHE="$DATA/frame_cache_200bin.pt" \
QM9_CACHE="$DATA/qm9_mols_cache_v3b_noh.pkl" \
OUTDIR="$OUTDIR" \
python3 gen_eval_v26.py > "$OUTDIR/run.log" 2>&1
echo "Done. See $OUTDIR/stats.json"
