#!/usr/bin/env bash
# Paper Drugs reproduction (adt.tex Tab 4 / Tab 5)
# MAX=30 (E142, Tab 4) or MAX=50 (E157, Tab 5), 7 scaffolds × N=GEN_N
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADT_ROOT="${ADT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

source ~/.venv/bin/activate
GEN_N="${GEN_N:-10000}"
MAX="${MAX:-30}"
SCAFFOLDS=(benzene pyridine pyrimidine pyrazine furan thiophene cyclohexane)

DATA="$ADT_ROOT/Drugs/data/freeorder_v26"
if [ "$MAX" = "30" ]; then
  CKPT="$DATA/checkpoints_v26_scratch/E142.pt"
  TAG="v26s"
else
  CKPT="$DATA/checkpoints_v26_max50_ft_v2/best.pt"
  TAG="v26g"
fi
OUTDIR="$DATA/${TAG}_scaffolds_n${GEN_N}_repro"
mkdir -p "$OUTDIR"
for SCAF in "${SCAFFOLDS[@]}"; do
  echo "[$(date +%H:%M:%S)] === scaffold=$SCAF (N=$GEN_N, MAX=$MAX) ==="
  cd "$SCRIPT_DIR"
  CUDA_VISIBLE_DEVICES=0 \
  GEN_N="$GEN_N" ADT_R_BINS=200 AROMATIZE_RINGS=1 \
  CKPT="$CKPT" \
  FRAME_CACHE="$DATA/scaffolds/frame_cache_${SCAF}.pt" \
  SCAFFOLD="$SCAF" \
  OUTDIR="$OUTDIR/$SCAF" \
  python3 gen_eval_v26.py \
  > "$OUTDIR/${SCAF}.log" 2>&1
done
echo "Done. Aggregate via reproduce/scripts/per_scaffold_stats.py"
