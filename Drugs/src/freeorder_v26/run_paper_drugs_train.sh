#!/usr/bin/env bash
# Paper Drugs training: MAX=30 (E142 scratch) / MAX=50 (E157 ft)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADT_ROOT="${ADT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

source ~/.venv/bin/activate
MAX="${MAX:-30}"
DATA="$ADT_ROOT/Drugs/data/freeorder_v26"
mkdir -p "$DATA/logs"
LOG="$DATA/logs/fo_train_v26_${MAX}atom_repro_$(date +%Y%m%d_%H%M%S).log"
cd "$SCRIPT_DIR"

if [ "$MAX" = "30" ]; then
  SAVE_DIR="${SAVE_DIR:-$DATA/checkpoints_v26_scratch_repro}"
  EPOCHS="${EPOCHS:-300}"
  mkdir -p "$SAVE_DIR"
  torchrun --nproc_per_node=2 fo_train_v26.py \
      --max_atoms 30 --max_pointer 64 --max_offset 50 --require_benzene 0 \
      --data_cache "$DATA/drugs_mols_v26_max30.pkl" \
      --frame_cache "$DATA/frame_cache_v21_benzene.pt" \
      --epochs "$EPOCHS" --batch_size 64 --lr 2e-4 --warmup_epochs 5 \
      --d_model 768 --n_layers 12 --n_heads 8 --d_ff 3072 \
      --dropout 0.2 --amp bf16 --nohydrogen --dynamic \
      --save_dir "$SAVE_DIR" --save_every 5 --num_workers 4 \
      --eval_after 9999 --eval_every 9999 > "$LOG" 2>&1
elif [ "$MAX" = "50" ]; then
  SAVE_DIR="${SAVE_DIR:-$DATA/checkpoints_v26_max50_ft_v2_repro}"
  PRETRAIN="${PRETRAIN:-$DATA/checkpoints_v26_scratch/E142.pt}"
  EPOCHS="${EPOCHS:-30}"
  mkdir -p "$SAVE_DIR"
  torchrun --nproc_per_node=2 fo_train_v26.py \
      --max_atoms 50 --max_pointer 64 --max_offset 50 --require_benzene 0 \
      --data_cache "$DATA/drugs_mols_v26_max50.pkl" \
      --pretrain "$PRETRAIN" \
      --frame_cache "$DATA/frame_cache_v21_benzene.pt" \
      --epochs "$EPOCHS" --batch_size 48 --lr 5e-5 --warmup_epochs 0 \
      --d_model 768 --n_layers 12 --n_heads 8 --d_ff 3072 \
      --dropout 0.2 --amp bf16 --nohydrogen --dynamic \
      --save_dir "$SAVE_DIR" --save_every 1 --num_workers 4 \
      --eval_after 9999 --eval_every 9999 > "$LOG" 2>&1
else
  echo "MAX must be 30 or 50, got: $MAX"; exit 1
fi
echo "Done. Log: $LOG"; echo "best.pt: $SAVE_DIR/best.pt"
