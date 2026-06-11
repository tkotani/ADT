#!/usr/bin/env bash
# Paper QM9 training reproduction (E562 を再生成、~16.5h on RTX 4090)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADT_ROOT="${ADT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

source ~/.venv/bin/activate
DATA="$ADT_ROOT/QM9/data/freeorder"
SAVE_DIR="${SAVE_DIR:-$DATA/checkpoints_v26_repro}"
EPOCHS="${EPOCHS:-600}"
mkdir -p "$SAVE_DIR" "$DATA/logs"
LOG="$DATA/logs/fo_train_v26_repro_$(date +%Y%m%d_%H%M%S).log"

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES=0 \
~/.venv/bin/python3 fo_train_v26.py \
    --epochs "$EPOCHS" --batch_size 128 --lr 2e-4 --warmup_epochs 5 \
    --d_model 768 --n_layers 12 --n_heads 8 --d_ff 3072 \
    --dropout 0.2 --amp bf16 --nohydrogen --dynamic \
    --max_offset 32 --num_workers 4 \
    --save_every 10 --eval_after 9999 --eval_every 9999 \
    --save_dir "$SAVE_DIR" \
    --frame_cache "$DATA/frame_cache_200bin.pt" \
    > "$LOG" 2>&1
echo "Done. Log: $LOG"; echo "best.pt: $SAVE_DIR/best.pt"
