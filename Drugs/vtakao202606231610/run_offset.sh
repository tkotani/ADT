#!/usr/bin/env bash
# emb_offset (vtakao202606231610) — scratch supervised pretrain, 2-GPU DDP (effective batch 2*64=128).
#   scratch (kt1):  CACHE=/mnt/data1/drugs_mols_v26_max30.pkl  SAVE=/mnt/data1/.../checkpoints_scratch_offset
# Recipe = paper drugs train, MAX=30. Set RESUME=<ckpt> to continue FULL state (optimizer/scheduler/epoch).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CACHE="${CACHE:?set CACHE=path to drugs_mols_v26_max30.pkl on THIS host}"
SAVE="${SAVE:?set SAVE=checkpoint dir on THIS host}"
mkdir -p "$SAVE"
LOG="$SAVE/train.log"

INIT_ARG=()
if [ -n "${RESUME:-}" ]; then
  INIT_ARG=(--resume "$RESUME")          # continue FULL state (optimizer/scheduler/epoch) from a ckpt
fi

cd "$HERE"
OMP_NUM_THREADS=1 AROMATIZE_RINGS=1 \
  nohup python3 -m torch.distributed.run --nproc_per_node=2 --master_port="${MASTER_PORT:-29510}" train.py \
    --max_atoms 30 --max_offset 50 --require_benzene 0 \
    --data_cache "$CACHE" "${INIT_ARG[@]}" \
    --epochs 300 --batch_size 64 --lr 2e-4 --warmup_epochs 5 \
    --d_model 768 --n_layers 12 --n_heads 8 --d_ff 3072 \
    --dropout 0.2 --amp bf16 --nohydrogen --dynamic \
    --save_dir "$SAVE" --save_every 5 --num_workers 4 \
    --eval_after 9999 --eval_every 9999 > "$LOG" 2>&1 &
echo "started scratch pretrain (2-GPU DDP) PID=$! -> $LOG"
