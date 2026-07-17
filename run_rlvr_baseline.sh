#!/usr/bin/env bash
# =============================================================================
#  ADT — Stage 2: data-free RLVR training  (baseline recipe = "kr2", 2026-07-17)
# -----------------------------------------------------------------------------
#  Refine a pretrained ADT generator (Stage 1 / E240) against the verifiable
#  GFN2-xTB reward (XTP = xTB-topology-preservation).  NO molecular dataset is
#  used in the loop; the only supervision is the physics reward.
#
#  Recipe (baseline): size is HELD FIXED (lam_size=1.0, size-PID OFF) with a
#  moment controller keeping the heavy-atom count at 25 +/- 5.45; the element
#  composition is matched to GEOM-Drugs by a ONE-SIDED comp-PID (base 2.0,
#  setpoint kappa=0.03, cap x1.5); a scaffold-diversity floor holds diversity
#  at 0.96.  Reaches batch XTP (Xema) ~0.985 (XTP ~98.5%); best.pt is auto-saved
#  at the running-max Xema.
#
#  Paths are taken from the environment (edit or export before running):
#     INIT_CKPT      pretrained generator to start from (Stage 1 / E240)
#     FRAME_CACHE    bootstrap frame cache (scaffold prefixes)
#     OUT_DIR        output directory (ckpts + train.log)
#     COMPLETER_CKPT ML n_H model (perception-free H, MLnH)      [reward pipeline]
#     MLHADD_CKPT    ML H-placer model (perception-free H)       [reward pipeline]
#     XTB_BIN        GFN2-xTB binary            (default: ~/xtb/bin/xtb)
#     PY             python with torch+rdkit    (default: python3)
#     N_STEPS        training steps             (default: 6000; plateau ~ here)
#
#  The XTP reward is PERCEPTION-FREE (no RDKit): H atoms are placed by the two
#  learned models COMPLETER_CKPT (n_H) + MLHADD_CKPT (H directions), then GFN2-xTB
#  relaxes and topology preservation is checked.  (Set H_PLACER=rdkit to use
#  RDKit-based H instead, e.g. if the ML H models are unavailable.)
# =============================================================================
set -euo pipefail

INIT_CKPT="${INIT_CKPT:?set INIT_CKPT = pretrained E240 checkpoint (epoch_240.pt)}"
FRAME_CACHE="${FRAME_CACHE:?set FRAME_CACHE = frame_cache_bootstrap3.pt}"
OUT_DIR="${OUT_DIR:?set OUT_DIR = output directory}"
export COMPLETER_CKPT="${COMPLETER_CKPT:?set COMPLETER_CKPT = MLnH completer best.pt}"
export MLHADD_CKPT="${MLHADD_CKPT:?set MLHADD_CKPT = ML H-placer (mlhadd v6prod) best.pt}"
export XVR_PFREE="${XVR_PFREE:-1}"     # 1 = perception-free reward (paper default)
export XTB_BIN="${XTB_BIN:-$HOME/xtb/bin/xtb}"
PY="${PY:-python3}"
N_STEPS="${N_STEPS:-6000}"

mkdir -p "$OUT_DIR"
cd "$(dirname "$0")"          # run from the code directory (rl_train_klrl_step1.py alongside)

"$PY" -u rl_train_klrl_step1.py \
  --ckpt        "$INIT_CKPT" \
  --frame_cache "$FRAME_CACHE" \
  --out_dir     "$OUT_DIR" \
  --reward xtb --size_baseline \
  --batch 48 --n_train_steps "$N_STEPS" --max_steps_per_mol 60 \
  --lr 2e-5 --temperature 1.0 --seed 42 \
  --save_every 200 --max_keep_ckpts 12 --xtb_workers 16 \
  --beta_kl 0.1 --ratchet_every 100 \
  `# --- size: FIXED at 1.0 (no size-PID) + moment controller to hold N~25+/-5.45 ---` \
  --lam_size 1.0 \
  --size_mu 25 --size_sigma 5.45 --size_nmax 56 --size_moment_ctl \
  --moment_mu_star 25 --moment_sigma_star 5.45 --moment_mu_init 25 \
  --moment_step0 1.0 --moment_step_floor 0.06 --moment_settle 0.3 --moment_off_cap 8.0 \
  `# --- composition: ONE-SIDED comp-PID -> match GEOM element fractions ---` \
  --lam_comp 2.0 --lam_comp_pid --kappa_comp 0.03 --lam_comp_max_mult 1.5 \
  `# --- scaffold-diversity floor ---` \
  --scaffold_floor --target_scaffdiv 0.96 --scaff_kp 3 --scaff_ki 0.3 --scaff_window 300 \
  2>&1 | tee -a "$OUT_DIR/train.log"

echo "[done] RLVR finished. Best checkpoint (max Xema): $OUT_DIR/best.pt"
