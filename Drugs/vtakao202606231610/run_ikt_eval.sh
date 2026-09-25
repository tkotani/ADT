#!/usr/bin/env bash
# run_ikt_eval.sh — Figure 7: XTP versus size for ADT alone and ADT + the frozen IKT corrector.
# Generation is biased toward long molecules (end_bias); every molecule is fresh; the IKT is not trained.
# Defaults are the conditions of the paper's run (n_mol 1500, min_na 20, end_bias 2.4, 48 proposals,
# top 6 verified with xTB, bootstrap3 frames).
# Set these env for THIS host:
#   GEN_CKPT       generator ckpt (the paper's rlvr_E240direct.pt)
#   IKT_CKPT       IKT corrector ckpt (ikt_torsion_bend.pt)
#   MLHADD_CKPT    mlhadd v6prod ckpt (ML H placer)
#   COMPLETER_CKPT completer ckpt (ML nH)
#   FRAME_DIR      dir holding frame_cache_bootstrap3.pt
#   OUT            output dir (persize_ikt.json, ikt_xtp_size.pdf)
# Optional: N_MOL, MIN_NA, XTB_WORKERS, CUDA_VISIBLE_DEVICES, XTB_BIN, PYBIN
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${GEN_CKPT:?set GEN_CKPT}"; : "${IKT_CKPT:?set IKT_CKPT}"
: "${MLHADD_CKPT:?set MLHADD_CKPT}"; : "${COMPLETER_CKPT:?set COMPLETER_CKPT}"
: "${FRAME_DIR:?set FRAME_DIR}"; : "${OUT:?set OUT}"
# XTP protocol of the paper (same as run_gen_records.sh)
export H_PLACER=mlhadd H_PRERELAX=1 XVR_ESTRAIN_TAU=2.0 AROMATIZE_RINGS=1 XVR_PFREE=1
export XVR_CLAMP="${XVR_CLAMP:-1}" XVR_CLAMP_ONLY="${XVR_CLAMP_ONLY:-1}" XVR_CLAMP_IDEAL="${XVR_CLAMP_IDEAL:-1}"
export XVR_FAIL_CREDIT="${XVR_FAIL_CREDIT:-0}" XVR_STRAIN_HPRE="${XVR_STRAIN_HPRE:-1}"
export MLNH_PARITY="${MLNH_PARITY:-1}" H_INTEGRITY="${H_INTEGRITY:-1}" PYTHONUNBUFFERED=1
export XTB_BIN="${XTB_BIN:-$HOME/xtb/bin/xtb}"
mkdir -p "$OUT"; cd "$HERE"
${PYBIN:-python3} ikt_eval_big.py --adt "$GEN_CKPT" --ikt "$IKT_CKPT" \
  --frame_cache "$FRAME_DIR/frame_cache_bootstrap3.pt" --out "$OUT/persize_ikt.json" \
  --n_mol "${N_MOL:-1500}" --min_na "${MIN_NA:-20}" --end_bias 2.4 --batch 96 \
  --n_cand 48 --xtb_cand 6 --cand_temp 1.2 --xtb_workers "${XTB_WORKERS:-16}" --xtb_bin "$XTB_BIN"
${PYBIN:-python3} plot_ikt_xtp_size.py --recs "$OUT/persize_ikt.json" --out "$OUT/ikt_xtp_size.pdf"
