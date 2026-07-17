#!/usr/bin/env bash
# run_gen_records.sh — generate N molecules/scaffold from a ckpt -> perception-free molrecord_v2 banks
# -> per-scaffold funnel table. H = ML nH (completer) + mlhadd v6prod ML placer (NO RDKit for H).
# Set these env for THIS host:
#   GEN_CKPT       generator ckpt (e.g. RL ckpt_stepNNN.pt, or the E240 base epoch_240.pt)
#   MLHADD_CKPT    mlhadd v6prod ckpt (ML H placer)
#   COMPLETER_CKPT completer ckpt (ML nH)
#   FRAME_DIR      dir holding frame_cache_<scaf>.pt
#   OUT            output bank dir (records land in $OUT/<scaf>/pfree_bank_<scaf>.pt)
# Optional: N (per scaffold, default 10), SCAFS, XTB_WORKERS, CUDA_VISIBLE_DEVICES, XTB_BIN
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; COMMON="$HERE/../../common"
: "${GEN_CKPT:?set GEN_CKPT}"; : "${MLHADD_CKPT:?set MLHADD_CKPT}"; : "${COMPLETER_CKPT:?set COMPLETER_CKPT}"
: "${FRAME_DIR:?set FRAME_DIR}"; : "${OUT:?set OUT}"
N="${N:-10}"
SCAFS="${SCAFS:-bootstrap3 benzene_real pyridine_real pyrimidine_real pyrazine_real furan_real thiophene_real cyclohexane_real}"
mkdir -p "$OUT"; cd "$COMMON"
for SCAF in $SCAFS; do
  GEN_CKPT="$GEN_CKPT" COMPLETER_CKPT="$COMPLETER_CKPT" \
  H_PLACER=mlhadd MLHADD_CKPT="$MLHADD_CKPT" \
  H_PRERELAX=1 XVR_ESTRAIN_TAU=2.0 AROMATIZE_RINGS=1 XVR_PFREE=1 \
  FRAME_DIR="$FRAME_DIR" XTB_WORKERS="${XTB_WORKERS:-8}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  SAVE_BANK="$OUT/$SCAF" \
  ${PYBIN:-python3} measure_pfree.py "$N" "$SCAF"
done
echo "==================== funnel table ===================="
${PYBIN:-python3} "$COMMON/funnel_stats.py" "$OUT"
