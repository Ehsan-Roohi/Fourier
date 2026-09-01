#!/bin/bash
# Submit the bounded raw-guarded, DAE-aware PTC growth rescue of failed N32.

set -euo pipefail

: "${R26_GE_JFM_N32_PTC_REF:?Set the immutable rescue source commit SHA}"
[[ "$R26_GE_JFM_N32_PTC_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_GE_JFM_N32_PTC_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}

R26_GE_JFM_N32_FAILED_DIR="${R26_GE_JFM_N32_FAILED_DIR:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/R26_GU_EMERSON_JFM_N32_20260831_213708}"
R26_GE_JFM_N32_PTC_OUT="${R26_GE_JFM_N32_PTC_OUT:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/R26_GU_EMERSON_JFM_N32_RAW_GUARDED_PTC_GROWTH_$(date +%Y%m%d_%H%M%S)}"
test -f "$R26_GE_JFM_N32_FAILED_DIR/N32/JFM_N32_TRANSFORMED_CANDIDATE_GATE.json"
test -f "$R26_GE_JFM_N32_FAILED_DIR/N32/gu_emerson_jfm_n32_candidate.npz"
test ! -e "$R26_GE_JFM_N32_PTC_OUT"
test ! -e "${R26_GE_JFM_N32_PTC_OUT}_RESULTS.zip"
mkdir -p "$(dirname "$R26_GE_JFM_N32_PTC_OUT")"

SCRIPT="$(mktemp "${TMPDIR:-/tmp}/r26-ge-jfm32-ptc.XXXXXX.slurm")"
cleanup() {
  rm -f "$SCRIPT"
}
trap cleanup EXIT

RAW_URL="https://raw.githubusercontent.com/Ehsan-Roohi/Fourier/${R26_GE_JFM_N32_PTC_REF}/r13-r26-cavity/code/r26/hpc/r26_gu_emerson_jfm_n32_ptc_rescue.slurm"
curl -fsSL "$RAW_URL" -o "$SCRIPT"
bash -n "$SCRIPT"

SUBMIT_RESULT="$(sbatch \
  --output="${R26_GE_JFM_N32_PTC_OUT}.slurm-%j.out" \
  --error="${R26_GE_JFM_N32_PTC_OUT}.slurm-%j.err" \
  --export=ALL,R26_GE_JFM_N32_PTC_OUT="$R26_GE_JFM_N32_PTC_OUT",R26_GE_JFM_N32_PTC_REF="$R26_GE_JFM_N32_PTC_REF",R26_GE_JFM_N32_FAILED_DIR="$R26_GE_JFM_N32_FAILED_DIR" \
  "$SCRIPT")"
JOB_ID="${SUBMIT_RESULT##* }"

echo "$SUBMIT_RESULT"
echo "JOB_ID=$JOB_ID"
echo "R26_OUTPUT=$R26_GE_JFM_N32_PTC_OUT"
echo "R26_RESULTS_ZIP=${R26_GE_JFM_N32_PTC_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_GE_JFM_N32_PTC_OUT}_RESULTS.zip.sha256.txt"
squeue -j "$JOB_ID" -o "%.18i %.14T %.12M %.12l %.30R"
