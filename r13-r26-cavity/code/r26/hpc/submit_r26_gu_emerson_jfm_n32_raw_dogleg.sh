#!/bin/bash
# Submit one bounded raw-merit dogleg rescue of the failed N32 state.

set -euo pipefail

: "${R26_GE_JFM_N32_DOGLEG_REF:?Set the immutable dogleg source commit SHA}"
[[ "$R26_GE_JFM_N32_DOGLEG_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_GE_JFM_N32_DOGLEG_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}

R26_GE_JFM_N32_FAILED_DIR="${R26_GE_JFM_N32_FAILED_DIR:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/R26_GU_EMERSON_JFM_N32_ALPHA_AWARE_PTC_20260902_135429}"
R26_GE_JFM_N32_DOGLEG_OUT="${R26_GE_JFM_N32_DOGLEG_OUT:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/R26_GU_EMERSON_JFM_N32_RAW_DOGLEG_$(date +%Y%m%d_%H%M%S)}"
test -f "$R26_GE_JFM_N32_FAILED_DIR/N32/JFM_N32_TRANSFORMED_CANDIDATE_GATE.json"
test -f "$R26_GE_JFM_N32_FAILED_DIR/N32/gu_emerson_jfm_n32_candidate.npz"
test ! -e "$R26_GE_JFM_N32_DOGLEG_OUT"
test ! -e "${R26_GE_JFM_N32_DOGLEG_OUT}_RESULTS.zip"
mkdir -p "$(dirname "$R26_GE_JFM_N32_DOGLEG_OUT")"

SCRIPT="$(mktemp "${TMPDIR:-/tmp}/r26-ge-jfm32-dogleg.XXXXXX.slurm")"
cleanup() {
  rm -f "$SCRIPT"
}
trap cleanup EXIT

RAW_URL="https://raw.githubusercontent.com/Ehsan-Roohi/Fourier/${R26_GE_JFM_N32_DOGLEG_REF}/r13-r26-cavity/code/r26/hpc/r26_gu_emerson_jfm_n32_raw_dogleg.slurm"
curl -fsSL "$RAW_URL" -o "$SCRIPT"
bash -n "$SCRIPT"

SUBMIT_RESULT="$(sbatch \
  --output="${R26_GE_JFM_N32_DOGLEG_OUT}.slurm-%j.out" \
  --error="${R26_GE_JFM_N32_DOGLEG_OUT}.slurm-%j.err" \
  --export=ALL,R26_GE_JFM_N32_DOGLEG_OUT="$R26_GE_JFM_N32_DOGLEG_OUT",R26_GE_JFM_N32_DOGLEG_REF="$R26_GE_JFM_N32_DOGLEG_REF",R26_GE_JFM_N32_FAILED_DIR="$R26_GE_JFM_N32_FAILED_DIR" \
  "$SCRIPT")"
JOB_ID="${SUBMIT_RESULT##* }"

echo "$SUBMIT_RESULT"
echo "JOB_ID=$JOB_ID"
echo "R26_OUTPUT=$R26_GE_JFM_N32_DOGLEG_OUT"
echo "R26_RESULTS_ZIP=${R26_GE_JFM_N32_DOGLEG_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_GE_JFM_N32_DOGLEG_OUT}_RESULTS.zip.sha256.txt"
squeue -j "$JOB_ID" -o "%.18i %.14T %.12M %.12l %.30R"
