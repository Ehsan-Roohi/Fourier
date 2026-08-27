#!/bin/bash
# Submit the source-locked, non-production Gu--Emerson published-method gate.

set -euo pipefail

: "${R26_GE_PUBLISHED_REF:?Set R26_GE_PUBLISHED_REF to the immutable 40-character commit SHA}"
[[ "$R26_GE_PUBLISHED_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_GE_PUBLISHED_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}

R26_GE_PUBLISHED_OUT="${R26_GE_PUBLISHED_OUT:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/R26_GU_EMERSON_PUBLISHED_GATE_N8_N16_$(date +%Y%m%d_%H%M%S)}"
test ! -e "$R26_GE_PUBLISHED_OUT"
test ! -e "${R26_GE_PUBLISHED_OUT}_RESULTS.zip"
mkdir -p "$(dirname "$R26_GE_PUBLISHED_OUT")"

SCRIPT="$(mktemp "${TMPDIR:-/tmp}/r26-ge-pub816.XXXXXX.slurm")"
cleanup() { rm -f "$SCRIPT"; }
trap cleanup EXIT

RAW_URL="https://raw.githubusercontent.com/Ehsan-Roohi/Fourier/${R26_GE_PUBLISHED_REF}/r13-r26-cavity/code/r26/hpc/r26_gu_emerson_published_gate_n8_n16.slurm"
curl -fsSL "$RAW_URL" -o "$SCRIPT"
bash -n "$SCRIPT"

SUBMIT_RESULT="$(sbatch \
  --output="${R26_GE_PUBLISHED_OUT}.slurm-%j.out" \
  --error="${R26_GE_PUBLISHED_OUT}.slurm-%j.err" \
  --export=ALL,R26_GE_PUBLISHED_OUT="$R26_GE_PUBLISHED_OUT",R26_GE_PUBLISHED_REF="$R26_GE_PUBLISHED_REF" \
  "$SCRIPT")"
JOB_ID="${SUBMIT_RESULT##* }"

echo "$SUBMIT_RESULT"
echo "JOB_ID=$JOB_ID"
echo "R26_OUTPUT=$R26_GE_PUBLISHED_OUT"
echo "R26_RESULTS_ZIP=${R26_GE_PUBLISHED_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_GE_PUBLISHED_OUT}_RESULTS.zip.sha256.txt"
squeue -j "$JOB_ID" -o "%.18i %.14T %.12M %.12l %.30R"
