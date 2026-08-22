#!/bin/bash
# Submit the gated fresh-equilibrium N30 production run.

set -euo pipefail

: "${R26_N16_GATE_DIR:?Set R26_N16_GATE_DIR to the passed N8/N16 gate directory}"
: "${R26_N30_REF:?Set R26_N30_REF to the immutable 40-character N30 Git commit SHA}"

[[ "$R26_N30_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_N30_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}
test -d "$R26_N16_GATE_DIR"
test -f "$R26_N16_GATE_DIR/N16_GATE_PASSED.json"

R26_N30_OUT="${R26_N30_OUT:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/R26_SER_PTC_FRESH_N30_$(date +%Y%m%d_%H%M%S)}"
test ! -e "$R26_N30_OUT"
test ! -e "${R26_N30_OUT}_RESULTS.zip"
mkdir -p "$(dirname "$R26_N30_OUT")"

SCRIPT="$(mktemp "${TMPDIR:-/tmp}/r26-n30-submit.XXXXXX.slurm")"
cleanup() {
  rm -f "$SCRIPT"
}
trap cleanup EXIT

RAW_URL="https://raw.githubusercontent.com/Ehsan-Roohi/Fourier/${R26_N30_REF}/r13-r26-cavity/code/r26/hpc/r26_kn020_ser_ptc_fresh_n30.slurm"
curl -fsSL "$RAW_URL" -o "$SCRIPT"
bash -n "$SCRIPT"

SUBMIT_RESULT="$(sbatch \
  --output="${R26_N30_OUT}.slurm-%j.out" \
  --error="${R26_N30_OUT}.slurm-%j.err" \
  --export=ALL,R26_N16_GATE_DIR="$R26_N16_GATE_DIR",R26_N30_OUT="$R26_N30_OUT",R26_N30_REF="$R26_N30_REF" \
  "$SCRIPT")"
JOB_ID="${SUBMIT_RESULT##* }"

echo "$SUBMIT_RESULT"
echo "JOB_ID=$JOB_ID"
echo "R26_OUTPUT=$R26_N30_OUT"
echo "R26_RESULTS_ZIP=${R26_N30_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_N30_OUT}_RESULTS.zip.sha256.txt"
squeue -j "$JOB_ID" -o "%.18i %.14T %.12M %.12l %.30R"
