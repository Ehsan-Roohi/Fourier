#!/bin/bash
# Submit the source-locked THOR-style N8/N16 candidate gate.

set -euo pipefail

: "${R26_THOR_REF:?Set R26_THOR_REF to the immutable 40-character commit SHA}"
[[ "$R26_THOR_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_THOR_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}

R26_THOR_OUT="${R26_THOR_OUT:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/R26_THOR_GATE_N8_N16_$(date +%Y%m%d_%H%M%S)}"
test ! -e "$R26_THOR_OUT"
test ! -e "${R26_THOR_OUT}_RESULTS.zip"
mkdir -p "$(dirname "$R26_THOR_OUT")"

SCRIPT="$(mktemp "${TMPDIR:-/tmp}/r26-thor816.XXXXXX.slurm")"
cleanup() {
  rm -f "$SCRIPT"
}
trap cleanup EXIT

RAW_URL="https://raw.githubusercontent.com/Ehsan-Roohi/Fourier/${R26_THOR_REF}/r13-r26-cavity/code/r26/hpc/r26_kn020_thor_gate_n8_n16.slurm"
curl -fsSL "$RAW_URL" -o "$SCRIPT"
bash -n "$SCRIPT"

SUBMIT_RESULT="$(sbatch \
  --output="${R26_THOR_OUT}.slurm-%j.out" \
  --error="${R26_THOR_OUT}.slurm-%j.err" \
  --export=ALL,R26_THOR_OUT="$R26_THOR_OUT",R26_THOR_REF="$R26_THOR_REF" \
  "$SCRIPT")"
JOB_ID="${SUBMIT_RESULT##* }"

echo "$SUBMIT_RESULT"
echo "JOB_ID=$JOB_ID"
echo "R26_OUTPUT=$R26_THOR_OUT"
echo "R26_RESULTS_ZIP=${R26_THOR_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_THOR_OUT}_RESULTS.zip.sha256.txt"
squeue -j "$JOB_ID" -o "%.18i %.14T %.12M %.12l %.30R"
