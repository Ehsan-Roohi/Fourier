#!/bin/bash
# Submit numerical-rank/cross-solver audit followed by one conditional N24 run.

set -euo pipefail

: "${R26_THOR_GATE_DIR:?Set R26_THOR_GATE_DIR to the passed THOR N8/N16 gate}"
: "${R26_LEGACY_GATE_DIR:?Set R26_LEGACY_GATE_DIR to the immutable SER-PTC N8/N16 gate}"
: "${R26_THOR_N24_REF:?Set R26_THOR_N24_REF to the immutable 40-character commit SHA}"
[[ "$R26_THOR_N24_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_THOR_N24_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}

R26_THOR_N24_OUT="${R26_THOR_N24_OUT:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/R26_THOR_CROSS_SOLVER_N24_$(date +%Y%m%d_%H%M%S)}"
test ! -e "$R26_THOR_N24_OUT"
test ! -e "${R26_THOR_N24_OUT}_RESULTS.zip"
mkdir -p "$(dirname "$R26_THOR_N24_OUT")"

SCRIPT="$(mktemp "${TMPDIR:-/tmp}/r26-thor24.XXXXXX.slurm")"
cleanup() {
  rm -f "$SCRIPT"
}
trap cleanup EXIT

RAW_URL="https://raw.githubusercontent.com/Ehsan-Roohi/Fourier/${R26_THOR_N24_REF}/r13-r26-cavity/code/r26/hpc/r26_kn020_thor_cross_solver_n24.slurm"
curl -fsSL "$RAW_URL" -o "$SCRIPT"
bash -n "$SCRIPT"

SUBMIT_RESULT="$(sbatch \
  --output="${R26_THOR_N24_OUT}.slurm-%j.out" \
  --error="${R26_THOR_N24_OUT}.slurm-%j.err" \
  --export=ALL,R26_THOR_GATE_DIR="$R26_THOR_GATE_DIR",R26_LEGACY_GATE_DIR="$R26_LEGACY_GATE_DIR",R26_THOR_N24_OUT="$R26_THOR_N24_OUT",R26_THOR_N24_REF="$R26_THOR_N24_REF" \
  "$SCRIPT")"
JOB_ID="${SUBMIT_RESULT##* }"

echo "$SUBMIT_RESULT"
echo "JOB_ID=$JOB_ID"
echo "R26_OUTPUT=$R26_THOR_N24_OUT"
echo "R26_RESULTS_ZIP=${R26_THOR_N24_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_THOR_N24_OUT}_RESULTS.zip.sha256.txt"
squeue -j "$JOB_ID" -o "%.18i %.14T %.12M %.12l %.30R"
