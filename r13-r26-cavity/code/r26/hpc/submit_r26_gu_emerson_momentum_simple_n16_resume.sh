#!/bin/bash
set -euo pipefail

: "${R26_GE_AP16_REF:?Set R26_GE_AP16_REF to an immutable 40-character commit SHA}"
: "${R26_GE_FAILED_STANDALONE_DIR:?Set R26_GE_FAILED_STANDALONE_DIR to Job 63721694 output}"
: "${R26_GE_FAILED_SOURCE16_DIR:?Set R26_GE_FAILED_SOURCE16_DIR to Job 63725331 output}"
[[ "$R26_GE_AP16_REF" =~ ^[0-9a-f]{40}$ ]] || exit 2
test -d "$R26_GE_FAILED_STANDALONE_DIR"
test -d "$R26_GE_FAILED_SOURCE16_DIR"

R26_GE_AP16_OUT="${R26_GE_AP16_OUT:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/R26_GU_EMERSON_MOMENTUM_SIMPLE_N16_RESUME_$(date +%Y%m%d_%H%M%S)}"
test ! -e "$R26_GE_AP16_OUT"
test ! -e "${R26_GE_AP16_OUT}_RESULTS.zip"

SCRIPT="$(mktemp "${TMPDIR:-/tmp}/r26-ge-ap16.XXXXXX.slurm")"
trap 'rm -f "$SCRIPT"' EXIT
RAW_URL="https://raw.githubusercontent.com/Ehsan-Roohi/Fourier/${R26_GE_AP16_REF}/r13-r26-cavity/code/r26/hpc/r26_gu_emerson_momentum_simple_n16_resume.slurm"
curl -fsSL "$RAW_URL" -o "$SCRIPT"
bash -n "$SCRIPT"

JOB_ID="$(sbatch --parsable \
  --output="${R26_GE_AP16_OUT}.slurm-%j.out" \
  --error="${R26_GE_AP16_OUT}.slurm-%j.err" \
  --export=ALL,R26_GE_AP16_OUT="$R26_GE_AP16_OUT",R26_GE_AP16_REF="$R26_GE_AP16_REF",R26_GE_FAILED_STANDALONE_DIR="$R26_GE_FAILED_STANDALONE_DIR",R26_GE_FAILED_SOURCE16_DIR="$R26_GE_FAILED_SOURCE16_DIR" \
  "$SCRIPT")"

echo "JOB_ID=$JOB_ID"
echo "R26_OUTPUT=$R26_GE_AP16_OUT"
echo "R26_RESULTS_ZIP=${R26_GE_AP16_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_GE_AP16_OUT}_RESULTS.zip.sha256.txt"
squeue -j "$JOB_ID" -o "%.18i %.14T %.12M %.12l %.30R" 2>/dev/null || true
