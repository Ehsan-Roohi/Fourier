#!/bin/bash
# Submit immutable-root reconciliation followed by one conditional THOR N28 run.

set -euo pipefail

: "${R26_THOR_N24_DIR:?Set R26_THOR_N24_DIR to the passed THOR N24 directory}"
: "${R26_LEGACY_N25_DIR:?Set R26_LEGACY_N25_DIR to the immutable accepted N25 directory}"
: "${R26_LEGACY_N27_DIR:?Set R26_LEGACY_N27_DIR to the immutable accepted N27 directory}"
: "${R26_LEGACY_N28_DIR:?Set R26_LEGACY_N28_DIR to the immutable accepted N28 directory}"
: "${R26_THOR_N28_REF:?Set R26_THOR_N28_REF to the immutable 40-character commit SHA}"
[[ "$R26_THOR_N28_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_THOR_N28_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}

R26_THOR_N28_OUT="${R26_THOR_N28_OUT:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/R26_THOR_ROOT_RECONCILIATION_N28_$(date +%Y%m%d_%H%M%S)}"
test ! -e "$R26_THOR_N28_OUT"
test ! -e "${R26_THOR_N28_OUT}_RESULTS.zip"
mkdir -p "$(dirname "$R26_THOR_N28_OUT")"

SCRIPT="$(mktemp "${TMPDIR:-/tmp}/r26-thor28.XXXXXX.slurm")"
cleanup() {
  unlink "$SCRIPT" 2>/dev/null || true
}
trap cleanup EXIT

RAW_URL="https://raw.githubusercontent.com/Ehsan-Roohi/Fourier/${R26_THOR_N28_REF}/r13-r26-cavity/code/r26/hpc/r26_kn020_thor_root_reconciliation_n28.slurm"
curl -fsSL "$RAW_URL" -o "$SCRIPT"
bash -n "$SCRIPT"

SUBMIT_RESULT="$(sbatch \
  --output="${R26_THOR_N28_OUT}.slurm-%j.out" \
  --error="${R26_THOR_N28_OUT}.slurm-%j.err" \
  --export=ALL,R26_THOR_N24_DIR="$R26_THOR_N24_DIR",R26_LEGACY_N25_DIR="$R26_LEGACY_N25_DIR",R26_LEGACY_N27_DIR="$R26_LEGACY_N27_DIR",R26_LEGACY_N28_DIR="$R26_LEGACY_N28_DIR",R26_THOR_N28_OUT="$R26_THOR_N28_OUT",R26_THOR_N28_REF="$R26_THOR_N28_REF" \
  "$SCRIPT")"
JOB_ID="${SUBMIT_RESULT##* }"

echo "$SUBMIT_RESULT"
echo "JOB_ID=$JOB_ID"
echo "R26_OUTPUT=$R26_THOR_N28_OUT"
echo "R26_RESULTS_ZIP=${R26_THOR_N28_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_THOR_N28_OUT}_RESULTS.zip.sha256.txt"
squeue -j "$JOB_ID" -o "%.18i %.14T %.12M %.12l %.30R" 2>/dev/null || true
