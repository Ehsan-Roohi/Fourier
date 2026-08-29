#!/bin/bash
# Submit one bounded THOR N29 refinement from the accepted THOR N28 root.

set -euo pipefail

: "${R26_THOR_ACCEPTED_N28_DIR:?Set R26_THOR_ACCEPTED_N28_DIR to the passed THOR N28 directory}"
: "${R26_THOR_N29_REF:?Set R26_THOR_N29_REF to the immutable 40-character commit SHA}"
[[ "$R26_THOR_N29_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_THOR_N29_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}

verify_file() {
  local path="$1"
  local expected="$2"
  local actual
  test -f "$path" || {
    echo "required accepted-N28 input is missing: $path" >&2
    return 2
  }
  actual="$(sha256sum "$path")"
  actual="${actual%% *}"
  test "$actual" = "$expected" || {
    echo "accepted-N28 input hash mismatch: $path" >&2
    return 2
  }
}

verify_file "$R26_THOR_ACCEPTED_N28_DIR/THOR_ROOT_RECONCILIATION_N28_PASSED.json" "6144476718f135a2bd9b6c2ea54da9ce6508397b0d6208bc3865f6b5d9186fd5"
verify_file "$R26_THOR_ACCEPTED_N28_DIR/THOR_N28_CROSS_SOLVER_VALIDATION.json" "ef28e067aca3a3be6a8c346554c7973819eb50186ee1664910a047763fd2a33c"
verify_file "$R26_THOR_ACCEPTED_N28_DIR/N28/thor_validation.json" "7e8669247da92932df81652d19aef08bc0c58854b3ff1e275c57413234af8ede"
verify_file "$R26_THOR_ACCEPTED_N28_DIR/N28/thor_state.npz" "21cf1a09daa3ef7a7ddca604bc508fa201dc67029fd85b8a41fe32e78947b5b0"

R26_THOR_N29_OUT="${R26_THOR_N29_OUT:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/R26_THOR_N29_FROM_ACCEPTED_N28_$(date +%Y%m%d_%H%M%S)}"
test ! -e "$R26_THOR_N29_OUT"
test ! -e "${R26_THOR_N29_OUT}_RESULTS.zip"
mkdir -p "$(dirname "$R26_THOR_N29_OUT")"

SCRIPT="$(mktemp "${TMPDIR:-/tmp}/r26-thor29.XXXXXX.slurm")"
cleanup() {
  unlink "$SCRIPT" 2>/dev/null || true
}
trap cleanup EXIT

RAW_URL="https://raw.githubusercontent.com/Ehsan-Roohi/Fourier/${R26_THOR_N29_REF}/r13-r26-cavity/code/r26/hpc/r26_kn020_thor_n29_from_accepted_n28.slurm"
curl -fsSL "$RAW_URL" -o "$SCRIPT"
bash -n "$SCRIPT"

SUBMIT_RESULT="$(sbatch \
  --output="${R26_THOR_N29_OUT}.slurm-%j.out" \
  --error="${R26_THOR_N29_OUT}.slurm-%j.err" \
  --export=ALL,R26_THOR_ACCEPTED_N28_DIR="$R26_THOR_ACCEPTED_N28_DIR",R26_THOR_N29_OUT="$R26_THOR_N29_OUT",R26_THOR_N29_REF="$R26_THOR_N29_REF" \
  "$SCRIPT")"
JOB_ID="${SUBMIT_RESULT##* }"

echo "$SUBMIT_RESULT"
echo "JOB_ID=$JOB_ID"
echo "R26_OUTPUT=$R26_THOR_N29_OUT"
echo "R26_RESULTS_ZIP=${R26_THOR_N29_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_THOR_N29_OUT}_RESULTS.zip.sha256.txt"
squeue -j "$JOB_ID" -o "%.18i %.14T %.12M %.12l %.30R" 2>/dev/null || true
