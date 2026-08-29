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

require_file() {
  local path="$1"
  test -f "$path" || {
    echo "required input is missing: $path" >&2
    return 2
  }
}

verify_root() {
  local path="$1"
  local expected="$2"
  local actual
  require_file "$path"
  actual="$(sha256sum "$path")"
  actual="${actual%% *}"
  test "$actual" = "$expected" || {
    echo "immutable root hash mismatch: $path" >&2
    return 2
  }
}

require_file "$R26_THOR_N24_DIR/THOR_CROSS_SOLVER_N24_PASSED.json"
require_file "$R26_THOR_N24_DIR/N8_N16_N24_GRID_SENSITIVITY.json"
for directory in "$R26_LEGACY_N25_DIR" "$R26_LEGACY_N27_DIR" "$R26_LEGACY_N28_DIR"; do
  require_file "$directory/run_summary.json"
done
verify_root "$R26_THOR_N24_DIR/N24/thor_state.npz" "94924cbf73d367418f32042f25e80fe1fc5d84aa572672776ef1728ed612d411"
verify_root "$R26_LEGACY_N25_DIR/last_accepted_state.npz" "1ab75d87bc21f37d5658d8c2d728eb909ce758bada0f205287beb7d6f2f18a13"
verify_root "$R26_LEGACY_N27_DIR/last_accepted_state.npz" "b64fcadcf8e7c0e17e1f07df348f5cf619f152662ff606024e11404c79010905"
verify_root "$R26_LEGACY_N28_DIR/last_accepted_state.npz" "a28951fed0063f66dac5e0ec481a108f9a4599fdcfd83d6865b2a160ffa4a409"

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
