#!/usr/bin/env bash
set -euo pipefail

BASE="${DSMC_CAMPAIGN_BASE:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="${BASE}/JFM_MAXWELL_KNGU020_ENSEMBLE_${STAMP}"
REPO="$ROOT/repo"
RESULTS="$ROOT/results"
REPO_URL="${DSMC_REPO_URL:-https://github.com/Ehsan-Roohi/Fourier.git}"
GIT_REF="${DSMC_GIT_REF:-main}"
SPARTA_BIN="${SPARTA_BIN:-/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/DSMC_Python_sparta_maxwell_kngu005_020_jfm/sparta_cavity_mohammadzadeh/third_party/sparta/src/spa_mpi}"
EXPECTED_SPARTA_COMMIT="912c9e163c38ea5c3562d039e65215f6e2a4f3f8"

[[ ! -e "$ROOT" ]] || { echo "Campaign root already exists: $ROOT" >&2; exit 2; }
[[ -x "$SPARTA_BIN" ]] || { echo "SPARTA executable not found: $SPARTA_BIN" >&2; exit 3; }
[[ "$(git -C "$(dirname "$SPARTA_BIN")" rev-parse HEAD)" == "$EXPECTED_SPARTA_COMMIT" ]] || {
  echo "SPARTA source is not the locked commit $EXPECTED_SPARTA_COMMIT" >&2; exit 4;
}
mkdir -p "$ROOT" "$RESULTS" "$ROOT/logs"
git clone --depth 1 --branch "$GIT_REF" "$REPO_URL" "$REPO"
CODE_DIR="$REPO/r13-r26-cavity/code/dsmc"
for file in unity_sparta_maxwell_kngu020_ensemble.slurm \
            unity_sparta_maxwell_kngu020_ensemble_gate.slurm \
            generate_jfm_maxwell_kngu020_case.py \
            validate_jfm_maxwell_kngu020_ensemble_case.py \
            analyze_jfm_maxwell_kngu020_ensemble.py; do
  [[ -s "$CODE_DIR/$file" ]] || { echo "Missing published campaign file: $file" >&2; exit 5; }
done

cd "$ROOT"
SMOKE_JOB=""
ARRAY_JOB=""
GATE_JOB=""
cleanup_partial_submission() {
  status=$?
  set +e
  [[ -n "$GATE_JOB" ]] && scancel "$GATE_JOB"
  [[ -n "$ARRAY_JOB" ]] && scancel "$ARRAY_JOB"
  [[ -n "$SMOKE_JOB" ]] && scancel "$SMOKE_JOB"
  echo "Submission failed; cancelled partial smoke/array/gate jobs." >&2
  exit "$status"
}
trap cleanup_partial_submission ERR

SMOKE_JOB="$(sbatch --parsable \
  --array=0 --time=00:30:00 \
  --export=ALL,DSMC_REPO_ROOT="$REPO",DSMC_RESULTS_BASE="$ROOT/smoke_results",SPARTA_BIN="$SPARTA_BIN",DSMC_WARMUP=2000,DSMC_SAMPLE=2000,DSMC_STRIDE=10,DSMC_BLOCK=2000 \
  "$CODE_DIR/unity_sparta_maxwell_kngu020_ensemble.slurm")"
ARRAY_JOB="$(sbatch --parsable \
  --dependency="afterok:${SMOKE_JOB}" \
  --export=ALL,DSMC_REPO_ROOT="$REPO",DSMC_RESULTS_BASE="$RESULTS",SPARTA_BIN="$SPARTA_BIN" \
  "$CODE_DIR/unity_sparta_maxwell_kngu020_ensemble.slurm")"
GATE_JOB="$(sbatch --parsable --dependency="afterok:${ARRAY_JOB}" \
  --export=ALL,DSMC_REPO_ROOT="$REPO",DSMC_RESULTS_BASE="$RESULTS",DSMC_ARRAY_JOB="$ARRAY_JOB" \
  "$CODE_DIR/unity_sparta_maxwell_kngu020_ensemble_gate.slurm")"
trap - ERR
{
  printf 'campaign_root=%s\nresults=%s\nsmoke_job=%s\narray_job=%s\ngate_job=%s\n' \
    "$ROOT" "$RESULTS" "$SMOKE_JOB" "$ARRAY_JOB" "$GATE_JOB"
  printf 'repo_commit=%s\nsparta_commit=%s\nsubmitted_utc=%s\n' \
    "$(git -C "$REPO" rev-parse HEAD)" "$EXPECTED_SPARTA_COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_url=%s\nrepo_ref=%s\n' "$REPO_URL" "$GIT_REF"
} | tee "$ROOT/SUBMISSION.txt"
echo "SUBMITTED smoke=${SMOKE_JOB} array=${ARRAY_JOB} gate=${GATE_JOB} root=${ROOT}"
