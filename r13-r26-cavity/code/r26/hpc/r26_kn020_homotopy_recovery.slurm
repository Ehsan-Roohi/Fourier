#!/bin/bash
#SBATCH --job-name=r26-homotopy
#SBATCH --partition=cpu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/r26-homotopy-%j.out
#SBATCH --error=logs/r26-homotopy-%j.err

set -euo pipefail

: "${R26_REPO_ROOT:?R26_REPO_ROOT is required}"
: "${R26_RECOVERY_ROOT:?R26_RECOVERY_ROOT is required}"
: "${R26_N28_SEED:?R26_N28_SEED is required}"
: "${R26_DELIVERABLES:?R26_DELIVERABLES is required}"

PKG="$R26_REPO_ROOT/r13-r26-cavity/code/r26"
RESULTS="$R26_RECOVERY_ROOT/results"
INITIAL_H="${R26_HOMOTOPY_INITIAL_STEP:-0.20}"
MINIMUM_H="${R26_HOMOTOPY_MINIMUM_STEP:-0.0125}"
MAX_NFEV="${R26_MAX_NFEV:-360}"
mkdir -p "$RESULTS" "$R26_DELIVERABLES"

export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

package_run() {
  status=$?
  set +e
  printf 'exit_status=%s\nfinished_utc=%s\n' "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$R26_RECOVERY_ROOT/RUN_STATUS.txt"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  zip_path="$R26_DELIVERABLES/R26_KNGU020_HOMOTOPY_JOB${SLURM_JOB_ID}_${stamp}.zip"
  (cd "$R26_RECOVERY_ROOT" && zip -qr "$zip_path" results logs SUBMISSION.txt RUN_STATUS.txt \
    repo_commit.txt repo/r13-r26-cavity/code/r26)
  (cd "$R26_DELIVERABLES" && sha256sum "$(basename "$zip_path")" \
    > "$(basename "$zip_path").sha256")
  echo "DOWNLOAD_ZIP=$zip_path"
  echo "DOWNLOAD_SHA256=${zip_path}.sha256"
  exit "$status"
}
trap package_run EXIT

test -f "$R26_N28_SEED"
python3 "$PKG/run_tests.py"
(cd "$PKG" && python3 -m unittest discover -s tests -p 'test_maxwell_contract.py' -v)
python3 "$PKG/tools/validate_r26_maxwell_run.py" "$(dirname "$R26_N28_SEED")" \
  --expected-kn 0.20 --expected-nodes 28 --raw-tolerance 1e-8

run_stage() {
  nodes="$1"
  seed="$2"
  out="$RESULTS/N${nodes}"
  echo "===== START N${nodes} FROM $seed ====="
  python3 "$PKG/analysis/run_jfm_observability_continuation.py" \
    --nodes "$nodes" --case-family jfm-maxwell --kn-gu 0.20 \
    --lid-speed-m-s 100 --wall-temperature-k 300 --beta 0.0 \
    --closure-mode jfm2009 --solver colored_newton \
    --raw-tolerance 1e-8 --solver-tolerance 1e-9 --max-nfev "$MAX_NFEV" \
    --initial-state "$seed" --reconcile-initial \
    --grid-reconciliation-homotopy \
    --homotopy-initial-step "$INITIAL_H" \
    --homotopy-minimum-step "$MINIMUM_H" --homotopy-growth 1.5 \
    --reconciliation-fallback-solver least_squares \
    --output-dir "$out"
  python3 "$PKG/tools/validate_r26_maxwell_run.py" "$out" \
    --expected-kn 0.20 --expected-nodes "$nodes" --raw-tolerance 1e-8
  echo "===== N${nodes} ACCEPTED ====="
}

run_stage 29 "$R26_N28_SEED"
run_stage 30 "$RESULTS/N29/last_accepted_state.npz"

python3 "$PKG/tools/compare_r26_grid_states.py" \
  --state "$R26_N28_SEED" --label N28 \
  --state "$RESULTS/N29/last_accepted_state.npz" --label N29 \
  --state "$RESULTS/N30/last_accepted_state.npz" --label N30 \
  --output "$RESULTS/grid_comparison_N28_N29_N30.json"

sha256sum "$R26_N28_SEED" \
  "$RESULTS/N29/last_accepted_state.npz" \
  "$RESULTS/N30/last_accepted_state.npz" \
  > "$RESULTS/accepted_states.sha256"
echo "R26_HOMOTOPY_RECOVERY_COMPLETE"
