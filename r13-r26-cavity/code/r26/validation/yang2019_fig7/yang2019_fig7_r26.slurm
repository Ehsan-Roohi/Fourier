#!/usr/bin/env bash
#SBATCH --job-name=r26-yang19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --qos=long
#SBATCH --output=r26-yang19-%j.out
#SBATCH --error=r26-yang19-%j.err

set -Eeuo pipefail

: "${YANG_ROOT:?YANG_ROOT is required}"
: "${YANG_REPO:?YANG_REPO is required}"

PKG="$YANG_REPO/r13-r26-cavity/code/r26"
VALIDATION="$PKG/validation/yang2019_fig7"
RUNS="$YANG_ROOT/runs"
REPORT="$YANG_ROOT/report"
mkdir -p "$RUNS" "$REPORT" "$YANG_ROOT/deliverables"

export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$YANG_ROOT/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

package_results() {
  rc=$?
  trap - EXIT
  set +e
  status=FAIL
  [[ "$rc" -eq 0 ]] && status=PASS
  printf 'exit_code=%s\nstatus=%s\njob_id=%s\nfinished_utc=%s\n' \
    "$rc" "$status" "${SLURM_JOB_ID:-unknown}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$YANG_ROOT/JOB_STATUS.txt"
  find "$RUNS" -type f \( -name 'last_accepted_state.npz' -o -name 'run_summary.json' \) \
    -print0 | sort -z | xargs -0 -r sha256sum > "$YANG_ROOT/accepted_outputs.sha256"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  zip_path="$YANG_ROOT/deliverables/YANG2019_FIG7_R26_JOB${SLURM_JOB_ID:-unknown}_${stamp}.zip"
  (
    cd "$YANG_ROOT"
    zip -q -r "$zip_path" JOB_STATUS.txt SOURCE_COMMIT.txt accepted_outputs.sha256 report \
      runs/N17/last_accepted_state.npz runs/N17/run_summary.json \
      runs/N21/last_accepted_state.npz runs/N21/run_summary.json \
      runs/N25/last_accepted_state.npz runs/N25/run_summary.json 2>/dev/null
  )
  sha256sum "$zip_path" > "${zip_path}.sha256"
  printf 'VALIDATION_STATUS=%s\nZIP=%s\nSHA256=%s\n' "$status" "$zip_path" "${zip_path}.sha256"
  exit "$rc"
}
trap package_results EXIT

git -C "$YANG_REPO" rev-parse HEAD | tee "$YANG_ROOT/SOURCE_COMMIT.txt"
python3 -m unittest discover -s "$PKG/tests" -p 'test_maxwell_contract.py'
python3 -m py_compile \
  "$VALIDATION/validate_yang2019_run.py" \
  "$VALIDATION/score_yang2019_fig7.py" \
  "$VALIDATION/extract_yang2019_fig7_vector.py"

run_grid() {
  nodes="$1"
  seed="${2:-}"
  output="$RUNS/N${nodes}"
  args=(
    --nodes "$nodes" --case-family jfm-maxwell --kn-gu 0.10
    --lid-speed-m-s 10 --wall-temperature-k 273 --beta 1.6
    --closure-mode jfm2009 --solver colored_newton
    --raw-tolerance 1e-8 --solver-tolerance 1e-9 --max-nfev 220
    --initial-step 0.01 --minimum-step 0.00015625 --output-dir "$output"
  )
  if [[ -n "$seed" ]]; then
    args+=(--initial-state "$seed" --reconcile-initial)
  fi
  python3 "$PKG/analysis/run_jfm_observability_continuation.py" "${args[@]}"
  python3 "$VALIDATION/validate_yang2019_run.py" "$output" \
    --expected-nodes "$nodes" --raw-tolerance 1e-8
}

# A ladder avoids the destabilising direct coarse-to-fine jump observed in the
# Kn=0.20 study.  Each accepted state is the only seed permitted downstream.
run_grid 17
run_grid 21 "$RUNS/N17/last_accepted_state.npz"
run_grid 25 "$RUNS/N21/last_accepted_state.npz"

python3 "$VALIDATION/score_yang2019_fig7.py" \
  --state "N17=$RUNS/N17/last_accepted_state.npz" \
  --state "N21=$RUNS/N21/last_accepted_state.npz" \
  --state "N25=$RUNS/N25/last_accepted_state.npz" \
  --output-dir "$REPORT"
