#!/bin/bash
#SBATCH --job-name=r26-k020-n32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --qos=long
#SBATCH --output=r26-k020-n32-%j.out
#SBATCH --error=r26-k020-n32-%j.err

set -euo pipefail

: "${R26_N28_DIR:?Set R26_N28_DIR to the accepted Kn=0.20 N28 run directory}"
: "${R26_REFINE_OUT:?Set R26_REFINE_OUT to a new output directory}"
: "${R26_REFINE_REF:?Set R26_REFINE_REF to the immutable 40-character Git commit SHA}"

[[ "$R26_REFINE_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_REFINE_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}

test -f "$R26_N28_DIR/last_accepted_state.npz"
test -f "$R26_N28_DIR/run_summary.json"
test ! -e "$R26_REFINE_OUT"
mkdir -p "$R26_REFINE_OUT"

REPO="$R26_REFINE_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_REFINE_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_REFINE_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_REFINE_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)
python3 "$PKG/tools/validate_r26_maxwell_run.py" "$R26_N28_DIR" \
  --expected-kn 0.20 --expected-nodes 28 --expected-beta 0.0 --raw-tolerance 1e-8

run_stage () {
  local nodes="$1"
  local seed="$2"
  local out="$R26_REFINE_OUT/N${nodes}"
  python3 "$PKG/analysis/run_jfm_observability_continuation.py" \
    --nodes "$nodes" --case-family jfm-maxwell --kn-gu 0.20 \
    --lid-speed-m-s 100 --wall-temperature-k 300 --beta 0.0 \
    --closure-mode jfm2009 --solver colored_newton \
    --raw-tolerance 1e-8 --solver-tolerance 1e-9 --max-nfev 48 \
    --max-objective-evaluations 12000 \
    --initial-state "$seed" --reconcile-initial --output-dir "$out"
  python3 "$PKG/tools/validate_r26_maxwell_run.py" "$out" \
    --expected-kn 0.20 --expected-nodes "$nodes" --expected-beta 0.0 --raw-tolerance 1e-8
}

# N29 previously stalled.  Move from the accepted even N28 grid to N30, then
# promote N30 to N32 only after strict validation.  Same parity is a numerical
# risk-reduction choice, not a claim that odd grids are physically invalid.
run_stage 30 "$R26_N28_DIR/last_accepted_state.npz"
run_stage 32 "$R26_REFINE_OUT/N30/last_accepted_state.npz"

sha256sum "$R26_N28_DIR/last_accepted_state.npz" \
          "$R26_REFINE_OUT/N30/last_accepted_state.npz" \
          "$R26_REFINE_OUT/N32/last_accepted_state.npz" \
  > "$R26_REFINE_OUT/accepted_states.sha256"
