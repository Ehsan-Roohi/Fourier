#!/bin/bash
#SBATCH --job-name=r26-k020-fn30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --qos=long
#SBATCH --output=r26-k020-fn30-%j.out
#SBATCH --error=r26-k020-fn30-%j.err

set -euo pipefail

: "${R26_FRESH_N30_OUT:?Set R26_FRESH_N30_OUT to a new output directory}"
: "${R26_FRESH_N30_REF:?Set R26_FRESH_N30_REF to the immutable 40-character Git commit SHA}"

[[ "$R26_FRESH_N30_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_FRESH_N30_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}

test ! -e "$R26_FRESH_N30_OUT"
mkdir -p "$R26_FRESH_N30_OUT"

REPO="$R26_FRESH_N30_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_FRESH_N30_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_FRESH_N30_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_FRESH_N30_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)

# Start on the N30 grid from its analytic equilibrium.  In particular, do not
# interpolate or reconcile the accepted N28 state: the failed JOB63408741 was
# globally improved but remained outside the easy Newton basin in the
# top-left fourth-order corner layer.  Fixed-grid lid continuation lets that
# layer form incrementally without changing the equations or acceptance gate.
python3 "$PKG/analysis/run_jfm_observability_continuation.py" \
  --nodes 30 --case-family jfm-maxwell --kn-gu 0.20 \
  --lid-speed-m-s 100 --wall-temperature-k 300 --beta 0.0 \
  --closure-mode jfm2009 --solver colored_newton \
  --raw-tolerance 1e-8 --solver-tolerance 1e-9 \
  --smoke-lid 0.001 --initial-step 0.02 --minimum-step 0.0025 \
  --max-nfev 48 --max-objective-evaluations 12000 \
  --output-dir "$R26_FRESH_N30_OUT/N30"

python3 "$PKG/tools/validate_r26_maxwell_run.py" "$R26_FRESH_N30_OUT/N30" \
  --expected-kn 0.20 --expected-nodes 30 --expected-beta 0.0 --raw-tolerance 1e-8

sha256sum "$R26_FRESH_N30_OUT/N30/last_accepted_state.npz" \
          "$R26_FRESH_N30_OUT/N30/run_summary.json" \
  > "$R26_FRESH_N30_OUT/accepted_n30.sha256"
