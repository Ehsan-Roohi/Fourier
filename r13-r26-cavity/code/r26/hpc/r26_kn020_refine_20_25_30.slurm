#!/bin/bash
#SBATCH --job-name=r26-k020-n30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --qos=long
#SBATCH --output=r26-k020-n30-%j.out
#SBATCH --error=r26-k020-n30-%j.err

set -euo pipefail

: "${R26_REFINE_SEED:?Set R26_REFINE_SEED to the accepted Kn=0.20 N20 last_accepted_state.npz}"
: "${R26_REFINE_OUT:?Set R26_REFINE_OUT to a new output directory}"
: "${R26_REFINE_REF:?Set R26_REFINE_REF to the immutable 40-character Git commit SHA}"

[[ "$R26_REFINE_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_REFINE_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}

test -f "$R26_REFINE_SEED"
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

(cd "$PKG" && python3 -m unittest discover -s tests -p 'test_maxwell_contract.py')

run_stage () {
  local nodes="$1"
  local seed="$2"
  local out="$R26_REFINE_OUT/N${nodes}"
  python3 "$PKG/analysis/run_jfm_observability_continuation.py" \
    --nodes "$nodes" --case-family jfm-maxwell --kn-gu 0.20 \
    --lid-speed-m-s 100 --wall-temperature-k 300 --beta 0.0 \
    --closure-mode jfm2009 --solver colored_newton \
    --raw-tolerance 1e-8 --solver-tolerance 1e-9 --max-nfev 240 \
    --max-objective-evaluations 12000 \
    --initial-state "$seed" --reconcile-initial --output-dir "$out"
  python3 "$PKG/tools/validate_r26_maxwell_run.py" "$out" \
    --expected-kn 0.20 --expected-nodes "$nodes" --expected-beta 0.0 --raw-tolerance 1e-8
}

# Do not jump directly from N20 to N30: acceptance at N25 is a hard gate.
run_stage 25 "$R26_REFINE_SEED"
run_stage 30 "$R26_REFINE_OUT/N25/last_accepted_state.npz"
sha256sum "$R26_REFINE_OUT/N25/last_accepted_state.npz" \
          "$R26_REFINE_OUT/N30/last_accepted_state.npz" \
  > "$R26_REFINE_OUT/accepted_states.sha256"
