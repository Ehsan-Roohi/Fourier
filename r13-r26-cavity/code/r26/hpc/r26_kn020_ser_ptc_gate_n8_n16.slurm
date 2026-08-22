#!/bin/bash
#SBATCH --job-name=r26-k020-g816
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --qos=long
#SBATCH --output=r26-k020-g816-%j.out
#SBATCH --error=r26-k020-g816-%j.err

set -euo pipefail

: "${R26_GATE_OUT:?Set R26_GATE_OUT to a new output directory}"
: "${R26_GATE_REF:?Set R26_GATE_REF to the immutable 40-character Git commit SHA}"

[[ "$R26_GATE_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_GATE_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}
test ! -e "$R26_GATE_OUT"
mkdir -p "$R26_GATE_OUT"

REPO="$R26_GATE_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_GATE_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_GATE_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_GATE_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)

run_gate() {
  local nodes="$1"
  local output="$R26_GATE_OUT/N${nodes}"
  python3 "$PKG/analysis/run_jfm_observability_continuation.py" \
    --nodes "$nodes" --case-family jfm-maxwell --kn-gu 0.20 \
    --lid-speed-m-s 100 --wall-temperature-k 300 --beta 0.0 \
    --closure-mode jfm2009 --solver colored_newton \
    --analytic-mass-jacobian --secant-predictor --ser-ptc \
    --pseudo-time-initial 1.0 --pseudo-time-minimum 1e-8 \
    --pseudo-time-maximum 1e8 --newton-switch-tolerance 1e-6 \
    --max-jacobians 5 --raw-tolerance 1e-8 --solver-tolerance 1e-9 \
    --smoke-lid 0.001 --initial-step 0.04 --minimum-step 0.0025 \
    --max-nfev 80 --max-objective-evaluations 4000 \
    --output-dir "$output"
  python3 "$PKG/tools/validate_r26_maxwell_run.py" "$output" \
    --expected-kn 0.20 --expected-nodes "$nodes" --expected-beta 0.0 \
    --raw-tolerance 1e-8 --require-modern-globalization
}

# Both grids start from their own analytic equilibrium.  N16 is launched only
# after the complete N8 run and its independent validator return success.
run_gate 8
run_gate 16

python3 - "$R26_GATE_OUT" "$ACTUAL_REF" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
commit = sys.argv[2]
record = {
    "status": "R26_N16_GATE_PASSED",
    "source_commit": commit,
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n8_summary": str((root / "N8" / "run_summary.json").resolve()),
    "n16_summary": str((root / "N16" / "run_summary.json").resolve()),
    "n30_authorized": False,
    "note": "This record validates only the N8/N16 globalization gate; no N30 launcher is present in this commit.",
}
(root / "N16_GATE_PASSED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

sha256sum \
  "$R26_GATE_OUT/N8/last_accepted_state.npz" \
  "$R26_GATE_OUT/N8/run_summary.json" \
  "$R26_GATE_OUT/N16/last_accepted_state.npz" \
  "$R26_GATE_OUT/N16/run_summary.json" \
  "$R26_GATE_OUT/N16_GATE_PASSED.json" \
  > "$R26_GATE_OUT/N8_N16_GATE.sha256"
