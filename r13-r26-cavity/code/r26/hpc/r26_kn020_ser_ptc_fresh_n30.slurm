#!/bin/bash
#SBATCH --job-name=r26-k020-spt30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --qos=long
#SBATCH --output=r26-k020-spt30-%j.out
#SBATCH --error=r26-k020-spt30-%j.err

set -euo pipefail

: "${R26_N16_GATE_DIR:?Set R26_N16_GATE_DIR to the passed N8/N16 gate directory}"
: "${R26_N30_OUT:?Set R26_N30_OUT to a new output directory}"
: "${R26_N30_REF:?Set R26_N30_REF to the immutable 40-character N30 Git commit SHA}"

EXPECTED_GATE_REF="8cbd874eea68dd475faa3f5e3fb318b49cc0c665"

[[ "$R26_N30_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_N30_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}
test -d "$R26_N16_GATE_DIR"
test -f "$R26_N16_GATE_DIR/N16_GATE_PASSED.json"
test -f "$R26_N16_GATE_DIR/N8_N16_GATE.sha256"
test ! -e "$R26_N30_OUT"
test ! -e "${R26_N30_OUT}_RESULTS.zip"
mkdir -p "$R26_N30_OUT"

# Keep complete logs inside the portable results archive.  Scheduler logs are
# still written to the paths supplied to sbatch by the submit helper.
exec > >(tee -a "$R26_N30_OUT/run.stdout.log") \
     2> >(tee -a "$R26_N30_OUT/run.stderr.log" >&2)

package_on_exit() {
  local return_code=$?
  local final_code=$return_code
  trap - EXIT
  set +e
  if (( return_code != 0 )); then
    python3 - "$R26_N30_OUT" "$R26_N30_REF" "$EXPECTED_GATE_REF" "$return_code" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
record = {
    "status": "R26_N30_PRODUCTION_FAILED",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_commit": sys.argv[2],
    "validated_gate_commit": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "note": "A failed archive is diagnostic only and contains no accepted N30 claim.",
}
(root / "N30_PRODUCTION_FAILED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
  fi

  python3 - "$R26_N30_OUT" "${R26_N30_OUT}_RESULTS.zip" <<'PY'
import pathlib
import sys
import zipfile

root = pathlib.Path(sys.argv[1]).resolve()
archive = pathlib.Path(sys.argv[2]).resolve()
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_dir() or ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        handle.write(path, pathlib.Path(root.name) / relative)
PY
  if (( $? != 0 )) && (( final_code == 0 )); then
    final_code=70
  fi
  (
    cd "$(dirname "$R26_N30_OUT")"
    sha256sum "$(basename "${R26_N30_OUT}_RESULTS.zip")" \
      > "$(basename "${R26_N30_OUT}_RESULTS.zip").sha256.txt"
  )
  if (( $? != 0 )) && (( final_code == 0 )); then
    final_code=71
  fi
  echo "R26_RESULTS_ZIP=${R26_N30_OUT}_RESULTS.zip"
  echo "R26_RESULTS_SHA=${R26_N30_OUT}_RESULTS.zip.sha256.txt"
  exit "$final_code"
}
trap package_on_exit EXIT

REPO="$R26_N30_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_N30_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_N30_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_N30_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)

# The production commit authorizes N30 only after revalidating the immutable
# N8/N16 gate.  The gate record itself correctly says n30_authorized=false:
# that older commit contained no N30 launcher.  Authorization is this separate
# reviewed commit plus the following strict gate checks.
python3 "$PKG/tools/validate_r26_globalization_gate.py" \
  "$R26_N16_GATE_DIR" \
  --expected-source-commit "$EXPECTED_GATE_REF" \
  --raw-tolerance 1e-8
for nodes in 8 16; do
  python3 "$PKG/tools/validate_r26_maxwell_run.py" \
    "$R26_N16_GATE_DIR/N${nodes}" \
    --expected-kn 0.20 --expected-nodes "$nodes" --expected-beta 0.0 \
    --raw-tolerance 1e-8 --require-modern-globalization
done

# Fresh fixed-grid continuation only: no N16/N28 interpolation, restart, or
# reconciliation.  The accepted N8/N16 settings are carried over unchanged.
python3 "$PKG/analysis/run_jfm_observability_continuation.py" \
  --nodes 30 --case-family jfm-maxwell --kn-gu 0.20 \
  --lid-speed-m-s 100 --wall-temperature-k 300 --beta 0.0 \
  --closure-mode jfm2009 --solver colored_newton \
  --analytic-mass-jacobian --secant-predictor --ser-ptc \
  --pseudo-time-initial 1.0 --pseudo-time-minimum 1e-8 \
  --pseudo-time-maximum 1e8 --newton-switch-tolerance 1e-6 \
  --max-jacobians 5 --raw-tolerance 1e-8 --solver-tolerance 1e-9 \
  --smoke-lid 0.001 --initial-step 0.04 --minimum-step 0.0025 \
  --max-nfev 80 --max-objective-evaluations 4000 \
  --output-dir "$R26_N30_OUT/N30"

python3 "$PKG/tools/validate_r26_maxwell_run.py" "$R26_N30_OUT/N30" \
  --expected-kn 0.20 --expected-nodes 30 --expected-beta 0.0 \
  --raw-tolerance 1e-8 --require-modern-globalization

python3 - "$R26_N30_OUT" "$ACTUAL_REF" "$EXPECTED_GATE_REF" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "N30" / "run_summary.json").read_text(encoding="utf-8"))
attempts = summary["attempts"]
record = {
    "status": "R26_N30_PRODUCTION_PASSED",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_commit": sys.argv[2],
    "validated_gate_commit": sys.argv[3],
    "nodes": 30,
    "kn_gu": 0.20,
    "target_lid": summary["case"]["lid_target"],
    "attempts": len(attempts),
    "accepted_attempts": sum(bool(row["accepted"]) for row in attempts),
    "rejected_attempts": sum(not bool(row["accepted"]) for row in attempts),
    "final_raw_gate": attempts[-1]["raw_acceptance_gate"],
    "maximum_jacobians_per_attempt": max(
        int(row["solver"]["jacobian_evaluations"]) for row in attempts
    ),
    "note": "Algebraically accepted N30 prediction; grid convergence and external validation remain separate requirements.",
}
(root / "N30_PRODUCTION_PASSED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

(
  cd "$R26_N30_OUT"
  sha256sum \
    N30/last_accepted_state.npz \
    N30/run_summary.json \
    N30_PRODUCTION_PASSED.json \
    source_commit.txt \
    > N30_PRODUCTION.sha256
)
