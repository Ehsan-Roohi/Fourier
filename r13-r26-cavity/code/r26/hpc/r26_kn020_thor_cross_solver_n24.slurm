#!/bin/bash
#SBATCH --job-name=r26-thor24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --output=r26-thor24-%j.out
#SBATCH --error=r26-thor24-%j.err

set -euo pipefail

: "${R26_THOR_GATE_DIR:?Set R26_THOR_GATE_DIR to the passed THOR N8/N16 gate}"
: "${R26_LEGACY_GATE_DIR:?Set R26_LEGACY_GATE_DIR to the immutable SER-PTC N8/N16 gate}"
: "${R26_THOR_N24_OUT:?Set R26_THOR_N24_OUT to a new output directory}"
: "${R26_THOR_N24_REF:?Set R26_THOR_N24_REF to the immutable 40-character commit SHA}"

EXPECTED_THOR_GATE_REF="a51d7e80aefea529d26933556222bfd068d83c16"
EXPECTED_LEGACY_GATE_REF="8cbd874eea68dd475faa3f5e3fb318b49cc0c665"
[[ "$R26_THOR_N24_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_THOR_N24_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}
test -f "$R26_THOR_GATE_DIR/THOR_N8_N16_GATE_PASSED.json"
test -f "$R26_THOR_GATE_DIR/N8/thor_state.npz"
test -f "$R26_THOR_GATE_DIR/N16/thor_state.npz"
test -f "$R26_LEGACY_GATE_DIR/N16_GATE_PASSED.json"
test -f "$R26_LEGACY_GATE_DIR/N8/last_accepted_state.npz"
test -f "$R26_LEGACY_GATE_DIR/N16/last_accepted_state.npz"
test ! -e "$R26_THOR_N24_OUT"
test ! -e "${R26_THOR_N24_OUT}_RESULTS.zip"
mkdir -p "$R26_THOR_N24_OUT"

exec > >(tee -a "$R26_THOR_N24_OUT/run.stdout.log") \
     2> >(tee -a "$R26_THOR_N24_OUT/run.stderr.log" >&2)

package_on_exit() {
  local return_code=$?
  trap - EXIT
  set +e
  if (( return_code != 0 )); then
    python3 - "$R26_THOR_N24_OUT" "$R26_THOR_N24_REF" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
record = {
    "status": "R26_THOR_CROSS_SOLVER_N24_FAILED",
    "source_commit": sys.argv[2],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n24_accepted": False,
    "n28_authorized": False,
    "n30_authorized": False,
    "production_accepted": False,
}
(root / "THOR_CROSS_SOLVER_N24_FAILED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  fi
  local zip_path="${R26_THOR_N24_OUT}_RESULTS.zip"
  local sha_path="${zip_path}.sha256.txt"
  (cd "$(dirname "$R26_THOR_N24_OUT")" && zip -qr "$zip_path" "$(basename "$R26_THOR_N24_OUT")")
  (cd "$(dirname "$zip_path")" && sha256sum "$(basename "$zip_path")" > "$(basename "$sha_path")")
  exit "$return_code"
}
trap package_on_exit EXIT

REPO="$R26_THOR_N24_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_THOR_N24_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_THOR_N24_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_THOR_N24_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)

python3 "$PKG/tools/validate_r26_globalization_gate.py" \
  "$R26_LEGACY_GATE_DIR" \
  --expected-source-commit "$EXPECTED_LEGACY_GATE_REF" \
  --raw-tolerance 1e-8

python3 "$PKG/tools/audit_r26_thor_cross_solver.py" \
  --thor-dir "$R26_THOR_GATE_DIR" \
  --legacy-gate-dir "$R26_LEGACY_GATE_DIR" \
  --output "$R26_THOR_N24_OUT/THOR_CROSS_SOLVER_AUDIT.json" \
  --expected-thor-source-commit "$EXPECTED_THOR_GATE_REF" \
  --expected-legacy-source-commit "$EXPECTED_LEGACY_GATE_REF" \
  --raw-tolerance 1e-8 \
  --minimum-scaled-rcond 1e-8 \
  --maximum-profile-nrms 0.05 \
  --maximum-line-nrms 0.15 \
  --maximum-DG-relative-difference 0.02

python3 - "$R26_THOR_N24_OUT/THOR_CROSS_SOLVER_AUDIT.json" <<'PY'
import json
import pathlib
import sys

record = json.loads(pathlib.Path(sys.argv[1]).read_text())
if record.get("status") != "R26_THOR_CROSS_SOLVER_AUDIT_PASSED":
    raise SystemExit("cross-solver audit did not pass")
if record.get("n24_authorized") is not True:
    raise SystemExit("N24 was not explicitly authorized")
if record.get("n28_authorized") is not False or record.get("n30_authorized") is not False:
    raise SystemExit("cross-solver audit exceeded its authorization scope")
PY

python3 "$PKG/analysis/run_r26_thor_validation.py" \
  --case-family jfm-maxwell \
  --nodes 24 \
  --kn-gu 0.2 \
  --lid-speed-m-s 100 \
  --initial-state "$R26_THOR_GATE_DIR/N16/thor_state.npz" \
  --output-dir "$R26_THOR_N24_OUT/N24" \
  --max-iterations 100 \
  --inner-max-iterations 200 \
  --raw-tolerance 1e-8 \
  --solver-tolerance 1e-9 \
  --ilu-drop-tolerance 1e-6 \
  --ilu-fill-factor 48

python3 "$PKG/analysis/compare_r26_thor_candidates.py" \
  "$R26_THOR_GATE_DIR/N8/thor_state.npz" \
  "$R26_THOR_GATE_DIR/N16/thor_state.npz" \
  "$R26_THOR_N24_OUT/N24/thor_state.npz" \
  --output "$R26_THOR_N24_OUT/N8_N16_N24_GRID_SENSITIVITY.json"

python3 - "$R26_THOR_N24_OUT" "$ACTUAL_REF" "$EXPECTED_THOR_GATE_REF" "$EXPECTED_LEGACY_GATE_REF" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
n24 = json.loads((root / "N24" / "thor_validation.json").read_text())
audit = json.loads((root / "THOR_CROSS_SOLVER_AUDIT.json").read_text())
if n24.get("status") != "R26_THOR_VALIDATION_CANDIDATE_PASSED":
    raise SystemExit("N24 THOR candidate did not pass")
if n24.get("physical_candidate_gate_passed") is not True:
    raise SystemExit("N24 physical gate is not boolean true")
if n24.get("production_accepted") is not False:
    raise SystemExit("N24 incorrectly made a production claim")
if float(n24.get("raw_acceptance_gate", float("inf"))) > 1.0e-8:
    raise SystemExit("N24 raw gate failed")
if audit.get("n24_authorized") is not True:
    raise SystemExit("N24 authorization disappeared")
record = {
    "status": "R26_THOR_CROSS_SOLVER_N24_PASSED",
    "source_commit": sys.argv[2],
    "thor_gate_source_commit": sys.argv[3],
    "legacy_gate_source_commit": sys.argv[4],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n24_accepted": True,
    "n24_state_sha256": n24["state_sha256"],
    "n24_raw_gate": n24["raw_acceptance_gate"],
    "cross_solver_audit": str((root / "THOR_CROSS_SOLVER_AUDIT.json").resolve()),
    "grid_sensitivity_record": str((root / "N8_N16_N24_GRID_SENSITIVITY.json").resolve()),
    "n28_authorized": False,
    "n30_authorized": False,
    "production_accepted": False,
    "next_required_stage": "compare N24 with immutable accepted N25/N27/N28 roots before any N28 run",
}
(root / "THOR_CROSS_SOLVER_N24_PASSED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "R26_THOR_CROSS_SOLVER_N24_COMPLETE"
echo "R26_RESULTS_ZIP=${R26_THOR_N24_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_THOR_N24_OUT}_RESULTS.zip.sha256.txt"
