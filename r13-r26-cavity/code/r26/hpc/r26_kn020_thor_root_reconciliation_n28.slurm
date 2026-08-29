#!/bin/bash
#SBATCH --job-name=r26-thor28
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --output=r26-thor28-%j.out
#SBATCH --error=r26-thor28-%j.err

set -euo pipefail

: "${R26_THOR_N24_DIR:?Set R26_THOR_N24_DIR to the passed THOR N24 directory}"
: "${R26_LEGACY_N25_DIR:?Set R26_LEGACY_N25_DIR to the immutable accepted N25 directory}"
: "${R26_LEGACY_N27_DIR:?Set R26_LEGACY_N27_DIR to the immutable accepted N27 directory}"
: "${R26_LEGACY_N28_DIR:?Set R26_LEGACY_N28_DIR to the immutable accepted N28 directory}"
: "${R26_THOR_N28_OUT:?Set R26_THOR_N28_OUT to a new output directory}"
: "${R26_THOR_N28_REF:?Set R26_THOR_N28_REF to the immutable 40-character commit SHA}"

EXPECTED_THOR_N24_REF="c47b43a838a4df2edf3cea4bbf56e67f038bd798"
[[ "$R26_THOR_N28_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_THOR_N28_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}
test ! -e "$R26_THOR_N28_OUT"
test ! -e "${R26_THOR_N28_OUT}_RESULTS.zip"
mkdir -p "$R26_THOR_N28_OUT"

exec > >(tee -a "$R26_THOR_N28_OUT/run.stdout.log") \
     2> >(tee -a "$R26_THOR_N28_OUT/run.stderr.log" >&2)

CURRENT_STAGE="preflight"
package_on_exit() {
  local return_code=$?
  trap - EXIT
  set +e
  if (( return_code != 0 )); then
    python3 - "$R26_THOR_N28_OUT" "$R26_THOR_N28_REF" "$CURRENT_STAGE" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
record = {
    "status": "R26_THOR_ROOT_RECONCILIATION_N28_FAILED",
    "source_commit": sys.argv[2],
    "failure_stage": sys.argv[3],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n28_accepted": False,
    "n29_authorized": False,
    "n30_authorized": False,
    "production_accepted": False,
    "legacy_roots_used_as_solver_seed": False,
}
(root / "THOR_ROOT_RECONCILIATION_N28_FAILED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  fi
  local zip_path="${R26_THOR_N28_OUT}_RESULTS.zip"
  local sha_path="${zip_path}.sha256.txt"
  (cd "$(dirname "$R26_THOR_N28_OUT")" && zip -qr "$zip_path" "$(basename "$R26_THOR_N28_OUT")")
  (cd "$(dirname "$zip_path")" && sha256sum "$(basename "$zip_path")" > "$(basename "$sha_path")")
  exit "$return_code"
}
trap package_on_exit EXIT

CURRENT_STAGE="input_preflight"
test -f "$R26_THOR_N24_DIR/THOR_CROSS_SOLVER_N24_PASSED.json"
test -f "$R26_THOR_N24_DIR/N24/thor_state.npz"
test -f "$R26_THOR_N24_DIR/N8_N16_N24_GRID_SENSITIVITY.json"
for directory in "$R26_LEGACY_N25_DIR" "$R26_LEGACY_N27_DIR" "$R26_LEGACY_N28_DIR"; do
  test -f "$directory/run_summary.json"
  test -f "$directory/last_accepted_state.npz"
done

CURRENT_STAGE="source_checkout"
REPO="$R26_THOR_N28_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_THOR_N28_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_THOR_N28_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_THOR_N28_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

CURRENT_STAGE="unit_and_contract_tests"
(cd "$PKG" && python3 run_tests.py)

CURRENT_STAGE="immutable_root_ladder_audit"
python3 "$PKG/tools/audit_r26_thor_root_ladder.py" \
  --thor-n24-dir "$R26_THOR_N24_DIR" \
  --legacy-n25-dir "$R26_LEGACY_N25_DIR" \
  --legacy-n27-dir "$R26_LEGACY_N27_DIR" \
  --legacy-n28-dir "$R26_LEGACY_N28_DIR" \
  --output "$R26_THOR_N28_OUT/THOR_ROOT_LADDER_AUDIT.json" \
  --expected-thor-n24-source-commit "$EXPECTED_THOR_N24_REF" \
  --raw-tolerance 1e-8 \
  --maximum-DG-relative-difference 0.02

python3 - "$R26_THOR_N28_OUT/THOR_ROOT_LADDER_AUDIT.json" <<'PY'
import json
import pathlib
import sys

record = json.loads(pathlib.Path(sys.argv[1]).read_text())
if record.get("status") != "R26_THOR_ROOT_LADDER_AUDIT_PASSED":
    raise SystemExit("immutable root ladder audit did not pass")
if record.get("n28_run_authorized") is not True:
    raise SystemExit("N28 run was not explicitly authorized")
if record.get("n29_authorized") is not False or record.get("n30_authorized") is not False:
    raise SystemExit("root ladder audit exceeded its authorization scope")
if record.get("legacy_roots_used_as_solver_seed") is not False:
    raise SystemExit("legacy root seed firewall is not explicit")
PY

CURRENT_STAGE="thor_n28_from_thor_n24"
python3 "$PKG/analysis/run_r26_thor_validation.py" \
  --case-family jfm-maxwell \
  --nodes 28 \
  --kn-gu 0.2 \
  --lid-speed-m-s 100 \
  --initial-state "$R26_THOR_N24_DIR/N24/thor_state.npz" \
  --output-dir "$R26_THOR_N28_OUT/N28" \
  --max-iterations 100 \
  --inner-max-iterations 200 \
  --raw-tolerance 1e-8 \
  --solver-tolerance 1e-9 \
  --ilu-drop-tolerance 1e-6 \
  --ilu-fill-factor 48

CURRENT_STAGE="n28_same_grid_cross_solver_validation"
python3 "$PKG/tools/validate_r26_thor_n28_cross_solver.py" \
  --thor-n28-dir "$R26_THOR_N28_OUT/N28" \
  --legacy-n28-dir "$R26_LEGACY_N28_DIR" \
  --root-ladder-audit "$R26_THOR_N28_OUT/THOR_ROOT_LADDER_AUDIT.json" \
  --output "$R26_THOR_N28_OUT/THOR_N28_CROSS_SOLVER_VALIDATION.json" \
  --raw-tolerance 1e-8 \
  --maximum-profile-nrms 0.05 \
  --maximum-line-nrms 0.15 \
  --maximum-DG-relative-difference 0.02

CURRENT_STAGE="final_record"
python3 - "$R26_THOR_N28_OUT" "$ACTUAL_REF" "$EXPECTED_THOR_N24_REF" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
n28 = json.loads((root / "N28" / "thor_validation.json").read_text())
audit = json.loads((root / "THOR_ROOT_LADDER_AUDIT.json").read_text())
cross = json.loads((root / "THOR_N28_CROSS_SOLVER_VALIDATION.json").read_text())
if n28.get("status") != "R26_THOR_VALIDATION_CANDIDATE_PASSED":
    raise SystemExit("N28 THOR candidate did not pass")
if n28.get("physical_candidate_gate_passed") is not True:
    raise SystemExit("N28 physical gate is not boolean true")
if n28.get("production_accepted") is not False:
    raise SystemExit("N28 candidate incorrectly made a production claim")
if audit.get("n28_run_authorized") is not True:
    raise SystemExit("N28 authorization disappeared")
if cross.get("status") != "R26_THOR_N28_CROSS_SOLVER_VALIDATION_PASSED":
    raise SystemExit("N28 cross-solver validation did not pass")
if cross.get("n29_authorized") is not True:
    raise SystemExit("N29 was not explicitly authorized")
record = {
    "status": "R26_THOR_ROOT_RECONCILIATION_N28_PASSED",
    "source_commit": sys.argv[2],
    "thor_n24_source_commit": sys.argv[3],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n28_accepted": True,
    "n28_state_sha256": n28["state_sha256"],
    "n28_raw_gate": n28["raw_acceptance_gate"],
    "root_ladder_audit": str((root / "THOR_ROOT_LADDER_AUDIT.json").resolve()),
    "cross_solver_validation": str((root / "THOR_N28_CROSS_SOLVER_VALIDATION.json").resolve()),
    "legacy_roots_used_as_solver_seed": False,
    "n29_authorized": True,
    "n30_authorized": False,
    "production_accepted": False,
    "next_required_stage": "one bounded THOR N29 solve from the accepted THOR N28 state; historical failed N29 states remain forbidden",
}
(root / "THOR_ROOT_RECONCILIATION_N28_PASSED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "R26_THOR_ROOT_RECONCILIATION_N28_COMPLETE"
echo "R26_RESULTS_ZIP=${R26_THOR_N28_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_THOR_N28_OUT}_RESULTS.zip.sha256.txt"
