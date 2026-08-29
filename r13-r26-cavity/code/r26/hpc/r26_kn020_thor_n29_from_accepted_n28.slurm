#!/bin/bash
#SBATCH --job-name=r26-thor29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --output=r26-thor29-%j.out
#SBATCH --error=r26-thor29-%j.err

set -euo pipefail

: "${R26_THOR_ACCEPTED_N28_DIR:?Set R26_THOR_ACCEPTED_N28_DIR to the passed THOR N28 directory}"
: "${R26_THOR_N29_OUT:?Set R26_THOR_N29_OUT to a new output directory}"
: "${R26_THOR_N29_REF:?Set R26_THOR_N29_REF to the immutable 40-character commit SHA}"

EXPECTED_N28_REF="743e284d89980cbedd188d4127aae133674e054e"
EXPECTED_N28_STATE_FILE_SHA256="21cf1a09daa3ef7a7ddca604bc508fa201dc67029fd85b8a41fe32e78947b5b0"
EXPECTED_N28_PASSED_FILE_SHA256="6144476718f135a2bd9b6c2ea54da9ce6508397b0d6208bc3865f6b5d9186fd5"
EXPECTED_N28_CROSS_FILE_SHA256="ef28e067aca3a3be6a8c346554c7973819eb50186ee1664910a047763fd2a33c"
EXPECTED_N28_VALIDATION_FILE_SHA256="7e8669247da92932df81652d19aef08bc0c58854b3ff1e275c57413234af8ede"
[[ "$R26_THOR_N29_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_THOR_N29_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}
test ! -e "$R26_THOR_N29_OUT"
test ! -e "${R26_THOR_N29_OUT}_RESULTS.zip"
mkdir -p "$R26_THOR_N29_OUT"

exec > >(tee -a "$R26_THOR_N29_OUT/run.stdout.log") \
     2> >(tee -a "$R26_THOR_N29_OUT/run.stderr.log" >&2)

CURRENT_STAGE="preflight"
package_on_exit() {
  local return_code=$?
  trap - EXIT
  set +e
  if (( return_code != 0 )); then
    python3 - "$R26_THOR_N29_OUT" "$R26_THOR_N29_REF" "$CURRENT_STAGE" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
record = {
    "status": "R26_THOR_N29_FROM_ACCEPTED_N28_FAILED",
    "source_commit": sys.argv[2],
    "failure_stage": sys.argv[3],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n29_accepted": False,
    "n30_authorized": False,
    "production_accepted": False,
    "accepted_thor_n28_is_only_permitted_seed": True,
    "legacy_or_failed_n29_used_as_solver_seed": False,
    "historical_failed_n30_used_as_solver_seed": False,
}
(root / "THOR_N29_FROM_ACCEPTED_N28_FAILED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  fi
  local zip_path="${R26_THOR_N29_OUT}_RESULTS.zip"
  local sha_path="${zip_path}.sha256.txt"
  (cd "$(dirname "$R26_THOR_N29_OUT")" && zip -qr "$zip_path" "$(basename "$R26_THOR_N29_OUT")")
  (cd "$(dirname "$zip_path")" && sha256sum "$(basename "$zip_path")" > "$(basename "$sha_path")")
  exit "$return_code"
}
trap package_on_exit EXIT

CURRENT_STAGE="accepted_n28_preflight"
N28_PASSED="$R26_THOR_ACCEPTED_N28_DIR/THOR_ROOT_RECONCILIATION_N28_PASSED.json"
N28_CROSS="$R26_THOR_ACCEPTED_N28_DIR/THOR_N28_CROSS_SOLVER_VALIDATION.json"
N28_VALIDATION="$R26_THOR_ACCEPTED_N28_DIR/N28/thor_validation.json"
N28_STATE="$R26_THOR_ACCEPTED_N28_DIR/N28/thor_state.npz"
test -f "$N28_PASSED"
test -f "$N28_CROSS"
test -f "$N28_VALIDATION"
test -f "$N28_STATE"
test "$(sha256sum "$N28_PASSED" | awk '{print $1}')" = "$EXPECTED_N28_PASSED_FILE_SHA256"
test "$(sha256sum "$N28_CROSS" | awk '{print $1}')" = "$EXPECTED_N28_CROSS_FILE_SHA256"
test "$(sha256sum "$N28_VALIDATION" | awk '{print $1}')" = "$EXPECTED_N28_VALIDATION_FILE_SHA256"
test "$(sha256sum "$N28_STATE" | awk '{print $1}')" = "$EXPECTED_N28_STATE_FILE_SHA256"
python3 - "$N28_PASSED" "$N28_CROSS" "$N28_VALIDATION" "$EXPECTED_N28_REF" <<'PY'
import json
import pathlib
import sys

passed = json.loads(pathlib.Path(sys.argv[1]).read_text())
cross = json.loads(pathlib.Path(sys.argv[2]).read_text())
candidate = json.loads(pathlib.Path(sys.argv[3]).read_text())
expected_ref = sys.argv[4]
if passed.get("status") != "R26_THOR_ROOT_RECONCILIATION_N28_PASSED":
    raise SystemExit("accepted N28 final record did not pass")
if passed.get("source_commit") != expected_ref:
    raise SystemExit("accepted N28 source commit changed")
if passed.get("n28_accepted") is not True or passed.get("n29_authorized") is not True:
    raise SystemExit("N29 is not explicitly authorized")
if passed.get("n30_authorized") is not False or passed.get("production_accepted") is not False:
    raise SystemExit("accepted N28 record exceeded its authorization scope")
if cross.get("status") != "R26_THOR_N28_CROSS_SOLVER_VALIDATION_PASSED":
    raise SystemExit("N28 cross-solver validation did not pass")
if cross.get("legacy_n28_used_as_solver_seed") is not False:
    raise SystemExit("legacy N28 seed firewall is not explicit")
if candidate.get("status") != "R26_THOR_VALIDATION_CANDIDATE_PASSED":
    raise SystemExit("accepted THOR N28 candidate record did not pass")
if candidate.get("production_accepted") is not False:
    raise SystemExit("accepted THOR N28 candidate made a production claim")
PY

CURRENT_STAGE="source_checkout"
REPO="$R26_THOR_N29_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_THOR_N29_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_THOR_N29_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_THOR_N29_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

CURRENT_STAGE="unit_and_contract_tests"
(cd "$PKG" && python3 run_tests.py)

CURRENT_STAGE="thor_n29_from_accepted_thor_n28"
python3 "$PKG/analysis/run_r26_thor_validation.py" \
  --case-family jfm-maxwell \
  --nodes 29 \
  --kn-gu 0.2 \
  --lid-speed-m-s 100 \
  --initial-state "$N28_STATE" \
  --output-dir "$R26_THOR_N29_OUT/N29" \
  --max-iterations 100 \
  --inner-max-iterations 200 \
  --raw-tolerance 1e-8 \
  --solver-tolerance 1e-9 \
  --ilu-drop-tolerance 1e-6 \
  --ilu-fill-factor 48

CURRENT_STAGE="n28_to_n29_independent_refinement_validation"
python3 "$PKG/tools/validate_r26_thor_n29_refinement.py" \
  --accepted-thor-n28-dir "$R26_THOR_ACCEPTED_N28_DIR" \
  --thor-n29-dir "$R26_THOR_N29_OUT/N29" \
  --output "$R26_THOR_N29_OUT/THOR_N29_REFINEMENT_VALIDATION.json" \
  --raw-tolerance 1e-8 \
  --maximum-profile-nrms 0.05 \
  --maximum-line-nrms 0.15 \
  --maximum-DG-relative-difference 0.02

CURRENT_STAGE="final_record"
python3 - "$R26_THOR_N29_OUT" "$ACTUAL_REF" "$EXPECTED_N28_REF" <<'PY'
import datetime
import json
import pathlib
import sys

from r26_thor_reconciliation import sha256_file

root = pathlib.Path(sys.argv[1])
n29 = json.loads((root / "N29" / "thor_validation.json").read_text())
validation = json.loads((root / "THOR_N29_REFINEMENT_VALIDATION.json").read_text())
if n29.get("status") != "R26_THOR_VALIDATION_CANDIDATE_PASSED":
    raise SystemExit("N29 THOR candidate did not pass")
if n29.get("physical_candidate_gate_passed") is not True:
    raise SystemExit("N29 physical gate is not boolean true")
if n29.get("production_accepted") is not False:
    raise SystemExit("N29 candidate made a production claim")
if validation.get("status") != "R26_THOR_N29_REFINEMENT_VALIDATION_PASSED":
    raise SystemExit("N29 refinement validation did not pass")
if validation.get("n29_accepted") is not True or validation.get("n30_authorized") is not True:
    raise SystemExit("N29 acceptance or N30 authorization is missing")
if validation.get("production_accepted") is not False:
    raise SystemExit("N29 validation made a production claim")
state_path = root / "N29" / "thor_state.npz"
record = {
    "status": "R26_THOR_N29_FROM_ACCEPTED_N28_PASSED",
    "source_commit": sys.argv[2],
    "accepted_n28_source_commit": sys.argv[3],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n29_accepted": True,
    "n29_state_sha256": n29["state_sha256"],
    "n29_state_file_sha256": sha256_file(state_path),
    "n29_raw_gate": n29["raw_acceptance_gate"],
    "refinement_validation": str((root / "THOR_N29_REFINEMENT_VALIDATION.json").resolve()),
    "accepted_thor_n28_used_as_solver_seed": True,
    "legacy_or_failed_n29_used_as_solver_seed": False,
    "historical_failed_n30_used_as_solver_seed": False,
    "n30_authorized": True,
    "production_accepted": False,
    "next_required_stage": "one bounded THOR N30 solve from the accepted THOR N29 state; historical failed N30 and rescue states remain forbidden",
}
(root / "THOR_N29_FROM_ACCEPTED_N28_PASSED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "R26_THOR_N29_FROM_ACCEPTED_N28_COMPLETE"
echo "R26_RESULTS_ZIP=${R26_THOR_N29_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_THOR_N29_OUT}_RESULTS.zip.sha256.txt"
