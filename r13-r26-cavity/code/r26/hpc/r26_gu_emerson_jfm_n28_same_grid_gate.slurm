#!/bin/bash
#SBATCH --job-name=r26-ge-jfm28
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=r26-ge-jfm28-%j.out
#SBATCH --error=r26-ge-jfm28-%j.err

set -euo pipefail

: "${R26_GE_JFM_N28_OUT:?Set R26_GE_JFM_N28_OUT to a new output directory}"
: "${R26_GE_JFM_N28_REF:?Set R26_GE_JFM_N28_REF to an immutable 40-character commit SHA}"
[[ "$R26_GE_JFM_N28_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_GE_JFM_N28_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}

REFERENCE="/project/pi_roohie_umass_edu/CavityColdToHotIdentify/R26_KNGU020_N26_N30_LADDER_JOB63184894/N28"
test -f "$REFERENCE/last_accepted_state.npz"
test -f "$REFERENCE/run_summary.json"
test ! -e "$R26_GE_JFM_N28_OUT"
test ! -e "${R26_GE_JFM_N28_OUT}_RESULTS.zip"
mkdir -p "$R26_GE_JFM_N28_OUT"

exec > >(tee -a "$R26_GE_JFM_N28_OUT/run.stdout.log") \
     2> >(tee -a "$R26_GE_JFM_N28_OUT/run.stderr.log" >&2)

package_on_exit() {
  local return_code=$?
  trap - EXIT
  set +e
  if (( return_code != 0 )) && [[ ! -f "$R26_GE_JFM_N28_OUT/JFM_N28_SAME_GRID_GATE.json" ]]; then
    python3 - "$R26_GE_JFM_N28_OUT" "$R26_GE_JFM_N28_REF" <<'PY'
import datetime, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
record = {
    "status": "R26_GU_EMERSON_JFM_N28_SAME_GRID_GATE_FAILED",
    "source_commit": sys.argv[2],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "same_grid_gate_passed": False,
    "candidate_accepted": False,
    "production_accepted": False,
    "n32_authorized": False,
    "n40_authorized": False,
    "n44_authorized": False,
}
(root / "JFM_N28_SAME_GRID_GATE.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  fi
  local zip_path="${R26_GE_JFM_N28_OUT}_RESULTS.zip"
  local sha_path="${zip_path}.sha256.txt"
  (cd "$(dirname "$R26_GE_JFM_N28_OUT")" && zip -qr "$zip_path" "$(basename "$R26_GE_JFM_N28_OUT")")
  (cd "$(dirname "$zip_path")" && sha256sum "$(basename "$zip_path")" > "$(basename "$sha_path")")
  exit "$return_code"
}
trap package_on_exit EXIT

REPO="$R26_GE_JFM_N28_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_GE_JFM_N28_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_GE_JFM_N28_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_GE_JFM_N28_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)

python3 "$PKG/tools/validate_r26_maxwell_run.py" \
  "$REFERENCE" \
  --expected-kn 0.20 \
  --expected-nodes 28 \
  --expected-beta 0.0 \
  --raw-tolerance 1e-8

python3 "$PKG/analysis/run_r26_gu_emerson_jfm_n28_same_grid_gate.py" \
  --reference-dir "$REFERENCE" \
  --output-dir "$R26_GE_JFM_N28_OUT/GATE" \
  --source-commit "$ACTUAL_REF"

python3 "$PKG/tools/validate_r26_gu_emerson_jfm_n28_same_grid_gate.py" \
  --run-dir "$R26_GE_JFM_N28_OUT/GATE" \
  --reference-dir "$REFERENCE" \
  --expected-source-commit "$ACTUAL_REF"

python3 - "$R26_GE_JFM_N28_OUT/GATE/JFM_N28_SAME_GRID_GATE.json" <<'PY'
import json, pathlib, sys
record = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if record.get("status") != "R26_GU_EMERSON_JFM_N28_SAME_GRID_GATE_PASSED":
    raise SystemExit("JFM N28 same-grid gate did not pass")
if record.get("same_grid_gate_passed") is not True:
    raise SystemExit("same-grid gate is not boolean true")
if record.get("candidate_accepted") is not True:
    raise SystemExit("N28 transformed candidate was not accepted")
if record.get("production_accepted") is not False:
    raise SystemExit("N28 gate made a premature production claim")
if record.get("n32_authorized") is not True:
    raise SystemExit("N32 was not explicitly authorized")
if record.get("n40_authorized") is not False or record.get("n44_authorized") is not False:
    raise SystemExit("N28 gate exceeded one-grid authorization")
PY

echo "R26_GU_EMERSON_JFM_N28_SAME_GRID_GATE_COMPLETE"
echo "R26_RESULTS_ZIP=${R26_GE_JFM_N28_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_GE_JFM_N28_OUT}_RESULTS.zip.sha256.txt"
