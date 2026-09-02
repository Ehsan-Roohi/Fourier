#!/bin/bash
# Resume N32 with raw-guarded, line-search-aware PTC growth toward Newton.
# No grid above N32 is present in this job.

#SBATCH --job-name=r26-ge-j32a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=r26-ge-j32a-%j.out
#SBATCH --error=r26-ge-j32a-%j.err

set -euo pipefail

: "${R26_GE_JFM_N32_PTC_OUT:?Set a new PTC rescue output directory}"
: "${R26_GE_JFM_N32_PTC_REF:?Set the immutable rescue source commit SHA}"
: "${R26_GE_JFM_N32_FAILED_DIR:?Set the failed N32 job directory}"
[[ "$R26_GE_JFM_N32_PTC_REF" =~ ^[0-9a-f]{40}$ ]] || exit 2

REFERENCE="/project/pi_roohie_umass_edu/CavityColdToHotIdentify/R26_KNGU020_N26_N30_LADDER_JOB63184894/N28"
FAILED_N32="$R26_GE_JFM_N32_FAILED_DIR/N32"
test -f "$REFERENCE/last_accepted_state.npz"
test -f "$REFERENCE/run_summary.json"
test -f "$FAILED_N32/JFM_N32_TRANSFORMED_CANDIDATE_GATE.json"
test -f "$FAILED_N32/gu_emerson_jfm_n32_candidate.npz"
test ! -e "$R26_GE_JFM_N32_PTC_OUT"
test ! -e "${R26_GE_JFM_N32_PTC_OUT}_RESULTS.zip"
mkdir -p "$R26_GE_JFM_N32_PTC_OUT"

exec > >(tee -a "$R26_GE_JFM_N32_PTC_OUT/run.stdout.log") \
     2> >(tee -a "$R26_GE_JFM_N32_PTC_OUT/run.stderr.log" >&2)

package_on_exit() {
  local return_code=$?
  trap - EXIT
  set +e
  if (( return_code != 0 )) && [[ ! -f "$R26_GE_JFM_N32_PTC_OUT/N32/JFM_N32_TRANSFORMED_CANDIDATE_GATE.json" ]]; then
    python3 - "$R26_GE_JFM_N32_PTC_OUT" "$R26_GE_JFM_N32_PTC_REF" <<'PY'
import datetime, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
record = {
    "status": "R26_GU_EMERSON_JFM_N32_ALPHA_AWARE_PTC_JOB_FAILED",
    "source_commit": sys.argv[2],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "candidate_accepted": False,
    "production_accepted": False,
    "n36_authorized": False,
    "n40_authorized": False,
    "n44_authorized": False,
    "maximum_grid_run": 32,
    "higher_than_n32_run_attempted": False,
}
(root / "JFM_N32_PTC_RESCUE_JOB_FAILED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  fi
  local zip_path="${R26_GE_JFM_N32_PTC_OUT}_RESULTS.zip"
  local sha_path="${zip_path}.sha256.txt"
  (cd "$(dirname "$R26_GE_JFM_N32_PTC_OUT")" && zip -qr "$zip_path" "$(basename "$R26_GE_JFM_N32_PTC_OUT")")
  (cd "$(dirname "$zip_path")" && sha256sum "$(basename "$zip_path")" > "$(basename "$sha_path")")
  exit "$return_code"
}
trap package_on_exit EXIT

REPO="$R26_GE_JFM_N32_PTC_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_GE_JFM_N32_PTC_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_GE_JFM_N32_PTC_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_GE_JFM_N32_PTC_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)

# Re-authorize the immutable N28 root with the exact rescue source tree.
python3 "$PKG/analysis/run_r26_gu_emerson_jfm_n28_same_grid_gate.py" \
  --reference-dir "$REFERENCE" \
  --output-dir "$R26_GE_JFM_N32_PTC_OUT/N28_GATE" \
  --source-commit "$ACTUAL_REF"

python3 "$PKG/tools/validate_r26_gu_emerson_jfm_n28_same_grid_gate.py" \
  --run-dir "$R26_GE_JFM_N32_PTC_OUT/N28_GATE" \
  --reference-dir "$REFERENCE" \
  --expected-source-commit "$ACTUAL_REF"

# The failed state is a seed only.  Final acceptance re-evaluates the complete
# unchanged physical and transformed R26 residuals at N32.
python3 "$PKG/analysis/run_r26_gu_emerson_jfm_n32_candidate.py" \
  --n28-gate-dir "$R26_GE_JFM_N32_PTC_OUT/N28_GATE" \
  --failed-n32-dir "$FAILED_N32" \
  --output-dir "$R26_GE_JFM_N32_PTC_OUT/N32" \
  --source-commit "$ACTUAL_REF"

python3 "$PKG/tools/validate_r26_gu_emerson_jfm_n32_candidate.py" \
  --run-dir "$R26_GE_JFM_N32_PTC_OUT/N32" \
  --n28-gate-dir "$R26_GE_JFM_N32_PTC_OUT/N28_GATE" \
  --expected-source-commit "$ACTUAL_REF"

python3 - "$R26_GE_JFM_N32_PTC_OUT/N32/JFM_N32_TRANSFORMED_CANDIDATE_GATE.json" <<'PY'
import json, pathlib, sys
record = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if record.get("status") != "R26_GU_EMERSON_JFM_N32_TRANSFORMED_CANDIDATE_PASSED":
    raise SystemExit("N32 PTC rescue did not pass")
if record.get("candidate_accepted") is not True:
    raise SystemExit("N32 PTC candidate is not accepted")
if record.get("production_accepted") is not False:
    raise SystemExit("N32 PTC rescue made a production claim")
if record.get("n36_authorized") is not True:
    raise SystemExit("N32 PTC rescue did not authorize N36")
if record.get("n40_authorized") is not False or record.get("n44_authorized") is not False:
    raise SystemExit("N32 PTC rescue over-authorized a higher grid")
if record.get("maximum_grid_run") != 32:
    raise SystemExit("N32 PTC rescue changed grid scope")
predecessor = record.get("predecessor_failure") or {}
if predecessor.get("source_commit") != "292017c2262fb3fe03a3c6a51af6cbbeb7a10552":
    raise SystemExit("N32 PTC predecessor provenance mismatch")
PY

echo "R26_GU_EMERSON_JFM_N32_ALPHA_AWARE_PTC_COMPLETE"
echo "R26_RESULTS_ZIP=${R26_GE_JFM_N32_PTC_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_GE_JFM_N32_PTC_OUT}_RESULTS.zip.sha256.txt"
