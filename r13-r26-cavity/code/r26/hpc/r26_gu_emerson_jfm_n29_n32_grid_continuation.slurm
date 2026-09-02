#!/bin/bash
# Track the accepted N28 root one grid node at a time through N32.
# Every stage is fail-closed; this job contains no N36 or higher solve.

#SBATCH --job-name=r26-ge-gc32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=r26-ge-gc32-%j.out
#SBATCH --error=r26-ge-gc32-%j.err

set -euo pipefail

: "${R26_GE_JFM_GRID_CONTINUATION_OUT:?Set a new grid-continuation output directory}"
: "${R26_GE_JFM_GRID_CONTINUATION_REF:?Set the immutable source commit SHA}"
[[ "$R26_GE_JFM_GRID_CONTINUATION_REF" =~ ^[0-9a-f]{40}$ ]] || exit 2

REFERENCE="/project/pi_roohie_umass_edu/CavityColdToHotIdentify/R26_KNGU020_N26_N30_LADDER_JOB63184894/N28"
test -f "$REFERENCE/last_accepted_state.npz"
test -f "$REFERENCE/run_summary.json"
test ! -e "$R26_GE_JFM_GRID_CONTINUATION_OUT"
test ! -e "${R26_GE_JFM_GRID_CONTINUATION_OUT}_RESULTS.zip"
mkdir -p "$R26_GE_JFM_GRID_CONTINUATION_OUT"

exec > >(tee -a "$R26_GE_JFM_GRID_CONTINUATION_OUT/run.stdout.log") \
     2> >(tee -a "$R26_GE_JFM_GRID_CONTINUATION_OUT/run.stderr.log" >&2)

package_on_exit() {
  local return_code=$?
  trap - EXIT
  set +e
  if (( return_code != 0 )) && [[ ! -f "$R26_GE_JFM_GRID_CONTINUATION_OUT/N32/JFM_N32_TRANSFORMED_CANDIDATE_GATE.json" ]]; then
    python3 - "$R26_GE_JFM_GRID_CONTINUATION_OUT" "$R26_GE_JFM_GRID_CONTINUATION_REF" <<'PY'
import datetime, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
record = {
    "status": "R26_GU_EMERSON_JFM_N29_N32_GRID_CONTINUATION_JOB_FAILED",
    "source_commit": sys.argv[2],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "candidate_accepted": False,
    "production_accepted": False,
    "n36_authorized": False,
    "n40_authorized": False,
    "n44_authorized": False,
    "maximum_grid_run": max(
        [28] + [n for n in (29, 30, 31) if (root / f"GRID/N{n}").is_dir()]
    ),
    "higher_than_n32_run_attempted": False,
}
(root / "JFM_N29_N32_GRID_CONTINUATION_JOB_FAILED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  fi
  local zip_path="${R26_GE_JFM_GRID_CONTINUATION_OUT}_RESULTS.zip"
  local sha_path="${zip_path}.sha256.txt"
  (cd "$(dirname "$R26_GE_JFM_GRID_CONTINUATION_OUT")" && zip -qr "$zip_path" "$(basename "$R26_GE_JFM_GRID_CONTINUATION_OUT")")
  (cd "$(dirname "$zip_path")" && sha256sum "$(basename "$zip_path")" > "$(basename "$sha_path")")
  exit "$return_code"
}
trap package_on_exit EXIT

REPO="$R26_GE_JFM_GRID_CONTINUATION_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_GE_JFM_GRID_CONTINUATION_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_GE_JFM_GRID_CONTINUATION_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_GE_JFM_GRID_CONTINUATION_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)

python3 "$PKG/analysis/run_r26_gu_emerson_jfm_n28_same_grid_gate.py" \
  --reference-dir "$REFERENCE" \
  --output-dir "$R26_GE_JFM_GRID_CONTINUATION_OUT/N28_GATE" \
  --source-commit "$ACTUAL_REF"

python3 "$PKG/tools/validate_r26_gu_emerson_jfm_n28_same_grid_gate.py" \
  --run-dir "$R26_GE_JFM_GRID_CONTINUATION_OUT/N28_GATE" \
  --reference-dir "$REFERENCE" \
  --expected-source-commit "$ACTUAL_REF"

python3 "$PKG/analysis/run_r26_gu_emerson_jfm_n29_n31_grid_continuation.py" \
  --n28-gate-dir "$R26_GE_JFM_GRID_CONTINUATION_OUT/N28_GATE" \
  --output-dir "$R26_GE_JFM_GRID_CONTINUATION_OUT/GRID" \
  --source-commit "$ACTUAL_REF"

python3 "$PKG/tools/validate_r26_gu_emerson_jfm_n29_n31_grid_continuation.py" \
  --run-dir "$R26_GE_JFM_GRID_CONTINUATION_OUT/GRID" \
  --n28-gate-dir "$R26_GE_JFM_GRID_CONTINUATION_OUT/N28_GATE" \
  --expected-source-commit "$ACTUAL_REF"

python3 "$PKG/analysis/run_r26_gu_emerson_jfm_n32_candidate.py" \
  --n28-gate-dir "$R26_GE_JFM_GRID_CONTINUATION_OUT/N28_GATE" \
  --accepted-n31-dir "$R26_GE_JFM_GRID_CONTINUATION_OUT/GRID/N31" \
  --output-dir "$R26_GE_JFM_GRID_CONTINUATION_OUT/N32" \
  --source-commit "$ACTUAL_REF"

python3 "$PKG/tools/validate_r26_gu_emerson_jfm_n32_candidate.py" \
  --run-dir "$R26_GE_JFM_GRID_CONTINUATION_OUT/N32" \
  --n28-gate-dir "$R26_GE_JFM_GRID_CONTINUATION_OUT/N28_GATE" \
  --expected-source-commit "$ACTUAL_REF"

python3 - "$R26_GE_JFM_GRID_CONTINUATION_OUT/N32/JFM_N32_TRANSFORMED_CANDIDATE_GATE.json" <<'PY'
import json, pathlib, sys
record = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if record.get("status") != "R26_GU_EMERSON_JFM_N32_TRANSFORMED_CANDIDATE_PASSED":
    raise SystemExit("N32 grid-continuation candidate did not pass")
if record.get("candidate_accepted") is not True:
    raise SystemExit("N32 grid-continuation candidate is not accepted")
if record.get("production_accepted") is not False:
    raise SystemExit("N32 grid continuation made a production claim")
if record.get("n36_authorized") is not True:
    raise SystemExit("accepted N32 did not authorize exactly N36")
if record.get("n40_authorized") is not False or record.get("n44_authorized") is not False:
    raise SystemExit("N32 grid continuation over-authorized a higher grid")
if record.get("maximum_grid_run") != 32:
    raise SystemExit("N32 grid continuation changed grid scope")
predecessor = record.get("predecessor_grid_stage") or {}
if predecessor.get("status") != "R26_GU_EMERSON_JFM_N31_GRID_CONTINUATION_PASSED":
    raise SystemExit("N32 did not consume the accepted N31 predecessor")
PY

echo "R26_GU_EMERSON_JFM_N29_N32_GRID_CONTINUATION_COMPLETE"
echo "R26_RESULTS_ZIP=${R26_GE_JFM_GRID_CONTINUATION_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_GE_JFM_GRID_CONTINUATION_OUT}_RESULTS.zip.sha256.txt"
