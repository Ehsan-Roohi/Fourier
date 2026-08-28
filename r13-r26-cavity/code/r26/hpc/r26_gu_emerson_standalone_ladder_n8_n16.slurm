#!/bin/bash
#SBATCH --job-name=r26-ge-stand
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=r26-ge-stand-%j.out
#SBATCH --error=r26-ge-stand-%j.err

set -euo pipefail

: "${R26_GE_STANDALONE_OUT:?Set R26_GE_STANDALONE_OUT to a new output directory}"
: "${R26_GE_STANDALONE_REF:?Set R26_GE_STANDALONE_REF to an immutable 40-character commit SHA}"
: "${R26_GE_CROSS_DIR:?Set R26_GE_CROSS_DIR to the completed reconstruction cross-gate directory}"
[[ "$R26_GE_STANDALONE_REF" =~ ^[0-9a-f]{40}$ ]] || exit 2
test -d "$R26_GE_CROSS_DIR"
test ! -e "$R26_GE_STANDALONE_OUT"
test ! -e "${R26_GE_STANDALONE_OUT}_RESULTS.zip"
mkdir -p "$R26_GE_STANDALONE_OUT"

exec > >(tee -a "$R26_GE_STANDALONE_OUT/run.stdout.log") \
     2> >(tee -a "$R26_GE_STANDALONE_OUT/run.stderr.log" >&2)

package_on_exit() {
  local return_code=$?
  trap - EXIT
  set +e
  if (( return_code != 0 )) && ! compgen -G "$R26_GE_STANDALONE_OUT/GU_EMERSON_STANDALONE_LADDER_*" >/dev/null; then
    python3 - "$R26_GE_STANDALONE_OUT" "$R26_GE_STANDALONE_REF" <<'PY'
import datetime, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
record = {
    "status": "R26_GU_EMERSON_STANDALONE_LADDER_FAILED",
    "source_commit": sys.argv[2],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "failure_stage": "preflight_or_unhandled_exception",
    "standalone_from_equilibrium_passed": False,
    "production_accepted": False,
    "n24_authorized": False,
    "n28_authorized": False,
    "n29_authorized": False,
    "n30_authorized": False,
}
(root / "GU_EMERSON_STANDALONE_LADDER_FAILED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n"
)
PY
  fi
  local zip_path="${R26_GE_STANDALONE_OUT}_RESULTS.zip"
  local sha_path="${zip_path}.sha256.txt"
  (cd "$(dirname "$R26_GE_STANDALONE_OUT")" && zip -qr "$zip_path" "$(basename "$R26_GE_STANDALONE_OUT")")
  (cd "$(dirname "$zip_path")" && sha256sum "$(basename "$zip_path")" > "$(basename "$sha_path")")
  exit "$return_code"
}
trap package_on_exit EXIT

REPO="$R26_GE_STANDALONE_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_GE_STANDALONE_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_GE_STANDALONE_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_GE_STANDALONE_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)

PRIOR_RECON="$R26_GE_CROSS_DIR"
if test -d "$R26_GE_CROSS_DIR/RECONSTRUCTION"; then
  PRIOR_RECON="$R26_GE_CROSS_DIR/RECONSTRUCTION"
fi
python3 "$PKG/tools/validate_r26_gu_emerson_reconstruction_gate.py" \
  --run-dir "$PRIOR_RECON" \
  --expected-source-commit c9c3bc07d14a691d2d4ed70533b46f8daed53726

python3 "$PKG/analysis/run_r26_gu_emerson_standalone_ladder.py" \
  --cross-gate-dir "$R26_GE_CROSS_DIR" \
  --output "$R26_GE_STANDALONE_OUT/STANDALONE" \
  --source-commit "$ACTUAL_REF"

python3 "$PKG/tools/validate_r26_gu_emerson_standalone_ladder.py" \
  --run-dir "$R26_GE_STANDALONE_OUT/STANDALONE" \
  --expected-source-commit "$ACTUAL_REF"

echo "R26_GU_EMERSON_STANDALONE_LADDER_COMPLETE"
echo "R26_RESULTS_ZIP=${R26_GE_STANDALONE_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_GE_STANDALONE_OUT}_RESULTS.zip.sha256.txt"
