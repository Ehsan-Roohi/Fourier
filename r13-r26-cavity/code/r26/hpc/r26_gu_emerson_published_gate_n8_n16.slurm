#!/bin/bash
#SBATCH --job-name=r26-ge-pub816
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=r26-ge-pub816-%j.out
#SBATCH --error=r26-ge-pub816-%j.err

set -euo pipefail

: "${R26_GE_PUBLISHED_OUT:?Set R26_GE_PUBLISHED_OUT to a new output directory}"
: "${R26_GE_PUBLISHED_REF:?Set R26_GE_PUBLISHED_REF to the immutable 40-character commit SHA}"
[[ "$R26_GE_PUBLISHED_REF" =~ ^[0-9a-f]{40}$ ]] || exit 2
test ! -e "$R26_GE_PUBLISHED_OUT"
test ! -e "${R26_GE_PUBLISHED_OUT}_RESULTS.zip"
mkdir -p "$R26_GE_PUBLISHED_OUT"

exec > >(tee -a "$R26_GE_PUBLISHED_OUT/run.stdout.log") \
     2> >(tee -a "$R26_GE_PUBLISHED_OUT/run.stderr.log" >&2)

package_on_exit() {
  local return_code=$?
  trap - EXIT
  set +e
  if (( return_code != 0 )); then
    python3 - "$R26_GE_PUBLISHED_OUT" "$R26_GE_PUBLISHED_REF" <<'PY'
import datetime, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
record = {
    "status": "R26_GU_EMERSON_PUBLISHED_ALGORITHM_GATE_FAILED",
    "source_commit": sys.argv[2],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "production_authorized": False,
    "n24_authorized": False,
    "n28_authorized": False,
    "n29_authorized": False,
    "n30_authorized": False,
}
(root / "GU_EMERSON_PUBLISHED_GATE_FAILED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n"
)
PY
  fi
  local zip_path="${R26_GE_PUBLISHED_OUT}_RESULTS.zip"
  local sha_path="${zip_path}.sha256.txt"
  (cd "$(dirname "$R26_GE_PUBLISHED_OUT")" && zip -qr "$zip_path" "$(basename "$R26_GE_PUBLISHED_OUT")")
  (cd "$(dirname "$zip_path")" && sha256sum "$(basename "$zip_path")" > "$(basename "$sha_path")")
  exit "$return_code"
}
trap package_on_exit EXIT

REPO="$R26_GE_PUBLISHED_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_GE_PUBLISHED_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_GE_PUBLISHED_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_GE_PUBLISHED_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)
python3 "$PKG/analysis/run_r26_gu_emerson_published_gate.py" \
  --source-commit "$ACTUAL_REF" \
  --output "$R26_GE_PUBLISHED_OUT/GU_EMERSON_PUBLISHED_GATE.json"

echo "R26_GU_EMERSON_PUBLISHED_ALGORITHM_GATE_COMPLETE"
echo "R26_RESULTS_ZIP=${R26_GE_PUBLISHED_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_GE_PUBLISHED_OUT}_RESULTS.zip.sha256.txt"
