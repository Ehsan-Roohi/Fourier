#!/bin/bash
#SBATCH --job-name=r26-ge-rec816
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=r26-ge-rec816-%j.out
#SBATCH --error=r26-ge-rec816-%j.err

set -euo pipefail

: "${R26_GE_RECON_OUT:?Set R26_GE_RECON_OUT to a new output directory}"
: "${R26_GE_RECON_REF:?Set R26_GE_RECON_REF to an immutable 40-character commit SHA}"
[[ "$R26_GE_RECON_REF" =~ ^[0-9a-f]{40}$ ]] || exit 2
test ! -e "$R26_GE_RECON_OUT"
test ! -e "${R26_GE_RECON_OUT}_RESULTS.zip"
mkdir -p "$R26_GE_RECON_OUT"

exec > >(tee -a "$R26_GE_RECON_OUT/run.stdout.log") \
     2> >(tee -a "$R26_GE_RECON_OUT/run.stderr.log" >&2)

package_on_exit() {
  local return_code=$?
  trap - EXIT
  set +e
  if (( return_code != 0 )); then
    python3 - "$R26_GE_RECON_OUT" "$R26_GE_RECON_REF" <<'PY'
import datetime, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
record = {
    "status": "R26_GU_EMERSON_RECONSTRUCTION_CROSS_GATE_FAILED",
    "source_commit": sys.argv[2],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "cross_solver_fixed_point_gate_passed": False,
    "standalone_from_equilibrium_passed": False,
    "production_accepted": False,
    "n24_authorized": False,
    "n28_authorized": False,
    "n29_authorized": False,
    "n30_authorized": False,
}
(root / "GU_EMERSON_RECONSTRUCTION_CROSS_GATE_FAILED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n"
)
PY
  fi
  local zip_path="${R26_GE_RECON_OUT}_RESULTS.zip"
  local sha_path="${zip_path}.sha256.txt"
  (cd "$(dirname "$R26_GE_RECON_OUT")" && zip -qr "$zip_path" "$(basename "$R26_GE_RECON_OUT")")
  (cd "$(dirname "$zip_path")" && sha256sum "$(basename "$zip_path")" > "$(basename "$sha_path")")
  exit "$return_code"
}
trap package_on_exit EXIT

REPO="$R26_GE_RECON_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_GE_RECON_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_GE_RECON_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_GE_RECON_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)

python3 "$PKG/analysis/run_r26_thor_validation.py" \
  --case-family gu-asme2009 --nodes 8 --kn-gu 0.1 --lid-speed-m-s 10 \
  --wall-temperature-k 273 --beta 0 --max-iterations 50 --inner-max-iterations 80 \
  --output-dir "$R26_GE_RECON_OUT/REFERENCE_N8"

python3 "$PKG/analysis/run_r26_thor_validation.py" \
  --case-family gu-asme2009 --nodes 16 --kn-gu 0.1 --lid-speed-m-s 10 \
  --wall-temperature-k 273 --beta 0 --max-iterations 50 --inner-max-iterations 80 \
  --output-dir "$R26_GE_RECON_OUT/REFERENCE_N16"

python3 "$PKG/analysis/run_r26_gu_emerson_reconstruction_cross_gate.py" \
  --reference-root "$R26_GE_RECON_OUT" \
  --output "$R26_GE_RECON_OUT/RECONSTRUCTION" \
  --source-commit "$ACTUAL_REF"

python3 "$PKG/tools/validate_r26_gu_emerson_reconstruction_gate.py" \
  --run-dir "$R26_GE_RECON_OUT/RECONSTRUCTION" \
  --expected-source-commit "$ACTUAL_REF"

echo "R26_GU_EMERSON_RECONSTRUCTION_CROSS_GATE_COMPLETE"
echo "R26_RESULTS_ZIP=${R26_GE_RECON_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_GE_RECON_OUT}_RESULTS.zip.sha256.txt"
