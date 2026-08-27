#!/bin/bash
#SBATCH --job-name=r26-thor816
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --output=r26-thor816-%j.out
#SBATCH --error=r26-thor816-%j.err

set -euo pipefail

: "${R26_THOR_OUT:?Set R26_THOR_OUT to a new output directory}"
: "${R26_THOR_REF:?Set R26_THOR_REF to the immutable 40-character commit SHA}"

[[ "$R26_THOR_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_THOR_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}
test ! -e "$R26_THOR_OUT"
test ! -e "${R26_THOR_OUT}_RESULTS.zip"
mkdir -p "$R26_THOR_OUT"

exec > >(tee -a "$R26_THOR_OUT/run.stdout.log") \
     2> >(tee -a "$R26_THOR_OUT/run.stderr.log" >&2)

package_on_exit() {
  local return_code=$?
  trap - EXIT
  set +e
  if (( return_code != 0 )); then
    python3 - "$R26_THOR_OUT" "$R26_THOR_REF" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
record = {
    "status": "R26_THOR_N8_N16_GATE_FAILED",
    "source_commit": sys.argv[2],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n28_authorized": False,
    "n30_authorized": False,
    "production_accepted": False,
}
(root / "THOR_N8_N16_GATE_FAILED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  fi
  local zip_path="${R26_THOR_OUT}_RESULTS.zip"
  local sha_path="${zip_path}.sha256.txt"
  (cd "$(dirname "$R26_THOR_OUT")" && zip -qr "$zip_path" "$(basename "$R26_THOR_OUT")")
  (cd "$(dirname "$zip_path")" && sha256sum "$(basename "$zip_path")" > "$(basename "$sha_path")")
  exit "$return_code"
}
trap package_on_exit EXIT

REPO="$R26_THOR_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_THOR_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_THOR_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_THOR_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)

python3 "$PKG/analysis/run_r26_thor_validation.py" \
  --case-family jfm-maxwell \
  --nodes 8 \
  --kn-gu 0.2 \
  --lid-speed-m-s 100 \
  --output-dir "$R26_THOR_OUT/N8" \
  --max-iterations 60 \
  --inner-max-iterations 140 \
  --raw-tolerance 1e-8 \
  --solver-tolerance 1e-9 \
  --ilu-drop-tolerance 1e-6 \
  --ilu-fill-factor 48

python3 "$PKG/analysis/run_r26_thor_validation.py" \
  --case-family jfm-maxwell \
  --nodes 16 \
  --kn-gu 0.2 \
  --lid-speed-m-s 100 \
  --initial-state "$R26_THOR_OUT/N8/thor_state.npz" \
  --output-dir "$R26_THOR_OUT/N16" \
  --max-iterations 80 \
  --inner-max-iterations 160 \
  --raw-tolerance 1e-8 \
  --solver-tolerance 1e-9 \
  --ilu-drop-tolerance 1e-6 \
  --ilu-fill-factor 48

python3 "$PKG/analysis/compare_r26_thor_candidates.py" \
  "$R26_THOR_OUT/N8/thor_state.npz" \
  "$R26_THOR_OUT/N16/thor_state.npz" \
  --output "$R26_THOR_OUT/N8_N16_GRID_SENSITIVITY.json"

python3 - "$R26_THOR_OUT" "$ACTUAL_REF" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
n8 = json.loads((root / "N8" / "thor_validation.json").read_text())
n16 = json.loads((root / "N16" / "thor_validation.json").read_text())
for nodes, record in ((8, n8), (16, n16)):
    if record.get("status") != "R26_THOR_VALIDATION_CANDIDATE_PASSED":
        raise SystemExit(f"N{nodes} THOR candidate did not pass")
    if record.get("physical_candidate_gate_passed") is not True:
        raise SystemExit(f"N{nodes} physical candidate gate is not boolean true")
    if record.get("production_accepted") is not False:
        raise SystemExit(f"N{nodes} incorrectly made a production claim")
    if float(record.get("raw_acceptance_gate", float("inf"))) > 1.0e-8:
        raise SystemExit(f"N{nodes} raw gate failed")
record = {
    "status": "R26_THOR_N8_N16_GATE_PASSED",
    "source_commit": sys.argv[2],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n8_state_sha256": n8["state_sha256"],
    "n16_state_sha256": n16["state_sha256"],
    "n8_raw_gate": n8["raw_acceptance_gate"],
    "n16_raw_gate": n16["raw_acceptance_gate"],
    "grid_sensitivity_record": str((root / "N8_N16_GRID_SENSITIVITY.json").resolve()),
    "n28_authorized": False,
    "n30_authorized": False,
    "production_accepted": False,
    "next_required_stage": "independent final numerical-Jacobian rank and profile comparison on N8/N16",
}
(root / "THOR_N8_N16_GATE_PASSED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "R26_THOR_N8_N16_GATE_COMPLETE"
echo "R26_RESULTS_ZIP=${R26_THOR_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_THOR_OUT}_RESULTS.zip.sha256.txt"
