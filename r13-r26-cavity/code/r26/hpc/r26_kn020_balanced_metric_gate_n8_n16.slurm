#!/bin/bash
#SBATCH --job-name=r26-arc-bal816
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --qos=long
#SBATCH --output=r26-arc-bal816-%j.out
#SBATCH --error=r26-arc-bal816-%j.err

set -euo pipefail

: "${R26_N16_GATE_DIR:?Set R26_N16_GATE_DIR to the passed historical N8/N16 gate}"
: "${R26_ARC_METRIC_OUT:?Set R26_ARC_METRIC_OUT to a new output directory}"
: "${R26_ARC_METRIC_REF:?Set R26_ARC_METRIC_REF to the immutable 40-character commit SHA}"

EXPECTED_GATE_REF="8cbd874eea68dd475faa3f5e3fb318b49cc0c665"
[[ "$R26_ARC_METRIC_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_ARC_METRIC_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}
test -f "$R26_N16_GATE_DIR/N16_GATE_PASSED.json"
test -f "$R26_N16_GATE_DIR/N8/run_summary.json"
test -f "$R26_N16_GATE_DIR/N16/run_summary.json"
test ! -e "$R26_ARC_METRIC_OUT"
test ! -e "${R26_ARC_METRIC_OUT}_RESULTS.zip"
mkdir -p "$R26_ARC_METRIC_OUT"

exec > >(tee -a "$R26_ARC_METRIC_OUT/run.stdout.log") \
     2> >(tee -a "$R26_ARC_METRIC_OUT/run.stderr.log" >&2)

package_on_exit() {
  local return_code=$?
  trap - EXIT
  set +e
  if (( return_code != 0 )); then
    python3 - "$R26_ARC_METRIC_OUT" "$R26_ARC_METRIC_REF" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
record = {
    "status": "R26_BALANCED_ARCLENGTH_METRIC_GATE_FAILED",
    "source_commit": sys.argv[2],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n30_authorized": False,
    "note": "Diagnostic failure only; no N30 job is authorized by this archive.",
}
(root / "BALANCED_METRIC_GATE_FAILED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  fi
  local zip_path="${R26_ARC_METRIC_OUT}_RESULTS.zip"
  local sha_path="${zip_path}.sha256.txt"
  (cd "$(dirname "$R26_ARC_METRIC_OUT")" && zip -qr "$zip_path" "$(basename "$R26_ARC_METRIC_OUT")")
  (cd "$(dirname "$zip_path")" && sha256sum "$(basename "$zip_path")" > "$(basename "$sha_path")")
  exit "$return_code"
}
trap package_on_exit EXIT

REPO="$R26_ARC_METRIC_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_ARC_METRIC_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_ARC_METRIC_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_ARC_METRIC_OUT/source_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)
python3 "$PKG/tools/validate_r26_globalization_gate.py" \
  "$R26_N16_GATE_DIR" \
  --expected-source-commit "$EXPECTED_GATE_REF" \
  --raw-tolerance 1e-8

python3 "$PKG/analysis/run_r26_balanced_metric_gate.py" \
  --gate-dir "$R26_N16_GATE_DIR" \
  --output-dir "$R26_ARC_METRIC_OUT/GATE" \
  --expected-gate-source-commit "$EXPECTED_GATE_REF" \
  --parameter-metric-fraction 0.5

python3 - "$R26_ARC_METRIC_OUT" "$ACTUAL_REF" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
gate = json.loads((root / "GATE" / "BALANCED_METRIC_GATE_PASSED.json").read_text())
if gate.get("status") != "R26_BALANCED_ARCLENGTH_METRIC_GATE_PASS":
    raise SystemExit("balanced metric gate did not pass")
if gate.get("n30_authorized") is not False:
    raise SystemExit("metric gate changed N30 authorization")
record = {
    "status": "R26_BALANCED_ARCLENGTH_METRIC_GATE_PASSED",
    "source_commit": sys.argv[2],
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "n30_authorized": False,
    "n8_record": str((root / "GATE" / "N8" / "balanced_metric_replay.json").resolve()),
    "n16_record": str((root / "GATE" / "N16" / "balanced_metric_replay.json").resolve()),
    "note": "N8/N16 balanced-metric replay passed; N30 remains intentionally unauthorized.",
}
(root / "BALANCED_METRIC_GATE_PASSED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "R26_BALANCED_METRIC_GATE_COMPLETE"
echo "R26_RESULTS_ZIP=${R26_ARC_METRIC_OUT}_RESULTS.zip"
echo "R26_RESULTS_SHA=${R26_ARC_METRIC_OUT}_RESULTS.zip.sha256.txt"
