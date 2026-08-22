#!/bin/bash
#SBATCH --job-name=r26-k020-arc30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --output=r26-k020-arc30-%j.out
#SBATCH --error=r26-k020-arc30-%j.err

set -euo pipefail

: "${R26_N30_FAILED_DIR:?Set R26_N30_FAILED_DIR to the failed fresh-N30 directory}"
: "${R26_ARC_OUT:?Set R26_ARC_OUT to a new output directory}"
: "${R26_ARC_REF:?Set R26_ARC_REF to the immutable 40-character rescue commit SHA}"

EXPECTED_FAILED_REF="312ee29799e5fdb4340d1146af5c408d72563d49"
TARGET_LID="0.40032038451271784"

[[ "$R26_ARC_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_ARC_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}
test -d "$R26_N30_FAILED_DIR"
test -f "$R26_N30_FAILED_DIR/N30_PRODUCTION_FAILED.json"
test -f "$R26_N30_FAILED_DIR/source_commit.txt"
test -f "$R26_N30_FAILED_DIR/N30/run_summary.json"
test ! -e "$R26_ARC_OUT"
test ! -e "${R26_ARC_OUT}_RESULTS.zip"
mkdir -p "$R26_ARC_OUT"

exec > >(tee -a "$R26_ARC_OUT/run.stdout.log") \
     2> >(tee -a "$R26_ARC_OUT/run.stderr.log" >&2)

package_on_exit() {
  local return_code=$?
  local final_code=$return_code
  trap - EXIT
  set +e
  if (( return_code != 0 )); then
    python3 - "$R26_ARC_OUT" "$R26_ARC_REF" "$EXPECTED_FAILED_REF" "$return_code" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
record = {
    "status": "R26_N30_ARCLENGTH_RESCUE_FAILED",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_commit": sys.argv[2],
    "failed_production_commit": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "note": "A failed archive is diagnostic only and contains no accepted N30 target claim.",
}
(root / "N30_ARCLENGTH_RESCUE_FAILED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
  fi

  python3 - "$R26_ARC_OUT" "${R26_ARC_OUT}_RESULTS.zip" <<'PY'
import pathlib
import sys
import zipfile

root = pathlib.Path(sys.argv[1]).resolve()
archive = pathlib.Path(sys.argv[2]).resolve()
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_dir() or ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        handle.write(path, pathlib.Path(root.name) / relative)
PY
  if (( $? != 0 )) && (( final_code == 0 )); then
    final_code=70
  fi
  (
    cd "$(dirname "$R26_ARC_OUT")"
    sha256sum "$(basename "${R26_ARC_OUT}_RESULTS.zip")" \
      > "$(basename "${R26_ARC_OUT}_RESULTS.zip").sha256.txt"
  )
  if (( $? != 0 )) && (( final_code == 0 )); then
    final_code=71
  fi
  echo "R26_RESULTS_ZIP=${R26_ARC_OUT}_RESULTS.zip"
  echo "R26_RESULTS_SHA=${R26_ARC_OUT}_RESULTS.zip.sha256.txt"
  exit "$final_code"
}
trap package_on_exit EXIT

REPO="$R26_ARC_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_ARC_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_ARC_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_ARC_OUT/source_commit.txt"
printf '%s\n' "$EXPECTED_FAILED_REF" > "$R26_ARC_OUT/failed_production_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)

# The failed production archive supplied two independently accepted roots on
# the same N30 grid.  Pseudo-arclength promotes lid velocity to an unknown and
# adds one geometric hyperplane equation.  SER-PTC only globalizes the path;
# no R26 equation or final raw acceptance gate is changed.
python3 "$PKG/analysis/run_r26_pseudo_arclength_rescue.py" \
  --failed-run-dir "$R26_N30_FAILED_DIR" \
  --output-dir "$R26_ARC_OUT/ARCLENGTH" \
  --expected-failed-source-commit "$EXPECTED_FAILED_REF" \
  --target-lid "$TARGET_LID" \
  --raw-tolerance 1e-8 --residual-tolerance 1e-9 \
  --parameter-scale 0.04 \
  --initial-step-factor 1.0 --minimum-step-factor 0.125 \
  --maximum-step-factor 2.0 --maximum-attempts 24 \
  --maximum-jacobians 7 --maximum-objective-evaluations 6000

# Pseudo-arclength is used only to bracket and predict the target.  Make one
# ordinary fixed-lid correction at exactly 100 m/s and independently validate
# that unmodified R26 root before writing a success record.
python3 "$PKG/analysis/run_jfm_observability_continuation.py" \
  --nodes 30 --case-family jfm-maxwell --kn-gu 0.20 \
  --lid-speed-m-s 100 --wall-temperature-k 300 --beta 0.0 \
  --closure-mode jfm2009 --solver colored_newton \
  --analytic-mass-jacobian --secant-predictor --ser-ptc \
  --pseudo-time-initial 1.0 --pseudo-time-minimum 1e-8 \
  --pseudo-time-maximum 1e8 --newton-switch-tolerance 1e-6 \
  --max-jacobians 8 --raw-tolerance 1e-8 --solver-tolerance 1e-9 \
  --max-nfev 120 --max-objective-evaluations 6000 \
  --initial-state "$R26_ARC_OUT/ARCLENGTH/landing_seed.npz" \
  --reconcile-initial --output-dir "$R26_ARC_OUT/N30_TARGET"

python3 "$PKG/tools/validate_r26_maxwell_run.py" "$R26_ARC_OUT/N30_TARGET" \
  --expected-kn 0.20 --expected-nodes 30 --expected-beta 0.0 \
  --raw-tolerance 1e-8
python3 "$PKG/tools/validate_r26_arclength_rescue.py" \
  "$R26_ARC_OUT/ARCLENGTH" "$R26_ARC_OUT/N30_TARGET" \
  --raw-tolerance 1e-8 --expected-target-lid "$TARGET_LID"

python3 - "$R26_ARC_OUT" "$ACTUAL_REF" "$EXPECTED_FAILED_REF" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
arc = json.loads((root / "ARCLENGTH" / "arclength_summary.json").read_text(encoding="utf-8"))
target = json.loads((root / "N30_TARGET" / "run_summary.json").read_text(encoding="utf-8"))
accepted_arc = [row for row in arc["attempts"] if bool(row["accepted"])]
target_attempt = target["attempts"][0]
record = {
    "status": "R26_N30_ARCLENGTH_RESCUE_PASSED",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_commit": sys.argv[2],
    "failed_production_commit": sys.argv[3],
    "nodes": 30,
    "kn_gu": 0.20,
    "target_lid": target["case"]["lid_target"],
    "accepted_arclength_points": len(accepted_arc),
    "rejected_arclength_attempts": sum(not bool(row["accepted"]) for row in arc["attempts"]),
    "maximum_arclength_jacobians": max(int(row["solver"]["jacobian_evaluations"]) for row in arc["attempts"]),
    "final_fixed_lid_raw_gate": target_attempt["raw_acceptance_gate"],
    "note": "Algebraically accepted N30 root; grid convergence and external validation remain separate requirements.",
}
(root / "N30_ARCLENGTH_RESCUE_PASSED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

(
  cd "$R26_ARC_OUT"
  sha256sum \
    ARCLENGTH/arclength_summary.json \
    ARCLENGTH/landing_seed.npz \
    N30_TARGET/last_accepted_state.npz \
    N30_TARGET/run_summary.json \
    N30_ARCLENGTH_RESCUE_PASSED.json \
    source_commit.txt \
    failed_production_commit.txt \
    > N30_ARCLENGTH_RESCUE.sha256
)
