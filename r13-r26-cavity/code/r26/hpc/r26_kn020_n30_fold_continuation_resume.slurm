#!/bin/bash
#SBATCH --job-name=r26-k020-fold30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --output=r26-k020-fold30-%j.out
#SBATCH --error=r26-k020-fold30-%j.err

set -euo pipefail

: "${R26_BALANCED_GATE_DIR:?Set the passed N8/N16 balanced-metric gate}"
: "${R26_N30_FAILED_DIR:?Set the failed fresh-N30 directory}"
: "${R26_ARC_FAILED_DIR:?Set the first failed N30 arclength directory}"
: "${R26_BALANCED_FAILED_DIR:?Set the failed balanced N30 directory}"
: "${R26_FOLD_N30_OUT:?Set a new output directory}"
: "${R26_FOLD_N30_REF:?Set the immutable 40-character source commit}"

EXPECTED_BALANCED_GATE_REF="93fd4b55b8932bcfedb36f1c66e90b443e7744e2"
EXPECTED_PRODUCTION_REF="312ee29799e5fdb4340d1146af5c408d72563d49"
EXPECTED_FAILED_ARC_REF="380a5cb05f0620813c255a086ca899b4051ea2ae"
EXPECTED_BALANCED_FAILURE_REF="7c69348b051d5515625a7746dd46b90ce5c06619"
TARGET_LID="0.40032038451271784"

[[ "$R26_FOLD_N30_REF" =~ ^[0-9a-f]{40}$ ]] || {
  echo "R26_FOLD_N30_REF must be a lowercase 40-character Git commit SHA" >&2
  exit 2
}
test -f "$R26_BALANCED_GATE_DIR/BALANCED_METRIC_GATE_PASSED.json"
test -f "$R26_BALANCED_GATE_DIR/GATE/BALANCED_METRIC_GATE_PASSED.json"
test -f "$R26_N30_FAILED_DIR/N30_PRODUCTION_FAILED.json"
test -f "$R26_N30_FAILED_DIR/N30/run_summary.json"
test -f "$R26_ARC_FAILED_DIR/N30_ARCLENGTH_RESCUE_FAILED.json"
test -f "$R26_ARC_FAILED_DIR/ARCLENGTH/arclength_summary.json"
test -f "$R26_BALANCED_FAILED_DIR/N30_BALANCED_ARCLENGTH_RESCUE_FAILED.json"
test -f "$R26_BALANCED_FAILED_DIR/source_commit.txt"
for stem in \
  arc_attempt_003_lid_0.37036616951 \
  arc_attempt_004_lid_0.370383245071
do
  test -f "$R26_BALANCED_FAILED_DIR/ARCLENGTH/${stem}.json"
  test -f "$R26_BALANCED_FAILED_DIR/ARCLENGTH/${stem}.npz"
done
test ! -e "$R26_FOLD_N30_OUT"
test ! -e "${R26_FOLD_N30_OUT}_RESULTS.zip"
mkdir -p "$R26_FOLD_N30_OUT"

exec > >(tee -a "$R26_FOLD_N30_OUT/run.stdout.log") \
     2> >(tee -a "$R26_FOLD_N30_OUT/run.stderr.log" >&2)

package_on_exit() {
  local return_code=$?
  local final_code=$return_code
  trap - EXIT
  set +e
  if (( return_code != 0 )); then
    python3 - "$R26_FOLD_N30_OUT" "$R26_FOLD_N30_REF" "$return_code" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
record = {
    "status": "R26_N30_FOLD_CONTINUATION_RESUME_FAILED",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_commit": sys.argv[2],
    "balanced_metric_gate_commit": "93fd4b55b8932bcfedb36f1c66e90b443e7744e2",
    "failed_production_commit": "312ee29799e5fdb4340d1146af5c408d72563d49",
    "failed_arclength_commit": "380a5cb05f0620813c255a086ca899b4051ea2ae",
    "balanced_failure_commit": "7c69348b051d5515625a7746dd46b90ce5c06619",
    "exit_code": int(sys.argv[3]),
    "n30_target_accepted": False,
    "note": "Diagnostic failure only; no accepted N30 target claim.",
}
(root / "N30_FOLD_CONTINUATION_RESUME_FAILED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
  fi
  python3 - "$R26_FOLD_N30_OUT" "${R26_FOLD_N30_OUT}_RESULTS.zip" <<'PY'
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
  if (( $? != 0 )) && (( final_code == 0 )); then final_code=70; fi
  (
    cd "$(dirname "$R26_FOLD_N30_OUT")"
    sha256sum "$(basename "${R26_FOLD_N30_OUT}_RESULTS.zip")" \
      > "$(basename "${R26_FOLD_N30_OUT}_RESULTS.zip").sha256.txt"
  )
  if (( $? != 0 )) && (( final_code == 0 )); then final_code=71; fi
  echo "R26_RESULTS_ZIP=${R26_FOLD_N30_OUT}_RESULTS.zip"
  echo "R26_RESULTS_SHA=${R26_FOLD_N30_OUT}_RESULTS.zip.sha256.txt"
  exit "$final_code"
}
trap package_on_exit EXIT

REPO="$R26_FOLD_N30_OUT/source"
git clone --filter=blob:none --no-checkout https://github.com/Ehsan-Roohi/Fourier.git "$REPO"
git -C "$REPO" fetch --depth=1 origin "$R26_FOLD_N30_REF"
git -C "$REPO" checkout --detach FETCH_HEAD
ACTUAL_REF="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_REF" = "$R26_FOLD_N30_REF"
printf '%s\n' "$ACTUAL_REF" > "$R26_FOLD_N30_OUT/source_commit.txt"
printf '%s\n' "$EXPECTED_BALANCED_GATE_REF" > "$R26_FOLD_N30_OUT/balanced_metric_gate_commit.txt"
printf '%s\n' "$EXPECTED_PRODUCTION_REF" > "$R26_FOLD_N30_OUT/failed_production_commit.txt"
printf '%s\n' "$EXPECTED_FAILED_ARC_REF" > "$R26_FOLD_N30_OUT/failed_arclength_commit.txt"
printf '%s\n' "$EXPECTED_BALANCED_FAILURE_REF" > "$R26_FOLD_N30_OUT/balanced_failure_commit.txt"

PKG="$REPO/r13-r26-cavity/code/r26"
export PYTHONPATH="$PKG"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

(cd "$PKG" && python3 run_tests.py)
python3 "$PKG/tools/validate_r26_balanced_metric_gate.py" \
  "$R26_BALANCED_GATE_DIR" \
  --expected-source-commit "$EXPECTED_BALANCED_GATE_REF" \
  --expected-parameter-fraction 0.5 --raw-tolerance 1e-8

mkdir -p "$R26_FOLD_N30_OUT/input_provenance"
cp "$R26_BALANCED_GATE_DIR/BALANCED_METRIC_GATE_PASSED.json" \
  "$R26_FOLD_N30_OUT/input_provenance/"
cp "$R26_BALANCED_FAILED_DIR/N30_BALANCED_ARCLENGTH_RESCUE_FAILED.json" \
  "$R26_FOLD_N30_OUT/input_provenance/"
cp "$R26_BALANCED_FAILED_DIR/source_commit.txt" \
  "$R26_FOLD_N30_OUT/input_provenance/balanced_failure_source_commit.txt"
for stem in \
  arc_attempt_003_lid_0.37036616951 \
  arc_attempt_004_lid_0.370383245071
do
  cp "$R26_BALANCED_FAILED_DIR/ARCLENGTH/${stem}.json" \
    "$R26_FOLD_N30_OUT/input_provenance/"
  cp "$R26_BALANCED_FAILED_DIR/ARCLENGTH/${stem}.npz" \
    "$R26_FOLD_N30_OUT/input_provenance/"
done
(
  cd "$R26_FOLD_N30_OUT"
  sha256sum input_provenance/* > input_provenance.sha256
)

# The scale is calibrated once from the original non-degenerate secant and is
# then held fixed.  A vanishing lid component is permitted because it is the
# defining geometry of a fold, not a loss of metric definiteness.
python3 "$PKG/analysis/run_r26_pseudo_arclength_rescue.py" \
  --failed-run-dir "$R26_N30_FAILED_DIR" \
  --failed-arclength-dir "$R26_ARC_FAILED_DIR" \
  --resume-arclength-dir "$R26_BALANCED_FAILED_DIR" \
  --output-dir "$R26_FOLD_N30_OUT/ARCLENGTH" \
  --expected-failed-source-commit "$EXPECTED_PRODUCTION_REF" \
  --expected-failed-arclength-source-commit "$EXPECTED_FAILED_ARC_REF" \
  --expected-resume-arclength-source-commit "$EXPECTED_BALANCED_FAILURE_REF" \
  --target-lid "$TARGET_LID" \
  --raw-tolerance 1e-8 --residual-tolerance 1e-9 \
  --parameter-metric-fraction 0.5 \
  --initial-step-factor 1.0 --minimum-step-factor 0.125 \
  --maximum-step-factor 2.0 --maximum-attempts 24 \
  --maximum-iterations 80 --maximum-jacobians 7 \
  --maximum-objective-evaluations 6000 \
  --pseudo-transient-chord-limit 12 --newton-chord-limit 3

python3 "$PKG/analysis/run_jfm_observability_continuation.py" \
  --nodes 30 --case-family jfm-maxwell --kn-gu 0.20 \
  --lid-speed-m-s 100 --wall-temperature-k 300 --beta 0.0 \
  --closure-mode jfm2009 --solver colored_newton \
  --analytic-mass-jacobian --secant-predictor --ser-ptc \
  --pseudo-time-initial 1.0 --pseudo-time-minimum 1e-8 \
  --pseudo-time-maximum 1e8 --newton-switch-tolerance 1e-6 \
  --max-jacobians 8 --raw-tolerance 1e-8 --solver-tolerance 1e-9 \
  --max-nfev 120 --max-objective-evaluations 6000 \
  --initial-state "$R26_FOLD_N30_OUT/ARCLENGTH/landing_seed.npz" \
  --reconcile-initial --output-dir "$R26_FOLD_N30_OUT/N30_TARGET"

python3 "$PKG/tools/validate_r26_maxwell_run.py" \
  "$R26_FOLD_N30_OUT/N30_TARGET" \
  --expected-kn 0.20 --expected-nodes 30 --expected-beta 0.0 \
  --raw-tolerance 1e-8
python3 "$PKG/tools/validate_r26_arclength_rescue.py" \
  "$R26_FOLD_N30_OUT/ARCLENGTH" "$R26_FOLD_N30_OUT/N30_TARGET" \
  --raw-tolerance 1e-8 --expected-target-lid "$TARGET_LID" \
  --expected-failed-arclength-source-commit "$EXPECTED_FAILED_ARC_REF" \
  --expected-continuation-resume-source-commit "$EXPECTED_BALANCED_FAILURE_REF" \
  --expected-parameter-metric-fraction 0.5

python3 - "$R26_FOLD_N30_OUT" "$ACTUAL_REF" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
arc = json.loads((root / "ARCLENGTH" / "arclength_summary.json").read_text())
target = json.loads((root / "N30_TARGET" / "run_summary.json").read_text())
accepted = [row for row in arc["attempts"] if bool(row["accepted"])]
record = {
    "status": "R26_N30_FOLD_CONTINUATION_RESUME_PASSED",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_commit": sys.argv[2],
    "balanced_metric_gate_commit": "93fd4b55b8932bcfedb36f1c66e90b443e7744e2",
    "failed_production_commit": "312ee29799e5fdb4340d1146af5c408d72563d49",
    "failed_arclength_commit": "380a5cb05f0620813c255a086ca899b4051ea2ae",
    "balanced_failure_commit": "7c69348b051d5515625a7746dd46b90ce5c06619",
    "resumed_from_lid": 0.37038324507064474,
    "nodes": 30,
    "kn_gu": 0.20,
    "target_lid": target["case"]["lid_target"],
    "metric_policy": arc["arclength_controls"]["metric_policy"],
    "accepted_arclength_points": len(accepted),
    "rejected_arclength_attempts": sum(not bool(row["accepted"]) for row in arc["attempts"]),
    "minimum_parameter_metric_fraction": min(
        float(row["parameter_metric_fraction"]) for row in arc["attempts"]
    ),
    "final_fixed_lid_raw_gate": target["attempts"][0]["raw_acceptance_gate"],
    "n30_target_accepted": True,
    "note": "Algebraically accepted N30 target; grid convergence and external validation remain separate requirements.",
}
(root / "N30_FOLD_CONTINUATION_RESUME_PASSED.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n"
)
PY

(
  cd "$R26_FOLD_N30_OUT"
  sha256sum \
    ARCLENGTH/arclength_summary.json ARCLENGTH/landing_seed.npz \
    N30_TARGET/last_accepted_state.npz N30_TARGET/run_summary.json \
    N30_FOLD_CONTINUATION_RESUME_PASSED.json input_provenance.sha256 \
    source_commit.txt balanced_metric_gate_commit.txt \
    failed_production_commit.txt failed_arclength_commit.txt \
    balanced_failure_commit.txt > N30_FOLD_CONTINUATION_RESUME.sha256
)

echo "R26_N30_FOLD_CONTINUATION_RESUME_COMPLETE"
