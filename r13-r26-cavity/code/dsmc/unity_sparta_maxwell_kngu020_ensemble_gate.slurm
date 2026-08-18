#!/usr/bin/env bash
#SBATCH --job-name=sp-mm020-gate
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sp-mm020-gate-%j.out
#SBATCH --error=logs/sp-mm020-gate-%j.err

set -euo pipefail
REPO_ROOT="${DSMC_REPO_ROOT:?DSMC_REPO_ROOT is required}"
RESULTS_BASE="${DSMC_RESULTS_BASE:?DSMC_RESULTS_BASE is required}"
ARRAY_JOB="${DSMC_ARRAY_JOB:?DSMC_ARRAY_JOB is required}"
CODE_DIR="${REPO_ROOT}/r13-r26-cavity/code/dsmc"
REPORT_DIR="${RESULTS_BASE}/ensemble_analysis"

python3 "$CODE_DIR/analyze_jfm_maxwell_kngu020_ensemble.py" \
  "$RESULTS_BASE" --output "$REPORT_DIR" --fail-on-gate
find "$RESULTS_BASE" -type f ! -name 'restart.*' -print0 | sort -z | xargs -0 sha256sum \
  > "$RESULTS_BASE/ALL_FILES.sha256"

ARCHIVE="$(dirname "$RESULTS_BASE")/DSMC_MAXWELL_KNGU020_ENSEMBLE_${ARRAY_JOB}.zip"
(cd "$(dirname "$RESULTS_BASE")" && zip -qr "$ARCHIVE" "$(basename "$RESULTS_BASE")" \
  -x '*/restart.maxwell020.*')
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
printf 'PUBLICATION_GATE_PASS\narchive=%s\n' "$ARCHIVE" \
  > "$RESULTS_BASE/PUBLICATION_GATE_PASS.txt"
echo "COMPLETE: $ARCHIVE"

