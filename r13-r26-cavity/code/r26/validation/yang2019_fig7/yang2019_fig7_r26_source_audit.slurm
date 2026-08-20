#!/usr/bin/env bash
#SBATCH --job-name=r26-y19-src
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --qos=long
#SBATCH --output=r26-y19-src-%j.out
#SBATCH --error=r26-y19-src-%j.err
set -Eeuo pipefail
: "${YANG_ROOT:?}"; : "${YANG_REPO:?}"
PKG="$YANG_REPO/r13-r26-cavity/code/r26"; VAL="$PKG/validation/yang2019_fig7"; RUNS="$YANG_ROOT/runs"; REPORT="$YANG_ROOT/report"
mkdir -p "$RUNS" "$REPORT" "$YANG_ROOT/deliverables" "$YANG_ROOT/.matplotlib"
export PYTHONPATH="$PKG" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}" MPLCONFIGDIR="$YANG_ROOT/.matplotlib"
package_results(){ rc=$?; trap - EXIT; set +e; status=FAIL; [[ $rc -eq 0 ]]&&status=PASS; printf 'exit_code=%s\nstatus=%s\njob_id=%s\n' "$rc" "$status" "${SLURM_JOB_ID:-unknown}" > "$YANG_ROOT/JOB_STATUS.txt"; find "$RUNS" -type f \( -name last_accepted_state.npz -o -name run_summary.json \) -print0|sort -z|xargs -0 -r sha256sum > "$YANG_ROOT/accepted_outputs.sha256"; stamp=$(date -u +%Y%m%dT%H%M%SZ); z="$YANG_ROOT/deliverables/YANG2019_R26_SOURCE_AUDIT_JOB${SLURM_JOB_ID:-unknown}_${stamp}.zip"; (cd "$YANG_ROOT"&&zip -q -r "$z" JOB_STATUS.txt SOURCE_COMMIT.txt accepted_outputs.sha256 report runs 2>/dev/null); sha256sum "$z">"$z.sha256"; printf 'SOURCE_AUDIT_STATUS=%s\nZIP=%s\nSHA256=%s\n' "$status" "$z" "$z.sha256"; exit $rc; }; trap package_results EXIT
git -C "$YANG_REPO" rev-parse HEAD|tee "$YANG_ROOT/SOURCE_COMMIT.txt"
python3 -m unittest discover -s "$PKG/tests" -p 'test_maxwell_contract.py'; python3 -m py_compile "$VAL/validate_yang2019_asme_run.py" "$VAL/score_yang2019_fig7.py"
run_grid(){ n=$1; seed=${2:-}; out="$RUNS/N$n"; args=(--nodes "$n" --case-family gu-asme2009 --kn-gu .10 --lid-speed-m-s 10 --wall-temperature-k 273 --beta 1.6 --closure-mode asme2009-cavity --solver colored_newton --raw-tolerance 1e-8 --solver-tolerance 1e-9 --max-nfev 220 --initial-step .01 --minimum-step .00015625 --output-dir "$out"); [[ -z $seed ]]||args+=(--initial-state "$seed" --reconcile-initial); python3 "$PKG/analysis/run_jfm_observability_continuation.py" "${args[@]}"; python3 "$VAL/validate_yang2019_asme_run.py" "$out" --expected-nodes "$n" --raw-tolerance 1e-8; }
run_grid 17; run_grid 21 "$RUNS/N17/last_accepted_state.npz"; run_grid 25 "$RUNS/N21/last_accepted_state.npz"
python3 "$VAL/score_yang2019_fig7.py" --state "N17=$RUNS/N17/last_accepted_state.npz" --state "N21=$RUNS/N21/last_accepted_state.npz" --state "N25=$RUNS/N25/last_accepted_state.npz" --output-dir "$REPORT"
