#!/usr/bin/env bash
#SBATCH --job-name=sp-mm020-ens
#SBATCH --partition=cpu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-23%8
#SBATCH --output=logs/sp-mm020-ens-%A_%a.out
#SBATCH --error=logs/sp-mm020-ens-%A_%a.err

set -euo pipefail

REPO_ROOT="${DSMC_REPO_ROOT:?DSMC_REPO_ROOT is required}"
RESULTS_BASE="${DSMC_RESULTS_BASE:?DSMC_RESULTS_BASE is required}"
SPARTA_BIN="${SPARTA_BIN:?SPARTA_BIN is required}"
CODE_DIR="${REPO_ROOT}/r13-r26-cavity/code/dsmc"
EXPECTED_SPARTA_COMMIT="912c9e163c38ea5c3562d039e65215f6e2a4f3f8"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"
WARMUP="${DSMC_WARMUP:-200000}"
SAMPLE="${DSMC_SAMPLE:-2000000}"
STRIDE="${DSMC_STRIDE:-100}"
BLOCK="${DSMC_BLOCK:-200000}"
SEEDS=(104729 130363 155921 196613 242081 32452843 49979687 67867967)

TASK="${SLURM_ARRAY_TASK_ID:?This file must be submitted as an array job}"
if (( TASK < 8 )); then
  DESIGN="primary"
  NX=160
  PPC=256
  SEED="${SEEDS[TASK]}"
else
  OFFSET=$((TASK - 8))
  GROUP=$((OFFSET / 4))
  SEED_INDEX=$((OFFSET % 4))
  SEED="${SEEDS[SEED_INDEX]}"
  case "$GROUP" in
    0) DESIGN="grid_coarse"; NX=120; PPC=256 ;;
    1) DESIGN="grid_fine";   NX=200; PPC=256 ;;
    2) DESIGN="ppc_low";     NX=160; PPC=128 ;;
    3) DESIGN="ppc_high";    NX=160; PPC=512 ;;
    *) echo "Invalid design group" >&2; exit 2 ;;
  esac
fi

module purge
module load "$OPENMPI_MODULE"
export OMP_NUM_THREADS=1
for path in "$CODE_DIR/generate_jfm_maxwell_kngu020_case.py" \
            "$CODE_DIR/validate_jfm_maxwell_kngu020_ensemble_case.py" "$SPARTA_BIN"; do
  [[ -f "$path" || -x "$path" ]] || { echo "Missing required path: $path" >&2; exit 3; }
done

SPARTA_SOURCE_DIR="$(dirname "$SPARTA_BIN")"
SPARTA_COMMIT="$(git -C "$SPARTA_SOURCE_DIR" rev-parse HEAD)"
[[ "$SPARTA_COMMIT" == "$EXPECTED_SPARTA_COMMIT" ]] || {
  echo "SPARTA commit mismatch: $SPARTA_COMMIT" >&2; exit 4;
}

RUN_DIR="${RESULTS_BASE}/${DESIGN}_N${NX}_P${PPC}/seed_${SEED}"
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite $RUN_DIR" >&2; exit 5; }
mkdir -p "$(dirname "$RUN_DIR")"
python3 "$CODE_DIR/generate_jfm_maxwell_kngu020_case.py" \
  --output "$RUN_DIR" --seed "$SEED" --kn-gu 0.20 --nx "$NX" --ppc "$PPC" \
  --warmup "$WARMUP" --sample "$SAMPLE" --stride "$STRIDE" \
  --checkpoint "$BLOCK" --block "$BLOCK"
python3 "$CODE_DIR/validate_jfm_maxwell_kngu020_ensemble_case.py" "$RUN_DIR" \
  --nx "$NX" --ppc "$PPC" --seed "$SEED" --warmup "$WARMUP" \
  --sample "$SAMPLE" --stride "$STRIDE" --block "$BLOCK"

{
  printf 'status=started\narray_job_id=%s\narray_task_id=%s\n' \
    "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" "$TASK"
  printf 'design=%s\nnx=%s\nppc=%s\nseed=%s\nkn_gu=0.20\n' "$DESIGN" "$NX" "$PPC" "$SEED"
  printf 'warmup_steps=%s\nsample_steps=%s\nstride=%s\nblock_steps=%s\n' \
    "$WARMUP" "$SAMPLE" "$STRIDE" "$BLOCK"
  printf 'collision_model=Maxwell-VSS_omega1_alpha2.140\n'
  printf 'sparta_commit=%s\ncode_commit=%s\n' "$SPARTA_COMMIT" "$(git -C "$REPO_ROOT" rev-parse HEAD)"
  printf 'hostname=%s\nslurm_ntasks=%s\nstarted_utc=%s\n' \
    "$(hostname)" "$SLURM_NTASKS" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RUN_DIR/unity_run_metadata.txt"

cd "$RUN_DIR"
set +e
mpirun -n 2 "$SPARTA_BIN" -help > mpi_preflight.out 2> mpi_preflight.err
PREFLIGHT=$?
set -e
grep -q '^Usage: sparta' mpi_preflight.out || { tail -80 mpi_preflight.err >&2; exit 6; }
for style in 'pflux/grid' 'sonine/grid' 'thermal/grid' 'eflux/grid'; do
  grep -Fq "$style" mpi_preflight.out || { echo "Missing SPARTA style $style" >&2; exit 6; }
done
printf 'mpi_preflight_status=%s\nmpi_preflight=pass\n' "$PREFLIGHT" >> unity_run_metadata.txt

set +e
mpirun -n "$SLURM_NTASKS" "$SPARTA_BIN" < in.cavity > screen.out 2> screen.err
STATUS=$?
set -e
if (( STATUS != 0 )); then
  printf 'finished_utc=%s\nstatus=failed\nsparta_exit=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$STATUS" >> unity_run_metadata.txt
  tail -120 screen.err >&2 || true
  exit "$STATUS"
fi
[[ -s log.sparta ]] || { echo "SPARTA log missing" >&2; exit 7; }
mv log.sparta log.cavity
printf 'finished_utc=%s\nstatus=complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> unity_run_metadata.txt

python3 "$CODE_DIR/validate_jfm_maxwell_kngu020_ensemble_case.py" "$RUN_DIR" \
  --nx "$NX" --ppc "$PPC" --seed "$SEED" --warmup "$WARMUP" \
  --sample "$SAMPLE" --stride "$STRIDE" --block "$BLOCK" --require-final \
  > validation_report.json
sha256sum case_metadata.json in.cavity maxwell.vss grid.final.* grid.block.* \
  > FILES.sha256
echo "COMPLETE: $RUN_DIR"
