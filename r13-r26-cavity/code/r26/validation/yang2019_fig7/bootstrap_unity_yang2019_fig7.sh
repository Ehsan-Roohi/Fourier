#!/usr/bin/env bash
set -euo pipefail

BASE="${R26_VALIDATION_BASE:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify}"
PINNED_COMMIT="9d442c00da3187f4a0b8a170cc03ad1171f79009"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="$BASE/YANG2019_R26_VALIDATION_${STAMP}"
REPO="$ROOT/repo"
SLURM_REL="r13-r26-cavity/code/r26/validation/yang2019_fig7/yang2019_fig7_r26.slurm"

[[ ! -e "$ROOT" ]] || { echo "Campaign root already exists: $ROOT" >&2; exit 2; }
mkdir -p "$ROOT"
git init -q "$REPO"
git -C "$REPO" remote add origin https://github.com/Ehsan-Roohi/Fourier.git
git -C "$REPO" fetch -q --depth 1 origin "$PINNED_COMMIT"
git -C "$REPO" checkout -q --detach FETCH_HEAD
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$PINNED_COMMIT" ]] || {
  echo "Pinned source checkout failed" >&2; exit 3;
}
[[ -s "$REPO/$SLURM_REL" ]] || { echo "Published Slurm driver missing" >&2; exit 4; }

cd "$ROOT"
JOB="$(sbatch --parsable \
  --export=ALL,YANG_ROOT="$ROOT",YANG_REPO="$REPO" \
  "$REPO/$SLURM_REL")"
{
  printf 'campaign_root=%s\njob=%s\nsource_commit=%s\nsubmitted_utc=%s\n' \
    "$ROOT" "$JOB" "$PINNED_COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$ROOT/SUBMISSION.txt"
echo "SUBMITTED job=$JOB root=$ROOT"
