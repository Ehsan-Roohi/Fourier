#!/usr/bin/env bash
set -euo pipefail

BASE="${R26_CAMPAIGN_BASE:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="$BASE/R26_KNGU020_HOMOTOPY_${STAMP}"
REPO="$ROOT/repo"
DELIVERABLES="$BASE/deliverables"
REPO_URL="${R26_REPO_URL:-https://github.com/Ehsan-Roohi/Fourier.git}"
GIT_REF="${R26_GIT_REF:-main}"
SEED="${R26_N28_SEED:-$BASE/R26_KNGU020_N26_N30_LADDER_JOB63184894/N28/last_accepted_state.npz}"
EXPECTED_SEED_SHA="a28951fed0063f66dac5e0ec481a108f9a4599fdcfd83d6865b2a160ffa4a409"

[[ ! -e "$ROOT" ]] || { echo "Campaign root already exists: $ROOT" >&2; exit 2; }
[[ -f "$SEED" ]] || { echo "Accepted N28 seed missing: $SEED" >&2; exit 3; }
actual_seed_sha="$(sha256sum "$SEED" | awk '{print $1}')"
[[ "$actual_seed_sha" == "$EXPECTED_SEED_SHA" ]] || {
  echo "N28 seed hash mismatch: $actual_seed_sha" >&2
  exit 4
}

mkdir -p "$ROOT/logs" "$DELIVERABLES"
git clone --depth 1 --branch "$GIT_REF" "$REPO_URL" "$REPO"
SCRIPT="$REPO/r13-r26-cavity/code/r26/hpc/r26_kn020_homotopy_recovery.slurm"
[[ -s "$SCRIPT" ]] || { echo "Published recovery script missing: $SCRIPT" >&2; exit 5; }
git -C "$REPO" rev-parse HEAD > "$ROOT/repo_commit.txt"

JOB="$(sbatch --parsable --chdir="$ROOT" \
  --export=ALL,R26_REPO_ROOT="$REPO",R26_RECOVERY_ROOT="$ROOT",R26_N28_SEED="$SEED",R26_DELIVERABLES="$DELIVERABLES" \
  "$SCRIPT")"
{
  printf 'campaign_root=%s\njob=%s\nseed=%s\nseed_sha256=%s\n' \
    "$ROOT" "$JOB" "$SEED" "$actual_seed_sha"
  printf 'repo_url=%s\nrepo_ref=%s\nrepo_commit=%s\nsubmitted_utc=%s\n' \
    "$REPO_URL" "$GIT_REF" "$(git -C "$REPO" rev-parse HEAD)" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$ROOT/SUBMISSION.txt"
echo "SUBMITTED job=$JOB root=$ROOT"
