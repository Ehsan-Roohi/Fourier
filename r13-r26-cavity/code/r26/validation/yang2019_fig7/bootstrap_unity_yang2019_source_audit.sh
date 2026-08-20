#!/usr/bin/env bash
set -euo pipefail
BASE="${R26_VALIDATION_BASE:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify}"; PINNED_COMMIT="eb82a9dea3583224f67054fa0706bbde8e9bd5c6"; STAMP=$(date -u +%Y%m%dT%H%M%SZ); ROOT="$BASE/YANG2019_R26_SOURCE_AUDIT_$STAMP"; REPO="$ROOT/repo"; REL="r13-r26-cavity/code/r26/validation/yang2019_fig7/yang2019_fig7_r26_source_audit.slurm"
mkdir -p "$ROOT"; git init -q "$REPO"; git -C "$REPO" remote add origin https://github.com/Ehsan-Roohi/Fourier.git; git -C "$REPO" fetch -q --depth 1 origin "$PINNED_COMMIT"; git -C "$REPO" checkout -q --detach FETCH_HEAD; [[ $(git -C "$REPO" rev-parse HEAD) == "$PINNED_COMMIT" ]]
cd "$ROOT"; JOB=$(sbatch --parsable --export=ALL,YANG_ROOT="$ROOT",YANG_REPO="$REPO" "$REPO/$REL"); printf 'campaign_root=%s\njob=%s\nsource_commit=%s\n' "$ROOT" "$JOB" "$PINNED_COMMIT"|tee "$ROOT/SUBMISSION.txt"; echo "SUBMITTED job=$JOB root=$ROOT"
