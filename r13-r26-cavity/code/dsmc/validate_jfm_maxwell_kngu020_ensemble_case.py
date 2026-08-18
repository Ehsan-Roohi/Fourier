#!/usr/bin/env python3
"""Fail-closed validator for one blocked Maxwell-VSS KnGu=0.20 run."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


SPARTA_SOURCE_COMMIT = "912c9e163c38ea5c3562d039e65215f6e2a4f3f8"
FIELDS = ["nrho", "u", "v", "w", "T", "qx", "qy", "Pxx", "Pxy",
          "Pyy", "Pzz", "B1xx", "B1xy", "B1yy", "B1zz"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"MAXWELL_ENSEMBLE_VALIDATION_FAILED: {message}")


def read_dump(path: Path, nx: int, fix_id: str) -> np.ndarray:
    require(path.is_file() and path.stat().st_size > 0, f"missing dump {path.name}")
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith("ITEM: CELLS")), -1)
    require(header_index >= 0, f"missing CELLS header in {path.name}")
    expected = ["ITEM:", "CELLS", "id", "xc", "yc", *
                [f"f_{fix_id}[{i}]" for i in range(1, len(FIELDS) + 1)]]
    require(lines[header_index].split() == expected, f"wrong schema in {path.name}")
    data = np.loadtxt(path, skiprows=header_index + 1)
    require(data.shape == (nx * nx, 3 + len(FIELDS)), f"wrong row/column count in {path.name}")
    require(np.isfinite(data).all(), f"non-finite values in {path.name}")
    require(np.unique(data[:, 0].astype(np.int64)).size == nx * nx, f"duplicate cell ids in {path.name}")
    require(float(data[:, 3].min()) > 0.0, f"non-positive density in {path.name}")
    require(float(data[:, 7].min()) > 0.0, f"non-positive temperature in {path.name}")
    return data


def validate(case: Path, args: argparse.Namespace) -> dict[str, object]:
    metadata_path = case / "case_metadata.json"
    require(metadata_path.is_file(), "case_metadata.json missing")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    exact = {
        "kn_convention": "gu_lambda_over_L",
        "molecular_model": "VSS transport approximation to IPL Maxwell molecules",
        "nx": args.nx,
        "ny": args.nx,
        "particles_per_cell": args.ppc,
        "seed": args.seed,
        "warmup_steps": args.warmup,
        "sample_steps": args.sample,
        "sample_stride": args.stride,
        "block_steps": args.block,
        "block_count": args.sample // args.block,
        "accumulated_samples_per_cell": args.sample // args.stride,
        "samples_per_block_per_cell": args.block // args.stride,
        "viscosity_index": 1.0,
        "vss_alpha": 2.14,
        "dump_schema_version": "maxwell_vss_antifourier_v2_blocked_15_fields",
        "dump_columns": FIELDS,
        "evidence_level": "blocked_realisation_for_ensemble_audit",
    }
    wrong = {key: (meta.get(key), expected) for key, expected in exact.items()
             if meta.get(key) != expected}
    require(not wrong, f"metadata contract mismatch: {wrong}")
    require(math.isclose(float(meta["kn_gu"]), 0.20, rel_tol=0.0, abs_tol=2e-15), "KnGu is not 0.20")
    require(math.isclose(float(meta["kn_gu_reconstructed"]), 0.20, rel_tol=0.0, abs_tol=2e-15), "reconstructed KnGu mismatch")
    require(float(meta["dx_over_lambda_gu"]) <= 1.0 / 20.0, "cell width exceeds lambdaGu/20")
    require(float(meta["dt_over_collision_time"]) <= 0.02, "time step exceeds 0.02 collision times")
    require(float(meta["initial_simulator_particles"]) == args.nx * args.nx * args.ppc,
            "initial particle count mismatch")

    deck = (case / "in.cavity").read_text(encoding="utf-8")
    for token in ("collide              vss gas maxwell.vss",
                  "fix                  blockavg ave/grid",
                  "fix                  finalavg ave/grid",
                  "f_blockavg[*]", "f_finalavg[*]"):
        require(token in deck, f"input-deck contract missing {token!r}")

    # Fail before the expensive DSMC run if a dump references a fix that the
    # generated input deck never defined.  The previous checkpoint dump used
    # ``f_fieldavg[*]`` without a matching ``fix fieldavg`` and reached a
    # SPARTA memory fault only at the first production checkpoint.
    fix_ids = {
        fields[1]
        for line in deck.splitlines()
        if (fields := line.split()) and fields[0] == "fix" and len(fields) > 1
    }
    referenced_fix_ids = set(re.findall(r"\bf_([A-Za-z0-9_]+)\[", deck))
    undefined_fix_ids = sorted(referenced_fix_ids - fix_ids)
    require(not undefined_fix_ids,
            f"input deck references undefined fix IDs: {undefined_fix_ids}")

    if not args.require_final:
        return {"status": "generated_case_pass", "case": str(case)}

    blocks = sorted(case.glob("grid.block.*"))
    require(len(blocks) == args.sample // args.block, "incomplete independent block series")
    for block in blocks:
        read_dump(block, args.nx, "blockavg")
    finals = sorted(case.glob("grid.final.*"))
    require(len(finals) == 1, "expected exactly one final production mean")
    final = read_dump(finals[0], args.nx, "finalavg")

    target_nrho = float(meta["number_density_m-3"])
    mass_error = float(final[:, 3].mean() / target_nrho - 1.0)
    require(abs(mass_error) <= 5.0e-3, f"domain mean density error {mass_error:.3e}")
    runmeta = (case / "unity_run_metadata.txt").read_text(encoding="utf-8")
    require(f"sparta_commit={SPARTA_SOURCE_COMMIT}" in runmeta, "unlocked SPARTA source commit")
    require("status=complete" in runmeta, "run not marked complete")
    require((case / "log.cavity").is_file(), "log.cavity missing")
    return {
        "status": "MAXWELL_ENSEMBLE_VALIDATION_PASS",
        "case": str(case),
        "nx": args.nx,
        "ppc": args.ppc,
        "seed": args.seed,
        "blocks": len(blocks),
        "samples_per_cell": args.sample // args.stride,
        "domain_mean_density_relative_error": mass_error,
        "final_dump": finals[0].name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ppc", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=200_000)
    parser.add_argument("--sample", type=int, default=2_000_000)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--block", type=int, default=200_000)
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args()
    require(args.sample % args.block == 0 and args.block % args.stride == 0,
            "invalid sample/block/stride relationship")
    result = validate(args.case.resolve(), args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
