#!/usr/bin/env python3
"""Fail-closed final N8/N16 THOR numerical-rank and SER--PTC comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from r26_cases import jfm_maxwell_cavity_case
from r26_discretization import R26NodeBVP
from r26_fv_backend import (
    compatible_fv_bulk_residual,
    wall_bounded_control_volume_weights,
)
from r26_thor_audit import (
    compare_cross_solver_profiles,
    numerical_jacobian_rank,
    raw_acceptance_gate,
    state_sha256,
)
from r26_thor_solver import make_thor_problem


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"R26_THOR_CROSS_SOLVER_AUDIT_FAILED: {message}")


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"required record missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"record is not an object: {path}")
    return value


def load_state(path: Path, *, require_accepted: bool) -> dict[str, object]:
    require(path.is_file(), f"state archive missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive["state"], dtype=float)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        lid = float(np.asarray(archive["lid_velocity"]).item())
        kn = float(np.asarray(archive["kn_input"]).item())
        beta = float(np.asarray(archive["beta"]).item())
        if require_accepted:
            require("accepted" in archive, f"accepted flag missing: {path}")
            require(bool(np.asarray(archive["accepted"]).item()), f"state rejected: {path}")
    return {
        "path": str(path.resolve()),
        "state": state,
        "x": x,
        "y": y,
        "lid": lid,
        "kn": kn,
        "beta": beta,
        "state_sha256": state_sha256(state),
    }


def make_legacy_problem(case: object) -> R26NodeBVP:
    return R26NodeBVP(
        case,
        bulk_operator=compatible_fv_bulk_residual,
        mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thor-dir", type=Path, required=True)
    parser.add_argument("--legacy-gate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-thor-source-commit", required=True)
    parser.add_argument("--expected-legacy-source-commit", required=True)
    parser.add_argument("--raw-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--minimum-scaled-rcond", type=float, default=1.0e-8)
    parser.add_argument("--maximum-profile-nrms", type=float, default=0.05)
    parser.add_argument("--maximum-line-nrms", type=float, default=0.15)
    parser.add_argument("--maximum-DG-relative-difference", type=float, default=0.02)
    args = parser.parse_args()
    require(args.raw_tolerance > 0.0, "raw tolerance must be positive")
    require(not args.output.exists(), "output record already exists")

    thor_gate = read_json(args.thor_dir / "THOR_N8_N16_GATE_PASSED.json")
    legacy_gate = read_json(args.legacy_gate_dir / "N16_GATE_PASSED.json")
    require(thor_gate.get("status") == "R26_THOR_N8_N16_GATE_PASSED", "THOR gate did not pass")
    require(
        thor_gate.get("source_commit") == args.expected_thor_source_commit,
        "THOR source commit mismatch",
    )
    require(legacy_gate.get("status") == "R26_N16_GATE_PASSED", "legacy gate did not pass")
    require(
        legacy_gate.get("source_commit") == args.expected_legacy_source_commit,
        "legacy source commit mismatch",
    )

    grids: list[dict[str, object]] = []
    for nodes in (8, 16):
        thor_record = read_json(args.thor_dir / f"N{nodes}" / "thor_validation.json")
        legacy_summary = read_json(args.legacy_gate_dir / f"N{nodes}" / "run_summary.json")
        require(
            thor_record.get("status") == "R26_THOR_VALIDATION_CANDIDATE_PASSED",
            f"N{nodes} THOR candidate did not pass",
        )
        require(thor_record.get("physical_candidate_gate_passed") is True, f"N{nodes} THOR physical gate failed")
        require(thor_record.get("production_accepted") is False, f"N{nodes} THOR premature production claim")
        require(legacy_summary.get("termination") == "target_accepted", f"N{nodes} legacy target not accepted")

        thor = load_state(
            args.thor_dir / f"N{nodes}" / "thor_state.npz",
            require_accepted=True,
        )
        legacy = load_state(
            args.legacy_gate_dir / f"N{nodes}" / "last_accepted_state.npz",
            require_accepted=False,
        )
        for label, candidate in (("THOR", thor), ("legacy", legacy)):
            require(np.asarray(candidate["state"]).shape == (nodes, nodes, 17), f"N{nodes} {label} state shape mismatch")
            require(math.isclose(float(candidate["kn"]), 0.2, rel_tol=0.0, abs_tol=2.0e-15), f"N{nodes} {label} Kn mismatch")
            require(math.isclose(float(candidate["beta"]), 0.0, rel_tol=0.0, abs_tol=2.0e-15), f"N{nodes} {label} beta mismatch")
        require(math.isclose(float(thor["lid"]), float(legacy["lid"]), rel_tol=0.0, abs_tol=2.0e-14), f"N{nodes} lid mismatch")
        require(thor["state_sha256"] == thor_record.get("state_sha256"), f"N{nodes} THOR state hash mismatch")
        legacy_attempts = legacy_summary.get("attempts", [])
        require(bool(legacy_attempts), f"N{nodes} legacy attempts missing")
        require(legacy["state_sha256"] == legacy_attempts[-1].get("state_sha256"), f"N{nodes} legacy state hash mismatch")

        case = jfm_maxwell_cavity_case(
            nodes,
            kn=0.2,
            lid_speed_m_per_s=100.0,
            wall_temperature_K=300.0,
            grid_stretch_beta=0.0,
        )
        require(math.isclose(float(thor["lid"]), case.lid_velocity, rel_tol=0.0, abs_tol=2.0e-14), f"N{nodes} target lid mismatch")
        thor_problem = make_thor_problem(case)
        legacy_problem = make_legacy_problem(case)
        thor_raw = raw_acceptance_gate(thor_problem, np.asarray(thor["state"]))
        legacy_raw = raw_acceptance_gate(legacy_problem, np.asarray(legacy["state"]))
        raw_passed = bool(
            thor_raw <= args.raw_tolerance and legacy_raw <= args.raw_tolerance
        )

        rank = numerical_jacobian_rank(
            thor_problem,
            np.asarray(thor["state"]),
            minimum_reciprocal_condition=args.minimum_scaled_rcond,
        )
        comparison = compare_cross_solver_profiles(
            np.asarray(legacy["state"]),
            np.asarray(legacy["x"]),
            np.asarray(legacy["y"]),
            np.asarray(thor["state"]),
            np.asarray(thor["x"]),
            np.asarray(thor["y"]),
            lid_velocity=case.lid_velocity,
        )
        comparison_passed = bool(
            float(comparison["maximum_normalized_rms_difference"]) <= args.maximum_profile_nrms
            and float(comparison["maximum_line_normalized_rms_difference"]) <= args.maximum_line_nrms
            and float(comparison["D_relative_difference"]) <= args.maximum_DG_relative_difference
            and float(comparison["G_relative_difference"]) <= args.maximum_DG_relative_difference
        )
        grid_passed = bool(raw_passed and rank.passed and comparison_passed)
        grids.append(
            {
                "nodes": nodes,
                "thor_state_sha256": thor["state_sha256"],
                "legacy_state_sha256": legacy["state_sha256"],
                "independent_thor_raw_gate": thor_raw,
                "independent_legacy_raw_gate": legacy_raw,
                "independent_raw_gates_passed": raw_passed,
                "numerical_rank": asdict(rank),
                "cross_solver_comparison": comparison,
                "cross_solver_comparison_passed": comparison_passed,
                "grid_audit_passed": grid_passed,
            }
        )

    n24_authorized = all(bool(grid["grid_audit_passed"]) for grid in grids)
    record = {
        "status": (
            "R26_THOR_CROSS_SOLVER_AUDIT_PASSED"
            if n24_authorized
            else "R26_THOR_CROSS_SOLVER_AUDIT_FAILED"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "thor_source_commit": args.expected_thor_source_commit,
        "legacy_source_commit": args.expected_legacy_source_commit,
        "grids": grids,
        "thresholds": {
            "raw_tolerance": args.raw_tolerance,
            "minimum_scaled_rcond": args.minimum_scaled_rcond,
            "maximum_profile_nrms": args.maximum_profile_nrms,
            "maximum_line_nrms": args.maximum_line_nrms,
            "maximum_DG_relative_difference": args.maximum_DG_relative_difference,
        },
        "n24_authorized": n24_authorized,
        "n28_authorized": False,
        "n30_authorized": False,
        "production_accepted": False,
        "note": (
            "N24 only is authorized; no grid-convergence or external-validation claim."
            if n24_authorized
            else "N24 is not authorized; inspect the per-grid audit diagnostics."
        ),
    }
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    raise SystemExit(0 if n24_authorized else 1)


if __name__ == "__main__":
    main()
