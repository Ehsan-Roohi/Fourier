#!/usr/bin/env python3
"""Validate one THOR N28 root against the immutable legacy N28 root."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r26_cases import jfm_maxwell_cavity_case
from r26_discretization import R26NodeBVP
from r26_fv_backend import compatible_fv_bulk_residual, wall_bounded_control_volume_weights
from r26_thor_audit import compare_cross_solver_profiles, state_sha256
from r26_thor_reconciliation import (
    EXPECTED_ROOT_FILE_SHA256,
    load_immutable_root,
    same_grid_cross_solver_passed,
)
from r26_thor_solver import make_thor_problem
from r26_validation import global_balance_diagnostics


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"R26_THOR_N28_CROSS_SOLVER_VALIDATION_FAILED: {message}")


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"required record missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"record is not an object: {path}")
    return value


def independent_gate(problem: object, state: np.ndarray, case: object, tolerance: float) -> dict[str, object]:
    result = problem.evaluate(state)
    diagnostics = asdict(result.diagnostics)
    raw = float(
        max(
            result.diagnostics.raw_total_linf,
            abs(result.diagnostics.held_out_continuity),
            abs(result.diagnostics.mass_error),
        )
    )
    balances = global_balance_diagnostics(state, case)
    passed = bool(
        raw <= tolerance
        and float(diagnostics["min_density"]) > 0.0
        and float(diagnostics["min_temperature"]) > 0.0
        and float(balances["wall_effective_pressure_min"]) > 0.0
        and float(balances["momentum_boundary_flux_linf"]) <= 10.0 * tolerance
        and abs(float(balances["internal_energy_balance_error"])) <= 10.0 * tolerance
    )
    return {
        "raw_acceptance_gate": raw,
        "diagnostics": diagnostics,
        "global_balances": balances,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thor-n28-dir", type=Path, required=True)
    parser.add_argument("--legacy-n28-dir", type=Path, required=True)
    parser.add_argument("--root-ladder-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--maximum-profile-nrms", type=float, default=0.05)
    parser.add_argument("--maximum-line-nrms", type=float, default=0.15)
    parser.add_argument("--maximum-DG-relative-difference", type=float, default=0.02)
    args = parser.parse_args()
    require(args.raw_tolerance > 0.0, "raw tolerance must be positive")
    require(args.maximum_profile_nrms > 0.0, "profile tolerance must be positive")
    require(args.maximum_line_nrms > 0.0, "line tolerance must be positive")
    require(args.maximum_DG_relative_difference > 0.0, "D/G tolerance must be positive")
    require(not args.output.exists(), "output record already exists")

    audit = read_json(args.root_ladder_audit)
    require(audit.get("status") == "R26_THOR_ROOT_LADDER_AUDIT_PASSED", "root ladder audit failed")
    require(audit.get("n28_run_authorized") is True, "N28 run was not authorized")
    require(audit.get("n29_authorized") is False, "root audit prematurely authorized N29")
    require(audit.get("production_accepted") is False, "root audit made a production claim")

    thor_record = read_json(args.thor_n28_dir / "thor_validation.json")
    require(thor_record.get("status") == "R26_THOR_VALIDATION_CANDIDATE_PASSED", "THOR N28 candidate failed")
    require(thor_record.get("physical_candidate_gate_passed") is True, "THOR N28 physical gate failed")
    require(thor_record.get("production_accepted") is False, "THOR N28 made a production claim")
    case_record = thor_record.get("case")
    require(isinstance(case_record, dict), "THOR N28 case record missing")
    require(int(case_record.get("nodes", -1)) == 28, "THOR N28 grid mismatch")
    require(case_record.get("kn_convention") == "gu_lambda_over_L", "THOR N28 Kn convention mismatch")

    thor_path = args.thor_n28_dir / "thor_state.npz"
    # The new THOR file cannot have a predeclared byte hash.  It is still
    # required to be accepted, finite and internally hash-matched to its record.
    require(thor_path.is_file(), "THOR N28 state missing")
    with np.load(thor_path, allow_pickle=False) as archive:
        require("accepted" in archive and bool(np.asarray(archive["accepted"]).item()), "THOR N28 state rejected")
        thor_state = np.asarray(archive["state"], dtype=float)
        thor_x = np.asarray(archive["x"], dtype=float)
        thor_y = np.asarray(archive["y"], dtype=float)
        thor_lid = float(np.asarray(archive["lid_velocity"]).item())
    require(thor_state.shape == (28, 28, 17), "THOR N28 state shape mismatch")
    require(np.isfinite(thor_state).all(), "THOR N28 state contains NaN/Inf")
    require(
        state_sha256(thor_state) == thor_record.get("state_sha256"),
        "THOR N28 record/state hash mismatch",
    )

    legacy = load_immutable_root(
        args.legacy_n28_dir / "last_accepted_state.npz",
        nodes=28,
        expected_file_sha256=EXPECTED_ROOT_FILE_SHA256[28],
        require_accepted_flag=False,
    )
    case = jfm_maxwell_cavity_case(
        28,
        kn=0.2,
        lid_speed_m_per_s=100.0,
        wall_temperature_K=300.0,
        grid_stretch_beta=0.0,
    )
    require(np.array_equal(thor_x, case.x) and np.array_equal(thor_y, case.y), "THOR N28 coordinates mismatch")
    require(math.isclose(thor_lid, case.lid_velocity, rel_tol=0.0, abs_tol=2.0e-14), "THOR N28 lid mismatch")
    require(np.array_equal(legacy.x, case.x) and np.array_equal(legacy.y, case.y), "legacy N28 coordinates mismatch")
    require(math.isclose(legacy.kn_input, 0.2, rel_tol=0.0, abs_tol=2.0e-15), "legacy N28 Kn mismatch")
    require(math.isclose(legacy.beta, 0.0, rel_tol=0.0, abs_tol=2.0e-15), "legacy N28 beta mismatch")
    require(math.isclose(legacy.lid_velocity, case.lid_velocity, rel_tol=0.0, abs_tol=2.0e-14), "legacy N28 lid mismatch")

    thor_gate = independent_gate(make_thor_problem(case), thor_state, case, args.raw_tolerance)
    legacy_problem = R26NodeBVP(
        case,
        bulk_operator=compatible_fv_bulk_residual,
        mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
    )
    legacy_gate = independent_gate(legacy_problem, legacy.state, case, args.raw_tolerance)
    comparison = compare_cross_solver_profiles(
        legacy.state,
        legacy.x,
        legacy.y,
        thor_state,
        thor_x,
        thor_y,
        lid_velocity=case.lid_velocity,
        target_n=128,
    )
    comparison_passed = same_grid_cross_solver_passed(
        comparison,
        maximum_profile_nrms=args.maximum_profile_nrms,
        maximum_line_nrms=args.maximum_line_nrms,
        maximum_dg_relative_difference=args.maximum_DG_relative_difference,
    )
    passed = bool(thor_gate["passed"] and legacy_gate["passed"] and comparison_passed)
    record = {
        "status": (
            "R26_THOR_N28_CROSS_SOLVER_VALIDATION_PASSED"
            if passed
            else "R26_THOR_N28_CROSS_SOLVER_VALIDATION_FAILED"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "thor_record_state_sha256": thor_record.get("state_sha256"),
        "thor_independent_gate": thor_gate,
        "legacy_file_sha256": legacy.file_sha256,
        "legacy_state_sha256": legacy.state_sha256,
        "legacy_independent_gate": legacy_gate,
        "cross_solver_comparison": comparison,
        "cross_solver_comparison_passed": comparison_passed,
        "thresholds": {
            "raw_tolerance": args.raw_tolerance,
            "maximum_profile_nrms": args.maximum_profile_nrms,
            "maximum_line_nrms": args.maximum_line_nrms,
            "maximum_DG_relative_difference": args.maximum_DG_relative_difference,
            "provenance": "unchanged thresholds from the passed N8/N16 THOR cross-solver audit",
        },
        "n28_accepted": passed,
        "n29_authorized": passed,
        "n30_authorized": False,
        "production_accepted": False,
        "legacy_n28_used_as_solver_seed": False,
    }
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
