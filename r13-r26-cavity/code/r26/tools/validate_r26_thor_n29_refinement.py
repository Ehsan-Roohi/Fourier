#!/usr/bin/env python3
"""Validate one bounded THOR N29 refinement from the accepted THOR N28 root.

This stage does not consult or load any historical N29/N30 state.  It
independently re-evaluates the accepted N28 and candidate N29 states with the
audited THOR residual and applies the already-established THOR profile, line,
and Rana D/G thresholds.  A pass authorizes one N30 validation stage but is
not itself a production claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.csgraph import structural_rank

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r26_cases import jfm_maxwell_cavity_case
from r26_solver import jacobian_sparsity
from r26_thor_audit import compare_cross_solver_profiles, state_sha256
from r26_thor_reconciliation import (
    json_native,
    same_grid_cross_solver_passed,
    sha256_file,
)
from r26_thor_solver import make_thor_problem
from r26_validation import global_balance_diagnostics


EXPECTED_N28_SOURCE_COMMIT = "743e284d89980cbedd188d4127aae133674e054e"
EXPECTED_N28_FILE_SHA256 = "21cf1a09daa3ef7a7ddca604bc508fa201dc67029fd85b8a41fe32e78947b5b0"
EXPECTED_N28_STATE_SHA256 = "76f9828a0543051744db8818197158f273d2fe3447e1ca7c550a5100ad6b32ec"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"R26_THOR_N29_REFINEMENT_VALIDATION_FAILED: {message}")


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"required record missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"record is not an object: {path}")
    return value


def load_state(
    path: Path,
    *,
    nodes: int,
    expected_file_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    require(path.is_file(), f"state archive missing: {path}")
    if expected_file_sha256 is not None:
        require(sha256_file(path) == expected_file_sha256, f"N{nodes} state byte hash changed")
    with np.load(path, allow_pickle=False) as archive:
        required = {"state", "x", "y", "lid_velocity", "accepted", "production_accepted"}
        require(required.issubset(archive.files), f"N{nodes} state keys are incomplete")
        accepted = np.asarray(archive["accepted"])
        production = np.asarray(archive["production_accepted"])
        require(accepted.shape == () and accepted.dtype.kind == "b", f"N{nodes} accepted flag invalid")
        require(production.shape == () and production.dtype.kind == "b", f"N{nodes} production flag invalid")
        require(bool(accepted.item()), f"N{nodes} state is rejected")
        require(not bool(production.item()), f"N{nodes} state made a premature production claim")
        state = np.asarray(archive["state"], dtype=float)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        lid_velocity = float(np.asarray(archive["lid_velocity"]).item())
    require(state.shape == (nodes, nodes, 17), f"N{nodes} state shape mismatch")
    require(x.shape == (nodes,) and y.shape == (nodes,), f"N{nodes} coordinate shape mismatch")
    require(np.isfinite(state).all(), f"N{nodes} state contains NaN/Inf")
    return state, x, y, lid_velocity


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
    balances = json_native(global_balance_diagnostics(state, case))
    require(isinstance(balances, dict), "global balance diagnostics are not an object")
    rank = int(structural_rank(jacobian_sparsity(problem)))
    passed = bool(
        raw <= tolerance
        and float(diagnostics["min_density"]) > 0.0
        and float(diagnostics["min_temperature"]) > 0.0
        and float(balances["wall_effective_pressure_min"]) > 0.0
        and float(balances["momentum_boundary_flux_linf"]) <= 10.0 * tolerance
        and abs(float(balances["internal_energy_balance_error"])) <= 10.0 * tolerance
        and rank == problem.unknown_count
    )
    return {
        "raw_acceptance_gate": raw,
        "diagnostics": diagnostics,
        "global_balances": balances,
        "structural_jacobian_rank": rank,
        "unknown_count": problem.unknown_count,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-thor-n28-dir", type=Path, required=True)
    parser.add_argument("--thor-n29-dir", type=Path, required=True)
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

    upstream = read_json(args.accepted_thor_n28_dir / "THOR_ROOT_RECONCILIATION_N28_PASSED.json")
    require(upstream.get("status") == "R26_THOR_ROOT_RECONCILIATION_N28_PASSED", "N28 upstream failed")
    require(upstream.get("source_commit") == EXPECTED_N28_SOURCE_COMMIT, "N28 source commit changed")
    require(upstream.get("n28_accepted") is True, "N28 was not accepted")
    require(upstream.get("n29_authorized") is True, "N29 was not authorized")
    require(upstream.get("n30_authorized") is False, "N28 prematurely authorized N30")
    require(upstream.get("production_accepted") is False, "N28 made a production claim")
    require(upstream.get("legacy_roots_used_as_solver_seed") is False, "N28 legacy-seed firewall missing")

    n28_cross = read_json(args.accepted_thor_n28_dir / "THOR_N28_CROSS_SOLVER_VALIDATION.json")
    require(n28_cross.get("status") == "R26_THOR_N28_CROSS_SOLVER_VALIDATION_PASSED", "N28 cross-solver gate failed")
    require(n28_cross.get("n29_authorized") is True, "N28 cross-solver gate did not authorize N29")
    require(n28_cross.get("n30_authorized") is False, "N28 cross-solver gate authorized N30")
    require(n28_cross.get("legacy_n28_used_as_solver_seed") is False, "legacy N28 seed firewall missing")

    n28_record = read_json(args.accepted_thor_n28_dir / "N28" / "thor_validation.json")
    require(n28_record.get("status") == "R26_THOR_VALIDATION_CANDIDATE_PASSED", "N28 THOR candidate failed")
    require(n28_record.get("physical_candidate_gate_passed") is True, "N28 physical gate failed")
    require(n28_record.get("state_sha256") == EXPECTED_N28_STATE_SHA256, "N28 state hash changed")
    require(n28_record.get("production_accepted") is False, "N28 candidate made a production claim")

    n29_record = read_json(args.thor_n29_dir / "thor_validation.json")
    require(n29_record.get("status") == "R26_THOR_VALIDATION_CANDIDATE_PASSED", "N29 THOR candidate failed")
    require(n29_record.get("physical_candidate_gate_passed") is True, "N29 physical gate failed")
    require(n29_record.get("production_accepted") is False, "N29 candidate made a production claim")
    n29_case_record = n29_record.get("case")
    require(isinstance(n29_case_record, dict), "N29 case record missing")
    require(int(n29_case_record.get("nodes", -1)) == 29, "N29 grid mismatch")
    require(n29_case_record.get("kn_convention") == "gu_lambda_over_L", "N29 Kn convention mismatch")
    n29_initial = n29_record.get("initial_state")
    require(isinstance(n29_initial, dict), "N29 initial-state record missing")
    require(n29_initial.get("kind") == "explicit_restart", "N29 did not use an explicit restart")
    require(n29_initial.get("action") == "interpolated_and_mass_corrected", "N29 seed was not N28 refinement")
    require(n29_initial.get("file_sha256") == EXPECTED_N28_FILE_SHA256, "N29 seed is not accepted THOR N28")

    n28_state, n28_x, n28_y, n28_lid = load_state(
        args.accepted_thor_n28_dir / "N28" / "thor_state.npz",
        nodes=28,
        expected_file_sha256=EXPECTED_N28_FILE_SHA256,
    )
    n29_state, n29_x, n29_y, n29_lid = load_state(
        args.thor_n29_dir / "thor_state.npz",
        nodes=29,
    )
    require(state_sha256(n28_state) == EXPECTED_N28_STATE_SHA256, "decoded N28 state hash changed")
    require(state_sha256(n29_state) == n29_record.get("state_sha256"), "N29 record/state hash mismatch")

    n28_case = jfm_maxwell_cavity_case(28, kn=0.2, lid_speed_m_per_s=100.0, wall_temperature_K=300.0, grid_stretch_beta=0.0)
    n29_case = jfm_maxwell_cavity_case(29, kn=0.2, lid_speed_m_per_s=100.0, wall_temperature_K=300.0, grid_stretch_beta=0.0)
    require(np.array_equal(n28_x, n28_case.x) and np.array_equal(n28_y, n28_case.y), "N28 coordinates changed")
    require(np.array_equal(n29_x, n29_case.x) and np.array_equal(n29_y, n29_case.y), "N29 coordinates changed")
    require(math.isclose(n28_lid, n28_case.lid_velocity, rel_tol=0.0, abs_tol=2.0e-14), "N28 lid mismatch")
    require(math.isclose(n29_lid, n29_case.lid_velocity, rel_tol=0.0, abs_tol=2.0e-14), "N29 lid mismatch")

    n28_gate = independent_gate(make_thor_problem(n28_case), n28_state, n28_case, args.raw_tolerance)
    n29_gate = independent_gate(make_thor_problem(n29_case), n29_state, n29_case, args.raw_tolerance)
    comparison = compare_cross_solver_profiles(
        n28_state,
        n28_x,
        n28_y,
        n29_state,
        n29_x,
        n29_y,
        lid_velocity=n29_case.lid_velocity,
        target_n=128,
    )
    comparison_passed = same_grid_cross_solver_passed(
        comparison,
        maximum_profile_nrms=args.maximum_profile_nrms,
        maximum_line_nrms=args.maximum_line_nrms,
        maximum_dg_relative_difference=args.maximum_DG_relative_difference,
    )
    passed = bool(n28_gate["passed"] and n29_gate["passed"] and comparison_passed)
    record = {
        "status": "R26_THOR_N29_REFINEMENT_VALIDATION_PASSED" if passed else "R26_THOR_N29_REFINEMENT_VALIDATION_FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_n28_source_commit": EXPECTED_N28_SOURCE_COMMIT,
        "accepted_n28_file_sha256": EXPECTED_N28_FILE_SHA256,
        "accepted_n28_state_sha256": EXPECTED_N28_STATE_SHA256,
        "n28_independent_gate": n28_gate,
        "n29_record_state_sha256": n29_record.get("state_sha256"),
        "n29_independent_gate": n29_gate,
        "n28_to_n29_comparison": comparison,
        "comparison_passed": comparison_passed,
        "thresholds": {
            "raw_tolerance": args.raw_tolerance,
            "maximum_profile_nrms": args.maximum_profile_nrms,
            "maximum_line_nrms": args.maximum_line_nrms,
            "maximum_DG_relative_difference": args.maximum_DG_relative_difference,
            "provenance": "unchanged thresholds from the passed N8/N16 THOR cross-solver audit",
        },
        "n29_accepted": passed,
        "n30_authorized": passed,
        "production_accepted": False,
        "accepted_thor_n28_used_as_solver_seed": True,
        "legacy_or_failed_n29_used_as_solver_seed": False,
        "historical_failed_n30_used_as_solver_seed": False,
    }
    native_record = json_native(record)
    args.output.write_text(
        json.dumps(native_record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(native_record, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
