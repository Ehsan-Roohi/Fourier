#!/usr/bin/env python3
"""Authenticate and reconcile THOR N24 with legacy N25/N27/N28 roots."""

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
from r26_thor_audit import compare_cross_solver_profiles
from r26_thor_reconciliation import (
    EXPECTED_ROOT_FILE_SHA256,
    EXPECTED_ROOT_STATE_SHA256,
    ImmutableRoot,
    json_native,
    ladder_comparison_passed,
    load_immutable_root,
    n16_n24_profile_envelope,
)
from r26_thor_solver import make_thor_problem
from r26_validation import global_balance_diagnostics


EXPECTED_CORE_HASHES = {
    "code/r26_bulk_equations.py": "9abe3943ce541e6c5243a61893c1428daea30cf8fae42ab3e90c140eb7ba6a06",
    "code/r26_tensor_closures.py": "13037256b49de8ce0737136c56ab31fa5b1641545a79a65e77c761c25bcbbbea",
    "code/r26_wall_conditions.py": "b3a7bf0bc4be58f3e0c42928c87f4b01802ae88d50055b58410e485e7bbcdd49",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"R26_THOR_ROOT_LADDER_AUDIT_FAILED: {message}")


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"required record missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"record is not an object: {path}")
    return value


def validate_case(root: ImmutableRoot) -> object:
    case = jfm_maxwell_cavity_case(
        root.nodes,
        kn=0.2,
        lid_speed_m_per_s=100.0,
        wall_temperature_K=300.0,
        grid_stretch_beta=0.0,
    )
    require(math.isclose(root.kn_input, 0.2, rel_tol=0.0, abs_tol=2.0e-15), f"N{root.nodes} Kn mismatch")
    require(math.isclose(root.beta, 0.0, rel_tol=0.0, abs_tol=2.0e-15), f"N{root.nodes} beta mismatch")
    require(math.isclose(root.lid_velocity, case.lid_velocity, rel_tol=0.0, abs_tol=2.0e-14), f"N{root.nodes} lid mismatch")
    require(np.array_equal(root.x, case.x), f"N{root.nodes} x coordinates mismatch")
    require(np.array_equal(root.y, case.y), f"N{root.nodes} y coordinates mismatch")
    return case


def validate_legacy_summary(path: Path, nodes: int) -> dict[str, object]:
    summary = read_json(path)
    require(summary.get("termination") == "target_accepted", f"N{nodes} target not accepted")
    case = summary.get("case")
    require(isinstance(case, dict), f"N{nodes} case record missing")
    require(int(case.get("nodes", -1)) == nodes, f"N{nodes} summary grid mismatch")
    require(case.get("family") == "jfm-maxwell", f"N{nodes} case family mismatch")
    require(case.get("molecular_model") == "maxwell_molecules", f"N{nodes} molecular model mismatch")
    require(case.get("closure_mode") == "jfm2009", f"N{nodes} closure mismatch")
    require(case.get("kn_convention") == "gu_lambda_over_L", f"N{nodes} Kn convention mismatch")
    require(float(case.get("lid_speed_m_per_s")) == 100.0, f"N{nodes} dimensional lid mismatch")
    require(float(case.get("wall_temperature_K")) == 300.0, f"N{nodes} wall temperature mismatch")
    manifest = summary.get("source_manifest")
    require(isinstance(manifest, dict), f"N{nodes} source manifest missing")
    for name, expected in EXPECTED_CORE_HASHES.items():
        require(manifest.get(name) == expected, f"N{nodes} source hash mismatch for {name}")
    attempts = summary.get("attempts")
    require(isinstance(attempts, list) and bool(attempts), f"N{nodes} attempts missing")
    require(bool(attempts[-1].get("accepted")), f"N{nodes} last attempt rejected")
    return summary


def independent_root_diagnostics(root: ImmutableRoot, *, thor: bool, raw_tolerance: float) -> dict[str, object]:
    case = validate_case(root)
    if thor:
        problem = make_thor_problem(case)
    else:
        problem = R26NodeBVP(
            case,
            bulk_operator=compatible_fv_bulk_residual,
            mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
        )
    result = problem.evaluate(root.state)
    diagnostics = asdict(result.diagnostics)
    raw_gate = float(
        max(
            result.diagnostics.raw_total_linf,
            abs(result.diagnostics.held_out_continuity),
            abs(result.diagnostics.mass_error),
        )
    )
    balances = json_native(global_balance_diagnostics(root.state, case))
    require(isinstance(balances, dict), "global balance diagnostics are not an object")
    passed = bool(
        raw_gate <= raw_tolerance
        and float(diagnostics["min_density"]) > 0.0
        and float(diagnostics["min_temperature"]) > 0.0
        and float(balances["wall_effective_pressure_min"]) > 0.0
        and float(balances["momentum_boundary_flux_linf"]) <= 10.0 * raw_tolerance
        and abs(float(balances["internal_energy_balance_error"])) <= 10.0 * raw_tolerance
    )
    return {
        "nodes": root.nodes,
        "solver_family": "thor" if thor else "legacy-compatible-fv",
        "path": str(root.path),
        "file_sha256": root.file_sha256,
        "state_sha256": root.state_sha256,
        "raw_acceptance_gate": raw_gate,
        "diagnostics": diagnostics,
        "global_balances": balances,
        "independent_physical_gate_passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thor-n24-dir", type=Path, required=True)
    parser.add_argument("--legacy-n25-dir", type=Path, required=True)
    parser.add_argument("--legacy-n27-dir", type=Path, required=True)
    parser.add_argument("--legacy-n28-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-thor-n24-source-commit", required=True)
    parser.add_argument("--raw-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--maximum-DG-relative-difference", type=float, default=0.02)
    args = parser.parse_args()
    require(args.raw_tolerance > 0.0, "raw tolerance must be positive")
    require(args.maximum_DG_relative_difference > 0.0, "D/G tolerance must be positive")
    require(not args.output.exists(), "output record already exists")

    final = read_json(args.thor_n24_dir / "THOR_CROSS_SOLVER_N24_PASSED.json")
    require(final.get("status") == "R26_THOR_CROSS_SOLVER_N24_PASSED", "N24 final record did not pass")
    require(final.get("source_commit") == args.expected_thor_n24_source_commit, "N24 source commit mismatch")
    require(final.get("n24_accepted") is True, "N24 acceptance is not boolean true")
    require(final.get("n28_authorized") is False, "N24 record exceeded its authorization")
    require(final.get("production_accepted") is False, "N24 made a production claim")

    n24_record = read_json(args.thor_n24_dir / "N24" / "thor_validation.json")
    require(n24_record.get("status") == "R26_THOR_VALIDATION_CANDIDATE_PASSED", "N24 THOR candidate failed")
    require(n24_record.get("physical_candidate_gate_passed") is True, "N24 physical gate failed")
    require(n24_record.get("production_accepted") is False, "N24 candidate made a production claim")

    try:
        roots = [
            load_immutable_root(
                args.thor_n24_dir / "N24" / "thor_state.npz",
                nodes=24,
                expected_file_sha256=EXPECTED_ROOT_FILE_SHA256[24],
            ),
            load_immutable_root(
                args.legacy_n25_dir / "last_accepted_state.npz",
                nodes=25,
                expected_file_sha256=EXPECTED_ROOT_FILE_SHA256[25],
                require_accepted_flag=False,
            ),
            load_immutable_root(
                args.legacy_n27_dir / "last_accepted_state.npz",
                nodes=27,
                expected_file_sha256=EXPECTED_ROOT_FILE_SHA256[27],
                require_accepted_flag=False,
            ),
            load_immutable_root(
                args.legacy_n28_dir / "last_accepted_state.npz",
                nodes=28,
                expected_file_sha256=EXPECTED_ROOT_FILE_SHA256[28],
                require_accepted_flag=False,
            ),
        ]
    except (OSError, ValueError) as error:
        require(False, str(error))
    for root in roots:
        expected_state = EXPECTED_ROOT_STATE_SHA256.get(root.nodes)
        if expected_state is not None:
            require(root.state_sha256 == expected_state, f"N{root.nodes} decoded state hash mismatch")
    require(roots[0].state_sha256 == n24_record.get("state_sha256"), "N24 record/state hash mismatch")

    for nodes, directory in (
        (25, args.legacy_n25_dir),
        (27, args.legacy_n27_dir),
        (28, args.legacy_n28_dir),
    ):
        validate_legacy_summary(directory / "run_summary.json", nodes)

    diagnostics = [
        independent_root_diagnostics(root, thor=(root.nodes == 24), raw_tolerance=args.raw_tolerance)
        for root in roots
    ]
    require(all(bool(row["independent_physical_gate_passed"]) for row in diagnostics), "one or more roots failed independent physical validation")

    sensitivity = read_json(args.thor_n24_dir / "N8_N16_N24_GRID_SENSITIVITY.json")
    require(sensitivity.get("status") == "R26_THOR_GRID_SENSITIVITY_REPORTED", "N24 grid-sensitivity record missing")
    profile_envelope = n16_n24_profile_envelope(sensitivity)
    pairs: list[dict[str, object]] = []
    for coarse, fine in zip(roots, roots[1:]):
        comparison = compare_cross_solver_profiles(
            coarse.state,
            coarse.x,
            coarse.y,
            fine.state,
            fine.x,
            fine.y,
            lid_velocity=coarse.lid_velocity,
            target_n=128,
        )
        passed = ladder_comparison_passed(
            comparison,
            maximum_profile_nrms=profile_envelope,
            maximum_dg_relative_difference=args.maximum_DG_relative_difference,
        )
        pairs.append(
            {
                "coarse_nodes": coarse.nodes,
                "fine_nodes": fine.nodes,
                "comparison": comparison,
                "comparison_passed": passed,
            }
        )
    n28_run_authorized = bool(all(bool(row["comparison_passed"]) for row in pairs))
    record = {
        "status": (
            "R26_THOR_ROOT_LADDER_AUDIT_PASSED"
            if n28_run_authorized
            else "R26_THOR_ROOT_LADDER_AUDIT_FAILED"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "thor_n24_source_commit": args.expected_thor_n24_source_commit,
        "roots": diagnostics,
        "pairs": pairs,
        "thresholds": {
            "raw_tolerance": args.raw_tolerance,
            "maximum_profile_nrms": profile_envelope,
            "profile_threshold_provenance": "immutable observed N16-to-N24 THOR grid-sensitivity envelope",
            "maximum_DG_relative_difference": args.maximum_DG_relative_difference,
            "DG_threshold_provenance": "existing N8/N16 cross-solver audit threshold",
        },
        "n28_run_authorized": n28_run_authorized,
        "n29_authorized": False,
        "n30_authorized": False,
        "production_accepted": False,
        "legacy_roots_used_as_solver_seed": False,
        "formal_asymptotic_grid_convergence_claim": False,
    }
    native_record = json_native(record)
    args.output.write_text(
        json.dumps(native_record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(native_record, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if n28_run_authorized else 1)


if __name__ == "__main__":
    main()
