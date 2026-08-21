#!/usr/bin/env python3
"""Validate an accepted pure-Maxwell R26 continuation output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

# Allow direct execution by absolute path during login-node preflight, without
# relying on the caller to export PYTHONPATH first.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r26_cases import jfm_maxwell_cavity_case
from r26_discretization import R26NodeBVP
from r26_fv_backend import compatible_fv_bulk_residual, wall_bounded_control_volume_weights
from r26_validation import global_balance_diagnostics


EXPECTED_CORE_HASHES = {
    "code/r26_bulk_equations.py": "9abe3943ce541e6c5243a61893c1428daea30cf8fae42ab3e90c140eb7ba6a06",
    "code/r26_tensor_closures.py": "13037256b49de8ce0737136c56ab31fa5b1641545a79a65e77c761c25bcbbbea",
    "code/r26_wall_conditions.py": "b3a7bf0bc4be58f3e0c42928c87f4b01802ae88d50055b58410e485e7bbcdd49",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"R26_MAXWELL_VALIDATION_FAILED: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-kn", type=float, required=True, choices=(0.05, 0.20))
    parser.add_argument("--expected-nodes", type=int, required=True)
    parser.add_argument("--expected-beta", type=float, required=True)
    parser.add_argument("--raw-tolerance", type=float, default=1.0e-8)
    parser.add_argument(
        "--expected-target-lid",
        type=float,
        default=100.0 / math.sqrt(208.0 * 300.0),
        help="override only for a deliberately reduced smoke run",
    )
    args = parser.parse_args()

    summary_path = args.output_dir / "run_summary.json"
    state_path = args.output_dir / "last_accepted_state.npz"
    require(summary_path.is_file(), "run_summary.json missing")
    require(state_path.is_file(), "last_accepted_state.npz missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    case = summary.get("case", {})

    require(summary.get("termination") == "target_accepted", "target root was not accepted")
    require(case.get("family") == "jfm-maxwell", "wrong case family")
    require(case.get("molecular_model") == "maxwell_molecules", "molecular-model lock missing")
    require(case.get("kn_convention") == "gu_lambda_over_L", "wrong Kn convention")
    require(math.isclose(float(case.get("kn_input")), args.expected_kn, rel_tol=0.0, abs_tol=2e-15), "wrong Kn")
    require(int(case.get("nodes")) == args.expected_nodes, "wrong grid size")
    require(case.get("viscosity_kind") == "power_law", "wrong viscosity law")
    require(float(case.get("viscosity_exponent")) == 1.0, "Maxwell exponent is not one")
    require(case.get("closure_mode") == "jfm2009", "wrong R26 closure coefficients")
    require(float(case.get("wall_accommodation")) == 1.0, "wall is not fully diffuse")
    require(float(case.get("wall_temperature_K")) == 300.0, "wrong wall temperature")
    require(float(case.get("lid_speed_m_per_s")) == 100.0, "wrong lid speed")
    require(math.isclose(float(case.get("beta")), args.expected_beta, rel_tol=0.0, abs_tol=2e-15), "wrong grid-stretch beta")
    require("Pure Maxwell-molecule" in str(case.get("provenance")), "Maxwell provenance missing")
    require(math.isclose(float(case.get("mu_equilibrium")), args.expected_kn * math.sqrt(2.0 / math.pi), rel_tol=0.0, abs_tol=2e-15), "Gu Kn-to-mu conversion mismatch")
    require(math.isclose(float(case.get("lid_target")), args.expected_target_lid, rel_tol=0.0, abs_tol=2e-14), "wrong target lid")
    require(math.isclose(float(case.get("lid_last_accepted")), args.expected_target_lid, rel_tol=0.0, abs_tol=2e-14), "accepted lid is not target")

    manifest = summary.get("source_manifest", {})
    for name, expected in EXPECTED_CORE_HASHES.items():
        require(manifest.get(name) == expected, f"hash mismatch for {name}")

    attempts = summary.get("attempts", [])
    require(bool(attempts), "no nonlinear attempt recorded")
    final = attempts[-1]
    require(bool(final.get("accepted")), "last nonlinear attempt rejected")
    require(float(final.get("raw_acceptance_gate")) <= args.raw_tolerance, "raw residual gate failed")
    diagnostic = final.get("diagnostics", {})
    require(float(diagnostic.get("min_density")) > 0.0, "non-positive density")
    require(float(diagnostic.get("min_temperature")) > 0.0, "non-positive temperature")
    balance = final.get("global_balances", {})
    require(float(balance.get("wall_effective_pressure_min")) > 0.0, "non-positive effective wall pressure")
    require(float(balance.get("momentum_boundary_flux_linf")) <= 10.0 * args.raw_tolerance, "momentum balance failed")
    require(abs(float(balance.get("internal_energy_balance_error"))) <= 10.0 * args.raw_tolerance, "energy balance failed")

    with np.load(state_path, allow_pickle=False) as archive:
        state = np.asarray(archive["state"], dtype=float)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        require(state.shape == (args.expected_nodes, args.expected_nodes, 17), "wrong state shape")
        require(x.shape == (args.expected_nodes,), "wrong x-coordinate shape")
        require(y.shape == (args.expected_nodes,), "wrong y-coordinate shape")
        require(np.isfinite(state).all(), "state contains NaN or infinity")
        require(float(np.min(state[..., 0])) > 0.0, "state density is non-positive")
        require(float(np.min(state[..., 3])) > 0.0, "state temperature is non-positive")
        require(str(np.asarray(archive["kn_convention"]).item()) == "gu_lambda_over_L", "state Kn convention mismatch")
        require(math.isclose(float(np.asarray(archive["kn_input"]).item()), args.expected_kn, rel_tol=0.0, abs_tol=2e-15), "state Kn mismatch")
        require(math.isclose(float(np.asarray(archive["beta"]).item()), args.expected_beta, rel_tol=0.0, abs_tol=2e-15), "state beta mismatch")
        require(math.isclose(float(np.asarray(archive["lid_velocity"]).item()), args.expected_target_lid, rel_tol=0.0, abs_tol=2e-14), "state lid mismatch")

    expected_case = jfm_maxwell_cavity_case(
        args.expected_nodes,
        kn=args.expected_kn,
        lid_speed_m_per_s=100.0,
        wall_temperature_K=300.0,
        grid_stretch_beta=args.expected_beta,
    ).with_lid_velocity(args.expected_target_lid, suffix="independent-validator")
    require(np.array_equal(x, expected_case.x), "x coordinates do not match the declared grid")
    require(np.array_equal(y, expected_case.y), "y coordinates do not match the declared grid")
    problem = R26NodeBVP(
        expected_case,
        bulk_operator=compatible_fv_bulk_residual,
        mass_weights=wall_bounded_control_volume_weights(expected_case.x, expected_case.y),
    )
    independent = problem.evaluate(state)
    independent_raw_gate = max(
        independent.diagnostics.raw_total_linf,
        abs(independent.diagnostics.held_out_continuity),
        abs(independent.diagnostics.mass_error),
    )
    require(independent_raw_gate <= args.raw_tolerance, "independently recomputed raw residual gate failed")
    independent_balance = global_balance_diagnostics(state, expected_case)
    require(float(independent_balance["wall_effective_pressure_min"]) > 0.0, "independently recomputed wall pressure failed")
    require(float(independent_balance["momentum_boundary_flux_linf"]) <= 10.0 * args.raw_tolerance, "independently recomputed momentum balance failed")
    require(abs(float(independent_balance["internal_energy_balance_error"])) <= 10.0 * args.raw_tolerance, "independently recomputed energy balance failed")

    digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    print(json.dumps({"status": "R26_MAXWELL_VALIDATION_PASS", "kn_gu": args.expected_kn, "nodes": args.expected_nodes, "state_file_sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
