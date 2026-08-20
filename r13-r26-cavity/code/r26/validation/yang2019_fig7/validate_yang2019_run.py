#!/usr/bin/env python3
"""Fail-closed validator for the Yang et al. (2019) R26 cavity target."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"YANG2019_R26_RUN_FAILED: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-nodes", type=int, required=True)
    parser.add_argument("--raw-tolerance", type=float, default=1.0e-8)
    args = parser.parse_args()

    summary_path = args.output_dir / "run_summary.json"
    state_path = args.output_dir / "last_accepted_state.npz"
    require(summary_path.is_file(), "run_summary.json missing")
    require(state_path.is_file(), "last_accepted_state.npz missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    case = summary.get("case", {})

    require(summary.get("termination") == "target_accepted", "target root not accepted")
    require(case.get("family") == "jfm-maxwell", "wrong case family")
    require(case.get("molecular_model") == "maxwell_molecules", "wrong molecular class")
    require(case.get("kn_convention") == "gu_lambda_over_L", "wrong Kn convention")
    require(math.isclose(float(case.get("kn_input")), 0.1, abs_tol=2e-15), "Kn is not 0.1")
    require(int(case.get("nodes")) == args.expected_nodes, "wrong grid size")
    require(case.get("closure_mode") == "jfm2009", "not the final JFM-2009 closure")
    require(case.get("viscosity_kind") == "power_law", "wrong viscosity law")
    require(math.isclose(float(case.get("viscosity_exponent")), 1.0, abs_tol=2e-15), "Maxwell exponent is not one")
    require(math.isclose(float(case.get("wall_accommodation")), 1.0, abs_tol=2e-15), "wall not fully diffuse")
    require(math.isclose(float(case.get("wall_temperature_K")), 273.0, abs_tol=2e-12), "wall temperature is not 273 K")
    require(math.isclose(float(case.get("lid_speed_m_per_s")), 10.0, abs_tol=2e-12), "lid speed is not 10 m/s")
    expected_lid = 10.0 / math.sqrt(208.0 * 273.0)
    require(math.isclose(float(case.get("lid_target")), expected_lid, abs_tol=2e-14), "wrong nondimensional lid")
    require(math.isclose(float(case.get("lid_last_accepted")), expected_lid, abs_tol=2e-14), "accepted lid is not target")
    require(math.isclose(float(case.get("mu_equilibrium")), 0.1 * math.sqrt(2.0 / math.pi), abs_tol=2e-15), "Gu Kn conversion mismatch")

    attempts = summary.get("attempts", [])
    require(bool(attempts), "no nonlinear attempt recorded")
    final = attempts[-1]
    require(bool(final.get("accepted")), "last attempt rejected")
    require(float(final.get("raw_acceptance_gate")) <= args.raw_tolerance, "raw residual gate failed")
    diagnostics = final.get("diagnostics", {})
    require(float(diagnostics.get("min_density")) > 0.0, "non-positive density")
    require(float(diagnostics.get("min_temperature")) > 0.0, "non-positive temperature")
    balances = final.get("global_balances", {})
    require(float(balances.get("wall_effective_pressure_min")) > 0.0, "non-positive wall pressure")
    require(float(balances.get("momentum_boundary_flux_linf")) <= 10.0 * args.raw_tolerance, "momentum balance failed")
    require(abs(float(balances.get("internal_energy_balance_error"))) <= 10.0 * args.raw_tolerance, "energy balance failed")

    with np.load(state_path, allow_pickle=False) as archive:
        state = np.asarray(archive["state"], dtype=float)
        require(state.shape == (args.expected_nodes, args.expected_nodes, 17), "wrong state shape")
        require(np.isfinite(state).all(), "state contains NaN or infinity")
        require(float(np.min(state[..., 0])) > 0.0, "density is non-positive")
        require(float(np.min(state[..., 3])) > 0.0, "temperature is non-positive")
        require(math.isclose(float(np.asarray(archive["kn_input"]).item()), 0.1, abs_tol=2e-15), "state Kn mismatch")
        require(str(np.asarray(archive["kn_convention"]).item()) == "gu_lambda_over_L", "state Kn convention mismatch")

    print(json.dumps({
        "status": "YANG2019_R26_RUN_PASS",
        "nodes": args.expected_nodes,
        "state_file_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
