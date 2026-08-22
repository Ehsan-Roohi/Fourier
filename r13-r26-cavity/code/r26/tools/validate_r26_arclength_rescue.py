#!/usr/bin/env python3
"""Validate pseudo-arclength provenance and the final fixed-lid correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"R26_ARCLENGTH_VALIDATION_FAILED: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arclength_dir", type=Path)
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("--raw-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--expected-target-lid", type=float, required=True)
    args = parser.parse_args()

    arc_summary_path = args.arclength_dir / "arclength_summary.json"
    landing_path = args.arclength_dir / "landing_seed.npz"
    target_summary_path = args.target_dir / "run_summary.json"
    target_state_path = args.target_dir / "last_accepted_state.npz"
    for path in (
        arc_summary_path,
        landing_path,
        target_summary_path,
        target_state_path,
    ):
        require(path.is_file(), f"required artifact missing: {path.name}")

    arc = json.loads(arc_summary_path.read_text(encoding="utf-8"))
    target = json.loads(target_summary_path.read_text(encoding="utf-8"))
    require(arc.get("termination") == "target_bracketed", "target was not bracketed")
    landing = arc.get("landing") or {}
    require(
        math.isclose(
            float(landing.get("target_parameter")),
            args.expected_target_lid,
            rel_tol=0.0,
            abs_tol=2.0e-14,
        ),
        "landing target mismatch",
    )
    require(
        landing.get("landing_file_sha256") == sha256(landing_path),
        "landing seed checksum mismatch",
    )
    controls = arc.get("arclength_controls", {})
    maximum_jacobians = int(controls.get("maximum_jacobians_per_attempt", -1))
    maximum_objectives = int(
        controls.get("maximum_objective_evaluations_per_attempt", -1)
    )
    require(1 <= maximum_jacobians <= 7, "arclength Jacobian cap is invalid")
    require(maximum_objectives == 6000, "arclength objective cap changed")
    attempts = arc.get("attempts", [])
    require(bool(attempts), "no arclength attempts recorded")
    require(any(bool(row.get("accepted")) for row in attempts), "no arclength point accepted")
    for row in attempts:
        solver = row.get("solver", {})
        require(
            int(solver.get("jacobian_evaluations", -1)) <= maximum_jacobians,
            "arclength Jacobian cap exceeded",
        )
        require(
            int(solver.get("objective_evaluations", -1)) <= maximum_objectives,
            "arclength objective cap exceeded",
        )
        if bool(row.get("accepted")):
            require(
                float(row.get("raw_acceptance_gate")) <= args.raw_tolerance,
                "accepted arclength point failed the raw gate",
            )

    require(target.get("termination") == "target_accepted", "fixed target not accepted")
    case = target.get("case", {})
    require(int(case.get("nodes", -1)) == 30, "final target is not N30")
    require(case.get("family") == "jfm-maxwell", "final case family mismatch")
    require(case.get("kn_convention") == "gu_lambda_over_L", "final Kn convention mismatch")
    require(case.get("molecular_model") == "maxwell_molecules", "final molecular model mismatch")
    require(
        math.isclose(float(case.get("kn_input")), 0.20, rel_tol=0.0, abs_tol=2.0e-15),
        "final Kn mismatch",
    )
    require(
        math.isclose(
            float(case.get("lid_last_accepted")),
            args.expected_target_lid,
            rel_tol=0.0,
            abs_tol=2.0e-14,
        ),
        "final accepted lid mismatch",
    )
    provenance = target.get("input_provenance", {})
    require(
        provenance.get("kind") == "explicit_local_restart_interpolated",
        "final target did not use the explicit landing predictor",
    )
    require(
        provenance.get("file_sha256") == sha256(landing_path),
        "final target landing provenance mismatch",
    )
    nonlinear = target.get("nonlinear_solver", {})
    require(bool(nonlinear.get("analytic_mass_jacobian")), "analytic mass disabled")
    require(bool(nonlinear.get("ser_pseudo_transient")), "SER-PTC disabled")
    require(bool(nonlinear.get("secant_predictor")), "secant mode not declared")
    require(
        1 <= int(nonlinear.get("max_jacobians_per_attempt", -1)) <= 8,
        "final-correction Jacobian cap invalid",
    )
    target_attempts = target.get("attempts", [])
    require(len(target_attempts) == 1, "final landing must use exactly one correction")
    require(bool(target_attempts[0].get("accepted")), "final correction rejected")
    require(
        float(target_attempts[0].get("raw_acceptance_gate")) <= args.raw_tolerance,
        "final correction raw gate failed",
    )
    print(
        json.dumps(
            {
                "status": "R26_ARCLENGTH_TARGET_VALIDATION_PASS",
                "accepted_arclength_points": sum(
                    bool(row.get("accepted")) for row in attempts
                ),
                "target_state_sha256": sha256(target_state_path),
                "landing_seed_sha256": sha256(landing_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
