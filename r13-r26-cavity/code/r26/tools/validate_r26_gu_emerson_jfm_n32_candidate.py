#!/usr/bin/env python3
"""Independently validate a passed transformed-coordinate JFM N32 run."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from scipy.sparse.csgraph import structural_rank

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r26_cases import jfm_maxwell_cavity_case
from r26_discretization import R26NodeBVP
from r26_fv_backend import (
    compatible_fv_bulk_residual,
    wall_bounded_control_volume_weights,
)
from r26_gu_emerson_jfm_same_grid_gate import jsonable
from r26_gu_emerson_transformed_fv import (
    gu_emerson_compatible_transformed_fv_residual,
)
from r26_gu_emerson_variables import (
    GuEmersonLogStateTransform,
    gu_emerson_fields_from_state,
)
from r26_solver import jacobian_sparsity
from r26_thor_audit import compare_cross_solver_profiles, state_sha256
from r26_validation import global_balance_diagnostics


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"R26_JFM_N32_VALIDATION_FAILED: {message}")


def close(actual: float, expected: object, name: str) -> None:
    require(
        math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=5.0e-12),
        f"recomputed metric differs: {name}",
    )


def raw_gate(diagnostics: object) -> float:
    return float(
        max(
            diagnostics.raw_total_linf,
            abs(diagnostics.held_out_continuity),
            abs(diagnostics.mass_error),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n28-gate-dir", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args()
    require(
        re.fullmatch(r"[0-9a-f]{40}", args.expected_source_commit) is not None,
        "expected source commit is not a full lowercase SHA",
    )
    output = args.run_dir / "JFM_N32_INDEPENDENT_VALIDATION.json"
    require(not output.exists(), "independent validation record already exists")
    record_path = args.run_dir / "JFM_N32_TRANSFORMED_CANDIDATE_GATE.json"
    archive_path = args.run_dir / "gu_emerson_jfm_n32_candidate.npz"
    require(record_path.is_file(), "primary N32 record is missing")
    require(archive_path.is_file(), "N32 candidate archive is missing")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    require(
        record.get("status")
        == "R26_GU_EMERSON_JFM_N32_TRANSFORMED_CANDIDATE_PASSED",
        "primary N32 status did not pass",
    )
    require(record.get("source_commit") == args.expected_source_commit, "source mismatch")
    require(record.get("candidate_accepted") is True, "N32 candidate is rejected")
    require(record.get("production_accepted") is False, "premature production claim")
    require(record.get("n36_authorized") is True, "N36 is not authorized")
    require(record.get("n40_authorized") is False, "N40 was prematurely authorized")
    require(record.get("n44_authorized") is False, "N44 was prematurely authorized")
    require(record.get("maximum_grid_run") == 32, "run exceeded or missed N32")
    require(
        record.get("higher_than_n32_run_attempted") is False,
        "run attempted a grid above N32",
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        candidate = np.asarray(archive["state"], dtype=float)
        encoded = np.asarray(archive["encoded_transformed_state"], dtype=float)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        require(bool(np.asarray(archive["accepted"]).item()), "archive is rejected")
        require(int(np.asarray(archive["nodes"]).item()) == 32, "archive node mismatch")
        require(
            str(np.asarray(archive["source_commit"]).item())
            == args.expected_source_commit,
            "archive source mismatch",
        )
        require(
            not bool(np.asarray(archive["production_accepted"]).item()),
            "archive is production-marked",
        )
        require(bool(np.asarray(archive["n36_authorized"]).item()), "archive blocks N36")
        require(not bool(np.asarray(archive["n40_authorized"]).item()), "archive authorizes N40")
    case = jfm_maxwell_cavity_case(
        32,
        kn=0.2,
        lid_speed_m_per_s=100.0,
        wall_temperature_K=300.0,
        grid_stretch_beta=0.0,
    )
    require(candidate.shape == (32, 32, 17), "candidate shape mismatch")
    require(np.array_equal(x, case.x) and np.array_equal(y, case.y), "grid mismatch")
    require(
        state_sha256(candidate) == record.get("candidate", {}).get("state_sha256"),
        "candidate state hash mismatch",
    )
    transform = GuEmersonLogStateTransform(case)
    rebuilt = transform.decode(encoded)
    require(
        float(np.max(np.abs(rebuilt - candidate), initial=0.0)) <= 5.0e-11,
        "stored transformed coordinates do not reconstruct the candidate",
    )
    problem = R26NodeBVP(
        case,
        bulk_operator=compatible_fv_bulk_residual,
        mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
    )
    evaluation = problem.evaluate(candidate)
    fields = gu_emerson_fields_from_state(
        candidate,
        x=case.x,
        y=case.y,
        mu=case.mu(candidate[..., 3]),
    )
    transformed = gu_emerson_compatible_transformed_fv_residual(
        fields, case=case, convection_scheme="central"
    )
    physical = compatible_fv_bulk_residual(
        candidate,
        case.x,
        case.y,
        case.mu(candidate[..., 3]),
        case=case,
        convection_scheme="central",
    )
    interior = np.s_[1:-1, 1:-1]
    transformed_linf = float(np.max(np.abs(transformed[interior]), initial=0.0))
    compatibility_linf = float(
        np.max(np.abs(transformed[interior] - physical[interior]), initial=0.0)
    )
    close(raw_gate(evaluation.diagnostics), record["candidate"]["raw_gate"], "raw_gate")
    close(
        transformed_linf,
        record["candidate"]["transformed_interior_linf"],
        "transformed_interior_linf",
    )
    close(
        compatibility_linf,
        record["candidate"]["transformed_vs_compatible_physical_linf"],
        "compatibility_linf",
    )
    require(raw_gate(evaluation.diagnostics) <= 1.0e-8, "independent raw gate failed")
    require(transformed_linf <= 1.0e-8, "independent transformed gate failed")
    require(compatibility_linf <= 5.0e-12, "independent compatibility gate failed")
    with np.load(
        args.n28_gate_dir / "gu_emerson_jfm_n28_same_grid_candidate.npz",
        allow_pickle=False,
    ) as archive:
        n28_state = np.asarray(archive["state"], dtype=float)
        n28_x = np.asarray(archive["x"], dtype=float)
        n28_y = np.asarray(archive["y"], dtype=float)
    comparison = compare_cross_solver_profiles(
        n28_state,
        n28_x,
        n28_y,
        candidate,
        case.x,
        case.y,
        lid_velocity=case.lid_velocity,
    )
    for key in (
        "maximum_normalized_rms_difference",
        "maximum_line_normalized_rms_difference",
        "D_relative_difference",
        "G_relative_difference",
    ):
        close(float(comparison[key]), record["n28_to_n32_comparison"][key], key)
    balance = global_balance_diagnostics(candidate, case)
    close(
        float(balance["internal_energy_balance_error"]),
        record["candidate"]["global_balances"]["internal_energy_balance_error"],
        "internal_energy_balance_error",
    )
    pattern = jacobian_sparsity(problem, stencil_radius=4, include_mass_border=True)
    rank = int(structural_rank(pattern))
    require(rank == problem.unknown_count, "N32 structural pattern is rank deficient")
    validation = {
        "status": "R26_GU_EMERSON_JFM_N32_INDEPENDENT_VALIDATION_PASSED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.expected_source_commit,
        "candidate_state_sha256": state_sha256(candidate),
        "raw_gate": raw_gate(evaluation.diagnostics),
        "transformed_interior_linf": transformed_linf,
        "compatibility_linf": compatibility_linf,
        "structural_rank": rank,
        "unknown_count": problem.unknown_count,
        "candidate_arrays_reconstructed": True,
        "production_accepted": False,
        "n36_authorized": True,
        "n40_authorized": False,
        "n44_authorized": False,
    }
    output.write_text(
        json.dumps(jsonable(validation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(validation), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
