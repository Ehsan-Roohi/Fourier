#!/usr/bin/env python3
"""Independently validate the accepted N28->N29->N30->N31 root chain."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from analysis.run_r26_gu_emerson_jfm_n32_candidate import (
    COMPATIBILITY_TOLERANCE,
    CONSERVATION_TOLERANCE,
    JACOBIAN_STENCIL_RADIUS,
    RAW_TOLERANCE,
    raw_gate,
)
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
from r26_gu_emerson_variables import GuEmersonLogStateTransform, gu_emerson_fields_from_state
from r26_solver import jacobian_sparsity
from r26_thor_audit import state_sha256
from r26_validation import global_balance_diagnostics


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"R26_JFM_GRID_CONTINUATION_VALIDATION_FAILED: {message}")


def close(actual: float, expected: object, name: str) -> None:
    require(
        math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=5.0e-12),
        f"recomputed metric differs: {name}",
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
    output = args.run_dir / "JFM_N29_N31_INDEPENDENT_VALIDATION.json"
    require(not output.exists(), "independent validation record already exists")
    summary_path = args.run_dir / "JFM_N29_N31_GRID_CONTINUATION.json"
    require(summary_path.is_file(), "grid-continuation summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(
        summary.get("status")
        == "R26_GU_EMERSON_JFM_N29_N31_GRID_CONTINUATION_PASSED",
        "primary continuation did not pass",
    )
    require(summary.get("source_commit") == args.expected_source_commit, "source mismatch")
    require(summary.get("n32_authorized") is True, "N32 is not authorized")
    require(summary.get("n36_authorized") is False, "N36 was over-authorized")
    require(summary.get("maximum_grid_run") == 31, "primary run exceeded or missed N31")
    with np.load(
        args.n28_gate_dir / "gu_emerson_jfm_n28_same_grid_candidate.npz",
        allow_pickle=False,
    ) as archive:
        predecessor = np.asarray(archive["state"], dtype=float)
        require(bool(np.asarray(archive["accepted"]).item()), "N28 archive is rejected")
    validated: list[dict[str, object]] = []
    for nodes in (29, 30, 31):
        stage_dir = args.run_dir / f"N{nodes}"
        record_path = stage_dir / f"JFM_N{nodes}_GRID_CONTINUATION_STAGE.json"
        archive_path = stage_dir / f"gu_emerson_jfm_n{nodes}_grid_continuation.npz"
        require(record_path.is_file(), f"N{nodes} record is missing")
        require(archive_path.is_file(), f"N{nodes} archive is missing")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        require(
            record.get("status")
            == f"R26_GU_EMERSON_JFM_N{nodes}_GRID_CONTINUATION_PASSED",
            f"N{nodes} status did not pass",
        )
        require(record.get("source_commit") == args.expected_source_commit, f"N{nodes} source mismatch")
        require(record.get("candidate_accepted") is True, f"N{nodes} is rejected")
        require(record.get("production_accepted") is False, f"N{nodes} is production-marked")
        require(record.get("n36_authorized") is False, f"N{nodes} authorized N36")
        require(record.get("maximum_grid_run") == nodes, f"N{nodes} grid scope mismatch")
        require(
            record.get("predecessor", {}).get("state_sha256") == state_sha256(predecessor),
            f"N{nodes} predecessor hash mismatch",
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            state = np.asarray(archive["state"], dtype=float)
            encoded = np.asarray(archive["encoded_transformed_state"], dtype=float)
            x = np.asarray(archive["x"], dtype=float)
            y = np.asarray(archive["y"], dtype=float)
            require(bool(np.asarray(archive["accepted"]).item()), f"N{nodes} archive rejected")
            require(int(np.asarray(archive["nodes"]).item()) == nodes, f"N{nodes} archive node mismatch")
            require(
                str(np.asarray(archive["source_commit"]).item()) == args.expected_source_commit,
                f"N{nodes} archive source mismatch",
            )
            require(not bool(np.asarray(archive["n36_authorized"]).item()), f"N{nodes} archive authorized N36")
        case = jfm_maxwell_cavity_case(
            nodes,
            kn=0.2,
            lid_speed_m_per_s=100.0,
            wall_temperature_K=300.0,
            grid_stretch_beta=0.0,
        )
        require(state.shape == (nodes, nodes, 17), f"N{nodes} candidate shape mismatch")
        require(np.array_equal(x, case.x) and np.array_equal(y, case.y), f"N{nodes} grid mismatch")
        require(
            state_sha256(state) == record.get("candidate", {}).get("state_sha256"),
            f"N{nodes} candidate hash mismatch",
        )
        transform = GuEmersonLogStateTransform(case)
        require(
            float(np.max(np.abs(transform.decode(encoded) - state), initial=0.0)) <= 5.0e-11,
            f"N{nodes} transformed archive does not reconstruct the state",
        )
        problem = R26NodeBVP(
            case,
            bulk_operator=compatible_fv_bulk_residual,
            mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
        )
        evaluation = problem.evaluate(state)
        candidate_raw = raw_gate(evaluation.diagnostics)
        fields = gu_emerson_fields_from_state(
            state,
            x=case.x,
            y=case.y,
            mu=case.mu(state[..., 3]),
        )
        transformed = gu_emerson_compatible_transformed_fv_residual(
            fields,
            case=case,
            convection_scheme="central",
        )
        physical = compatible_fv_bulk_residual(
            state,
            case.x,
            case.y,
            case.mu(state[..., 3]),
            case=case,
            convection_scheme="central",
        )
        interior = np.s_[1:-1, 1:-1]
        transformed_linf = float(np.max(np.abs(transformed[interior]), initial=0.0))
        compatibility_linf = float(
            np.max(np.abs(transformed[interior] - physical[interior]), initial=0.0)
        )
        close(candidate_raw, record["candidate"]["raw_gate"], f"N{nodes} raw gate")
        close(
            transformed_linf,
            record["candidate"]["transformed_interior_linf"],
            f"N{nodes} transformed gate",
        )
        close(
            compatibility_linf,
            record["candidate"]["transformed_vs_compatible_physical_linf"],
            f"N{nodes} compatibility gate",
        )
        require(candidate_raw <= RAW_TOLERANCE, f"N{nodes} raw gate failed")
        require(transformed_linf <= RAW_TOLERANCE, f"N{nodes} transformed gate failed")
        require(compatibility_linf <= COMPATIBILITY_TOLERANCE, f"N{nodes} compatibility failed")
        require(abs(evaluation.diagnostics.mass_error) <= 1.0e-10, f"N{nodes} mass gate failed")
        balance = global_balance_diagnostics(state, case)
        require(float(balance["wall_effective_pressure_min"]) > 0.0, f"N{nodes} wall pressure failed")
        require(
            float(balance["momentum_boundary_flux_linf"]) <= CONSERVATION_TOLERANCE,
            f"N{nodes} momentum conservation failed",
        )
        require(
            abs(float(balance["internal_energy_balance_error"])) <= CONSERVATION_TOLERANCE,
            f"N{nodes} energy conservation failed",
        )
        require(float(balance["wall_normal_velocity_linf"]) <= RAW_TOLERANCE, f"N{nodes} wall velocity failed")
        pattern = jacobian_sparsity(
            problem,
            stencil_radius=JACOBIAN_STENCIL_RADIUS,
            include_mass_border=True,
        )
        rank = int(structural_rank(pattern))
        require(rank == problem.unknown_count, f"N{nodes} structural rank failed")
        validated.append(
            {
                "nodes": nodes,
                "candidate_state_sha256": state_sha256(state),
                "raw_gate": candidate_raw,
                "transformed_interior_linf": transformed_linf,
                "compatibility_linf": compatibility_linf,
                "structural_rank": rank,
                "unknown_count": problem.unknown_count,
            }
        )
        predecessor = state
    validation = {
        "status": "R26_GU_EMERSON_JFM_N29_N31_INDEPENDENT_VALIDATION_PASSED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.expected_source_commit,
        "validated_stages": validated,
        "candidate_arrays_reconstructed": True,
        "production_accepted": False,
        "n32_authorized": True,
        "n36_authorized": False,
        "n40_authorized": False,
        "n44_authorized": False,
        "maximum_grid_run": 31,
    }
    output.write_text(
        json.dumps(jsonable(validation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(validation), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
