#!/usr/bin/env python3
"""Run one bounded JFM-Maxwell N32 solve in Gu--Emerson coordinates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
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
from r26_gu_emerson_jfm_same_grid_gate import (
    JFM_N28_REFERENCE_STATE_SHA256,
    jsonable,
)
from r26_gu_emerson_transformed_fv import (
    gu_emerson_compatible_transformed_fv_residual,
)
from r26_gu_emerson_variables import (
    GuEmersonLogStateTransform,
    gu_emerson_fields_from_state,
)
from r26_solver import (
    SolveOptions,
    interpolate_state_grid,
    jacobian_sparsity,
    solve_r26_bvp,
)
from r26_thor_audit import compare_cross_solver_profiles, state_sha256
from r26_validation import global_balance_diagnostics


RAW_TOLERANCE = 1.0e-8
COMPATIBILITY_TOLERANCE = 5.0e-12
MAX_PROFILE_NRMS = 5.0e-2
MAX_LINE_NRMS = 1.5e-1
MAX_DG_RELATIVE_DIFFERENCE = 2.0e-2
CONSERVATION_TOLERANCE = 1.0e-7
MAX_JACOBIANS = 8
MAX_OBJECTIVE_EVALUATIONS = 16000
MAX_NEWTON_ITERATIONS = 32
RESCUE_MAX_JACOBIANS = 12
RESCUE_MAX_OBJECTIVE_EVALUATIONS = 26000
RESCUE_MAX_NEWTON_ITERATIONS = 48
JACOBIAN_STENCIL_RADIUS = 4
EXPECTED_FAILED_N32_SOURCE_COMMIT = "3cd50dc5a45f9bf086ac99dfc8e8762dc5b7d402"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_gate(diagnostics: object) -> float:
    return float(
        max(
            diagnostics.raw_total_linf,
            abs(diagnostics.held_out_continuity),
            abs(diagnostics.mass_error),
        )
    )


def load_authorized_n28(
    gate_dir: Path, source_commit: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    record_path = gate_dir / "JFM_N28_SAME_GRID_GATE.json"
    archive_path = gate_dir / "gu_emerson_jfm_n28_same_grid_candidate.npz"
    require(record_path.is_file(), "N28 same-grid record is missing")
    require(archive_path.is_file(), "N28 transformed candidate is missing")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    require(isinstance(record, dict), "N28 record is not an object")
    require(
        record.get("status") == "R26_GU_EMERSON_JFM_N28_SAME_GRID_GATE_PASSED",
        "N28 same-grid status did not pass",
    )
    require(record.get("source_commit") == source_commit, "N28 source commit mismatch")
    require(record.get("same_grid_gate_passed") is True, "N28 gate is not true")
    require(record.get("candidate_accepted") is True, "N28 candidate is rejected")
    require(record.get("n32_authorized") is True, "N32 is not authorized")
    require(record.get("n40_authorized") is False, "N28 gate over-authorized N40")
    require(record.get("production_accepted") is False, "N28 gate is production-marked")
    require(
        record.get("reference", {}).get("state_sha256")
        == JFM_N28_REFERENCE_STATE_SHA256,
        "N28 frozen reference hash mismatch",
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        require(bool(np.asarray(archive["accepted"]).item()), "N28 archive is rejected")
        require(
            str(np.asarray(archive["source_commit"]).item()) == source_commit,
            "N28 archive source commit mismatch",
        )
        state = np.asarray(archive["state"], dtype=float)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
    require(state.shape == (28, 28, 17), "N28 candidate shape mismatch")
    require(
        state_sha256(state) == record.get("candidate", {}).get("state_sha256"),
        "N28 candidate state hash mismatch",
    )
    return state, x, y, record


def load_failed_n32_candidate(
    failed_dir: Path,
    *,
    case: object,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load only the exact bounded N32 failure that motivated this rescue."""

    record_path = failed_dir / "JFM_N32_TRANSFORMED_CANDIDATE_GATE.json"
    archive_path = failed_dir / "gu_emerson_jfm_n32_candidate.npz"
    require(record_path.is_file(), "failed N32 record is missing")
    require(archive_path.is_file(), "failed N32 candidate archive is missing")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    require(isinstance(record, dict), "failed N32 record is not an object")
    require(
        record.get("status")
        == "R26_GU_EMERSON_JFM_N32_TRANSFORMED_CANDIDATE_FAILED",
        "predecessor N32 status is not FAILED",
    )
    require(
        record.get("source_commit") == EXPECTED_FAILED_N32_SOURCE_COMMIT,
        "predecessor N32 source commit mismatch",
    )
    require(record.get("candidate_accepted") is False, "failed N32 was accepted")
    require(record.get("n36_authorized") is False, "failed N32 authorized N36")
    require(record.get("n40_authorized") is False, "failed N32 authorized N40")
    require(record.get("maximum_grid_run") == 32, "failed run grid mismatch")
    require(
        record.get("higher_than_n32_run_attempted") is False,
        "failed run attempted a grid above N32",
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        state = np.asarray(archive["state"], dtype=float)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        require(not bool(np.asarray(archive["accepted"]).item()), "failed archive is accepted")
        require(int(np.asarray(archive["nodes"]).item()) == 32, "failed archive node mismatch")
        require(
            str(np.asarray(archive["source_commit"]).item())
            == EXPECTED_FAILED_N32_SOURCE_COMMIT,
            "failed archive source mismatch",
        )
        require(
            not bool(np.asarray(archive["n36_authorized"]).item()),
            "failed archive authorizes N36",
        )
        require(
            not bool(np.asarray(archive["n40_authorized"]).item()),
            "failed archive authorizes N40",
        )
    require(state.shape == (32, 32, 17), "failed N32 state shape mismatch")
    require(
        np.array_equal(x, case.x) and np.array_equal(y, case.y),
        "failed N32 grid mismatch",
    )
    require(
        state_sha256(state) == record.get("candidate", {}).get("state_sha256"),
        "failed N32 candidate hash mismatch",
    )
    return state, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n28-gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--failed-n32-dir",
        type=Path,
        help="resume only from the exact source-locked failed N32 candidate",
    )
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        parser.error("source commit must be an immutable lowercase 40-character SHA")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    args.output_dir.mkdir(parents=True)
    record_path = args.output_dir / "JFM_N32_TRANSFORMED_CANDIDATE_GATE.json"
    passed = False
    try:
        reference, old_x, old_y, n28_record = load_authorized_n28(
            args.n28_gate_dir, args.source_commit
        )
        case = jfm_maxwell_cavity_case(
            32,
            kn=0.2,
            lid_speed_m_per_s=100.0,
            wall_temperature_K=300.0,
            grid_stretch_beta=0.0,
        )
        require(case.name == "jfm-maxwell-Kn0.2-U100-N32", "N32 case name mismatch")
        require(case.r26_closure_mode == "jfm2009", "N32 coefficient mode mismatch")
        require(case.viscosity.exponent == 1.0, "N32 is not Maxwell molecules")
        require(case.grid_stretch_beta == 0.0, "N32 grid is not uniform")
        require(case.accommodation == 1.0, "N32 wall is not fully diffuse")
        require(
            math.isclose(case.kn, 0.2, rel_tol=0.0, abs_tol=2.0e-15),
            "N32 Kn mismatch",
        )
        problem = R26NodeBVP(
            case,
            bulk_operator=compatible_fv_bulk_residual,
            mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
        )
        predecessor_failure = None
        if args.failed_n32_dir is None:
            seed = interpolate_state_grid(
                reference,
                32,
                target_mean_density=case.mean_density,
                mass_weights=problem.mass_weights,
                old_x=old_x,
                old_y=old_y,
                new_x=case.x,
                new_y=case.y,
            )
            seed_kind = "accepted_transformed_N28_interpolated_to_N32"
        else:
            seed, predecessor_failure = load_failed_n32_candidate(
                args.failed_n32_dir,
                case=case,
            )
            seed_kind = "source_locked_failed_N32_candidate_for_physical_PTC_rescue"
        transform = GuEmersonLogStateTransform(case)
        seed_roundtrip = transform.decode(transform.encode(seed))
        seed_roundtrip_linf = float(
            np.max(np.abs(seed_roundtrip - seed), initial=0.0)
        )
        seed_evaluation = problem.evaluate(seed)
        seed_raw = raw_gate(seed_evaluation.diagnostics)
        if predecessor_failure is not None:
            require(
                math.isclose(
                    seed_raw,
                    float(predecessor_failure["candidate"]["raw_gate"]),
                    rel_tol=0.0,
                    abs_tol=5.0e-12,
                ),
                "failed N32 seed raw gate does not replay",
            )
        rescue = predecessor_failure is not None
        options = SolveOptions(
            method="colored_newton",
            residual_tolerance=1.0e-9,
            held_out_continuity_tolerance=RAW_TOLERANCE,
            max_iterations=(
                RESCUE_MAX_NEWTON_ITERATIONS if rescue else MAX_NEWTON_ITERATIONS
            ),
            max_objective_evaluations=(
                RESCUE_MAX_OBJECTIVE_EVALUATIONS
                if rescue
                else MAX_OBJECTIVE_EVALUATIONS
            ),
            analytic_mass_jacobian=True,
            pseudo_transient=rescue,
            pseudo_time_initial=1.0e-2,
            pseudo_time_minimum=1.0e-8,
            pseudo_time_maximum=1.0e8,
            pseudo_time_ser_exponent=1.0,
            pseudo_time_growth_limit=2.0,
            newton_switch_tolerance=1.0e-6,
            display=rescue,
            max_jacobian_evaluations=(
                RESCUE_MAX_JACOBIANS if rescue else MAX_JACOBIANS
            ),
            jacobian_stencil_radius=JACOBIAN_STENCIL_RADIUS,
        )
        result = solve_r26_bvp(
            problem,
            seed,
            options=options,
            state_transform=transform,
        )
        candidate = result.state
        evaluation = problem.evaluate(candidate)
        candidate_raw = raw_gate(evaluation.diagnostics)
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
        transformed_linf = float(
            np.max(np.abs(transformed[interior]), initial=0.0)
        )
        compatibility_linf = float(
            np.max(np.abs(transformed[interior] - physical[interior]), initial=0.0)
        )
        comparison = compare_cross_solver_profiles(
            reference,
            old_x,
            old_y,
            candidate,
            case.x,
            case.y,
            lid_velocity=case.lid_velocity,
        )
        balance = global_balance_diagnostics(candidate, case)
        conservation_passed = bool(
            float(balance["wall_effective_pressure_min"]) > 0.0
            and float(balance["momentum_boundary_flux_linf"])
            <= CONSERVATION_TOLERANCE
            and abs(float(balance["internal_energy_balance_error"]))
            <= CONSERVATION_TOLERANCE
            and float(balance["wall_normal_velocity_linf"]) <= RAW_TOLERANCE
        )
        comparison_passed = bool(
            float(comparison["maximum_normalized_rms_difference"])
            <= MAX_PROFILE_NRMS
            and float(comparison["maximum_line_normalized_rms_difference"])
            <= MAX_LINE_NRMS
            and float(comparison["D_relative_difference"])
            <= MAX_DG_RELATIVE_DIFFERENCE
            and float(comparison["G_relative_difference"])
            <= MAX_DG_RELATIVE_DIFFERENCE
        )
        pattern = jacobian_sparsity(
            problem,
            stencil_radius=JACOBIAN_STENCIL_RADIUS,
            include_mass_border=True,
        )
        rank = int(structural_rank(pattern))
        passed = bool(
            result.converged
            and result.scipy_success
            and candidate_raw <= RAW_TOLERANCE
            and evaluation.diagnostics.total_linf <= RAW_TOLERANCE
            and abs(evaluation.diagnostics.held_out_continuity) <= RAW_TOLERANCE
            and abs(evaluation.diagnostics.mass_error) <= 1.0e-10
            and evaluation.diagnostics.min_density > 0.0
            and evaluation.diagnostics.min_temperature > 0.0
            and transformed_linf <= RAW_TOLERANCE
            and compatibility_linf <= COMPATIBILITY_TOLERANCE
            and seed_roundtrip_linf <= 5.0e-11
            and conservation_passed
            and comparison_passed
            and rank == problem.unknown_count
        )
        record = {
            "status": (
                "R26_GU_EMERSON_JFM_N32_TRANSFORMED_CANDIDATE_PASSED"
                if passed
                else "R26_GU_EMERSON_JFM_N32_TRANSFORMED_CANDIDATE_FAILED"
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "case": asdict(case),
            "n28_authorization": {
                "record": str((args.n28_gate_dir / "JFM_N28_SAME_GRID_GATE.json").resolve()),
                "reference_state_sha256": JFM_N28_REFERENCE_STATE_SHA256,
                "candidate_state_sha256": state_sha256(reference),
                "same_grid_gate_passed": n28_record["same_grid_gate_passed"],
                "n32_authorized": n28_record["n32_authorized"],
            },
            "seed": {
                "kind": seed_kind,
                "state_sha256": state_sha256(seed),
                "raw_gate": seed_raw,
                "diagnostics": asdict(seed_evaluation.diagnostics),
                "transformed_coordinate_roundtrip_linf": seed_roundtrip_linf,
            },
            "predecessor_failure": (
                None
                if predecessor_failure is None
                else {
                    "source_commit": predecessor_failure["source_commit"],
                    "status": predecessor_failure["status"],
                    "candidate_state_sha256": predecessor_failure["candidate"][
                        "state_sha256"
                    ],
                    "candidate_raw_gate": predecessor_failure["candidate"][
                        "raw_gate"
                    ],
                    "solver_message": predecessor_failure["solver"]["message"],
                }
            ),
            "candidate": {
                "state_sha256": state_sha256(candidate),
                "raw_gate": candidate_raw,
                "transformed_interior_linf": transformed_linf,
                "transformed_vs_compatible_physical_linf": compatibility_linf,
                "diagnostics": asdict(evaluation.diagnostics),
                "global_balances": balance,
            },
            "solver": {
                "coordinate_system": "Gu--Emerson equations (48)--(55); log rho and theta",
                "residual": "historical compatible central physical R26 BVP",
                "equivalence_gate": "N28 transformed equation-(63) same-grid identity",
                "options": asdict(options),
                "converged": result.converged,
                "scipy_success": result.scipy_success,
                "message": result.message,
                "iterations": result.iterations,
                "function_evaluations": result.function_evaluations,
                "jacobian_evaluations": result.jacobian_evaluations,
                "invalid_evaluations": result.invalid_evaluations,
                "last_invalid_error": result.last_invalid_error,
            },
            "structural_rank": {
                "rank": rank,
                "unknown_count": problem.unknown_count,
                "full_rank": rank == problem.unknown_count,
                "jacobian_pattern_nonzeros": int(pattern.nnz),
            },
            "n28_to_n32_comparison": comparison,
            "comparison_passed": comparison_passed,
            "conservation_passed": conservation_passed,
            "candidate_accepted": passed,
            "production_accepted": False,
            "n36_authorized": passed,
            "n40_authorized": False,
            "n44_authorized": False,
            "maximum_grid_run": 32,
            "higher_than_n32_run_attempted": False,
            "next_required_stage": (
                "independently validate this N32 state, then run one bounded N36 candidate"
                if passed
                else "stop at N32 and inspect the recorded transformed-Newton failure"
            ),
            "thresholds": {
                "raw_tolerance": RAW_TOLERANCE,
                "compatibility_linf": COMPATIBILITY_TOLERANCE,
                "maximum_profile_nrms": MAX_PROFILE_NRMS,
                "maximum_line_nrms": MAX_LINE_NRMS,
                "maximum_DG_relative_difference": MAX_DG_RELATIVE_DIFFERENCE,
                "conservation_tolerance": CONSERVATION_TOLERANCE,
            },
            "source_manifest": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in (
                    ROOT / "r26_solver.py",
                    ROOT / "r26_gu_emerson_variables.py",
                    ROOT / "r26_gu_emerson_transformed_fv.py",
                    Path(__file__).resolve(),
                )
            },
        }
        np.savez_compressed(
            args.output_dir / "gu_emerson_jfm_n32_candidate.npz",
            state=candidate,
            seed=seed,
            encoded_transformed_state=transform.encode(candidate),
            x=case.x,
            y=case.y,
            nodes=32,
            kn_input=case.kn,
            source_commit=args.source_commit,
            accepted=passed,
            production_accepted=False,
            n36_authorized=passed,
            n40_authorized=False,
        )
    except Exception as exc:
        record = {
            "status": "R26_GU_EMERSON_JFM_N32_TRANSFORMED_CANDIDATE_FAILED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "failure": f"{type(exc).__name__}: {exc}",
            "candidate_accepted": False,
            "production_accepted": False,
            "n36_authorized": False,
            "n40_authorized": False,
            "n44_authorized": False,
            "maximum_grid_run": 32,
            "higher_than_n32_run_attempted": False,
        }
    record_path.write_text(
        json.dumps(jsonable(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(record), sort_keys=True), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
