#!/usr/bin/env python3
"""Track the accepted JFM N28 root through N29, N30, and N31.

Every stage is fail-closed.  A later grid is never attempted unless the
immediately preceding state is a source-locked, fully converged algebraic root.
N32 is deliberately outside this driver and must consume the accepted N31
archive through ``run_r26_gu_emerson_jfm_n32_candidate.py``.
"""

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

from analysis.run_r26_gu_emerson_jfm_n32_candidate import (
    COMPATIBILITY_TOLERANCE,
    CONSERVATION_TOLERANCE,
    JACOBIAN_STENCIL_RADIUS,
    RAW_TOLERANCE,
    load_authorized_n28,
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
from r26_thor_audit import state_sha256
from r26_validation import global_balance_diagnostics


STAGE_NODES = (29, 30, 31)
MAX_JACOBIANS_PER_STAGE = 8
MAX_OBJECTIVE_EVALUATIONS_PER_STAGE = 20000
MAX_ITERATIONS_PER_STAGE = 96


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_paths(stage_dir: Path, nodes: int) -> tuple[Path, Path]:
    return (
        stage_dir / f"JFM_N{nodes}_GRID_CONTINUATION_STAGE.json",
        stage_dir / f"gu_emerson_jfm_n{nodes}_grid_continuation.npz",
    )


def solve_stage(
    predecessor: np.ndarray,
    predecessor_x: np.ndarray,
    predecessor_y: np.ndarray,
    *,
    predecessor_nodes: int,
    nodes: int,
    output_dir: Path,
    source_commit: str,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if nodes != predecessor_nodes + 1:
        raise RuntimeError("grid continuation must advance by exactly one node")
    output_dir.mkdir(parents=True)
    record_path, archive_path = stage_paths(output_dir, nodes)
    case = jfm_maxwell_cavity_case(
        nodes,
        kn=0.2,
        lid_speed_m_per_s=100.0,
        wall_temperature_K=300.0,
        grid_stretch_beta=0.0,
    )
    if case.r26_closure_mode != "jfm2009":
        raise RuntimeError(f"N{nodes} coefficient mode mismatch")
    if case.viscosity.exponent != 1.0 or case.grid_stretch_beta != 0.0:
        raise RuntimeError(f"N{nodes} is not the uniform Maxwell-molecule target")
    if not math.isclose(case.kn, 0.2, rel_tol=0.0, abs_tol=2.0e-15):
        raise RuntimeError(f"N{nodes} Kn mismatch")
    problem = R26NodeBVP(
        case,
        bulk_operator=compatible_fv_bulk_residual,
        mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
    )
    seed = interpolate_state_grid(
        predecessor,
        nodes,
        target_mean_density=case.mean_density,
        mass_weights=problem.mass_weights,
        old_x=predecessor_x,
        old_y=predecessor_y,
        new_x=case.x,
        new_y=case.y,
    )
    transform = GuEmersonLogStateTransform(case)
    seed_roundtrip_linf = float(
        np.max(np.abs(transform.decode(transform.encode(seed)) - seed), initial=0.0)
    )
    seed_evaluation = problem.evaluate(seed)
    options = SolveOptions(
        method="colored_newton",
        residual_tolerance=1.0e-9,
        held_out_continuity_tolerance=RAW_TOLERANCE,
        max_iterations=MAX_ITERATIONS_PER_STAGE,
        max_objective_evaluations=MAX_OBJECTIVE_EVALUATIONS_PER_STAGE,
        analytic_mass_jacobian=True,
        pseudo_transient=False,
        require_raw_linf_decrease=True,
        direction_strategy="raw_dogleg",
        trust_region_initial_newton_fraction=2.0**-8,
        trust_region_minimum_radius=1.0e-10,
        trust_region_maximum_radius=1.0e10,
        display=True,
        max_jacobian_evaluations=MAX_JACOBIANS_PER_STAGE,
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
        fields,
        case=case,
        convection_scheme="central",
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
    balance = global_balance_diagnostics(candidate, case)
    conservation_passed = bool(
        float(balance["wall_effective_pressure_min"]) > 0.0
        and float(balance["momentum_boundary_flux_linf"])
        <= CONSERVATION_TOLERANCE
        and abs(float(balance["internal_energy_balance_error"]))
        <= CONSERVATION_TOLERANCE
        and float(balance["wall_normal_velocity_linf"]) <= RAW_TOLERANCE
    )
    pattern = jacobian_sparsity(
        problem,
        stencil_radius=JACOBIAN_STENCIL_RADIUS,
        include_mass_border=True,
    )
    rank = int(structural_rank(pattern))
    accepted = bool(
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
        and rank == problem.unknown_count
    )
    next_nodes = nodes + 1
    record = {
        "status": (
            f"R26_GU_EMERSON_JFM_N{nodes}_GRID_CONTINUATION_PASSED"
            if accepted
            else f"R26_GU_EMERSON_JFM_N{nodes}_GRID_CONTINUATION_FAILED"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "case": asdict(case),
        "predecessor": {
            "nodes": predecessor_nodes,
            "state_sha256": state_sha256(predecessor),
        },
        "seed": {
            "kind": f"accepted_N{predecessor_nodes}_root_interpolated_one_cell_to_N{nodes}",
            "state_sha256": state_sha256(seed),
            "raw_gate": raw_gate(seed_evaluation.diagnostics),
            "diagnostics": asdict(seed_evaluation.diagnostics),
            "transformed_coordinate_roundtrip_linf": seed_roundtrip_linf,
        },
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
        "conservation_passed": conservation_passed,
        "candidate_accepted": accepted,
        "production_accepted": False,
        f"n{next_nodes}_authorized": accepted,
        "n32_authorized": bool(accepted and nodes == 31),
        "n36_authorized": False,
        "n40_authorized": False,
        "n44_authorized": False,
        "maximum_grid_run": nodes,
        "higher_than_n32_run_attempted": False,
        "next_required_stage": (
            f"run exactly one N{next_nodes} continuation stage"
            if accepted
            else f"stop at N{nodes}; do not attempt N{next_nodes}"
        ),
        "thresholds": {
            "raw_tolerance": RAW_TOLERANCE,
            "compatibility_linf": COMPATIBILITY_TOLERANCE,
            "conservation_tolerance": CONSERVATION_TOLERANCE,
        },
    }
    record_path.write_text(
        json.dumps(jsonable(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        archive_path,
        state=candidate,
        seed=seed,
        encoded_transformed_state=transform.encode(candidate),
        x=case.x,
        y=case.y,
        nodes=nodes,
        source_commit=source_commit,
        accepted=accepted,
        production_accepted=False,
        n32_authorized=bool(accepted and nodes == 31),
        n36_authorized=False,
        n40_authorized=False,
    )
    print(
        f"R26_GRID_STAGE N={nodes} accepted={str(accepted).lower()} "
        f"seed_raw={raw_gate(seed_evaluation.diagnostics):.16e} "
        f"candidate_raw={candidate_raw:.16e} "
        f"jacobians={result.jacobian_evaluations} message={result.message}",
        flush=True,
    )
    return accepted, candidate, case.x, case.y, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n28-gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        parser.error("source commit must be an immutable lowercase 40-character SHA")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    args.output_dir.mkdir(parents=True)
    summary_path = args.output_dir / "JFM_N29_N31_GRID_CONTINUATION.json"
    stages: list[dict[str, object]] = []
    passed = False
    failure: str | None = None
    maximum_grid_run = 28
    try:
        predecessor, predecessor_x, predecessor_y, n28_record = load_authorized_n28(
            args.n28_gate_dir,
            args.source_commit,
        )
        predecessor_nodes = 28
        for nodes in STAGE_NODES:
            maximum_grid_run = nodes
            accepted, candidate, x, y, record = solve_stage(
                predecessor,
                predecessor_x,
                predecessor_y,
                predecessor_nodes=predecessor_nodes,
                nodes=nodes,
                output_dir=args.output_dir / f"N{nodes}",
                source_commit=args.source_commit,
            )
            stages.append(
                {
                    "nodes": nodes,
                    "status": record["status"],
                    "candidate_accepted": accepted,
                    "candidate_state_sha256": record["candidate"]["state_sha256"],
                    "candidate_raw_gate": record["candidate"]["raw_gate"],
                    "record": str(stage_paths(args.output_dir / f"N{nodes}", nodes)[0]),
                }
            )
            if not accepted:
                failure = f"N{nodes} stage rejected; N{nodes + 1} was not attempted"
                break
            predecessor = candidate
            predecessor_x = x
            predecessor_y = y
            predecessor_nodes = nodes
        passed = bool(len(stages) == len(STAGE_NODES) and stages[-1]["candidate_accepted"])
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        n28_record = None
    summary = {
        "status": (
            "R26_GU_EMERSON_JFM_N29_N31_GRID_CONTINUATION_PASSED"
            if passed
            else "R26_GU_EMERSON_JFM_N29_N31_GRID_CONTINUATION_FAILED"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "strategy": "accepted-root one-node grid continuation N28->N29->N30->N31",
        "n28_gate_status": None if n28_record is None else n28_record.get("status"),
        "stages": stages,
        "failure": failure,
        "candidate_accepted": passed,
        "production_accepted": False,
        "n32_authorized": passed,
        "n36_authorized": False,
        "n40_authorized": False,
        "n44_authorized": False,
        "maximum_grid_run": maximum_grid_run,
        "higher_than_n32_run_attempted": False,
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
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(summary), sort_keys=True), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
