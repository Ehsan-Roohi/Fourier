#!/usr/bin/env python3
"""Fail-closed Gu--Emerson standalone ladder on N8 and N16.

The ladder exercises the documented segregated reconstruction without a root
from another nonlinear solver.  N8 is solved from analytic equilibrium and
from one deterministic, smooth perturbation.  N16 is attempted only after
both N8 paths pass the complete raw R26 gate and agree.  Each N16 path is
initialized only by interpolation of its corresponding accepted N8 root.

This is one ordered validation job, not lid-speed continuation: Kn, lid speed,
wall temperature, closure model, and numerical controls remain fixed.  No
THOR state, global Newton/Krylov solve, homotopy, or pseudo-arclength state is
used as an initial condition.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re

import numpy as np

from r26_cases import gu_asme2009_cavity_case
from r26_gu_emerson_algorithm import GU_EMERSON_STAGE_ORDER
from r26_gu_emerson_reconstruction import (
    GuEmersonReconstructionOptions,
    make_gu_emerson_reconstruction_problem,
    solve_gu_emerson_reconstruction,
)
from r26_solver import interpolate_state_grid
from r26_thor_audit import state_sha256


CASE = {
    "kn_gu": 0.1,
    "lid_speed_m_s": 10.0,
    "wall_temperature_K": 273.0,
    "grid_stretch_beta": 0.0,
}
PRIOR_CROSS_GATE_STATUS = "R26_GU_EMERSON_RECONSTRUCTION_CROSS_GATE_PASSED"
PRIOR_CROSS_GATE_COMMIT = "c9c3bc07d14a691d2d4ed70533b46f8daed53726"
ROOT_ABSOLUTE_TOLERANCE = 1.0e-6

# The parameter source declares these per-equation linear tolerances.  The
# separate ``cs_user_modules.f90`` source does declare fixed nonlinear-source
# history factors; those were not used by this failed ladder and are introduced
# only by the source-linearized N16 resume.
RANA_CODE_SATURNE_CONTROL_EVIDENCE = {
    "archive": "SRCR26_22nd_NOV.tar",
    "source": "SRCR26_22nd_NOV/cs_user_parameters_R26.f90",
    "source_sha256": "0ce53e0811b00154fc0b3c7cb370cfce92a9382d53741a04758499c8132a13ca",
    "velocity_linear_relative_tolerance": 1.0e-6,
    "pressure_linear_relative_tolerance": 1.0e-6,
    "temperature_linear_relative_tolerance": 1.0e-6,
    "user_scalar_linear_relative_tolerance": 1.0e-5,
    "custom_under_relaxation_factors_declared_in_source": False,
    "nonlinear_source_history_declared_in_cs_user_modules": True,
    "nonlinear_source_history_factors": {
        "stress": 1.0e-2,
        "heat_flux": 1.0e-2,
        "m": 5.0e-1,
        "R": 5.0e-1,
        "Delta": 1.0e-1,
    },
    "cs_user_modules_sha256": "d92e0142776d90499e2beea4a8b3b37b590597f66b61f43bb49f58ade73a884b",
}


def jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Enum):
        return value.value
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")


def resolve_cross_gate(path: Path) -> tuple[Path, dict[str, object]]:
    candidates = (
        path / "GU_EMERSON_RECONSTRUCTION_CROSS_GATE.json",
        path / "RECONSTRUCTION" / "GU_EMERSON_RECONSTRUCTION_CROSS_GATE.json",
    )
    matches = [candidate for candidate in candidates if candidate.is_file()]
    if len(matches) != 1:
        raise RuntimeError("exactly one prior reconstruction cross-gate record is required")
    record = json.loads(matches[0].read_text())
    if record.get("status") != PRIOR_CROSS_GATE_STATUS:
        raise RuntimeError("prior reconstruction cross-gate status is not PASSED")
    if record.get("source_commit") != PRIOR_CROSS_GATE_COMMIT:
        raise RuntimeError("prior reconstruction cross-gate source commit mismatch")
    if record.get("cross_solver_fixed_point_gate_passed") is not True:
        raise RuntimeError("prior fixed-point gate boolean is not true")
    if record.get("production_accepted") is not False:
        raise RuntimeError("prior gate must remain explicitly non-production")
    return matches[0], record


def deterministic_perturbation(case: object) -> np.ndarray:
    """Return a smooth, positive, mass-corrected start with fixed amplitude."""

    state = case.equilibrium_state()
    x = np.asarray(case.x, dtype=float)
    y = np.asarray(case.y, dtype=float)
    X, Y = np.meshgrid(x, y)
    shape = np.sin(np.pi * X) * np.sin(np.pi * Y)
    state[..., 0] *= 1.0 + 1.0e-3 * shape
    state[..., 3] *= 1.0 - 5.0e-4 * shape
    state[..., 1] += 1.0e-3 * float(case.lid_velocity) * shape
    weights = make_gu_emerson_reconstruction_problem(case).mass_weights
    state[..., 0] *= float(case.mean_density) / float(np.sum(weights * state[..., 0]))
    if np.any(state[..., 0] <= 0.0) or np.any(state[..., 3] <= 0.0):
        raise RuntimeError("declared deterministic perturbation violated positivity")
    return state


def comparison(first: np.ndarray, second: np.ndarray) -> dict[str, object]:
    difference = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    component_max = np.max(np.abs(difference), axis=(0, 1))
    maximum = float(np.max(component_max, initial=0.0))
    return {
        "maximum_absolute_state_difference": maximum,
        "component_maximum_absolute_differences": component_max,
        "required_maximum_absolute_state_difference": ROOT_ABSOLUTE_TOLERANCE,
        "passed": bool(maximum <= ROOT_ABSOLUTE_TOLERANCE),
    }


def accepted_result(problem: object, result: object, options: object) -> tuple[bool, object, float]:
    diagnostics = problem.evaluate(result.state).diagnostics
    raw = float(
        max(
            diagnostics.raw_total_linf,
            abs(diagnostics.held_out_continuity),
            abs(diagnostics.mass_error),
        )
    )
    passed = bool(
        result.converged
        and raw <= options.raw_tolerance
        and diagnostics.total_linf <= options.scaled_tolerance
        and abs(diagnostics.held_out_continuity) <= options.held_continuity_tolerance
        and abs(diagnostics.mass_error) <= options.mass_tolerance
        and diagnostics.min_density > 0.0
        and diagnostics.min_temperature > 0.0
        and result.records
        and result.records[-1].stage_order == GU_EMERSON_STAGE_ORDER
    )
    return passed, diagnostics, raw


def run_path(
    output: Path,
    *,
    nodes: int,
    path_name: str,
    initial_state: np.ndarray,
    initial_kind: str,
    options: GuEmersonReconstructionOptions,
) -> tuple[np.ndarray, dict[str, object]]:
    case = gu_asme2009_cavity_case(
        nodes,
        kn=CASE["kn_gu"],
        lid_speed_m_per_s=CASE["lid_speed_m_s"],
        wall_temperature_K=CASE["wall_temperature_K"],
        grid_stretch_beta=CASE["grid_stretch_beta"],
    )
    problem = make_gu_emerson_reconstruction_problem(case)
    target = output / path_name
    target.mkdir(parents=True, exist_ok=False)
    initial_diagnostics = problem.evaluate(initial_state).diagnostics
    initial_raw = float(
        max(
            initial_diagnostics.raw_total_linf,
            abs(initial_diagnostics.held_out_continuity),
            abs(initial_diagnostics.mass_error),
        )
    )
    progress_path = target / "sweep_progress.jsonl"

    def checkpoint(item: object, state: np.ndarray) -> None:
        payload = asdict(item)
        with progress_path.open("a") as stream:
            stream.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")
        if item.outer_iteration % 5 == 0:
            np.savez_compressed(
                target / "latest_checkpoint.npz",
                state=state,
                nodes=nodes,
                initial_kind=initial_kind,
                outer_iteration=item.outer_iteration,
                raw_gate=item.raw_gate,
                accepted=False,
                production_accepted=False,
            )
        print(
            f"{path_name} sweep={item.outer_iteration} raw={item.raw_gate:.6e} "
            f"scaled={item.scaled_linf:.6e} held={item.held_continuity:.6e}",
            flush=True,
        )

    result = solve_gu_emerson_reconstruction(
        problem, initial_state, options=options, record_callback=checkpoint
    )
    passed, diagnostics, raw = accepted_result(problem, result, options)
    np.savez_compressed(
        target / "standalone_state.npz",
        state=result.state,
        x=case.x,
        y=case.y,
        nodes=nodes,
        lid_speed_m_s=CASE["lid_speed_m_s"],
        lid_velocity=case.lid_velocity,
        kn_input=case.kn,
        initial_kind=initial_kind,
        initial_state_sha256=state_sha256(initial_state),
        accepted=passed,
        production_accepted=False,
    )
    record = {
        "status": "R26_GU_EMERSON_STANDALONE_PATH_PASSED" if passed else "R26_GU_EMERSON_STANDALONE_PATH_FAILED",
        "nodes": nodes,
        "path_name": path_name,
        "initial_kind": initial_kind,
        "initial_state_sha256": state_sha256(initial_state),
        "initial_raw_gate": initial_raw,
        "state_sha256": state_sha256(result.state),
        "raw_acceptance_gate": raw,
        "diagnostics": asdict(diagnostics),
        "accepted": passed,
        "production_accepted": False,
        "solver": {
            "message": result.message,
            "outer_iterations": result.outer_iterations,
            "residual_evaluations": result.residual_evaluations,
            "block_factorizations": result.block_factorizations,
            "lsmr_fallbacks": result.lsmr_fallbacks,
            "wall_solves": result.wall_solves,
            "wall_function_evaluations": result.wall_function_evaluations,
        },
        "history": [asdict(item) for item in result.records],
    }
    write_json(target / "standalone_path.json", record)
    print(json.dumps(jsonable({key: record[key] for key in ("status", "nodes", "path_name", "raw_acceptance_gate", "solver")}), sort_keys=True), flush=True)
    return result.state, record


def final_record(
    *,
    status: str,
    source_commit: str,
    prior_path: Path,
    completed_stages: list[str],
    failure_stage: str | None,
    comparisons: dict[str, object],
) -> dict[str, object]:
    passed = status == "R26_GU_EMERSON_STANDALONE_LADDER_PASSED"
    return {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "prior_cross_gate_record": str(prior_path),
        "prior_cross_gate_commit": PRIOR_CROSS_GATE_COMMIT,
        "case_fixed": CASE,
        "published_stage_order": list(GU_EMERSON_STAGE_ORDER),
        "rana_code_saturne_control_evidence": RANA_CODE_SATURNE_CONTROL_EVIDENCE,
        "completed_stages": completed_stages,
        "failure_stage": failure_stage,
        "root_comparisons": comparisons,
        "standalone_from_equilibrium_attempted": True,
        "standalone_from_equilibrium_passed": passed,
        "independent_start_consistency_passed": passed,
        "production_accepted": False,
        "n24_authorized": passed,
        "n28_authorized": False,
        "n29_authorized": False,
        "n30_authorized": False,
        "next_required_stage": (
            "bounded N24 Gu--Emerson reconstruction from the independently accepted N16 roots"
            if passed
            else "inspect the first failed standalone stage; refined grids remain blocked"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-gate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        parser.error("source commit must be an immutable lowercase 40-character SHA")
    args.output.mkdir(parents=True, exist_ok=False)
    prior_path, _ = resolve_cross_gate(args.cross_gate_dir)
    # The measured N8 equilibrium path reduced the raw gate from 9.45e-2 to
    # 5.89e-4 in 120 fixed sweeps.  A 480-sweep fail-closed ceiling reaches
    # beyond the extrapolated raw-gate crossing without changing the method.
    options = GuEmersonReconstructionOptions(max_outer_iterations=480)
    write_json(
        args.output / "STANDALONE_LADDER_PLAN.json",
        {
            "source_commit": args.source_commit,
            "case_fixed": CASE,
            "ordered_stages": [
                "N8_FROM_EQUILIBRIUM",
                "N8_FROM_PERTURBED",
                "N8_ROOT_COMPARISON",
                "N16_FROM_N8_EQUILIBRIUM",
                "N16_FROM_N8_PERTURBED",
                "N16_ROOT_COMPARISON",
            ],
            "stop_on_first_failed_stage": True,
            "options": asdict(options),
            "measured_work_budget_basis": {
                "grid": 8,
                "initial_kind": "analytic_equilibrium",
                "raw_gate_after_sweep_1": 9.44523802593184e-2,
                "raw_gate_after_sweep_120": 5.889760194992856e-4,
                "sweep_ceiling": 480,
                "method_or_tolerance_changed_from_measurement": False,
            },
            "reconstruction_controls": {
                key: asdict(value) for key, value in (options.disclosure.controls or {}).items()
            },
            "rana_code_saturne_control_evidence": RANA_CODE_SATURNE_CONTROL_EVIDENCE,
        },
    )
    completed: list[str] = []
    comparisons: dict[str, object] = {}

    def stop(stage: str) -> None:
        record = final_record(
            status="R26_GU_EMERSON_STANDALONE_LADDER_FAILED",
            source_commit=args.source_commit,
            prior_path=prior_path,
            completed_stages=completed,
            failure_stage=stage,
            comparisons=comparisons,
        )
        write_json(args.output / "GU_EMERSON_STANDALONE_LADDER_FAILED.json", record)
        print(json.dumps(jsonable(record), sort_keys=True), flush=True)
        raise SystemExit(1)

    case8 = gu_asme2009_cavity_case(
        8, kn=CASE["kn_gu"], lid_speed_m_per_s=CASE["lid_speed_m_s"],
        wall_temperature_K=CASE["wall_temperature_K"], grid_stretch_beta=0.0,
    )
    n8_eq, record = run_path(
        args.output, nodes=8, path_name="N8_FROM_EQUILIBRIUM",
        initial_state=case8.equilibrium_state(), initial_kind="analytic_equilibrium",
        options=options,
    )
    if record["accepted"] is not True:
        stop("N8_FROM_EQUILIBRIUM")
    completed.append("N8_FROM_EQUILIBRIUM")

    n8_pert, record = run_path(
        args.output, nodes=8, path_name="N8_FROM_PERTURBED",
        initial_state=deterministic_perturbation(case8),
        initial_kind="deterministic_smooth_mass_corrected_perturbation",
        options=options,
    )
    if record["accepted"] is not True:
        stop("N8_FROM_PERTURBED")
    completed.append("N8_FROM_PERTURBED")
    comparisons["N8"] = comparison(n8_eq, n8_pert)
    write_json(args.output / "N8_ROOT_COMPARISON.json", comparisons["N8"])
    if comparisons["N8"]["passed"] is not True:
        stop("N8_ROOT_COMPARISON")
    completed.append("N8_ROOT_COMPARISON")

    case16 = gu_asme2009_cavity_case(
        16, kn=CASE["kn_gu"], lid_speed_m_per_s=CASE["lid_speed_m_s"],
        wall_temperature_K=CASE["wall_temperature_K"], grid_stretch_beta=0.0,
    )
    weights16 = make_gu_emerson_reconstruction_problem(case16).mass_weights
    seed16_eq = interpolate_state_grid(
        n8_eq, 16, target_mean_density=case16.mean_density, mass_weights=weights16,
        old_x=case8.x, old_y=case8.y, new_x=case16.x, new_y=case16.y,
    )
    seed16_pert = interpolate_state_grid(
        n8_pert, 16, target_mean_density=case16.mean_density, mass_weights=weights16,
        old_x=case8.x, old_y=case8.y, new_x=case16.x, new_y=case16.y,
    )
    n16_eq, record = run_path(
        args.output, nodes=16, path_name="N16_FROM_N8_EQUILIBRIUM",
        initial_state=seed16_eq, initial_kind="accepted_N8_equilibrium_path_interpolated",
        options=options,
    )
    if record["accepted"] is not True:
        stop("N16_FROM_N8_EQUILIBRIUM")
    completed.append("N16_FROM_N8_EQUILIBRIUM")

    n16_pert, record = run_path(
        args.output, nodes=16, path_name="N16_FROM_N8_PERTURBED",
        initial_state=seed16_pert, initial_kind="accepted_N8_perturbed_path_interpolated",
        options=options,
    )
    if record["accepted"] is not True:
        stop("N16_FROM_N8_PERTURBED")
    completed.append("N16_FROM_N8_PERTURBED")
    comparisons["N16"] = comparison(n16_eq, n16_pert)
    write_json(args.output / "N16_ROOT_COMPARISON.json", comparisons["N16"])
    if comparisons["N16"]["passed"] is not True:
        stop("N16_ROOT_COMPARISON")
    completed.append("N16_ROOT_COMPARISON")

    final = final_record(
        status="R26_GU_EMERSON_STANDALONE_LADDER_PASSED",
        source_commit=args.source_commit,
        prior_path=prior_path,
        completed_stages=completed,
        failure_stage=None,
        comparisons=comparisons,
    )
    write_json(args.output / "GU_EMERSON_STANDALONE_LADDER_PASSED.json", final)
    print(json.dumps(jsonable(final), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
