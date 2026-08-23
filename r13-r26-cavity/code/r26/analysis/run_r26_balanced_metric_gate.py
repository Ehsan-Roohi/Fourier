#!/usr/bin/env python3
"""Replay accepted N8/N16 roots with a secant-balanced arclength metric."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from r26_arclength import (
    ArcLengthCorrectorOptions,
    ArcLengthMetric,
    balanced_parameter_scale,
    normalized_secant_tangent,
    secant_metric_diagnostics,
    solve_r26_pseudo_arclength_step,
)
from r26_cases import jfm_maxwell_cavity_case
from r26_discretization import R26NodeBVP
from r26_fv_backend import (
    compatible_fv_bulk_residual,
    wall_bounded_control_volume_weights,
)
from r26_solver import LogStateTransform


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"R26_BALANCED_METRIC_GATE_FAILED: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha256(state: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(state, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(b"|<f8|")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def attempt_archive(run_dir: Path, attempt: dict[str, object]) -> Path:
    path = run_dir / (
        f"attempt_{int(attempt['attempt']):03d}_"
        f"lid_{float(attempt['proposed_lid']):.12g}.npz"
    )
    require(path.is_file(), f"accepted seed archive missing: {path.name}")
    return path


def load_seed(
    run_dir: Path,
    attempt: dict[str, object],
    nodes: int,
) -> tuple[np.ndarray, float, dict[str, object]]:
    require(bool(attempt.get("accepted")), "metric-gate seed is not accepted")
    path = attempt_archive(run_dir, attempt)
    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive["state"], dtype=float)
        parameter = float(np.asarray(archive["lid_velocity"]).item())
        accepted = bool(np.asarray(archive["accepted"]).item())
    require(accepted, f"{path.name} is explicitly rejected")
    require(state.shape == (nodes, nodes, 17), f"{path.name} has wrong shape")
    require(
        abs(parameter - float(attempt["proposed_lid"])) <= 2.0e-14,
        f"{path.name} lid mismatch",
    )
    return state, parameter, {
        "attempt": int(attempt["attempt"]),
        "parameter": parameter,
        "archive": str(path.resolve()),
        "archive_sha256": sha256(path),
        "state_sha256": state_sha256(state),
    }


def make_problem(case: object) -> R26NodeBVP:
    return R26NodeBVP(
        case,
        bulk_operator=compatible_fv_bulk_residual,
        mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
    )


def replay_grid(
    gate_dir: Path,
    output_dir: Path,
    nodes: int,
    parameter_fraction: float,
) -> dict[str, object]:
    run_dir = gate_dir / f"N{nodes}"
    summary_path = run_dir / "run_summary.json"
    require(summary_path.is_file(), f"N{nodes} run summary missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary.get("termination") == "target_accepted", f"N{nodes} source gate failed")
    accepted = [row for row in summary.get("attempts", []) if bool(row.get("accepted"))]
    require(len(accepted) >= 3, f"N{nodes} needs three accepted roots")
    previous_state, previous_parameter, previous_provenance = load_seed(
        run_dir, accepted[-3], nodes
    )
    current_state, current_parameter, current_provenance = load_seed(
        run_dir, accepted[-2], nodes
    )
    reference_state, reference_parameter, reference_provenance = load_seed(
        run_dir, accepted[-1], nodes
    )
    require(
        previous_parameter < current_parameter < reference_parameter,
        f"N{nodes} source roots are not ordered",
    )

    transform = LogStateTransform((nodes, nodes, 17))
    previous_encoded = transform.encode(previous_state)
    current_encoded = transform.encode(current_state)
    parameter_scale = balanced_parameter_scale(
        previous_encoded,
        previous_parameter,
        current_encoded,
        current_parameter,
        parameter_fraction=parameter_fraction,
    )
    metric = ArcLengthMetric(nodes * nodes * 17, parameter_scale=parameter_scale)
    metric_record = secant_metric_diagnostics(
        previous_encoded,
        previous_parameter,
        current_encoded,
        current_parameter,
        metric,
    )
    require(
        abs(metric_record.parameter_fraction - parameter_fraction) <= 2.0e-14,
        f"N{nodes} metric did not reach its requested balance",
    )
    tangent = normalized_secant_tangent(
        previous_encoded,
        previous_parameter,
        current_encoded,
        current_parameter,
        metric,
    )
    require(tangent.parameter > 0.0, f"N{nodes} tangent does not advance the lid")
    step_length = (reference_parameter - current_parameter) / tangent.parameter
    require(step_length > 0.0, f"N{nodes} replay step is not positive")

    case = jfm_maxwell_cavity_case(
        nodes,
        kn=0.20,
        lid_speed_m_per_s=100.0,
        wall_temperature_K=300.0,
        grid_stretch_beta=0.0,
    )
    result = solve_r26_pseudo_arclength_step(
        case,
        previous_state,
        previous_parameter,
        current_state,
        current_parameter,
        step_length,
        options=ArcLengthCorrectorOptions(
            residual_tolerance=1.0e-9,
            raw_tolerance=1.0e-8,
            parameter_scale=parameter_scale,
            maximum_iterations=80,
            maximum_jacobians=7,
            maximum_objective_evaluations=6000,
        ),
        reference_tangent=tangent,
    )
    require(result.accepted, f"N{nodes} balanced replay was rejected: {result.message}")
    require(result.parameter > current_parameter, f"N{nodes} replay did not advance")
    permitted_parameter_error = 0.25 * (reference_parameter - current_parameter)
    require(
        abs(result.parameter - reference_parameter) <= permitted_parameter_error,
        f"N{nodes} corrector moved too far from the known next branch point",
    )
    independent_case = case.with_lid_velocity(result.parameter, suffix="metric-gate-check")
    evaluation = make_problem(independent_case).evaluate(result.state)
    independent_raw = max(
        evaluation.diagnostics.raw_total_linf,
        abs(evaluation.diagnostics.held_out_continuity),
        abs(evaluation.diagnostics.mass_error),
    )
    require(independent_raw <= 1.0e-8, f"N{nodes} independent raw gate failed")

    grid_output = output_dir / f"N{nodes}"
    grid_output.mkdir(parents=True)
    np.savez_compressed(
        grid_output / "balanced_metric_state.npz",
        state=result.state,
        lid_velocity=result.parameter,
        kn_input=0.20,
        beta=0.0,
        accepted=result.accepted,
    )
    record: dict[str, object] = {
        "status": f"R26_N{nodes}_BALANCED_METRIC_REPLAY_PASS",
        "nodes": nodes,
        "source_roots": {
            "previous": previous_provenance,
            "current": current_provenance,
            "known_next": reference_provenance,
        },
        "metric": asdict(metric_record),
        "requested_parameter_fraction": parameter_fraction,
        "step_length": step_length,
        "predicted_parameter": result.predicted_parameter,
        "corrected_parameter": result.parameter,
        "known_next_parameter": reference_parameter,
        "raw_acceptance_gate": result.raw_acceptance_gate,
        "independent_raw_gate": independent_raw,
        "solver": {
            "message": result.message,
            "iterations": result.iterations,
            "jacobian_evaluations": result.jacobian_evaluations,
            "objective_evaluations": result.objective_evaluations,
            "pseudo_transient_steps": result.pseudo_transient_steps,
            "iteration_trace": result.iteration_trace,
        },
        "diagnostics": asdict(result.diagnostics),
        "state_sha256": state_sha256(result.state),
    }
    write_json(grid_output / "balanced_metric_replay.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gate-source-commit", required=True)
    parser.add_argument("--parameter-metric-fraction", type=float, default=0.5)
    args = parser.parse_args()

    require(args.gate_dir.is_dir(), "historical N8/N16 gate directory missing")
    require(not args.output_dir.exists(), "metric-gate output directory already exists")
    require(
        0.1 <= args.parameter_metric_fraction <= 0.9,
        "parameter metric fraction must lie within [0.1, 0.9]",
    )
    gate_record_path = args.gate_dir / "N16_GATE_PASSED.json"
    require(gate_record_path.is_file(), "historical N16 gate record missing")
    gate_record = json.loads(gate_record_path.read_text(encoding="utf-8"))
    require(
        gate_record.get("source_commit") == args.expected_gate_source_commit,
        "historical gate source commit mismatch",
    )
    args.output_dir.mkdir(parents=True)
    records = [
        replay_grid(
            args.gate_dir,
            args.output_dir,
            nodes,
            args.parameter_metric_fraction,
        )
        for nodes in (8, 16)
    ]
    summary = {
        "status": "R26_BALANCED_ARCLENGTH_METRIC_GATE_PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "historical_gate_source_commit": args.expected_gate_source_commit,
        "parameter_metric_fraction": args.parameter_metric_fraction,
        "n30_authorized": False,
        "records": records,
        "note": (
            "This gate validates the balanced arclength metric only; it does "
            "not run or authorize N30."
        ),
    }
    write_json(args.output_dir / "BALANCED_METRIC_GATE_PASSED.json", summary)
    print(json.dumps({"status": summary["status"], "n30_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
