#!/usr/bin/env python3
"""Resume a stalled N30 branch with bounded pseudo-arclength continuation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

from r26_arclength import (
    ArcLengthCorrectorOptions,
    ArcLengthMetric,
    balanced_parameter_scale,
    interpolate_bracketed_state,
    normalized_secant_tangent,
    secant_metric_diagnostics,
    solve_r26_pseudo_arclength_step,
)
from r26_cases import jfm_maxwell_cavity_case
from r26_discretization import R26NodeBVP
from r26_fv_backend import compatible_fv_bulk_residual, wall_bounded_control_volume_weights
from r26_solver import LogStateTransform
from r26_validation import global_balance_diagnostics


ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = tuple(sorted(ROOT.glob("r26_*.py")))
SOURCE_FILES = CORE_FILES + (Path(__file__).resolve(),)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"R26_ARCLENGTH_RESCUE_FAILED: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(state, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(b"|<f8|")
    digest.update(value.tobytes())
    return digest.hexdigest()


def jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def source_manifest() -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in SOURCE_FILES:
        prefix = "analysis" if path.parent.name == "analysis" else "code"
        manifest[f"{prefix}/{path.name}"] = sha256_file(path)
    return manifest


def make_problem(case: object) -> R26NodeBVP:
    return R26NodeBVP(
        case,
        bulk_operator=compatible_fv_bulk_residual,
        mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
    )


def attempt_archive(run_dir: Path, attempt: dict[str, object]) -> Path:
    stem = (
        f"attempt_{int(attempt['attempt']):03d}_"
        f"lid_{float(attempt['proposed_lid']):.12g}.npz"
    )
    path = run_dir / stem
    require(path.is_file(), f"accepted seed archive missing: {stem}")
    return path


def arclength_attempt_archive(
    run_dir: Path,
    attempt: dict[str, object],
) -> Path:
    stem = (
        f"arc_attempt_{int(attempt['attempt']):03d}_"
        f"lid_{float(attempt['corrected_parameter']):.12g}.npz"
    )
    path = run_dir / stem
    require(path.is_file(), f"accepted arclength seed archive missing: {stem}")
    return path


def load_and_validate_seed(
    run_dir: Path,
    attempt: dict[str, object],
    template: object,
    raw_tolerance: float,
) -> tuple[np.ndarray, float, dict[str, object]]:
    path = attempt_archive(run_dir, attempt)
    parameter = float(attempt["proposed_lid"])
    require(bool(attempt.get("accepted")), "selected seed attempt is not accepted")
    require(
        float(attempt.get("raw_acceptance_gate")) <= raw_tolerance,
        "selected seed failed its recorded raw gate",
    )
    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive["state"], dtype=float)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        lid = float(np.asarray(archive["lid_velocity"]).item())
        accepted = bool(np.asarray(archive["accepted"]).item())
        kn = float(np.asarray(archive["kn_input"]).item())
        beta = float(np.asarray(archive["beta"]).item())
    require(accepted, "seed archive is explicitly marked rejected")
    require(state.shape == (30, 30, 17), "seed state shape is not N30")
    require(np.array_equal(x, template.x), "seed x coordinates changed")
    require(np.array_equal(y, template.y), "seed y coordinates changed")
    require(abs(lid - parameter) <= 2.0e-14, "seed lid metadata mismatch")
    require(abs(kn - 0.20) <= 2.0e-15, "seed Kn metadata mismatch")
    require(abs(beta) <= 2.0e-15, "seed beta metadata mismatch")
    require(state_sha256(state) == attempt.get("state_sha256"), "seed state hash mismatch")
    case = template.with_lid_velocity(parameter, suffix="arc-seed-validator")
    problem = make_problem(case)
    evaluation = problem.evaluate(state)
    balances = global_balance_diagnostics(state, case)
    raw_gate = max(
        evaluation.diagnostics.raw_total_linf,
        abs(evaluation.diagnostics.held_out_continuity),
        abs(evaluation.diagnostics.mass_error),
    )
    require(raw_gate <= raw_tolerance, "independently recomputed seed raw gate failed")
    require(evaluation.diagnostics.min_density > 0.0, "seed density is non-positive")
    require(evaluation.diagnostics.min_temperature > 0.0, "seed temperature is non-positive")
    require(float(balances["wall_effective_pressure_min"]) > 0.0, "seed wall pressure failed")
    require(
        float(balances["momentum_boundary_flux_linf"]) <= 10.0 * raw_tolerance,
        "seed momentum balance failed",
    )
    require(
        abs(float(balances["internal_energy_balance_error"]))
        <= 10.0 * raw_tolerance,
        "seed energy balance failed",
    )
    provenance = {
        "attempt": int(attempt["attempt"]),
        "parameter": parameter,
        "archive": str(path.resolve()),
        "archive_sha256": sha256_file(path),
        "state_sha256": state_sha256(state),
        "independent_raw_gate": raw_gate,
    }
    return state, parameter, provenance


def load_and_validate_arclength_seed(
    run_dir: Path,
    attempt: dict[str, object],
    template: object,
    raw_tolerance: float,
) -> tuple[np.ndarray, float, dict[str, object]]:
    path = arclength_attempt_archive(run_dir, attempt)
    parameter = float(attempt["corrected_parameter"])
    require(bool(attempt.get("accepted")), "selected arclength seed is not accepted")
    require(
        float(attempt.get("raw_acceptance_gate")) <= raw_tolerance,
        "selected arclength seed failed its recorded raw gate",
    )
    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive["state"], dtype=float)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        lid = float(np.asarray(archive["lid_velocity"]).item())
        accepted = bool(np.asarray(archive["accepted"]).item())
        kn = float(np.asarray(archive["kn_input"]).item())
        beta = float(np.asarray(archive["beta"]).item())
    require(accepted, "arclength seed archive is explicitly marked rejected")
    require(state.shape == (30, 30, 17), "arclength seed shape is not N30")
    require(np.array_equal(x, template.x), "arclength seed x coordinates changed")
    require(np.array_equal(y, template.y), "arclength seed y coordinates changed")
    require(abs(lid - parameter) <= 2.0e-14, "arclength seed lid mismatch")
    require(abs(kn - 0.20) <= 2.0e-15, "arclength seed Kn mismatch")
    require(abs(beta) <= 2.0e-15, "arclength seed beta mismatch")
    require(
        state_sha256(state) == attempt.get("state_sha256"),
        "arclength seed state hash mismatch",
    )
    case = template.with_lid_velocity(parameter, suffix="arc-resume-validator")
    evaluation = make_problem(case).evaluate(state)
    balances = global_balance_diagnostics(state, case)
    raw_gate = max(
        evaluation.diagnostics.raw_total_linf,
        abs(evaluation.diagnostics.held_out_continuity),
        abs(evaluation.diagnostics.mass_error),
    )
    require(raw_gate <= raw_tolerance, "recomputed arclength seed raw gate failed")
    require(evaluation.diagnostics.min_density > 0.0, "resume density is non-positive")
    require(evaluation.diagnostics.min_temperature > 0.0, "resume temperature is non-positive")
    require(float(balances["wall_effective_pressure_min"]) > 0.0, "resume wall pressure failed")
    require(
        float(balances["momentum_boundary_flux_linf"]) <= 10.0 * raw_tolerance,
        "resume momentum balance failed",
    )
    require(
        abs(float(balances["internal_energy_balance_error"]))
        <= 10.0 * raw_tolerance,
        "resume energy balance failed",
    )
    provenance = {
        "attempt": int(attempt["attempt"]),
        "parameter": parameter,
        "archive": str(path.resolve()),
        "archive_sha256": sha256_file(path),
        "state_sha256": state_sha256(state),
        "independent_raw_gate": raw_gate,
    }
    return state, parameter, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-run-dir", type=Path, required=True)
    parser.add_argument("--failed-arclength-dir", type=Path)
    parser.add_argument("--resume-arclength-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-failed-source-commit", required=True)
    parser.add_argument("--expected-failed-arclength-source-commit")
    parser.add_argument("--expected-resume-arclength-source-commit")
    parser.add_argument(
        "--target-lid",
        type=float,
        default=100.0 / np.sqrt(208.0 * 300.0),
    )
    parser.add_argument("--raw-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--parameter-scale", type=float)
    parser.add_argument("--parameter-metric-fraction", type=float, default=0.5)
    parser.add_argument("--initial-step-factor", type=float, default=1.0)
    parser.add_argument("--minimum-step-factor", type=float, default=0.125)
    parser.add_argument("--maximum-step-factor", type=float, default=2.0)
    parser.add_argument("--maximum-attempts", type=int, default=24)
    parser.add_argument("--maximum-iterations", type=int, default=80)
    parser.add_argument("--maximum-jacobians", type=int, default=7)
    parser.add_argument("--maximum-objective-evaluations", type=int, default=6000)
    parser.add_argument("--pseudo-transient-chord-limit", type=int, default=12)
    parser.add_argument("--newton-chord-limit", type=int, default=3)
    args = parser.parse_args()

    require(args.failed_run_dir.is_dir(), "failed production directory missing")
    require(
        (args.failed_arclength_dir is None)
        == (args.expected_failed_arclength_source_commit is None),
        "failed arclength directory and expected commit must be supplied together",
    )
    require(
        (args.resume_arclength_dir is None)
        == (args.expected_resume_arclength_source_commit is None),
        "resume arclength directory and expected commit must be supplied together",
    )
    require(
        args.resume_arclength_dir is None or args.failed_arclength_dir is not None,
        "continuation resume requires the original failed-arclength metric seed",
    )
    if args.failed_arclength_dir is not None:
        require(args.failed_arclength_dir.is_dir(), "failed arclength directory missing")
        require(
            len(args.expected_failed_arclength_source_commit) == 40,
            "failed arclength commit SHA has wrong length",
        )
    if args.resume_arclength_dir is not None:
        require(args.resume_arclength_dir.is_dir(), "resume arclength directory missing")
        require(
            len(args.expected_resume_arclength_source_commit) == 40,
            "resume arclength commit SHA has wrong length",
        )
    require(not args.output_dir.exists(), "arclength output directory already exists")
    require(len(args.expected_failed_source_commit) == 40, "failed commit SHA has wrong length")
    require(args.maximum_attempts >= 1, "maximum attempts must be positive")
    require(args.maximum_iterations >= 1, "maximum iterations must be positive")
    require(
        0.1 <= args.parameter_metric_fraction <= 0.9,
        "parameter metric fraction must lie within [0.1, 0.9]",
    )
    require(
        args.pseudo_transient_chord_limit >= 1 and args.newton_chord_limit >= 1,
        "chord limits must be positive",
    )
    require(
        0.0 < args.minimum_step_factor <= args.initial_step_factor <= args.maximum_step_factor,
        "arclength step factors must be positive and ordered",
    )
    args.output_dir.mkdir(parents=True)

    failed_record_path = args.failed_run_dir / "N30_PRODUCTION_FAILED.json"
    source_commit_path = args.failed_run_dir / "source_commit.txt"
    run_dir = args.failed_run_dir / "N30"
    summary_path = run_dir / "run_summary.json"
    for path in (failed_record_path, source_commit_path, summary_path):
        require(path.is_file(), f"required failed-run artifact missing: {path.name}")
    failed_record = json.loads(failed_record_path.read_text(encoding="utf-8"))
    failed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_commit = source_commit_path.read_text(encoding="utf-8").strip()
    require(failed_record.get("status") == "R26_N30_PRODUCTION_FAILED", "wrong failed-run status")
    require(
        failed_record.get("source_commit") == args.expected_failed_source_commit,
        "failed record source commit mismatch",
    )
    require(source_commit == args.expected_failed_source_commit, "source_commit.txt mismatch")
    require(failed_summary.get("termination") == "minimum_step_rejected", "unexpected failure mode")
    case_record = failed_summary.get("case", {})
    require(case_record.get("family") == "jfm-maxwell", "wrong case family")
    require(int(case_record.get("nodes", -1)) == 30, "failed run is not N30")
    require(abs(float(case_record.get("kn_input")) - 0.20) <= 2.0e-15, "wrong Kn")
    require(abs(float(case_record.get("beta"))) <= 2.0e-15, "wrong beta")
    nonlinear = failed_summary.get("nonlinear_solver", {})
    for key in ("analytic_mass_jacobian", "secant_predictor", "ser_pseudo_transient"):
        require(bool(nonlinear.get(key)), f"failed run did not enable {key}")

    template = jfm_maxwell_cavity_case(
        30,
        kn=0.20,
        lid_speed_m_per_s=100.0,
        wall_temperature_K=300.0,
        grid_stretch_beta=0.0,
    )
    accepted_attempts = [
        row for row in failed_summary.get("attempts", []) if bool(row.get("accepted"))
    ]
    require(len(accepted_attempts) >= 2, "fewer than two accepted N30 seeds")
    resume_provenance: dict[str, object] | None = None
    if args.failed_arclength_dir is None:
        previous_state, previous_parameter, previous_provenance = (
            load_and_validate_seed(
                run_dir,
                accepted_attempts[-2],
                template,
                args.raw_tolerance,
            )
        )
        current_state, current_parameter, current_provenance = load_and_validate_seed(
            run_dir,
            accepted_attempts[-1],
            template,
            args.raw_tolerance,
        )
    else:
        failed_arc_record_path = (
            args.failed_arclength_dir / "N30_ARCLENGTH_RESCUE_FAILED.json"
        )
        failed_arc_source_path = args.failed_arclength_dir / "source_commit.txt"
        prior_arc_dir = args.failed_arclength_dir / "ARCLENGTH"
        prior_arc_summary_path = prior_arc_dir / "arclength_summary.json"
        for path in (
            failed_arc_record_path,
            failed_arc_source_path,
            prior_arc_summary_path,
        ):
            require(path.is_file(), f"required failed-arclength artifact missing: {path.name}")
        failed_arc_record = json.loads(
            failed_arc_record_path.read_text(encoding="utf-8")
        )
        prior_arc_summary = json.loads(
            prior_arc_summary_path.read_text(encoding="utf-8")
        )
        failed_arc_source = failed_arc_source_path.read_text(encoding="utf-8").strip()
        require(
            failed_arc_record.get("status") == "R26_N30_ARCLENGTH_RESCUE_FAILED",
            "wrong failed arclength status",
        )
        require(
            failed_arc_record.get("source_commit")
            == args.expected_failed_arclength_source_commit,
            "failed arclength record source commit mismatch",
        )
        require(
            failed_arc_source == args.expected_failed_arclength_source_commit,
            "failed arclength source_commit.txt mismatch",
        )
        require(
            failed_arc_record.get("failed_production_commit")
            == args.expected_failed_source_commit,
            "failed arclength production provenance mismatch",
        )
        require(
            prior_arc_summary.get("termination")
            == "minimum_arclength_step_rejected",
            "unexpected failed arclength termination",
        )
        prior_case = prior_arc_summary.get("case", {})
        require(int(prior_case.get("nodes", -1)) == 30, "failed arclength is not N30")
        require(abs(float(prior_case.get("kn_input")) - 0.20) <= 2.0e-15, "wrong resume Kn")
        require(abs(float(prior_case.get("beta"))) <= 2.0e-15, "wrong resume beta")
        prior_failed_provenance = prior_arc_summary.get("failed_run_provenance", {})
        require(
            prior_failed_provenance.get("source_commit")
            == args.expected_failed_source_commit,
            "resume production source mismatch",
        )
        require(
            prior_failed_provenance.get("failed_summary_sha256")
            == sha256_file(summary_path),
            "resume production summary hash mismatch",
        )
        prior_accepted = [
            row
            for row in prior_arc_summary.get("attempts", [])
            if bool(row.get("accepted"))
        ]
        require(prior_accepted, "failed arclength run has no accepted resume root")
        previous_state, previous_parameter, previous_provenance = (
            load_and_validate_seed(
                run_dir,
                accepted_attempts[-1],
                template,
                args.raw_tolerance,
            )
        )
        current_state, current_parameter, current_provenance = (
            load_and_validate_arclength_seed(
                prior_arc_dir,
                prior_accepted[-1],
                template,
                args.raw_tolerance,
            )
        )
        prior_controls = prior_arc_summary.get("arclength_controls", {})
        prior_accepted_step = float(prior_accepted[-1]["step_length"])
        prior_minimum_step = float(prior_controls["minimum_step"])
        require(
            0.0 < prior_minimum_step <= prior_accepted_step,
            "invalid prior arclength step bounds",
        )
        resume_provenance = {
            "directory": str(args.failed_arclength_dir.resolve()),
            "source_commit": failed_arc_source,
            "summary_sha256": sha256_file(prior_arc_summary_path),
            "accepted_seed": current_provenance,
            "prior_minimum_step": prior_minimum_step,
            "prior_accepted_step": prior_accepted_step,
        }

    # Freeze the metric calibrated from the original non-degenerate accepted
    # secant.  A later near-fold secant is continuation geometry, not a reason
    # to redefine the norm.
    calibration_previous_state = previous_state
    calibration_previous_parameter = previous_parameter
    calibration_previous_provenance = previous_provenance
    calibration_current_state = current_state
    calibration_current_parameter = current_parameter
    calibration_current_provenance = current_provenance
    continuation_resume_provenance: dict[str, object] | None = None

    if args.resume_arclength_dir is not None:
        resume_record_path = (
            args.resume_arclength_dir / "N30_BALANCED_ARCLENGTH_RESCUE_FAILED.json"
        )
        resume_source_path = args.resume_arclength_dir / "source_commit.txt"
        resume_attempt_dir = args.resume_arclength_dir / "ARCLENGTH"
        for path in (resume_record_path, resume_source_path, resume_attempt_dir):
            require(path.exists(), f"required continuation-resume artifact missing: {path.name}")
        resume_record = json.loads(resume_record_path.read_text(encoding="utf-8"))
        resume_source = resume_source_path.read_text(encoding="utf-8").strip()
        require(
            resume_record.get("status") == "R26_N30_BALANCED_ARCLENGTH_RESCUE_FAILED",
            "wrong continuation-resume status",
        )
        require(
            resume_record.get("source_commit")
            == args.expected_resume_arclength_source_commit,
            "continuation-resume record source mismatch",
        )
        require(
            resume_source == args.expected_resume_arclength_source_commit,
            "continuation-resume source_commit.txt mismatch",
        )
        require(
            resume_record.get("failed_production_commit")
            == args.expected_failed_source_commit,
            "continuation-resume production provenance mismatch",
        )
        require(
            resume_record.get("failed_arclength_commit")
            == args.expected_failed_arclength_source_commit,
            "continuation-resume arclength provenance mismatch",
        )
        require(
            resume_record.get("n30_target_accepted") is False,
            "continuation-resume failure record changed target acceptance",
        )
        attempt_paths = sorted(resume_attempt_dir.glob("arc_attempt_*.json"))
        require(bool(attempt_paths), "continuation-resume attempts are missing")
        resume_attempts = [
            json.loads(path.read_text(encoding="utf-8")) for path in attempt_paths
        ]
        resume_attempts.sort(key=lambda row: int(row["attempt"]))
        require(
            [int(row["attempt"]) for row in resume_attempts]
            == list(range(1, len(resume_attempts) + 1)),
            "continuation-resume attempt sequence is incomplete",
        )
        resume_accepted = [row for row in resume_attempts if bool(row.get("accepted"))]
        require(
            len(resume_accepted) >= 2,
            "continuation-resume has fewer than two accepted roots",
        )
        previous_state, previous_parameter, previous_provenance = (
            load_and_validate_arclength_seed(
                resume_attempt_dir,
                resume_accepted[-2],
                template,
                args.raw_tolerance,
            )
        )
        current_state, current_parameter, current_provenance = (
            load_and_validate_arclength_seed(
                resume_attempt_dir,
                resume_accepted[-1],
                template,
                args.raw_tolerance,
            )
        )
        continuation_resume_provenance = {
            "directory": str(args.resume_arclength_dir.resolve()),
            "status": resume_record["status"],
            "source_commit": resume_source,
            "failure_record_sha256": sha256_file(resume_record_path),
            "attempt_record_sha256": {
                path.name: sha256_file(path) for path in attempt_paths
            },
            "previous_accepted_seed": previous_provenance,
            "current_accepted_seed": current_provenance,
        }

    require(
        abs(current_parameter - previous_parameter) > np.finfo(float).eps,
        "last accepted seed pair has zero parameter increment",
    )
    require(current_parameter < args.target_lid, "failed run already reached the target")

    transform = LogStateTransform((30, 30, 17))
    calibration_previous_encoded = transform.encode(calibration_previous_state)
    calibration_current_encoded = transform.encode(calibration_current_state)
    previous_encoded = transform.encode(previous_state)
    current_encoded = transform.encode(current_state)
    selected_parameter_scale = (
        balanced_parameter_scale(
            calibration_previous_encoded,
            calibration_previous_parameter,
            calibration_current_encoded,
            calibration_current_parameter,
            parameter_fraction=args.parameter_metric_fraction,
        )
        if args.parameter_scale is None
        else float(args.parameter_scale)
    )
    metric = ArcLengthMetric(
        30 * 30 * 17,
        parameter_scale=selected_parameter_scale,
    )
    calibration_metric_diagnostics = secant_metric_diagnostics(
        calibration_previous_encoded,
        calibration_previous_parameter,
        calibration_current_encoded,
        calibration_current_parameter,
        metric,
    )
    initial_metric_diagnostics = secant_metric_diagnostics(
        previous_encoded,
        previous_parameter,
        current_encoded,
        current_parameter,
        metric,
    )
    require(
        0.1 <= calibration_metric_diagnostics.parameter_fraction <= 0.9,
        "calibrated pseudo-arclength metric degenerates toward fixed-state or "
        "fixed-parameter continuation",
    )
    reference_tangent = normalized_secant_tangent(
        previous_encoded,
        previous_parameter,
        current_encoded,
        current_parameter,
        metric,
    )
    initial_secant_length = reference_tangent.secant_length
    # Absolute step lengths from an older metric cannot be reused after
    # rebalancing. Rebuild the bounded schedule from this accepted secant.
    step_length = args.initial_step_factor * initial_secant_length
    minimum_step = args.minimum_step_factor * initial_secant_length
    maximum_step = args.maximum_step_factor * initial_secant_length
    declared_initial_step = step_length
    controls = ArcLengthCorrectorOptions(
        residual_tolerance=args.residual_tolerance,
        raw_tolerance=args.raw_tolerance,
        parameter_scale=selected_parameter_scale,
        maximum_iterations=args.maximum_iterations,
        maximum_jacobians=args.maximum_jacobians,
        maximum_objective_evaluations=args.maximum_objective_evaluations,
        pseudo_transient_chord_limit=args.pseudo_transient_chord_limit,
        newton_chord_limit=args.newton_chord_limit,
    )
    manifest = source_manifest()
    records: list[dict[str, object]] = []
    termination = "maximum_attempts_reached"
    landing_record: dict[str, object] | None = None

    for attempt_number in range(1, args.maximum_attempts + 1):
        started = time.time()
        try:
            result = solve_r26_pseudo_arclength_step(
                template,
                previous_state,
                previous_parameter,
                current_state,
                current_parameter,
                step_length,
                options=controls,
                reference_tangent=reference_tangent,
            )
        except Exception as error:
            fatal_metric = secant_metric_diagnostics(
                transform.encode(previous_state),
                previous_parameter,
                transform.encode(current_state),
                current_parameter,
                metric,
            )
            write_json(
                args.output_dir / "fatal_continuation_error.json",
                {
                    "attempt": attempt_number,
                    "from_parameter": current_parameter,
                    "step_length": step_length,
                    "parameter_scale": selected_parameter_scale,
                    "state_metric_fraction": fatal_metric.state_fraction,
                    "parameter_metric_fraction": fatal_metric.parameter_fraction,
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "accepted_attempts_before_error": sum(
                        bool(row["accepted"]) for row in records
                    ),
                },
            )
            raise
        record: dict[str, object] = {
            "attempt": attempt_number,
            "from_parameter": current_parameter,
            "predicted_parameter": result.predicted_parameter,
            "corrected_parameter": result.parameter,
            "step_length": step_length,
            "parameter_scale": selected_parameter_scale,
            "accepted": result.accepted,
            "elapsed_seconds": time.time() - started,
            "raw_acceptance_gate": result.raw_acceptance_gate,
            "scaled_residual_linf": result.scaled_residual_linf,
            "arclength_residual": result.arclength_residual,
            "solver": {
                "method": "bordered_pseudo_arclength_chord_ser_ptc",
                "message": result.message,
                "iterations": result.iterations,
                "jacobian_evaluations": result.jacobian_evaluations,
                "objective_evaluations": result.objective_evaluations,
                "invalid_evaluations": result.invalid_evaluations,
                "last_invalid_error": result.last_invalid_error,
                "linear_solver": result.linear_solver,
                "pseudo_transient_steps": result.pseudo_transient_steps,
                "final_pseudo_time_step": result.final_pseudo_time_step,
                "iteration_trace": result.iteration_trace,
            },
            "tangent_parameter_component": result.tangent.parameter,
            "state_metric_fraction": result.state_metric_fraction,
            "parameter_metric_fraction": result.parameter_metric_fraction,
            "diagnostics": asdict(result.diagnostics),
            "global_balances": result.global_balances,
            "state_sha256": state_sha256(result.state),
            "source_manifest": manifest,
        }
        stem = f"arc_attempt_{attempt_number:03d}_lid_{result.parameter:.12g}"
        np.savez_compressed(
            args.output_dir / f"{stem}.npz",
            state=result.state,
            x=template.x,
            y=template.y,
            lid_velocity=result.parameter,
            kn_input=template.kn,
            kn_convention=template.kn_convention.value,
            beta=template.grid_stretch_beta,
            accepted=result.accepted,
        )
        write_json(args.output_dir / f"{stem}.json", record)
        records.append(record)

        if not result.accepted:
            if step_length <= minimum_step * (1.0 + 8.0 * np.finfo(float).eps):
                termination = "minimum_arclength_step_rejected"
                break
            step_length = max(minimum_step, 0.5 * step_length)
            continue

        old_current_state = current_state
        old_current_parameter = current_parameter
        previous_state = current_state
        previous_parameter = current_parameter
        current_state = result.state
        current_parameter = result.parameter
        reference_tangent = normalized_secant_tangent(
            transform.encode(previous_state),
            previous_parameter,
            transform.encode(current_state),
            current_parameter,
            metric,
            reference=result.tangent,
        )

        crossed_target = (
            (old_current_parameter - args.target_lid)
            * (current_parameter - args.target_lid)
            <= 0.0
            and old_current_parameter != current_parameter
        )
        if crossed_target:
            target_problem = make_problem(
                template.with_lid_velocity(args.target_lid, suffix="arc-landing")
            )
            landing_state = interpolate_bracketed_state(
                target_problem,
                old_current_state,
                old_current_parameter,
                current_state,
                current_parameter,
                args.target_lid,
            )
            np.savez_compressed(
                args.output_dir / "landing_seed.npz",
                state=landing_state,
                x=template.x,
                y=template.y,
                lid_velocity=args.target_lid,
                kn_input=template.kn,
                kn_convention=template.kn_convention.value,
                beta=template.grid_stretch_beta,
                predictor_kind="pseudo_arclength_bracket_interpolation",
                lower_bracket_parameter=old_current_parameter,
                upper_bracket_parameter=current_parameter,
            )
            landing_record = {
                "target_parameter": args.target_lid,
                "bracket_parameters": [
                    old_current_parameter,
                    current_parameter,
                ],
                "landing_state_sha256": state_sha256(landing_state),
                "landing_file_sha256": sha256_file(
                    args.output_dir / "landing_seed.npz"
                ),
            }
            termination = "target_bracketed"
            break

        if result.jacobian_evaluations <= 3:
            step_length = min(maximum_step, 1.25 * step_length)
        elif result.jacobian_evaluations >= controls.maximum_jacobians - 1:
            step_length = max(minimum_step, 0.75 * step_length)

    summary: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "case": {
            "family": "jfm-maxwell",
            "nodes": 30,
            "kn_input": 0.20,
            "kn_convention": "gu_lambda_over_L",
            "beta": 0.0,
            "target_lid": args.target_lid,
        },
        "failed_run_provenance": {
            "directory": str(args.failed_run_dir.resolve()),
            "source_commit": source_commit,
            "failed_summary_sha256": sha256_file(summary_path),
            "previous_seed": calibration_previous_provenance,
            "current_seed": calibration_current_provenance,
        },
        "failed_arclength_provenance": resume_provenance,
        "continuation_resume_provenance": continuation_resume_provenance,
        "arclength_controls": {
            "parameter_scale": selected_parameter_scale,
            "parameter_scale_mode": (
                "secant_calibrated_fixed" if args.parameter_scale is None else "explicit"
            ),
            "metric_policy": "fixed_after_non_degenerate_calibration",
            "requested_parameter_metric_fraction": args.parameter_metric_fraction,
            "calibration_state_metric_fraction": calibration_metric_diagnostics.state_fraction,
            "calibration_parameter_metric_fraction": calibration_metric_diagnostics.parameter_fraction,
            "calibration_state_rms": calibration_metric_diagnostics.state_rms,
            "calibration_parameter_increment": calibration_metric_diagnostics.parameter_increment,
            "initial_state_metric_fraction": initial_metric_diagnostics.state_fraction,
            "initial_parameter_metric_fraction": initial_metric_diagnostics.parameter_fraction,
            "initial_state_rms": initial_metric_diagnostics.state_rms,
            "initial_parameter_increment": initial_metric_diagnostics.parameter_increment,
            "initial_secant_length": initial_secant_length,
            "initial_step": declared_initial_step,
            "minimum_step": minimum_step,
            "maximum_step": maximum_step,
            "maximum_attempts": args.maximum_attempts,
            "maximum_iterations_per_attempt": args.maximum_iterations,
            "maximum_jacobians_per_attempt": args.maximum_jacobians,
            "maximum_objective_evaluations_per_attempt": args.maximum_objective_evaluations,
            "pseudo_transient_chord_limit": args.pseudo_transient_chord_limit,
            "newton_chord_limit": args.newton_chord_limit,
            "pseudo_time_initial": controls.pseudo_time_initial,
            "pseudo_time_minimum": controls.pseudo_time_minimum,
            "pseudo_time_maximum": controls.pseudo_time_maximum,
            "newton_switch_tolerance": controls.newton_switch_tolerance,
            "raw_tolerance": args.raw_tolerance,
            "residual_tolerance": args.residual_tolerance,
        },
        "attempts": records,
        "landing": landing_record,
        "termination": termination,
        "source_manifest": manifest,
        "claim_boundary": (
            "Pseudo-arclength supplies a target predictor only; final fixed-parameter "
            "R26 correction and independent raw validation remain mandatory."
        ),
    }
    write_json(args.output_dir / "arclength_summary.json", summary)
    print(
        json.dumps(
            {
                "termination": termination,
                "accepted_arclength_points": sum(
                    bool(row["accepted"]) for row in records
                ),
                "last_parameter": current_parameter,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if termination == "target_bracketed" else 1)


if __name__ == "__main__":
    main()
