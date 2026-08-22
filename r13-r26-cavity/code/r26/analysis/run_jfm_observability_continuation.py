#!/usr/bin/env python3
"""Fail-closed R26 continuation for the JFM anti-Fourier cavity.

The driver starts from analytic equilibrium unless an explicit accepted local
restart is supplied, records every rejected attempt, and never promotes a
small optimizer norm to model validation.  The source-locked ``jfm-maxwell``
target uses the Gu equilibrium mean-free-path convention, Maxwell
``mu/mu0=T/T0``, Tw=300 K, a 100 m/s lid and fully diffuse walls.  The
historical VHS-transport hybrid, legacy Rana/John and Gu-ASME modes are
retained only for regression comparisons and must be selected explicitly.
"""

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
from scipy.sparse.csgraph import structural_rank

from r26_cases import (
    SQRT_2_OVER_PI,
    gu_asme2009_cavity_case,
    jfm_maxwell_cavity_case,
    jfm_observability_cavity_case,
    rana_john_case,
)
from r26_discretization import R26NodeBVP
from r26_fv_backend import compatible_fv_bulk_residual, wall_bounded_control_volume_weights
from r26_postprocess import rana_global_metrics
from r26_solver import (
    SolveOptions,
    interpolate_state_grid,
    jacobian_sparsity,
    secant_predict_state,
    solve_r26_bvp,
)
from r26_state import STATE_INDEX, planar_state_to_tensors
from r26_tensor_closures import closures_from_tensors, finite_difference_gradients
from r26_validation import global_balance_diagnostics, leading_r13_nsf_diagnostics
from r26_wall_conditions import (
    WallParameters,
    effective_pressure,
    extract_face_quantities,
    project_closures,
    square_wall_frame,
)


ROOT = Path(__file__).resolve().parents[1]
# The archived campaign used source/r26/code/*.py, whereas the public GitHub
# package keeps the modules directly in code/r26/*.py.  Resolve both layouts
# but emit one canonical manifest schema so validation is path-independent.
CODE = ROOT / "code" if (ROOT / "code" / "r26_cases.py").is_file() else ROOT
SOURCE_FILES = tuple(sorted(CODE.glob("r26_*.py"))) + (Path(__file__).resolve(),)
RANA_JOHN_LID = 50.0 / np.sqrt(208.0 * 273.0)
GU_ASME_DEFAULT_LID = 10.0 / np.sqrt(208.0 * 273.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_source_manifest() -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in SOURCE_FILES:
        if path.parent == CODE:
            key = f"code/{path.name}"
        elif path == Path(__file__).resolve():
            key = f"analysis/{path.name}"
        else:  # pragma: no cover - defensive future layout
            key = str(path.relative_to(ROOT))
        manifest[key] = sha256_file(path)
    return manifest


def state_sha256(state: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(state, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(b"|<f8|")
    digest.update(value.tobytes())
    return digest.hexdigest()


def jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def termination_exit_code(termination: str) -> int:
    """Return success only when the requested target root was accepted."""

    return 0 if termination == "target_accepted" else 1


def continuation_step_after_rejection(
    attempted_increment: float,
    minimum_step: float,
) -> float | None:
    """Halve a rejected increment but guarantee one exact floor attempt.

    Returning ``None`` means that the rejected proposal was already at (or
    below, because the target was closer than) the declared floor.  The old
    driver terminated when a halved step crossed the floor and therefore
    never tried the floor itself.
    """

    if not np.isfinite(attempted_increment) or attempted_increment <= 0.0:
        raise ValueError("attempted increment must be finite and positive")
    if not np.isfinite(minimum_step) or minimum_step <= 0.0:
        raise ValueError("minimum step must be finite and positive")
    tolerance = 16.0 * np.finfo(float).eps * max(1.0, minimum_step)
    if attempted_increment <= minimum_step + tolerance:
        return None
    return max(minimum_step, 0.5 * attempted_increment)


def make_problem(case: object) -> R26NodeBVP:
    return R26NodeBVP(
        case,
        bulk_operator=compatible_fv_bulk_residual,
        mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
    )


def centerline(array: np.ndarray, coordinate: np.ndarray, location: float, axis: int) -> np.ndarray:
    if axis == 1:
        return np.asarray([np.interp(location, coordinate, row) for row in array])
    return np.asarray([np.interp(location, coordinate, array[:, i]) for i in range(array.shape[1])])


def primary_vortex(state: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    rho = state[..., STATE_INDEX["rho"]]
    ux = state[..., STATE_INDEX["vx"]]
    integrand = rho * ux
    psi = np.zeros_like(integrand)
    increments = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(y)[:, None]
    psi[1:] = np.cumsum(increments, axis=0)
    interior = psi[1:-1, 1:-1]
    flat = int(np.argmax(np.abs(interior)))
    j, i = np.unravel_index(flat, interior.shape)
    j += 1
    i += 1
    return {
        "x_over_L_node": float(x[i]),
        "y_over_L_node": float(y[j]),
        "streamfunction": float(psi[j, i]),
        "definition": "largest-|psi| interior node; psi integrated from bottom using rho*ux",
    }


def wall_observables(state: np.ndarray, case: object) -> dict[str, object]:
    tensors = planar_state_to_tensors(state)
    gradients = finite_difference_gradients(state, x=case.x, y=case.y)
    closures = closures_from_tensors(
        tensors,
        gradients,
        mu=case.mu(tensors.theta),
        coefficient_mode=case.r26_closure_mode,
    )
    rows: dict[str, object] = {}
    entries = {
        "left": [(j, 0) for j in range(1, case.nodes - 1)],
        "right": [(j, case.nodes - 1) for j in range(1, case.nodes - 1)],
        "bottom": [(0, i) for i in range(1, case.nodes - 1)],
        "top": [(case.nodes - 1, i) for i in range(1, case.nodes - 1)],
    }
    for side, locations in entries.items():
        frame = square_wall_frame(side)
        params = WallParameters(
            wall_temperature=case.wall_temperature,
            accommodation=case.accommodation,
            gas_constant=case.gas_constant,
            wall_velocity=case.wall_velocity(side),
        )
        side_rows: list[dict[str, float]] = []
        for j, i in locations:
            point_tensors = planar_state_to_tensors(state[j, i])
            point_closure = type(closures)(
                phi=closures.phi[j, i],
                psi=closures.psi[j, i],
                Omega=closures.Omega[j, i],
                equation25_mode=closures.equation25_mode,
                provenance=closures.provenance,
                coefficient_mode=closures.coefficient_mode,
            )
            free, unknowns = extract_face_quantities(point_tensors, frame)
            palpha = effective_pressure(free, unknowns, project_closures(point_closure, frame), params)
            side_rows.append(
                {
                    "x": float(case.x[i]),
                    "y": float(case.y[j]),
                    "T": float(state[j, i, STATE_INDEX["theta"]]),
                    "p": float(state[j, i, STATE_INDEX["rho"]] * state[j, i, STATE_INDEX["theta"]]),
                    "p_alpha": palpha,
                    "qx": float(state[j, i, STATE_INDEX["qx"]]),
                    "qy": float(state[j, i, STATE_INDEX["qy"]]),
                    "sigma_xx": float(state[j, i, STATE_INDEX["sigma_xx"]]),
                    "sigma_xy": float(state[j, i, STATE_INDEX["sigma_xy"]]),
                    "sigma_yy": float(state[j, i, STATE_INDEX["sigma_yy"]]),
                }
            )
        rows[side] = side_rows
    return rows


def all_observables(state: np.ndarray, case: object) -> dict[str, object]:
    x, y = case.x, case.y
    pressure = state[..., STATE_INDEX["rho"]] * state[..., STATE_INDEX["theta"]]
    vertical: dict[str, object] = {"coordinate_y": y}
    horizontal: dict[str, object] = {"coordinate_x": x}
    for name in (
        "rho", "vx", "vy", "theta", "qx", "qy", "sigma_xx", "sigma_xy",
        "sigma_yy", "R_xx", "R_xy", "R_yy", "m_xxx", "m_xxy", "m_xyy",
        "m_yyy", "Delta",
    ):
        field = state[..., STATE_INDEX[name]]
        vertical[name] = centerline(field, x, 0.5, axis=1)
        horizontal[name] = centerline(field, y, 0.5, axis=0)
    vertical["pressure"] = centerline(pressure, x, 0.5, axis=1)
    horizontal["pressure"] = centerline(pressure, y, 0.5, axis=0)
    gradients = finite_difference_gradients(state, x=x, y=y)
    q = state[..., [STATE_INDEX["qx"], STATE_INDEX["qy"]]]
    grad_t = np.asarray(gradients.theta)[..., :2]
    dot = np.einsum("...i,...i->...", q, grad_t)
    qmag = np.linalg.norm(q, axis=-1)
    interior = (slice(1, -1), slice(1, -1))
    q_index = np.unravel_index(int(np.argmin(qmag[interior])), qmag[interior].shape)
    qj, qi = q_index[0] + 1, q_index[1] + 1
    return {
        "vertical_centerline_x_0p5": vertical,
        "horizontal_centerline_y_0p5": horizontal,
        "walls_smooth_nodes": wall_observables(state, case),
        "rana_D_G": rana_global_metrics(state, lid_velocity=case.lid_velocity, x=x, y=y),
        "vortex": primary_vortex(state, x, y),
        "heat_flux_topology": {
            "counter_gradient_fraction_interior": float(np.mean(dot[interior] > 0.0)),
            "q_minimum_node_x": float(x[qi]),
            "q_minimum_node_y": float(y[qj]),
            "q_minimum": float(qmag[qj, qi]),
            "definition": "q dot grad(T)>0 is counter-gradient; node minimum is not a subgrid stagnation proof",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument(
        "--case-family",
        choices=("jfm-maxwell", "jfm-observability", "rana-john2010", "gu-asme2009"),
        default="jfm-maxwell",
        help=(
            "pure-Maxwell matched target, historical VHS-transport hybrid, "
            "Rana/John comparison, or direct source-locked Gu-ASME case"
        ),
    )
    parser.add_argument("--kn-rana", type=float)
    parser.add_argument("--kn-gu", type=float)
    parser.add_argument("--lid-speed-m-s", type=float)
    parser.add_argument("--wall-temperature-k", type=float)
    parser.add_argument(
        "--vhs-omega",
        type=float,
        default=None,
        help="historical jfm-observability mode only; forbidden for jfm-maxwell",
    )
    parser.add_argument("--beta", type=float, default=2.5)
    parser.add_argument(
        "--closure-mode",
        choices=("jfm2009", "asme2009-cavity"),
        help=(
            "complete R26 closure coefficient set; Rana/John defaults to the "
            "final JFM-2009 model, while direct Gu-ASME is locked to its "
            "printed preliminary cavity set"
        ),
    )
    parser.add_argument(
        "--target-lid",
        type=float,
        help="optional nondimensional override; physical case speed is the default",
    )
    parser.add_argument("--smoke-lid", type=float, default=0.001)
    parser.add_argument("--initial-step", type=float, default=0.04)
    parser.add_argument("--minimum-step", type=float, default=0.0025)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument(
        "--max-objective-evaluations",
        type=int,
        help=(
            "hard fail-closed budget for residual evaluations in colored Newton; "
            "use this during refined-grid reconciliation to prevent stalled jobs"
        ),
    )
    parser.add_argument(
        "--solver",
        choices=("colored_newton", "least_squares", "krylov"),
        default="colored_newton",
    )
    parser.add_argument(
        "--analytic-mass-jacobian",
        action="store_true",
        help="differentiate the dense global-mass border exactly instead of coloring it",
    )
    parser.add_argument(
        "--secant-predictor",
        action="store_true",
        help="predict each proposal from the two most recent accepted fixed-grid roots",
    )
    parser.add_argument(
        "--ser-ptc",
        action="store_true",
        help="globalize colored Newton with a bulk-only SER pseudo-transient shift",
    )
    parser.add_argument("--pseudo-time-initial", type=float, default=1.0e-2)
    parser.add_argument("--pseudo-time-minimum", type=float, default=1.0e-8)
    parser.add_argument("--pseudo-time-maximum", type=float, default=1.0e8)
    parser.add_argument("--newton-switch-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--max-jacobians",
        type=int,
        help="fail-closed maximum colored Jacobian builds in one continuation attempt",
    )
    parser.add_argument("--raw-tolerance", type=float, default=1.0e-8)
    parser.add_argument(
        "--solver-tolerance",
        type=float,
        help="scaled nonlinear tolerance; defaults to one tenth of the raw acceptance gate",
    )
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument(
        "--reconcile-initial",
        action="store_true",
        help="force one nonlinear solve at the restart lid (required after grid refinement)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.solver != "colored_newton"
        and (args.analytic_mass_jacobian or args.ser_ptc)
    ):
        parser.error("analytic mass Jacobian and SER-PTC require --solver colored_newton")
    if args.case_family == "jfm-maxwell":
        if args.kn_gu is None or args.kn_rana is not None:
            parser.error("jfm-maxwell requires --kn-gu and forbids --kn-rana")
        if args.closure_mode not in (None, "jfm2009"):
            parser.error("jfm-maxwell requires final --closure-mode jfm2009")
        if args.vhs_omega is not None:
            parser.error("jfm-maxwell fixes mu/mu0=T/T0; do not pass --vhs-omega")
        physical_lid = 100.0 if args.lid_speed_m_s is None else args.lid_speed_m_s
        physical_wall_temperature = (
            300.0 if args.wall_temperature_k is None else args.wall_temperature_k
        )
        base = jfm_maxwell_cavity_case(
            args.nodes,
            kn=args.kn_gu,
            lid_speed_m_per_s=physical_lid,
            wall_temperature_K=physical_wall_temperature,
            grid_stretch_beta=args.beta,
        )
        equivalent_gu_kn = float(args.kn_gu)
    elif args.case_family == "jfm-observability":
        if args.kn_gu is None or args.kn_rana is not None:
            parser.error("jfm-observability requires --kn-gu and forbids --kn-rana")
        if args.closure_mode not in (None, "jfm2009"):
            parser.error("jfm-observability requires final --closure-mode jfm2009")
        physical_lid = 100.0 if args.lid_speed_m_s is None else args.lid_speed_m_s
        physical_wall_temperature = (
            300.0 if args.wall_temperature_k is None else args.wall_temperature_k
        )
        base = jfm_observability_cavity_case(
            args.nodes,
            kn=args.kn_gu,
            lid_speed_m_per_s=physical_lid,
            wall_temperature_K=physical_wall_temperature,
            viscosity_exponent=(0.81 if args.vhs_omega is None else args.vhs_omega),
            grid_stretch_beta=args.beta,
        )
        equivalent_gu_kn = float(args.kn_gu)
    elif args.case_family == "rana-john2010":
        if args.kn_rana is None or args.kn_gu is not None:
            parser.error("rana-john2010 requires --kn-rana and forbids --kn-gu")
        if args.wall_temperature_k is not None:
            parser.error("rana-john2010 has source-locked Tw=273 K")
        physical_lid = 50.0 if args.lid_speed_m_s is None else args.lid_speed_m_s
        physical_wall_temperature = 273.0
        base = rana_john_case(
            args.nodes,
            kn=args.kn_rana,
            lid_speed_m_per_s=physical_lid,
            grid_stretch_beta=args.beta,
            closure_mode=("jfm2009" if args.closure_mode is None else args.closure_mode),
        )
        equivalent_gu_kn = float(args.kn_rana / SQRT_2_OVER_PI)
    else:
        if args.kn_gu is None or args.kn_rana is not None:
            parser.error("gu-asme2009 requires --kn-gu and forbids --kn-rana")
        if args.closure_mode not in (None, "asme2009-cavity"):
            parser.error("gu-asme2009 is source-locked to --closure-mode asme2009-cavity")
        physical_lid = 10.0 if args.lid_speed_m_s is None else args.lid_speed_m_s
        physical_wall_temperature = (
            273.0 if args.wall_temperature_k is None else args.wall_temperature_k
        )
        base = gu_asme2009_cavity_case(
            args.nodes,
            kn=args.kn_gu,
            lid_speed_m_per_s=physical_lid,
            wall_temperature_K=physical_wall_temperature,
            grid_stretch_beta=args.beta,
        )
        equivalent_gu_kn = float(args.kn_gu)
    target_lid = float(base.lid_velocity if args.target_lid is None else args.target_lid)
    if not np.isfinite(target_lid) or target_lid < 0.0:
        parser.error("target lid must be finite and nonnegative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be absent or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.initial_state is None:
        state = base.equilibrium_state()
        provenance: dict[str, object] = {"kind": "analytic_equilibrium", "sha256": state_sha256(state)}
        current_lid = 0.0
        needs_reconciliation = False
        state_accepted_on_target_grid = True
        unreconciled_initial_state: np.ndarray | None = None
    else:
        with np.load(args.initial_state, allow_pickle=False) as archive:
            if "accepted" in archive and not bool(np.asarray(archive["accepted"]).item()):
                parser.error("initial-state archive is explicitly marked rejected")
            old = np.asarray(archive["state"], dtype=float)
            old_x = np.asarray(archive["x"], dtype=float)
            old_y = np.asarray(archive["y"], dtype=float)
            current_lid = float(archive["lid_velocity"])
        grid_changed = bool(
            old.shape[0] != base.nodes
            or old_x.shape != base.x.shape
            or old_y.shape != base.y.shape
            or (old_x.shape == base.x.shape and not np.array_equal(old_x, base.x))
            or (old_y.shape == base.y.shape and not np.array_equal(old_y, base.y))
        )
        state = interpolate_state_grid(
            old,
            base.nodes,
            old_x=old_x,
            old_y=old_y,
            new_x=base.x,
            new_y=base.y,
            mass_weights=wall_bounded_control_volume_weights(base.x, base.y),
        )
        provenance = {
            "kind": "explicit_local_restart_interpolated",
            "path": str(args.initial_state.resolve()),
            "file_sha256": sha256_file(args.initial_state),
            "state_sha256_after_interpolation": state_sha256(state),
            "grid_changed": grid_changed,
        }
        needs_reconciliation = bool(grid_changed or args.reconcile_initial)
        state_accepted_on_target_grid = not needs_reconciliation
        unreconciled_initial_state = state.copy() if needs_reconciliation else None
    if current_lid > target_lid + 1.0e-15:
        parser.error("restart lid exceeds the requested target lid")
    if (
        args.initial_state is not None
        and not needs_reconciliation
        and current_lid >= target_lid - 1.0e-15
    ):
        parser.error(
            "a restart already at the target lid must be revalidated with "
            "--reconcile-initial"
        )

    manifest = build_source_manifest()
    solver_tolerance = (
        0.1 * args.raw_tolerance
        if args.solver_tolerance is None
        else float(args.solver_tolerance)
    )
    if not np.isfinite(solver_tolerance) or solver_tolerance <= 0.0:
        parser.error("solver tolerance must be finite and positive")
    options = SolveOptions(
        method=args.solver,
        residual_tolerance=solver_tolerance,
        held_out_continuity_tolerance=args.raw_tolerance,
        max_iterations=args.max_nfev,
        max_function_evaluations=args.max_nfev,
        max_objective_evaluations=args.max_objective_evaluations,
        analytic_mass_jacobian=args.analytic_mass_jacobian,
        pseudo_transient=args.ser_ptc,
        pseudo_time_initial=args.pseudo_time_initial,
        pseudo_time_minimum=args.pseudo_time_minimum,
        pseudo_time_maximum=args.pseudo_time_maximum,
        newton_switch_tolerance=args.newton_switch_tolerance,
        max_jacobian_evaluations=args.max_jacobians,
    )
    attempts: list[dict[str, object]] = []
    previous_accepted_state: np.ndarray | None = None
    previous_accepted_lid: float | None = None
    step = float(args.initial_step)
    attempt = 0
    termination = (
        "target_accepted"
        if args.initial_state is None
        and state_accepted_on_target_grid
        and current_lid >= target_lid - 1.0e-15
        else "not_started"
    )
    while needs_reconciliation or current_lid < target_lid - 1.0e-15:
        attempt += 1
        reconciling_this_attempt = needs_reconciliation
        if reconciling_this_attempt:
            proposed = current_lid
        elif attempt == 1 and current_lid == 0.0:
            proposed = min(args.smoke_lid, target_lid)
        else:
            proposed = min(current_lid + step, target_lid)
        case = base.with_lid_velocity(proposed, suffix=f"attempt{attempt:03d}")
        problem = make_problem(case)
        solve_seed = state
        predictor: dict[str, object] = {"kind": "last_accepted_state"}
        if (
            args.secant_predictor
            and not reconciling_this_attempt
            and previous_accepted_state is not None
            and previous_accepted_lid is not None
            and proposed > current_lid
        ):
            factor = (proposed - current_lid) / (current_lid - previous_accepted_lid)
            if factor <= 2.0:
                solve_seed = secant_predict_state(
                    problem,
                    previous_accepted_state,
                    state,
                    previous_parameter=previous_accepted_lid,
                    current_parameter=current_lid,
                    target_parameter=proposed,
                )
                predictor = {
                    "kind": "encoded_secant_mass_preserving",
                    "previous_lid": previous_accepted_lid,
                    "current_lid": current_lid,
                    "target_lid": proposed,
                    "extrapolation_factor": factor,
                    "seed_state_sha256": state_sha256(solve_seed),
                    "seed_mass_error": problem.mass_constraint(solve_seed),
                }
            else:
                predictor = {
                    "kind": "last_accepted_state",
                    "secant_skipped": "extrapolation_factor_above_two",
                    "extrapolation_factor": factor,
                }
        started = time.time()
        result = solve_r26_bvp(problem, solve_seed, options=options)
        balances = global_balance_diagnostics(result.state, case)
        raw_gate = max(
            result.diagnostics.raw_total_linf,
            abs(result.diagnostics.held_out_continuity),
            abs(result.diagnostics.mass_error),
        )
        accepted = bool(
            result.converged
            and result.scipy_success
            and raw_gate <= args.raw_tolerance
            and result.diagnostics.min_density > 0.0
            and result.diagnostics.min_temperature > 0.0
            and balances["wall_effective_pressure_min"] > 0.0
            and balances["momentum_boundary_flux_linf"] <= 10.0 * args.raw_tolerance
            and abs(balances["internal_energy_balance_error"]) <= 10.0 * args.raw_tolerance
        )
        report: dict[str, object] = {
            "attempt": attempt,
            "from_lid": current_lid,
            "proposed_lid": proposed,
            "accepted": accepted,
            "elapsed_seconds": time.time() - started,
            "solver": {
                "method": result.solver_method,
                "scipy_success": result.scipy_success,
                "message": result.message,
                "function_evaluations": result.function_evaluations,
                "jacobian_evaluations": result.jacobian_evaluations,
                "pseudo_transient_steps": result.pseudo_transient_steps,
                "final_pseudo_time_step": result.final_pseudo_time_step,
                "invalid_evaluations": result.invalid_evaluations,
                "last_invalid_error": result.last_invalid_error,
            },
            "predictor": predictor,
            "diagnostics": asdict(result.diagnostics),
            "raw_acceptance_gate": raw_gate,
            "global_balances": balances,
            "leading_r13_nsf": leading_r13_nsf_diagnostics(result.state, case),
            "state_sha256": state_sha256(result.state),
            "structural_jacobian_rank": int(structural_rank(jacobian_sparsity(problem))),
            "unknown_count": problem.unknown_count,
            "rank_caveat": "structural rank is not numerical Jacobian rank",
            "source_manifest": manifest,
        }
        stem = f"attempt_{attempt:03d}_lid_{proposed:.12g}"
        np.savez_compressed(
            args.output_dir / f"{stem}.npz",
            state=result.state,
            x=case.x,
            y=case.y,
            lid_velocity=proposed,
            kn_input=case.kn,
            kn_convention=case.kn_convention.value,
            mu_equilibrium=case.mu_equilibrium,
            beta=case.grid_stretch_beta,
            accepted=accepted,
        )
        write_json(args.output_dir / f"{stem}.json", report)
        attempts.append(report)
        if accepted:
            old_accepted_state = state.copy()
            old_accepted_lid = current_lid
            state = result.state.copy()
            current_lid = proposed
            state_accepted_on_target_grid = True
            if reconciling_this_attempt:
                needs_reconciliation = False
            else:
                previous_accepted_state = old_accepted_state
                previous_accepted_lid = old_accepted_lid
                if attempt > 1:
                    step = min(args.initial_step, 1.4 * step)
            if not needs_reconciliation and current_lid >= target_lid - 1.0e-15:
                termination = "target_accepted"
                break
        else:
            if reconciling_this_attempt:
                termination = "grid_reconciliation_rejected"
                break
            if current_lid == 0.0 and proposed == min(args.smoke_lid, target_lid):
                termination = "smoke_rejected"
                break
            next_step = continuation_step_after_rejection(
                proposed - current_lid,
                args.minimum_step,
            )
            if next_step is None:
                termination = "minimum_step_rejected"
                break
            step = next_step

    final_case = base.with_lid_velocity(current_lid, suffix="last-accepted")
    final_payload: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "case": {
            "family": args.case_family,
            "nodes": base.nodes,
            "kn_input": base.kn,
            "kn_convention": base.kn_convention.value,
            "equivalent_gu_lambda_over_L_at_equilibrium": equivalent_gu_kn,
            "mu_equilibrium": base.mu_equilibrium,
            "lid_speed_m_per_s": physical_lid,
            "wall_temperature_K": physical_wall_temperature,
            "viscosity_exponent": base.viscosity.exponent,
            "molecular_model": (
                "maxwell_molecules"
                if args.case_family == "jfm-maxwell"
                else "legacy_case_specific"
            ),
            "wall_accommodation": base.accommodation,
            "lid_target": target_lid,
            "lid_last_accepted": current_lid if state_accepted_on_target_grid else None,
            "restart_lid_not_accepted_on_target_grid": (
                None if state_accepted_on_target_grid else current_lid
            ),
            "beta": base.grid_stretch_beta,
            "minimum_spacing": float(np.min(np.diff(base.x))),
            "minimum_spacing_over_Kn": float(np.min(np.diff(base.x)) / base.kn),
            "closure_mode": base.r26_closure_mode,
            "viscosity_kind": base.viscosity.kind.value,
            "provenance": base.provenance,
        },
        "input_provenance": provenance,
        "nonlinear_solver": {
            "method": args.solver,
            "analytic_mass_jacobian": args.analytic_mass_jacobian,
            "secant_predictor": args.secant_predictor,
            "ser_pseudo_transient": args.ser_ptc,
            "pseudo_time_initial": args.pseudo_time_initial,
            "pseudo_time_minimum": args.pseudo_time_minimum,
            "pseudo_time_maximum": args.pseudo_time_maximum,
            "newton_switch_tolerance": args.newton_switch_tolerance,
            "max_jacobians_per_attempt": args.max_jacobians,
            "raw_acceptance_tolerance": args.raw_tolerance,
        },
        "termination": termination,
        "attempts": attempts,
        "validation_status": (
            "algebraically accepted prediction; external validation and grid convergence still required"
            if termination == "target_accepted"
            else "no accepted target root"
        ),
        "source_manifest": manifest,
    }
    if state_accepted_on_target_grid and current_lid > 0.0:
        final_payload["global_balances"] = global_balance_diagnostics(state, final_case)
        final_payload["leading_r13_nsf"] = leading_r13_nsf_diagnostics(state, final_case)
        final_payload["observables"] = all_observables(state, final_case)
        np.savez_compressed(
            args.output_dir / "last_accepted_state.npz",
            state=state,
            x=base.x,
            y=base.y,
            lid_velocity=current_lid,
            kn_input=base.kn,
            kn_convention=base.kn_convention.value,
            mu_equilibrium=base.mu_equilibrium,
            beta=base.grid_stretch_beta,
        )
    elif unreconciled_initial_state is not None:
        np.savez_compressed(
            args.output_dir / "unreconciled_interpolated_seed.npz",
            state=unreconciled_initial_state,
            x=base.x,
            y=base.y,
            lid_velocity=current_lid,
            kn_input=base.kn,
            kn_convention=base.kn_convention.value,
            mu_equilibrium=base.mu_equilibrium,
            beta=base.grid_stretch_beta,
            accepted=False,
        )
    write_json(args.output_dir / "run_summary.json", final_payload)
    print(
        json.dumps(
            jsonable(
                {
                    "termination": termination,
                    "last_accepted_lid": (
                        current_lid if state_accepted_on_target_grid else None
                    ),
                }
            ),
            sort_keys=True,
        )
    )
    raise SystemExit(termination_exit_code(termination))


if __name__ == "__main__":
    main()
