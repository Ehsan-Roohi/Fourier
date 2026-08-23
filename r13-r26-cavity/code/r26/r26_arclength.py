#!/usr/bin/env python3
"""Pseudo-arclength continuation utilities for the R26 cavity BVP.

The fixed-lid continuation can fail when the solution branch becomes nearly
vertical in lid speed.  This module treats lid speed as an additional unknown
and replaces fixed-parameter correction by a bordered arclength equation.  No
R26 equation is removed or regularized, and every accepted point is checked
against the original raw residual and global balance gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import bmat, csc_matrix, diags, issparse
from scipy.sparse.linalg import lsmr, splu

from r26_cases import CavityCase
from r26_discretization import R26NodeBVP, ResidualDiagnostics
from r26_fv_backend import (
    compatible_fv_bulk_residual,
    fv_absolute_difference_step,
    wall_bounded_control_volume_weights,
)
from r26_solver import (
    LogStateTransform,
    analytic_mass_jacobian_row,
    jacobian_sparsity,
    pseudo_transient_diagonal,
)
from r26_state import validate_planar_state
from r26_validation import global_balance_diagnostics


@dataclass(frozen=True)
class ArcLengthMetric:
    """Mesh-independent RMS state metric plus a scaled parameter metric."""

    state_size: int
    parameter_scale: float = 0.04

    def __post_init__(self) -> None:
        if self.state_size < 1:
            raise ValueError("state size must be positive")
        if not np.isfinite(self.parameter_scale) or self.parameter_scale <= 0.0:
            raise ValueError("parameter scale must be finite and positive")

    @property
    def state_weight(self) -> float:
        return 1.0 / float(self.state_size)

    @property
    def parameter_weight(self) -> float:
        return 1.0 / float(self.parameter_scale**2)

    def inner(
        self,
        state_a: np.ndarray,
        parameter_a: float,
        state_b: np.ndarray,
        parameter_b: float,
    ) -> float:
        a = np.asarray(state_a, dtype=float)
        b = np.asarray(state_b, dtype=float)
        if a.shape != (self.state_size,) or b.shape != a.shape:
            raise ValueError("metric state vectors have the wrong shape")
        values = (parameter_a, parameter_b)
        if not np.isfinite(a).all() or not np.isfinite(b).all() or not all(
            np.isfinite(value) for value in values
        ):
            raise ValueError("metric inputs must be finite")
        return float(
            self.state_weight * np.dot(a, b)
            + self.parameter_weight * parameter_a * parameter_b
        )

    def norm(self, state: np.ndarray, parameter: float) -> float:
        value = self.inner(state, parameter, state, parameter)
        if value <= 0.0:
            raise ValueError("arclength vector must be nonzero")
        return float(np.sqrt(value))


@dataclass(frozen=True)
class ArcLengthTangent:
    state: np.ndarray
    parameter: float
    secant_length: float


def normalized_secant_tangent(
    previous_state: np.ndarray,
    previous_parameter: float,
    current_state: np.ndarray,
    current_parameter: float,
    metric: ArcLengthMetric,
    *,
    reference: ArcLengthTangent | None = None,
) -> ArcLengthTangent:
    """Return an oriented unit tangent in the declared arclength metric."""

    previous = np.asarray(previous_state, dtype=float)
    current = np.asarray(current_state, dtype=float)
    if previous.shape != (metric.state_size,) or current.shape != previous.shape:
        raise ValueError("secant states have the wrong shape")
    delta_state = current - previous
    delta_parameter = float(current_parameter - previous_parameter)
    length = metric.norm(delta_state, delta_parameter)
    tangent_state = delta_state / length
    tangent_parameter = delta_parameter / length
    if reference is not None:
        orientation = metric.inner(
            tangent_state,
            tangent_parameter,
            reference.state,
            reference.parameter,
        )
        if orientation < 0.0:
            tangent_state = -tangent_state
            tangent_parameter = -tangent_parameter
    return ArcLengthTangent(
        state=tangent_state,
        parameter=float(tangent_parameter),
        secant_length=length,
    )


def arclength_constraint(
    state: np.ndarray,
    parameter: float,
    predicted_state: np.ndarray,
    predicted_parameter: float,
    tangent: ArcLengthTangent,
    metric: ArcLengthMetric,
    *,
    scale: float,
) -> tuple[float, np.ndarray, float]:
    """Return the scaled hyperplane residual and its exact derivatives."""

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("arclength constraint scale must be finite and positive")
    delta_state = np.asarray(state, dtype=float) - np.asarray(
        predicted_state, dtype=float
    )
    delta_parameter = float(parameter - predicted_parameter)
    value = metric.inner(
        tangent.state,
        tangent.parameter,
        delta_state,
        delta_parameter,
    ) / scale
    state_row = metric.state_weight * tangent.state / scale
    parameter_entry = metric.parameter_weight * tangent.parameter / scale
    return float(value), state_row, float(parameter_entry)


def solve_bordered_newton_direction(
    jacobian: object,
    parameter_column: np.ndarray,
    arclength_row: np.ndarray,
    arclength_parameter: float,
    physical_residual: np.ndarray,
    arclength_residual: float,
) -> tuple[np.ndarray, float, str]:
    """Solve the full bordered system, which remains regular at a simple fold."""

    matrix = jacobian if issparse(jacobian) else csc_matrix(np.asarray(jacobian))
    residual = np.asarray(physical_residual, dtype=float)
    column = np.asarray(parameter_column, dtype=float)
    row = np.asarray(arclength_row, dtype=float)
    size = residual.size
    if matrix.shape != (size, size):
        raise ValueError("physical Jacobian has the wrong shape")
    if column.shape != (size,) or row.shape != (size,):
        raise ValueError("border vectors have the wrong shape")
    if not all(
        np.isfinite(value)
        for value in (arclength_parameter, arclength_residual)
    ):
        raise ValueError("border scalars must be finite")
    augmented = bmat(
        [
            [matrix, csc_matrix(column.reshape(-1, 1))],
            [csc_matrix(row.reshape(1, -1)), csc_matrix([[arclength_parameter]])],
        ],
        format="csc",
    )
    right_hand_side = -np.concatenate(
        (residual, np.asarray((arclength_residual,), dtype=float))
    )
    try:
        direction = splu(augmented).solve(right_hand_side)
        linear_solver = "bordered-splu"
    except RuntimeError:
        direction = lsmr(
            augmented,
            right_hand_side,
            atol=1.0e-12,
            btol=1.0e-12,
        )[0]
        linear_solver = "bordered-lsmr-fallback"
    if direction.shape != (size + 1,) or not np.isfinite(direction).all():
        raise FloatingPointError("bordered linear solve produced a non-finite direction")
    return direction[:-1], float(direction[-1]), linear_solver


@dataclass(frozen=True)
class ArcLengthCorrectorOptions:
    residual_tolerance: float = 1.0e-9
    raw_tolerance: float = 1.0e-8
    arclength_tolerance: float = 1.0e-9
    parameter_scale: float = 0.04
    parameter_minimum: float = 0.0
    parameter_maximum: float = 0.8
    parameter_difference_step: float = 1.0e-6
    maximum_iterations: int = 80
    maximum_jacobians: int = 7
    maximum_objective_evaluations: int = 6000
    pseudo_transient_chord_limit: int = 12
    newton_chord_limit: int = 3
    minimum_line_search_factor: float = 2.0**-18
    invalid_penalty: float = 1.0e8
    pseudo_time_initial: float = 1.0
    pseudo_time_minimum: float = 1.0e-8
    pseudo_time_maximum: float = 1.0e8
    pseudo_time_ser_exponent: float = 1.0
    pseudo_time_growth_limit: float = 2.0
    newton_switch_tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        positive = (
            self.residual_tolerance,
            self.raw_tolerance,
            self.arclength_tolerance,
            self.parameter_scale,
            self.parameter_difference_step,
            self.minimum_line_search_factor,
            self.invalid_penalty,
            self.pseudo_time_initial,
            self.pseudo_time_minimum,
            self.pseudo_time_maximum,
            self.pseudo_time_ser_exponent,
            self.pseudo_time_growth_limit,
            self.newton_switch_tolerance,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("arclength tolerances and scales must be finite and positive")
        if not (
            np.isfinite(self.parameter_minimum)
            and np.isfinite(self.parameter_maximum)
            and self.parameter_minimum < self.parameter_maximum
        ):
            raise ValueError("parameter bounds must be finite and ordered")
        integer_limits = (
            self.maximum_iterations,
            self.maximum_jacobians,
            self.maximum_objective_evaluations,
            self.pseudo_transient_chord_limit,
            self.newton_chord_limit,
        )
        if any(value < 1 for value in integer_limits):
            raise ValueError("arclength work limits must be positive")
        if not (
            self.pseudo_time_minimum
            <= self.pseudo_time_initial
            <= self.pseudo_time_maximum
        ):
            raise ValueError("pseudo-time limits must contain the initial step")
        if self.pseudo_time_growth_limit < 1.0:
            raise ValueError("pseudo-time growth limit must be at least one")


@dataclass(frozen=True)
class ArcLengthCorrectorResult:
    state: np.ndarray
    parameter: float
    accepted: bool
    message: str
    iterations: int
    jacobian_evaluations: int
    objective_evaluations: int
    invalid_evaluations: int
    last_invalid_error: str | None
    raw_acceptance_gate: float
    scaled_residual_linf: float
    arclength_residual: float
    diagnostics: ResidualDiagnostics
    global_balances: dict[str, object]
    tangent: ArcLengthTangent
    predicted_parameter: float
    linear_solver: str | None
    pseudo_transient_steps: int
    final_pseudo_time_step: float
    iteration_trace: tuple[dict[str, object], ...]


def _make_problem(case: CavityCase) -> R26NodeBVP:
    return R26NodeBVP(
        case,
        bulk_operator=compatible_fv_bulk_residual,
        mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
    )


def solve_r26_pseudo_arclength_step(
    case_template: CavityCase,
    previous_state: np.ndarray,
    previous_parameter: float,
    current_state: np.ndarray,
    current_parameter: float,
    step_length: float,
    *,
    options: ArcLengthCorrectorOptions | None = None,
    reference_tangent: ArcLengthTangent | None = None,
) -> ArcLengthCorrectorResult:
    """Correct one R26 predictor with a bordered pseudo-arclength Newton solve."""

    controls = ArcLengthCorrectorOptions() if options is None else options
    if not np.isfinite(step_length) or step_length <= 0.0:
        raise ValueError("arclength step must be finite and positive")
    template_problem = _make_problem(
        case_template.with_lid_velocity(current_parameter, suffix="arc-template")
    )
    transform = LogStateTransform(template_problem.shape)
    previous_encoded = transform.encode(previous_state)
    current_encoded = transform.encode(current_state)
    metric = ArcLengthMetric(
        template_problem.unknown_count,
        parameter_scale=controls.parameter_scale,
    )
    tangent = normalized_secant_tangent(
        previous_encoded,
        previous_parameter,
        current_encoded,
        current_parameter,
        metric,
        reference=reference_tangent,
    )
    predicted_encoded = current_encoded + step_length * tangent.state
    predicted_parameter = current_parameter + step_length * tangent.parameter
    lower, upper = transform.least_squares_bounds()
    predicted_encoded = np.clip(predicted_encoded, lower, upper)
    encoded = predicted_encoded.copy()
    parameter = float(predicted_parameter)
    pattern = jacobian_sparsity(
        template_problem,
        include_mass_border=False,
    )
    objective_evaluations = 0
    invalid_evaluations = 0
    last_invalid_error: str | None = None

    def counted(vector: np.ndarray, lid: float) -> np.ndarray:
        nonlocal objective_evaluations, invalid_evaluations, last_invalid_error
        if objective_evaluations >= controls.maximum_objective_evaluations:
            raise RuntimeError("pseudo-arclength objective-evaluation limit reached")
        objective_evaluations += 1
        try:
            state = transform.decode(vector)
            case = case_template.with_lid_velocity(lid, suffix="arc-corrector")
            return _make_problem(case).residual(state)
        except (FloatingPointError, ValueError, OverflowError) as error:
            invalid_evaluations += 1
            last_invalid_error = f"{type(error).__name__}: {error}"
            values = np.asarray(vector, dtype=float)
            sign = np.where(np.isfinite(values) & (values < 0.0), -1.0, 1.0)
            magnitude = 1.0 + np.minimum(
                np.nan_to_num(np.abs(values), nan=100.0, posinf=100.0),
                100.0,
            )
            return controls.invalid_penalty * sign * magnitude

    residual = counted(encoded, parameter)
    message = "pseudo-arclength nonlinear-iteration limit reached"
    last_linear_solver: str | None = None
    iterations = 0
    jacobian_evaluations = 0
    final_arc_residual = float("inf")
    pseudo_time_step = float(controls.pseudo_time_initial)
    pseudo_transient_steps = 0
    physical_jacobian = None
    parameter_column = None
    chord_steps = 0
    force_jacobian_refresh = False
    iteration_trace: list[dict[str, object]] = []
    trace_parameter_before = parameter
    trace_scaled_before = float(np.max(np.abs(residual), initial=0.0))
    trace_arc_before = float("inf")
    trace_jacobian_refreshed = False
    trace_chord_before = 0
    trace_use_pseudo_transient = False
    trace_pseudo_time_before = pseudo_time_step

    def append_trace(
        *,
        iteration: int,
        parameter_before: float,
        scaled_before: float,
        arc_before: float,
        jacobian_refreshed: bool,
        chord_before: int,
        use_pseudo_transient: bool,
        pseudo_time_before: float,
        line_search_factor: float | None,
        accepted_step: bool,
        merit_ratio: float,
        outcome: str,
    ) -> None:
        iteration_trace.append(
            {
                "iteration": iteration,
                "parameter_before": parameter_before,
                "scaled_residual_linf_before": scaled_before,
                "arclength_residual_before": arc_before,
                "jacobian_evaluations": jacobian_evaluations,
                "jacobian_refreshed": jacobian_refreshed,
                "chord_steps_before": chord_before,
                "pseudo_transient": use_pseudo_transient,
                "pseudo_time_step_before": pseudo_time_before,
                "line_search_factor": line_search_factor,
                "step_accepted": accepted_step,
                "parameter_after": parameter,
                "scaled_residual_linf_after": float(
                    np.max(np.abs(residual), initial=0.0)
                ),
                "arclength_residual_after": final_arc_residual,
                "pseudo_time_step_after": pseudo_time_step,
                "merit_ratio": merit_ratio,
                "linear_solver": last_linear_solver,
                "outcome": outcome,
            }
        )

    try:
        for iteration in range(1, controls.maximum_iterations + 1):
            iterations = iteration
            scaled_linf = float(np.max(np.abs(residual), initial=0.0))
            arc_value, arc_row, arc_parameter = arclength_constraint(
                encoded,
                parameter,
                predicted_encoded,
                predicted_parameter,
                tangent,
                metric,
                scale=step_length,
            )
            final_arc_residual = arc_value
            trace_parameter_before = parameter
            trace_scaled_before = scaled_linf
            trace_arc_before = arc_value
            trace_jacobian_refreshed = False
            trace_chord_before = chord_steps
            trace_use_pseudo_transient = False
            trace_pseudo_time_before = pseudo_time_step
            if (
                scaled_linf <= controls.residual_tolerance
                and abs(arc_value) <= controls.arclength_tolerance
            ):
                message = "pseudo-arclength residual tolerance reached"
                append_trace(
                    iteration=iteration,
                    parameter_before=parameter,
                    scaled_before=scaled_linf,
                    arc_before=arc_value,
                    jacobian_refreshed=False,
                    chord_before=chord_steps,
                    use_pseudo_transient=False,
                    pseudo_time_before=pseudo_time_step,
                    line_search_factor=None,
                    accepted_step=False,
                    merit_ratio=1.0,
                    outcome="residual_tolerance_reached",
                )
                break

            combined_linf = max(scaled_linf, abs(arc_value))
            use_pseudo_transient = (
                combined_linf > controls.newton_switch_tolerance
            )
            chord_limit = (
                controls.pseudo_transient_chord_limit
                if use_pseudo_transient
                else controls.newton_chord_limit
            )
            refresh_jacobian = bool(
                physical_jacobian is None
                or parameter_column is None
                or force_jacobian_refresh
                or chord_steps >= chord_limit
            )
            chord_steps_before = chord_steps
            trace_chord_before = chord_steps_before
            trace_use_pseudo_transient = use_pseudo_transient
            if refresh_jacobian:
                if jacobian_evaluations >= controls.maximum_jacobians:
                    message = (
                        "pseudo-arclength Jacobian-evaluation limit reached "
                        f"({controls.maximum_jacobians})"
                    )
                    append_trace(
                        iteration=iteration,
                        parameter_before=parameter,
                        scaled_before=scaled_linf,
                        arc_before=arc_value,
                        jacobian_refreshed=False,
                        chord_before=chord_steps,
                        use_pseudo_transient=use_pseudo_transient,
                        pseudo_time_before=pseudo_time_step,
                        line_search_factor=None,
                        accepted_step=False,
                        merit_ratio=1.0,
                        outcome="jacobian_evaluation_limit_reached",
                    )
                    break

                jacobian_parameter = float(parameter)
                jacobian_encoded = encoded.copy()
                jacobian_residual = residual.copy()
                jacobian_problem = _make_problem(
                    case_template.with_lid_velocity(
                        jacobian_parameter,
                        suffix="arc-jacobian",
                    )
                )
                mass_row, density_columns, mass_values = analytic_mass_jacobian_row(
                    jacobian_problem,
                    transform,
                    jacobian_encoded,
                )

                def without_mass_border(vector: np.ndarray) -> np.ndarray:
                    values = counted(vector, jacobian_parameter).copy()
                    values[mass_row] = 0.0
                    return values

                finite_difference_base = jacobian_residual.copy()
                finite_difference_base[mass_row] = 0.0
                finite_difference_jacobian = approx_derivative(
                    without_mass_border,
                    jacobian_encoded,
                    method="2-point",
                    abs_step=fv_absolute_difference_step(jacobian_encoded),
                    bounds=(lower, upper),
                    sparsity=pattern,
                    f0=finite_difference_base,
                ).tolil()
                finite_difference_jacobian[mass_row, :] = 0.0
                finite_difference_jacobian[mass_row, density_columns] = mass_values
                physical_jacobian = finite_difference_jacobian.tocsc()

                parameter_step = controls.parameter_difference_step * max(
                    1.0,
                    abs(jacobian_parameter),
                )
                can_step_forward = (
                    jacobian_parameter + parameter_step
                    <= controls.parameter_maximum
                )
                can_step_backward = (
                    jacobian_parameter - parameter_step
                    >= controls.parameter_minimum
                )
                if can_step_forward and can_step_backward:
                    plus = counted(
                        jacobian_encoded,
                        jacobian_parameter + parameter_step,
                    )
                    minus = counted(
                        jacobian_encoded,
                        jacobian_parameter - parameter_step,
                    )
                    parameter_column = (plus - minus) / (2.0 * parameter_step)
                elif can_step_forward:
                    plus = counted(
                        jacobian_encoded,
                        jacobian_parameter + parameter_step,
                    )
                    parameter_column = (plus - jacobian_residual) / parameter_step
                elif can_step_backward:
                    minus = counted(
                        jacobian_encoded,
                        jacobian_parameter - parameter_step,
                    )
                    parameter_column = (jacobian_residual - minus) / parameter_step
                else:
                    raise RuntimeError(
                        "pseudo-arclength parameter-difference step exceeds bounds"
                    )
                parameter_column[mass_row] = 0.0
                jacobian_evaluations += 1
                chord_steps = 0
                chord_steps_before = 0
                force_jacobian_refresh = False

            trace_jacobian_refreshed = refresh_jacobian

            assert physical_jacobian is not None
            assert parameter_column is not None
            merit = 0.5 * (float(np.dot(residual, residual)) + arc_value**2)
            old_linf = combined_linf
            pseudo_time_before = pseudo_time_step
            trace_pseudo_time_before = pseudo_time_before
            parameter_before = parameter
            linear_jacobian = physical_jacobian
            if use_pseudo_transient:
                current_problem = _make_problem(
                    case_template.with_lid_velocity(
                        parameter,
                        suffix="arc-pseudo-time",
                    )
                )
                pseudo_diagonal = pseudo_transient_diagonal(
                    current_problem,
                    transform,
                    encoded,
                )
                linear_jacobian = (
                    physical_jacobian
                    + diags(pseudo_diagonal / pseudo_time_step, format="csc")
                ).tocsc()
            state_direction, parameter_direction, last_linear_solver = (
                solve_bordered_newton_direction(
                    linear_jacobian,
                    parameter_column,
                    arc_row,
                    arc_parameter,
                    residual,
                    arc_value,
                )
            )

            factor = 1.0
            accepted_line_search = False
            trial_merit = merit
            trial_arc = arc_value
            while factor >= controls.minimum_line_search_factor:
                trial_parameter = parameter + factor * parameter_direction
                if not (
                    controls.parameter_minimum
                    <= trial_parameter
                    <= controls.parameter_maximum
                ):
                    factor *= 0.5
                    continue
                trial_encoded = np.clip(
                    encoded + factor * state_direction,
                    lower,
                    upper,
                )
                trial_residual = counted(trial_encoded, trial_parameter)
                trial_arc, _, _ = arclength_constraint(
                    trial_encoded,
                    trial_parameter,
                    predicted_encoded,
                    predicted_parameter,
                    tangent,
                    metric,
                    scale=step_length,
                )
                trial_merit = 0.5 * (
                    float(np.dot(trial_residual, trial_residual)) + trial_arc**2
                )
                sufficient_decrease = (
                    trial_merit < merit
                    if use_pseudo_transient
                    else trial_merit < merit * (1.0 - 1.0e-4 * factor)
                )
                if np.isfinite(trial_merit) and sufficient_decrease:
                    encoded = trial_encoded
                    parameter = float(trial_parameter)
                    residual = trial_residual
                    final_arc_residual = trial_arc
                    accepted_line_search = True
                    chord_steps += 1
                    if use_pseudo_transient:
                        pseudo_transient_steps += 1
                        new_linf = max(
                            float(np.max(np.abs(residual), initial=0.0)),
                            abs(trial_arc),
                        )
                        ser_ratio = old_linf / max(
                            new_linf,
                            np.finfo(float).tiny,
                        )
                        growth = min(
                            controls.pseudo_time_growth_limit,
                            max(
                                0.25,
                                ser_ratio ** controls.pseudo_time_ser_exponent,
                            ),
                        )
                        pseudo_time_step = float(
                            np.clip(
                                pseudo_time_step * growth,
                                controls.pseudo_time_minimum,
                                controls.pseudo_time_maximum,
                            )
                        )
                    if (
                        (not use_pseudo_transient and trial_merit > 0.25 * merit)
                        or (use_pseudo_transient and trial_merit > 0.95 * merit)
                    ):
                        force_jacobian_refresh = True
                    break
                factor *= 0.5

            if accepted_line_search:
                append_trace(
                    iteration=iteration,
                    parameter_before=parameter_before,
                    scaled_before=scaled_linf,
                    arc_before=arc_value,
                    jacobian_refreshed=refresh_jacobian,
                    chord_before=chord_steps_before,
                    use_pseudo_transient=use_pseudo_transient,
                    pseudo_time_before=pseudo_time_before,
                    line_search_factor=factor,
                    accepted_step=True,
                    merit_ratio=trial_merit / max(merit, np.finfo(float).tiny),
                    outcome=(
                        "accepted_refresh_requested"
                        if force_jacobian_refresh
                        else "accepted_chord_step"
                    ),
                )
                continue

            retry_with_smaller_pseudo_time = bool(
                use_pseudo_transient
                and pseudo_time_step > controls.pseudo_time_minimum
            )
            retry_with_refresh = chord_steps > 0
            outcome = "line_search_failed"
            if retry_with_smaller_pseudo_time:
                pseudo_time_step = max(
                    controls.pseudo_time_minimum,
                    0.25 * pseudo_time_step,
                )
                outcome = "pseudo_time_reduced"
            elif retry_with_refresh:
                force_jacobian_refresh = True
                chord_steps = 0
                outcome = "jacobian_refresh_requested"
            append_trace(
                iteration=iteration,
                parameter_before=parameter_before,
                scaled_before=scaled_linf,
                arc_before=arc_value,
                jacobian_refreshed=refresh_jacobian,
                chord_before=chord_steps_before,
                use_pseudo_transient=use_pseudo_transient,
                pseudo_time_before=pseudo_time_before,
                line_search_factor=None,
                accepted_step=False,
                merit_ratio=1.0,
                outcome=outcome,
            )
            if retry_with_smaller_pseudo_time or retry_with_refresh:
                continue
            message = (
                "pseudo-arclength SER-PTC/Newton line search failed on a fresh "
                f"Jacobian after {last_linear_solver}"
            )
            break
        else:
            message = (
                "pseudo-arclength nonlinear-iteration limit reached "
                f"({controls.maximum_iterations})"
            )
    except RuntimeError as error:
        message = str(error)
        if len(iteration_trace) < iterations:
            append_trace(
                iteration=iterations,
                parameter_before=trace_parameter_before,
                scaled_before=trace_scaled_before,
                arc_before=trace_arc_before,
                jacobian_refreshed=trace_jacobian_refreshed,
                chord_before=trace_chord_before,
                use_pseudo_transient=trace_use_pseudo_transient,
                pseudo_time_before=trace_pseudo_time_before,
                line_search_factor=None,
                accepted_step=False,
                merit_ratio=1.0,
                outcome="runtime_limit_or_failure",
            )

    state = transform.decode(encoded)
    final_case = case_template.with_lid_velocity(parameter, suffix="arc-final")
    final_problem = _make_problem(final_case)
    evaluation = final_problem.evaluate(state)
    diagnostics = evaluation.diagnostics
    balances = global_balance_diagnostics(state, final_case)
    raw_gate = max(
        diagnostics.raw_total_linf,
        abs(diagnostics.held_out_continuity),
        abs(diagnostics.mass_error),
    )
    scaled_linf = float(np.max(np.abs(evaluation.residual), initial=0.0))
    accepted = bool(
        scaled_linf <= controls.residual_tolerance
        and abs(final_arc_residual) <= controls.arclength_tolerance
        and raw_gate <= controls.raw_tolerance
        and diagnostics.min_density > 0.0
        and diagnostics.min_temperature > 0.0
        and float(balances["wall_effective_pressure_min"]) > 0.0
        and float(balances["momentum_boundary_flux_linf"])
        <= 10.0 * controls.raw_tolerance
        and abs(float(balances["internal_energy_balance_error"]))
        <= 10.0 * controls.raw_tolerance
    )
    if accepted:
        message = "pseudo-arclength raw physical gate reached"
    return ArcLengthCorrectorResult(
        state=validate_planar_state(state),
        parameter=float(parameter),
        accepted=accepted,
        message=message,
        iterations=iterations,
        jacobian_evaluations=jacobian_evaluations,
        objective_evaluations=objective_evaluations,
        invalid_evaluations=invalid_evaluations,
        last_invalid_error=last_invalid_error,
        raw_acceptance_gate=float(raw_gate),
        scaled_residual_linf=scaled_linf,
        arclength_residual=float(final_arc_residual),
        diagnostics=diagnostics,
        global_balances=balances,
        tangent=tangent,
        predicted_parameter=float(predicted_parameter),
        linear_solver=last_linear_solver,
        pseudo_transient_steps=pseudo_transient_steps,
        final_pseudo_time_step=float(pseudo_time_step),
        iteration_trace=tuple(iteration_trace),
    )


def interpolate_bracketed_state(
    problem: R26NodeBVP,
    lower_state: np.ndarray,
    lower_parameter: float,
    upper_state: np.ndarray,
    upper_parameter: float,
    target_parameter: float,
) -> np.ndarray:
    """Interpolate a target seed in log coordinates and restore exact mass."""

    values = (lower_parameter, upper_parameter, target_parameter)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("bracket parameters must be finite")
    spacing = float(upper_parameter - lower_parameter)
    if spacing == 0.0:
        raise ValueError("bracket parameters must be distinct")
    fraction = float((target_parameter - lower_parameter) / spacing)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("target parameter lies outside the bracket")
    transform = LogStateTransform(problem.shape)
    lower = transform.encode(lower_state)
    upper = transform.encode(upper_state)
    bounds = transform.least_squares_bounds()
    encoded = np.clip(lower + fraction * (upper - lower), *bounds)
    state = transform.decode(encoded)
    mean_density = problem.mean_density(state)
    if not np.isfinite(mean_density) or mean_density <= 0.0:
        raise FloatingPointError("bracket interpolation produced invalid mass")
    state[..., 0] *= problem.case.mean_density / mean_density
    return validate_planar_state(state)


__all__ = [
    "ArcLengthCorrectorOptions",
    "ArcLengthCorrectorResult",
    "ArcLengthMetric",
    "ArcLengthTangent",
    "arclength_constraint",
    "interpolate_bracketed_state",
    "normalized_secant_tangent",
    "solve_bordered_newton_direction",
    "solve_r26_pseudo_arclength_step",
]
