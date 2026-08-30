#!/usr/bin/env python3
"""Bounded monolithic oracle for the reconstructed Gu--Emerson system.

This is intentionally separate from the published-order segregated solver.
It answers one diagnostic question on N8: does the assembled transformed-FV
system, completed by the physical wall/corner equations and mass border, have
an algebraic root near the best segregated checkpoint?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from r26_discretization import R26NodeBVP
from r26_gu_emerson_transformed_fv import gu_emerson_transformed_fv_residual
from r26_gu_emerson_variables import (
    GuEmersonFields,
    gu_emerson_fields_as_planar17,
    gu_emerson_fields_from_planar17,
    state_from_gu_emerson_fields,
)
from r26_solver import LogStateTransform


@dataclass(frozen=True)
class GuEmersonMonolithicOracleOptions:
    residual_tolerance: float = 1.0e-10
    max_outer_iterations: int = 8
    invalid_penalty: float = 1.0e8
    display: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.residual_tolerance) or self.residual_tolerance <= 0.0:
            raise ValueError("oracle residual tolerance must be finite and positive")
        if self.max_outer_iterations < 1:
            raise ValueError("oracle work budget must be positive")
        if not np.isfinite(self.invalid_penalty) or self.invalid_penalty <= 0.0:
            raise ValueError("oracle invalid penalty must be finite and positive")


@dataclass(frozen=True)
class GuEmersonMonolithicOracleResult:
    fields: GuEmersonFields
    state: np.ndarray
    scipy_success: bool
    transformed_root_passed: bool
    complete_physical_gate_passed: bool
    objective_linf: float
    initial_objective_linf: float
    best_outer_iteration: int
    transformed_linf: float
    physical_raw_linf: float
    held_continuity: float
    mass_error: float
    function_evaluations: int
    jacobian_evaluations: int
    invalid_evaluations: int
    message: str


class EncodedGuEmersonMonolithicObjective:
    """Raw square residual with transformed bulk and physical boundaries."""

    def __init__(
        self,
        problem: R26NodeBVP,
        transform: LogStateTransform,
        penalty: float,
    ) -> None:
        self.problem = problem
        self.transform = transform
        self.penalty = float(penalty)
        self.invalid_evaluations = 0

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        try:
            packed = self.transform.decode(vector)
            fields = gu_emerson_fields_from_planar17(packed)
            case = self.problem.case
            state = state_from_gu_emerson_fields(
                fields, x=case.x, y=case.y, mu=case.mu(fields.theta)
            )
            physical = self.problem.evaluate(state)
            residual = np.asarray(physical.unscaled_residual, dtype=float).copy()
            transformed = gu_emerson_transformed_fv_residual(fields, case=case)
            residual[1:-1, 1:-1, :] = transformed[1:-1, 1:-1, :]
            residual[self.problem.mass_j, self.problem.mass_i, 0] = (
                physical.diagnostics.mass_error
            )
            if not np.isfinite(residual).all():
                raise FloatingPointError("monolithic objective is non-finite")
            return residual.ravel()
        except (FloatingPointError, ValueError, OverflowError):
            self.invalid_evaluations += 1
            x = np.asarray(vector, dtype=float)
            magnitude = 1.0 + np.minimum(
                np.nan_to_num(np.abs(x), nan=100.0, posinf=100.0), 100.0
            )
            return self.penalty * magnitude


def solve_gu_emerson_monolithic_oracle(
    problem: R26NodeBVP,
    initial_fields: GuEmersonFields,
    *,
    options: GuEmersonMonolithicOracleOptions | None = None,
) -> GuEmersonMonolithicOracleResult:
    """Run one bounded sparse least-squares oracle; never authorize refinement."""

    options = GuEmersonMonolithicOracleOptions() if options is None else options
    if problem.case.nodes > 8:
        raise ValueError("monolithic diagnostic oracle is restricted to N8 or smaller")
    packed = gu_emerson_fields_as_planar17(initial_fields)
    transform = LogStateTransform(problem.shape)
    x0 = transform.encode(packed)
    objective = EncodedGuEmersonMonolithicObjective(
        problem, transform, options.invalid_penalty
    )
    initial_objective_linf = float(np.max(np.abs(objective(x0))))
    best_vector = x0.copy()
    best_objective_linf = initial_objective_linf
    best_outer_iteration = 0
    callback_iterations = 0

    def retain_best(vector: np.ndarray, residual: np.ndarray) -> None:
        nonlocal best_vector, best_objective_linf, best_outer_iteration
        nonlocal callback_iterations
        callback_iterations += 1
        merit = float(np.max(np.abs(residual)))
        if merit < best_objective_linf:
            best_vector = np.asarray(vector, dtype=float).copy()
            best_objective_linf = merit
            best_outer_iteration = callback_iterations

    result = root(
        objective,
        x0,
        method="krylov",
        callback=retain_best,
        options={
            "fatol": options.residual_tolerance,
            "maxiter": options.max_outer_iterations,
            "line_search": "armijo",
            "disp": options.display,
            "jac_options": {"inner_maxiter": 12},
        },
    )
    final_packed = transform.decode(best_vector)
    fields = gu_emerson_fields_from_planar17(final_packed)
    case = problem.case
    state = state_from_gu_emerson_fields(
        fields, x=case.x, y=case.y, mu=case.mu(fields.theta)
    )
    diagnostics = problem.evaluate(state).diagnostics
    transformed = gu_emerson_transformed_fv_residual(fields, case=case)
    transformed_linf = float(np.max(np.abs(transformed[1:-1, 1:-1, :])))
    physical_raw = float(
        max(
            diagnostics.raw_total_linf,
            abs(diagnostics.held_out_continuity),
            abs(diagnostics.mass_error),
        )
    )
    objective_linf = float(np.max(np.abs(objective(best_vector))))
    tolerance = options.residual_tolerance
    return GuEmersonMonolithicOracleResult(
        fields=fields,
        state=state,
        scipy_success=bool(result.success),
        transformed_root_passed=bool(
            transformed_linf <= tolerance
            and abs(diagnostics.mass_error) <= tolerance
            and diagnostics.min_density > 0.0
            and diagnostics.min_temperature > 0.0
        ),
        complete_physical_gate_passed=bool(physical_raw <= tolerance),
        objective_linf=objective_linf,
        initial_objective_linf=initial_objective_linf,
        best_outer_iteration=best_outer_iteration,
        transformed_linf=transformed_linf,
        physical_raw_linf=physical_raw,
        held_continuity=float(diagnostics.held_out_continuity),
        mass_error=float(diagnostics.mass_error),
        function_evaluations=int(result.nfev),
        jacobian_evaluations=0,
        invalid_evaluations=objective.invalid_evaluations,
        message=(
            f"{result.message}; returning best monolithic iterate "
            f"{best_outer_iteration}"
        ),
    )


__all__ = [
    "EncodedGuEmersonMonolithicObjective",
    "GuEmersonMonolithicOracleOptions",
    "GuEmersonMonolithicOracleResult",
    "solve_gu_emerson_monolithic_oracle",
]
