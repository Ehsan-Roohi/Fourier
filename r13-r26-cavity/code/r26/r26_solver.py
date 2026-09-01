#!/usr/bin/env python3
"""Stateless nonlinear solver utilities for the private node-grid R26 BVP."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import OptimizeResult, least_squares, root
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import lsmr, splu

from r26_cases import CavityCase
from r26_discretization import R26NodeBVP, ResidualDiagnostics, trapezoidal_node_weights
from r26_fv_backend import fv_absolute_difference_step
from r26_state import NVAR, validate_planar_state


class R26StateTransform(Protocol):
    """Coordinate map accepted by the stateless nonlinear solver.

    Implementations retain the planar-17 vector layout, encode density and
    temperature logarithmically, and reconstruct a physical R26 state in
    :meth:`decode`.  The latter permits Newton iterations in alternative
    primary variables without changing any residual or acceptance row.
    """

    shape: tuple[int, int, int]
    supports_physical_pseudo_transient: bool

    def encode(self, state: np.ndarray) -> np.ndarray: ...

    def decode(self, vector: np.ndarray) -> np.ndarray: ...

    def least_squares_bounds(self) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class LogStateTransform:
    """Logarithmic rho/T coordinates and raw coordinates for all other fields."""

    shape: tuple[int, int, int]
    maximum_log_magnitude: float = 50.0
    supports_physical_pseudo_transient: bool = True

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or self.shape[-1] != NVAR:
            raise ValueError("transform shape must be (ny,nx,17)")
        if self.maximum_log_magnitude <= 0.0:
            raise ValueError("maximum log magnitude must be positive")

    def encode(self, state: np.ndarray) -> np.ndarray:
        u = validate_planar_state(state)
        if u.shape != self.shape:
            raise ValueError(f"state shape must be {self.shape}")
        encoded = u.copy()
        encoded[..., 0] = np.log(u[..., 0])
        encoded[..., 3] = np.log(u[..., 3])
        return encoded.ravel()

    def decode(self, vector: np.ndarray) -> np.ndarray:
        x = np.asarray(vector, dtype=float)
        if x.shape != (int(np.prod(self.shape)),) or not np.isfinite(x).all():
            raise ValueError("encoded state has incorrect shape or non-finite values")
        encoded = x.reshape(self.shape)
        logs = encoded[..., (0, 3)]
        if np.max(np.abs(logs), initial=0.0) > self.maximum_log_magnitude:
            raise FloatingPointError("rho/T log coordinate exceeded the declared solver domain")
        state = encoded.copy()
        state[..., 0] = np.exp(encoded[..., 0])
        state[..., 3] = np.exp(encoded[..., 3])
        return validate_planar_state(state)

    def least_squares_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.shape, -np.inf)
        upper = np.full(self.shape, np.inf)
        lower[..., 0] = lower[..., 3] = -self.maximum_log_magnitude
        upper[..., 0] = upper[..., 3] = self.maximum_log_magnitude
        return lower.ravel(), upper.ravel()


@dataclass(frozen=True)
class SolveOptions:
    method: str = "krylov"
    residual_tolerance: float = 1.0e-9
    held_out_continuity_tolerance: float = 1.0e-7
    max_iterations: int = 200
    max_function_evaluations: int = 5000
    max_objective_evaluations: int | None = None
    line_search: str = "armijo"
    display: bool = False
    invalid_penalty: float = 1.0e8
    analytic_mass_jacobian: bool = False
    pseudo_transient: bool = False
    pseudo_time_initial: float = 1.0e-2
    pseudo_time_minimum: float = 1.0e-8
    pseudo_time_maximum: float = 1.0e8
    pseudo_time_ser_exponent: float = 1.0
    pseudo_time_growth_limit: float = 2.0
    pseudo_time_minimum_accepted_alpha: float = 0.0
    pseudo_time_small_alpha_growth: float = 4.0
    newton_switch_tolerance: float = 1.0e-6
    require_raw_linf_decrease: bool = False
    max_jacobian_evaluations: int | None = None
    jacobian_stencil_radius: int = 2

    def __post_init__(self) -> None:
        if self.method not in {"krylov", "least_squares", "colored_newton"}:
            raise ValueError("method must be krylov, least_squares, or colored_newton")
        if self.residual_tolerance <= 0.0 or self.held_out_continuity_tolerance <= 0.0:
            raise ValueError("solver tolerances must be positive")
        if self.max_iterations < 1 or self.max_function_evaluations < 1:
            raise ValueError("solver iteration limits must be positive")
        if self.max_objective_evaluations is not None and self.max_objective_evaluations < 1:
            raise ValueError("objective evaluation limit must be positive when supplied")
        if (self.analytic_mass_jacobian or self.pseudo_transient) and self.method != "colored_newton":
            raise ValueError(
                "analytic mass Jacobian and pseudo-transient modes require colored_newton"
            )
        pseudo_values = (
            self.pseudo_time_initial,
            self.pseudo_time_minimum,
            self.pseudo_time_maximum,
            self.pseudo_time_ser_exponent,
            self.pseudo_time_growth_limit,
            self.pseudo_time_small_alpha_growth,
            self.newton_switch_tolerance,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in pseudo_values):
            raise ValueError("pseudo-time controls and Newton switch tolerance must be positive")
        if not self.pseudo_time_minimum <= self.pseudo_time_initial <= self.pseudo_time_maximum:
            raise ValueError("pseudo_time_initial must lie within the declared pseudo-time bounds")
        if self.pseudo_time_growth_limit < 1.0:
            raise ValueError("pseudo_time_growth_limit must be at least one")
        if self.pseudo_time_small_alpha_growth <= 1.0:
            raise ValueError("small-alpha pseudo-time growth must exceed one")
        if not 0.0 <= self.pseudo_time_minimum_accepted_alpha <= 1.0:
            raise ValueError("minimum accepted pseudo-time alpha must lie in [0, 1]")
        if self.require_raw_linf_decrease and not self.pseudo_transient:
            raise ValueError("raw-Linf line-search protection requires pseudo-transient mode")
        if self.max_jacobian_evaluations is not None and self.max_jacobian_evaluations < 1:
            raise ValueError("Jacobian evaluation limit must be positive when supplied")
        if self.jacobian_stencil_radius < 2:
            raise ValueError("Jacobian stencil radius must be at least two")


@dataclass(frozen=True)
class R26SolveResult:
    case: CavityCase
    state: np.ndarray
    encoded_state: np.ndarray
    diagnostics: ResidualDiagnostics
    converged: bool
    scipy_success: bool
    message: str
    iterations: int
    function_evaluations: int
    invalid_evaluations: int
    last_invalid_error: str | None
    solver_method: str
    jacobian_evaluations: int = 0
    pseudo_transient_steps: int = 0
    final_pseudo_time_step: float | None = None


class _ObjectiveEvaluationLimitReached(RuntimeError):
    """Internal fail-closed stop for an exhausted nonlinear work budget."""


class EncodedR26Objective:
    """Pure vector objective plus a finite guard for rejected line-search steps."""

    def __init__(self, problem: R26NodeBVP, transform: R26StateTransform, penalty: float) -> None:
        self.problem = problem
        self.transform = transform
        self.penalty = float(penalty)
        self.invalid_evaluations = 0
        self.last_invalid_error: str | None = None
        self.last_raw_linf = float("inf")

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        try:
            state = self.transform.decode(vector)
            evaluation = self.problem.evaluate(state)
            self.last_raw_linf = float(
                max(
                    evaluation.diagnostics.raw_total_linf,
                    abs(evaluation.diagnostics.held_out_continuity),
                    abs(evaluation.diagnostics.mass_error),
                )
            )
            return evaluation.flat
        except (FloatingPointError, ValueError, OverflowError) as exc:
            self.invalid_evaluations += 1
            self.last_invalid_error = f"{type(exc).__name__}: {exc}"
            self.last_raw_linf = float("inf")
            x = np.asarray(vector, dtype=float)
            sign = np.where(np.isfinite(x) & (x < 0.0), -1.0, 1.0)
            magnitude = 1.0 + np.minimum(np.nan_to_num(np.abs(x), nan=100.0, posinf=100.0), 100.0)
            return self.penalty * sign * magnitude

    def jvp(self, vector: np.ndarray, direction: np.ndarray, *, relative_step: float = 1.0e-6) -> np.ndarray:
        """Centered finite-difference Jacobian-vector product of the full BVP."""

        x = np.asarray(vector, dtype=float)
        v = np.asarray(direction, dtype=float)
        if x.shape != v.shape or not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError("JVP vectors must have the same finite shape")
        norm_v = np.linalg.norm(v)
        if norm_v == 0.0:
            return np.zeros_like(x)
        step = relative_step * (1.0 + np.linalg.norm(x)) / norm_v
        return (self(x + step * v) - self(x - step * v)) / (2.0 * step)


class EncodedR26MassContinuityObjective:
    """Return all physical BVP rows plus the independent mass constraint.

    ``R26NodeBVP`` keeps a square system by replacing one interior continuity
    row with the global mass border.  For an overdetermined consistency solve,
    this objective restores that held continuity row at its original index and
    appends the mass equation as row ``unknown_count``.  No physical equation
    is removed, duplicated, or converted into a penalty.
    """

    def __init__(self, problem: R26NodeBVP, transform: R26StateTransform, penalty: float) -> None:
        self.problem = problem
        self.transform = transform
        self.penalty = float(penalty)
        self.invalid_evaluations = 0
        self.last_invalid_error: str | None = None

    @property
    def equation_count(self) -> int:
        return self.problem.unknown_count + 1

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        try:
            state = self.transform.decode(vector)
            evaluation = self.problem.evaluate(state)
            values = evaluation.residual.ravel().copy()
            mass_index = int(np.ravel_multi_index(evaluation.mass_row, self.problem.shape))
            values[mass_index] = (
                evaluation.diagnostics.held_out_continuity
                / self.problem.case.scaling.bulk[0]
            )
            appended_mass = (
                evaluation.diagnostics.mass_error / self.problem.case.scaling.mass
            )
            return np.concatenate((values, np.asarray((appended_mass,), dtype=float)))
        except (FloatingPointError, ValueError, OverflowError) as exc:
            self.invalid_evaluations += 1
            self.last_invalid_error = f"{type(exc).__name__}: {exc}"
            x = np.asarray(vector, dtype=float)
            sign = np.where(np.isfinite(x) & (x < 0.0), -1.0, 1.0)
            magnitude = 1.0 + np.minimum(
                np.nan_to_num(np.abs(x), nan=100.0, posinf=100.0), 100.0
            )
            guarded = self.penalty * sign * magnitude
            return np.concatenate((guarded, np.asarray((self.penalty,), dtype=float)))


class EncodedR26RawMassContinuityObjective:
    """Return every *unscaled* physical row plus the raw mass constraint.

    This is the fail-closed merit function for final nonlinear reconciliation.
    Component/family scaling can improve a predictor or Newton step, but a
    small scaled residual does not imply that every dimensionaless raw moment
    balance is small.  As in :class:`EncodedR26MassContinuityObjective`, the
    displaced local continuity row is restored at its original index and mass
    is appended as an independent equation.  No physical row is omitted and
    no penalty/regularization row is added for a valid state.
    """

    def __init__(self, problem: R26NodeBVP, transform: R26StateTransform, penalty: float) -> None:
        self.problem = problem
        self.transform = transform
        self.penalty = float(penalty)
        self.invalid_evaluations = 0
        self.last_invalid_error: str | None = None

    @property
    def equation_count(self) -> int:
        return self.problem.unknown_count + 1

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        try:
            state = self.transform.decode(vector)
            evaluation = self.problem.evaluate(state)
            values = evaluation.unscaled_residual.ravel().copy()
            mass_index = int(np.ravel_multi_index(evaluation.mass_row, self.problem.shape))
            values[mass_index] = evaluation.diagnostics.held_out_continuity
            return np.concatenate(
                (values, np.asarray((evaluation.diagnostics.mass_error,), dtype=float))
            )
        except (FloatingPointError, ValueError, OverflowError) as exc:
            self.invalid_evaluations += 1
            self.last_invalid_error = f"{type(exc).__name__}: {exc}"
            x = np.asarray(vector, dtype=float)
            sign = np.where(np.isfinite(x) & (x < 0.0), -1.0, 1.0)
            magnitude = 1.0 + np.minimum(
                np.nan_to_num(np.abs(x), nan=100.0, posinf=100.0), 100.0
            )
            guarded = self.penalty * sign * magnitude
            return np.concatenate((guarded, np.asarray((self.penalty,), dtype=float)))


def residual_family_row_scales(
    problem: R26NodeBVP,
    jacobian: np.ndarray,
    *,
    relative_floor: float = 1.0e-10,
    absolute_floor: float = 1.0e-12,
) -> np.ndarray:
    """Equilibrate Jacobian rows without changing a physical R26 equation.

    A single global norm is unsafe for this BVP: collision-dominated bulk
    moments, wall relations, numerical boundary completion, corners, and the
    bordered mass equation have very different Jacobian magnitudes.  This
    helper assigns one robust median row norm to each component *within* each
    residual family.  The scales are fixed for a nonlinear solve and final
    acceptance remains based on the original raw residual.

    The mass-border row is treated separately because it couples every
    density degree of freedom.  Floors only prevent division by zero for an
    exactly inactive row in a diagnostic/mock problem; they do not add a row
    or alter the BVP.
    """

    matrix = np.asarray(jacobian, dtype=float)
    expected = (problem.unknown_count, problem.unknown_count)
    if matrix.shape != expected or not np.isfinite(matrix).all():
        raise ValueError(f"jacobian must be finite with shape {expected}")
    if not np.isfinite(relative_floor) or relative_floor <= 0.0:
        raise ValueError("relative_floor must be finite and positive")
    if not np.isfinite(absolute_floor) or absolute_floor <= 0.0:
        raise ValueError("absolute_floor must be finite and positive")

    norms = np.linalg.norm(matrix, axis=1).reshape(problem.shape)
    scales = np.ones(problem.shape)

    interior = norms[1:-1, 1:-1].copy()
    interior[problem.mass_j - 1, problem.mass_i - 1, 0] = np.nan
    scales[1:-1, 1:-1] = np.nanmedian(interior, axis=(0, 1))

    wall = np.median(
        np.stack([norms[node.j, node.i, :11] for node in problem.boundary_nodes]),
        axis=0,
    )
    extrapolation = np.median(
        np.stack([norms[node.j, node.i, 11:] for node in problem.boundary_nodes]),
        axis=0,
    )
    for node in problem.boundary_nodes:
        scales[node.j, node.i, :11] = wall
        scales[node.j, node.i, 11:] = extrapolation

    corners = ((0, 0), (0, -1), (-1, 0), (-1, -1))
    corner = np.median(np.stack([norms[j, i] for j, i in corners]), axis=0)
    for j, i in corners:
        scales[j, i] = corner

    scales[problem.mass_j, problem.mass_i, 0] = norms[
        problem.mass_j, problem.mass_i, 0
    ]
    maximum = float(np.nanmax(scales, initial=0.0))
    floor = max(float(absolute_floor), float(relative_floor) * maximum)
    return np.maximum(np.nan_to_num(scales.ravel(), nan=floor), floor)


def analytic_mass_jacobian_row(
    problem: R26NodeBVP,
    transform: R26StateTransform,
    encoded_state: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Return the exact bordered-mass row in logarithmic solver coordinates.

    The square BVP replaces one continuity equation by

    ``sum(weights * rho) - target_mean_density = 0``.

    Density is represented by ``log(rho)``, so the only nonzero derivatives
    are ``weights * rho / mass_scale``.  Supplying this dense border exactly
    prevents it from forcing every density column into a different finite-
    difference color while leaving the nonlinear residual unchanged.
    """

    if transform.shape != problem.shape:
        raise ValueError("transform and problem shapes must match")
    state = transform.decode(encoded_state)
    mass_row = int(
        np.ravel_multi_index((problem.mass_j, problem.mass_i, 0), problem.shape)
    )
    rho_columns = np.arange(0, problem.unknown_count, NVAR, dtype=np.int64)
    values = (
        problem.mass_weights.ravel()
        * state[..., 0].ravel()
        / float(problem.case.scaling.mass)
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("analytic mass Jacobian row is non-finite")
    return mass_row, rho_columns, values


def pseudo_transient_diagonal(
    problem: R26NodeBVP,
    transform: R26StateTransform,
    encoded_state: np.ndarray,
) -> np.ndarray:
    """Return the scaled pseudo-mass diagonal for physical bulk rows only.

    Smooth-wall, extrapolation, corner, and global-mass equations remain
    algebraic constraints.  On the interior, the diagonal represents
    ``dU/d(encoded U)`` divided by the audited bulk residual scaling.  Hence
    logarithmic density and temperature coordinates receive factors ``rho``
    and ``T`` rather than an arbitrary identity shift.
    """

    if transform.shape != problem.shape:
        raise ValueError("transform and problem shapes must match")
    state = transform.decode(encoded_state)
    coordinate_derivative = np.ones(problem.shape)
    coordinate_derivative[..., 0] = state[..., 0]
    coordinate_derivative[..., 3] = state[..., 3]
    bulk_scale = np.asarray(problem.case.scaling.bulk, dtype=float)
    if bulk_scale.shape != (NVAR,) or not np.isfinite(bulk_scale).all() or np.any(bulk_scale <= 0.0):
        raise ValueError("bulk residual scaling must contain 17 positive finite entries")
    diagonal = np.zeros(problem.shape)
    diagonal[1:-1, 1:-1] = coordinate_derivative[1:-1, 1:-1] / bulk_scale
    diagonal[problem.mass_j, problem.mass_i, 0] = 0.0
    return diagonal.ravel()


def physical_pseudo_transient_matrix(
    problem: R26NodeBVP,
    transform: R26StateTransform,
    encoded_state: np.ndarray,
) -> csc_matrix:
    """Return the physical pseudo-mass chain rule in solver coordinates.

    The ordinary logarithmic state transform is point-local, so its exact
    pseudo-mass matrix is diagonal.  Gu--Emerson coordinates are different:
    reconstructing ``sigma, q, m, R, Delta`` from ``g, h, omega, gamma, chi``
    contains first and second spatial derivatives.  Consequently
    ``d(physical state)/d(transformed state)`` is sparse but not diagonal.

    This routine evaluates that chain rule by colored finite differences of
    the coordinate map only.  No R26 residual is evaluated and no physical
    equation is modified.  Boundary, corner, and bordered-mass rows remain
    algebraic; only the 17 interior physical evolution rows receive the
    pseudo-time mass matrix.  Letting the pseudo-time step grow removes this
    matrix exactly and leaves the original steady root as the sole target.
    """

    if transform.shape != problem.shape:
        raise ValueError("transform and problem shapes must match")
    if not bool(getattr(transform, "supports_physical_pseudo_transient", False)):
        raise ValueError(
            "selected state transform does not define the physical pseudo-time chain rule"
        )
    encoded = np.asarray(encoded_state, dtype=float)
    if encoded.shape != (problem.unknown_count,) or not np.isfinite(encoded).all():
        raise ValueError("encoded pseudo-transient state has incorrect shape")

    # Preserve the inexpensive exact diagonal used by the historical physical
    # coordinates.  Alternative transforms use the complete sparse chain rule
    # below rather than pretending their gradient reconstruction is local.
    if isinstance(transform, LogStateTransform):
        return diags(
            pseudo_transient_diagonal(problem, transform, encoded),
            format="csc",
        )

    radius = int(
        getattr(transform, "physical_pseudo_transient_stencil_radius", 0)
    )
    if radius < 2:
        raise ValueError(
            "nonlocal state transform must declare a pseudo-transient stencil radius >= 2"
        )
    bulk_scale = np.asarray(problem.case.scaling.bulk, dtype=float)
    if (
        bulk_scale.shape != (NVAR,)
        or not np.isfinite(bulk_scale).all()
        or np.any(bulk_scale <= 0.0)
    ):
        raise ValueError("bulk residual scaling must contain 17 positive finite entries")

    def scaled_physical_pseudo_state(vector: np.ndarray) -> np.ndarray:
        state = transform.decode(vector)
        values = np.zeros(problem.shape)
        values[1:-1, 1:-1] = state[1:-1, 1:-1] / bulk_scale
        values[problem.mass_j, problem.mass_i, 0] = 0.0
        return values.ravel()

    lower, upper = transform.least_squares_bounds()
    pattern = jacobian_sparsity(
        problem,
        stencil_radius=radius,
        include_mass_border=False,
    )
    base = scaled_physical_pseudo_state(encoded)
    matrix = approx_derivative(
        scaled_physical_pseudo_state,
        encoded,
        method="2-point",
        abs_step=fv_absolute_difference_step(encoded),
        bounds=(lower, upper),
        sparsity=pattern,
        f0=base,
    ).tocsc()
    if matrix.shape != (problem.unknown_count, problem.unknown_count):
        raise RuntimeError("pseudo-transient chain-rule matrix has incorrect shape")
    if not np.isfinite(matrix.data).all():
        raise FloatingPointError("pseudo-transient chain-rule matrix is non-finite")
    return matrix


def secant_predict_state(
    problem: R26NodeBVP,
    previous_state: np.ndarray,
    current_state: np.ndarray,
    *,
    previous_parameter: float,
    current_parameter: float,
    target_parameter: float,
    maximum_extrapolation: float = 2.0,
) -> np.ndarray:
    """Extrapolate two accepted roots in encoded coordinates.

    Logarithmic rho/T coordinates preserve positivity.  The predicted density
    is then renormalized with the problem's exact quadrature weights, so the
    global mass border is satisfied before the nonlinear solve begins.
    """

    values = (previous_parameter, current_parameter, target_parameter)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("continuation parameters must be finite")
    spacing = float(current_parameter - previous_parameter)
    advance = float(target_parameter - current_parameter)
    if spacing <= 0.0 or advance < 0.0:
        raise ValueError("secant predictor requires monotone accepted parameters")
    factor = advance / spacing
    if not np.isfinite(maximum_extrapolation) or maximum_extrapolation <= 0.0:
        raise ValueError("maximum_extrapolation must be finite and positive")
    if factor > maximum_extrapolation:
        raise ValueError(
            f"secant extrapolation factor {factor:.6g} exceeds {maximum_extrapolation:.6g}"
        )
    transform = LogStateTransform(problem.shape)
    previous = transform.encode(previous_state)
    current = transform.encode(current_state)
    lower, upper = transform.least_squares_bounds()
    predicted = transform.decode(np.clip(current + factor * (current - previous), lower, upper))
    mean_density = problem.mean_density(predicted)
    if not np.isfinite(mean_density) or mean_density <= 0.0:
        raise FloatingPointError("secant predictor produced invalid mean density")
    predicted[..., 0] *= problem.case.mean_density / mean_density
    return validate_planar_state(predicted)


def jacobian_sparsity(
    problem: R26NodeBVP,
    *,
    stencil_radius: int = 2,
    include_mass_border: bool = True,
) -> csr_matrix:
    """Conservative dependency pattern for colored finite differences.

    The raw R26 rows contain a derivative of a closure that itself contains
    first derivatives, hence a radius-two dependency is required.  The mass
    border row additionally depends on every density degree of freedom.  This
    is a sparsity declaration only; it does not approximate or omit a term in
    the residual.
    """

    if stencil_radius < 2:
        raise ValueError("R26 closure derivatives require stencil_radius >= 2")
    ny, nx, nv = problem.shape
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    for j in range(ny):
        j0, j1 = max(0, j - stencil_radius), min(ny, j + stencil_radius + 1)
        for i in range(nx):
            i0, i1 = max(0, i - stencil_radius), min(nx, i + stencil_radius + 1)
            output_rows = (np.ravel_multi_index((j, i, 0), problem.shape) + np.arange(nv))
            neighbour_columns: list[int] = []
            for jj in range(j0, j1):
                for ii in range(i0, i1):
                    base = int(np.ravel_multi_index((jj, ii, 0), problem.shape))
                    neighbour_columns.extend(range(base, base + nv))
            cols = np.asarray(neighbour_columns, dtype=np.int64)
            rows.append(np.repeat(output_rows, cols.size))
            columns.append(np.tile(cols, output_rows.size))
    # The continuity row replaced by the integral mass constraint sees every
    # rho node, including all four explicitly modelled corners.  When the row
    # is supplied analytically, remove it completely from the finite-
    # difference pattern so it cannot serialize the density colors.
    mass_row = int(np.ravel_multi_index((problem.mass_j, problem.mass_i, 0), problem.shape))
    rho_columns = np.arange(0, problem.unknown_count, nv, dtype=np.int64)
    if include_mass_border:
        rows.append(np.full(rho_columns.shape, mass_row, dtype=np.int64))
        columns.append(rho_columns)
    row = np.concatenate(rows)
    column = np.concatenate(columns)
    data = np.ones(row.size, dtype=bool)
    pattern = coo_matrix(
        (data, (row, column)),
        shape=(problem.unknown_count, problem.unknown_count),
    ).tocsr()
    if not include_mass_border:
        pattern = pattern.tolil()
        pattern[mass_row, :] = False
        pattern = pattern.tocsr()
        pattern.eliminate_zeros()
    return pattern


def solve_r26_bvp(
    problem: R26NodeBVP,
    initial_state: np.ndarray,
    *,
    options: SolveOptions | None = None,
    state_transform: R26StateTransform | None = None,
) -> R26SolveResult:
    """Solve one fixed case from one explicit initial state.

    Algebraic acceptance additionally requires the continuity equation removed
    for the mass border to remain small; optimizer success by itself is never
    reported as R26 convergence.
    """

    options = SolveOptions() if options is None else options
    transform = (
        LogStateTransform(problem.shape)
        if state_transform is None
        else state_transform
    )
    if transform.shape != problem.shape:
        raise ValueError("state transform and R26 problem shapes must match")
    if options.pseudo_transient and not bool(
        getattr(transform, "supports_physical_pseudo_transient", False)
    ):
        raise ValueError(
            "selected state transform does not define the physical pseudo-time diagonal"
        )
    x0 = transform.encode(initial_state)
    objective = EncodedR26Objective(problem, transform, options.invalid_penalty)
    jacobian_evaluations = 0
    pseudo_transient_steps = 0
    final_pseudo_time_step: float | None = None

    if options.method == "krylov":
        scipy_result: OptimizeResult = root(
            objective,
            x0,
            method="krylov",
            options={
                "fatol": options.residual_tolerance,
                "maxiter": options.max_iterations,
                "line_search": options.line_search,
                "disp": options.display,
            },
        )
        encoded = np.asarray(scipy_result.x, dtype=float)
        iterations = int(getattr(scipy_result, "nit", 0))
        evaluations = int(getattr(scipy_result, "nfev", 0))
    elif options.method == "least_squares":
        lower, upper = transform.least_squares_bounds()
        # On a very small grid the radius-two pattern is effectively dense
        # and a direct dense trust-region solve is faster/more accurate.  From
        # N=7 onward colored finite differences avoid the quadratic number of
        # residual calls.
        sparsity = (
            None
            if problem.case.nodes <= 6
            else jacobian_sparsity(
                problem, stencil_radius=options.jacobian_stencil_radius
            )
        )
        result = least_squares(
            objective,
            x0,
            bounds=(lower, upper),
            method="trf",
            jac="2-point",
            jac_sparsity=sparsity,
            x_scale="jac",
            ftol=options.residual_tolerance,
            xtol=options.residual_tolerance,
            gtol=options.residual_tolerance,
            max_nfev=options.max_function_evaluations,
            verbose=2 if options.display else 0,
        )
        scipy_result = result
        encoded = np.asarray(result.x, dtype=float)
        iterations = int(getattr(result, "njev", 0) or 0)
        evaluations = int(result.nfev)
    else:
        # The steady equations are unchanged.  Optional SER pseudo-transient
        # continuation only globalizes the route to the same algebraic root;
        # it shifts physical bulk rows while wall, corner, and mass rows stay
        # algebraic.  Once the residual enters the declared Newton basin the
        # shift is removed exactly and the final root is polished by Newton.
        lower, upper = transform.least_squares_bounds()
        pattern = jacobian_sparsity(
            problem,
            stencil_radius=options.jacobian_stencil_radius,
            include_mass_border=not options.analytic_mass_jacobian,
        )
        encoded = x0.copy()
        evaluations = 0

        def counted(vector: np.ndarray) -> np.ndarray:
            nonlocal evaluations
            if (
                options.max_objective_evaluations is not None
                and evaluations >= options.max_objective_evaluations
            ):
                raise _ObjectiveEvaluationLimitReached
            evaluations += 1
            return objective(vector)

        residual = counted(encoded)
        raw_linf = objective.last_raw_linf
        success = False
        message = "colored sparse Newton iteration limit reached"
        iterations = 0
        jacobian = None
        newton_factorization = None
        chord_steps = 0
        force_jacobian_refresh = False
        pseudo_time_step = float(options.pseudo_time_initial)
        try:
            for iteration in range(1, options.max_iterations + 1):
                iterations = iteration
                residual_linf = float(np.max(np.abs(residual), initial=0.0))
                if options.display:
                    print(
                        "R26_NEWTON "
                        f"iteration={iteration} residual_linf={residual_linf:.16e} "
                        f"raw_linf={raw_linf:.16e} "
                        f"jacobians={jacobian_evaluations} evaluations={evaluations} "
                        f"pseudo_time={pseudo_time_step:.16e}",
                        flush=True,
                    )
                if residual_linf <= options.residual_tolerance:
                    success = True
                    message = (
                        "SER pseudo-transient/Newton residual tolerance reached"
                        if options.pseudo_transient
                        else "colored sparse Newton residual tolerance reached"
                    )
                    break
                use_pseudo_transient = bool(
                    options.pseudo_transient
                    and residual_linf > options.newton_switch_tolerance
                )
                chord_limit = 12 if use_pseudo_transient else 3
                if (
                    jacobian is None
                    or force_jacobian_refresh
                    or chord_steps >= chord_limit
                ):
                    if (
                        options.max_jacobian_evaluations is not None
                        and jacobian_evaluations >= options.max_jacobian_evaluations
                    ):
                        message = (
                            "colored sparse Newton Jacobian-evaluation limit reached "
                            f"({options.max_jacobian_evaluations})"
                        )
                        break

                    # R26 high-order moments can be many orders of magnitude
                    # smaller than unity.  A relative-only perturbation rounds
                    # away there, so retain the audited absolute perturbation.
                    derivative_kwargs: dict[str, object] = {
                        "method": "2-point",
                        "abs_step": fv_absolute_difference_step(encoded),
                        "bounds": (lower, upper),
                        "sparsity": pattern,
                    }
                    if options.analytic_mass_jacobian:
                        mass_row, rho_columns, mass_values = analytic_mass_jacobian_row(
                            problem, transform, encoded
                        )

                        def counted_without_mass_border(vector: np.ndarray) -> np.ndarray:
                            values = counted(vector).copy()
                            values[mass_row] = 0.0
                            return values

                        finite_difference_base = residual.copy()
                        finite_difference_base[mass_row] = 0.0
                        derivative_kwargs["f0"] = finite_difference_base
                        finite_difference_jacobian = approx_derivative(
                            counted_without_mass_border,
                            encoded,
                            **derivative_kwargs,
                        ).tolil()
                        finite_difference_jacobian[mass_row, :] = 0.0
                        finite_difference_jacobian[mass_row, rho_columns] = mass_values
                        jacobian = finite_difference_jacobian.tocsc()
                    else:
                        jacobian = approx_derivative(
                            counted,
                            encoded,
                            **derivative_kwargs,
                        ).tocsc()
                    jacobian_evaluations += 1
                    if options.display:
                        print(
                            "R26_JACOBIAN "
                            f"count={jacobian_evaluations} evaluations={evaluations}",
                            flush=True,
                        )
                    newton_factorization = None
                    chord_steps = 0
                    force_jacobian_refresh = False

                assert jacobian is not None
                linear_matrix = jacobian
                factorization = None
                if use_pseudo_transient:
                    pseudo_matrix = physical_pseudo_transient_matrix(
                        problem, transform, encoded
                    )
                    linear_matrix = (
                        jacobian + pseudo_matrix / pseudo_time_step
                    ).tocsc()
                    try:
                        factorization = splu(linear_matrix)
                    except RuntimeError:
                        factorization = None
                else:
                    if newton_factorization is None:
                        try:
                            newton_factorization = splu(jacobian)
                        except RuntimeError:
                            newton_factorization = None
                    factorization = newton_factorization
                try:
                    if factorization is None:
                        raise RuntimeError("sparse LU unavailable")
                    direction = factorization.solve(-residual)
                    linear_solver = "splu-ptc" if use_pseudo_transient else "splu"
                except RuntimeError:
                    direction = lsmr(
                        linear_matrix,
                        -residual,
                        atol=1.0e-12,
                        btol=1.0e-12,
                    )[0]
                    linear_solver = (
                        "lsmr-ptc-fallback"
                        if use_pseudo_transient
                        else "lsmr-fallback"
                    )
                if not np.isfinite(direction).all():
                    message = f"{linear_solver} produced a non-finite Newton direction"
                    break
                merit = 0.5 * float(np.dot(residual, residual))
                old_residual_linf = residual_linf
                alpha = 1.0
                accepted_step = False
                rejected_small_alpha = False
                while alpha >= 2.0**-20:
                    trial = np.clip(encoded + alpha * direction, lower, upper)
                    trial_residual = counted(trial)
                    trial_raw_linf = objective.last_raw_linf
                    trial_merit = 0.5 * float(np.dot(trial_residual, trial_residual))
                    sufficient_decrease = (
                        trial_merit < merit
                        if use_pseudo_transient
                        else trial_merit < merit * (1.0 - 1.0e-4 * alpha)
                    )
                    raw_sufficient_decrease = bool(
                        not options.require_raw_linf_decrease
                        or trial_raw_linf
                        < raw_linf * (1.0 - 1.0e-4 * alpha)
                    )
                    acceptable_merit = bool(
                        np.isfinite(trial_merit)
                        and sufficient_decrease
                        and raw_sufficient_decrease
                    )
                    if (
                        acceptable_merit
                        and use_pseudo_transient
                        and alpha < options.pseudo_time_minimum_accepted_alpha
                    ):
                        rejected_small_alpha = True
                        break
                    if acceptable_merit:
                        encoded = trial
                        residual = trial_residual
                        raw_linf = trial_raw_linf
                        accepted_step = True
                        chord_steps += 1
                        if use_pseudo_transient:
                            pseudo_transient_steps += 1
                            new_residual_linf = float(
                                np.max(np.abs(residual), initial=0.0)
                            )
                            ser_ratio = old_residual_linf / max(
                                new_residual_linf,
                                np.finfo(float).tiny,
                            )
                            growth = min(
                                options.pseudo_time_growth_limit,
                                max(
                                    0.25,
                                    ser_ratio ** options.pseudo_time_ser_exponent,
                                ),
                            )
                            pseudo_time_step = float(
                                np.clip(
                                    pseudo_time_step * growth,
                                    options.pseudo_time_minimum,
                                    options.pseudo_time_maximum,
                                )
                            )
                        if options.display:
                            print(
                                "R26_STEP accepted=true "
                                f"alpha={alpha:.16e} "
                                f"residual_linf={float(np.max(np.abs(residual), initial=0.0)):.16e} "
                                f"raw_linf={raw_linf:.16e} "
                                f"linear_solver={linear_solver} "
                                f"pseudo_transient={str(use_pseudo_transient).lower()} "
                                f"pseudo_time={pseudo_time_step:.16e} "
                                f"invalid_evaluations={objective.invalid_evaluations}",
                                flush=True,
                            )
                        if (
                            (not use_pseudo_transient and trial_merit > 0.25 * merit)
                            or (use_pseudo_transient and trial_merit > 0.95 * merit)
                        ):
                            force_jacobian_refresh = True
                        break
                    alpha *= 0.5
                if not accepted_step:
                    if options.display:
                        print(
                            "R26_STEP accepted=false "
                            f"reason={'small_alpha' if rejected_small_alpha else 'merit'} "
                            f"linear_solver={linear_solver} "
                            f"pseudo_transient={str(use_pseudo_transient).lower()} "
                            f"pseudo_time={pseudo_time_step:.16e} "
                            f"raw_linf={raw_linf:.16e} "
                            f"invalid_evaluations={objective.invalid_evaluations}",
                            flush=True,
                        )
                    if use_pseudo_transient:
                        # At a physical/algebraic R26 boundary, decreasing the
                        # pseudo-time step does not merely shorten the trial
                        # step: it changes the direction toward a singular DAE
                        # limit because wall, corner and mass rows carry no
                        # pseudo-time term.  If a raw-decreasing step exists but
                        # only below the declared alpha floor, move toward the
                        # steady Newton direction instead.  The N32 audit that
                        # motivated this branch found alpha=1/32 at dt=1e-2,
                        # then lost every descent direction after dt was
                        # incorrectly reduced to 6.25e-4 and below.
                        if rejected_small_alpha:
                            if pseudo_time_step < options.pseudo_time_maximum:
                                pseudo_time_step = min(
                                    options.pseudo_time_maximum,
                                    options.pseudo_time_small_alpha_growth
                                    * pseudo_time_step,
                                )
                                continue
                            message = (
                                "SER pseudo-transient step remained below the alpha "
                                f"floor at maximum pseudo-time after {linear_solver}"
                            )
                            break
                        # A failed chord direction after one or more accepted
                        # steps is evidence about the stale Jacobian, not the
                        # pseudo-time scale.  Refresh at the unchanged state
                        # before changing dt; otherwise the same obsolete
                        # direction is retried all the way to the DAE limit.
                        if chord_steps > 0:
                            force_jacobian_refresh = True
                            newton_factorization = None
                            chord_steps = 0
                            continue
                        if options.require_raw_linf_decrease:
                            if pseudo_time_step < options.pseudo_time_maximum:
                                pseudo_time_step = min(
                                    options.pseudo_time_maximum,
                                    options.pseudo_time_small_alpha_growth
                                    * pseudo_time_step,
                                )
                                continue
                            message = (
                                "SER pseudo-transient found no raw-decreasing "
                                f"step at maximum pseudo-time after {linear_solver}"
                            )
                            break
                        if pseudo_time_step > options.pseudo_time_minimum:
                            pseudo_time_step = max(
                                options.pseudo_time_minimum,
                                0.25 * pseudo_time_step,
                            )
                            continue
                        message = (
                            "SER pseudo-transient line search failed at minimum "
                            f"pseudo-time step after {linear_solver}"
                        )
                        break
                    if chord_steps > 0:
                        force_jacobian_refresh = True
                        newton_factorization = None
                        chord_steps = 0
                        continue
                    message = f"colored sparse Newton line search failed after {linear_solver}"
                    break
            else:
                residual_linf = float(np.max(np.abs(residual), initial=0.0))
                if residual_linf <= options.residual_tolerance:
                    success = True
                    message = (
                        "SER pseudo-transient/Newton residual tolerance reached"
                        if options.pseudo_transient
                        else "colored sparse Newton residual tolerance reached"
                    )
        except _ObjectiveEvaluationLimitReached:
            message = (
                "colored sparse Newton objective-evaluation limit reached "
                f"({options.max_objective_evaluations})"
            )
        final_pseudo_time_step = (
            pseudo_time_step if options.pseudo_transient else None
        )
        scipy_result = OptimizeResult(
            x=encoded,
            success=success,
            message=message,
            nit=iterations,
            nfev=evaluations,
        )

    try:
        state = transform.decode(encoded)
        evaluation = problem.evaluate(state)
        diagnostics = evaluation.diagnostics
        physical_final = True
    except (FloatingPointError, ValueError) as exc:
        state = np.asarray(initial_state, dtype=float).copy()
        diagnostics = problem.evaluate(state).diagnostics
        physical_final = False
        objective.last_invalid_error = f"final state invalid: {type(exc).__name__}: {exc}"

    converged = bool(
        physical_final
        and diagnostics.total_linf <= options.residual_tolerance
        and abs(diagnostics.held_out_continuity) <= options.held_out_continuity_tolerance
    )
    message = str(getattr(scipy_result, "message", ""))
    if bool(getattr(scipy_result, "success", False)) and not converged:
        message += (
            "; optimizer stopped but strict R26 acceptance failed "
            f"(residual={diagnostics.total_linf:.3e}, "
            f"held-out continuity={diagnostics.held_out_continuity:.3e})"
        )
    return R26SolveResult(
        case=problem.case,
        state=state,
        encoded_state=encoded,
        diagnostics=diagnostics,
        converged=converged,
        scipy_success=bool(getattr(scipy_result, "success", False)),
        message=message,
        iterations=iterations,
        function_evaluations=evaluations,
        invalid_evaluations=objective.invalid_evaluations,
        last_invalid_error=objective.last_invalid_error,
        solver_method=options.method,
        jacobian_evaluations=jacobian_evaluations,
        pseudo_transient_steps=pseudo_transient_steps,
        final_pseudo_time_step=final_pseudo_time_step,
    )


def interpolate_state_grid(
    state: np.ndarray,
    new_nodes: int,
    *,
    target_mean_density: float = 1.0,
    mass_weights: np.ndarray | None = None,
    old_x: np.ndarray | None = None,
    old_y: np.ndarray | None = None,
    new_x: np.ndarray | None = None,
    new_y: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate a square restart, preserving rho/T positivity and mass.

    Coordinate arrays make refinement between uniform and wall-stretched
    grids explicit.  Omitting all four arrays retains the legacy unit-square
    uniform behavior.
    """

    old = validate_planar_state(state)
    if old.ndim != 3 or old.shape[0] != old.shape[1]:
        raise ValueError("restart interpolation expects a square state grid")
    if new_nodes < 5:
        raise ValueError("new grid needs at least five nodes")
    old_nodes = old.shape[0]
    old_xv = np.linspace(0.0, 1.0, old_nodes) if old_x is None else np.asarray(old_x, dtype=float)
    old_yv = np.linspace(0.0, 1.0, old_nodes) if old_y is None else np.asarray(old_y, dtype=float)
    new_xv = np.linspace(0.0, 1.0, new_nodes) if new_x is None else np.asarray(new_x, dtype=float)
    new_yv = np.linspace(0.0, 1.0, new_nodes) if new_y is None else np.asarray(new_y, dtype=float)
    for name, coordinate, size in (
        ("old_x", old_xv, old_nodes),
        ("old_y", old_yv, old_nodes),
        ("new_x", new_xv, new_nodes),
        ("new_y", new_yv, new_nodes),
    ):
        if coordinate.shape != (size,) or not np.isfinite(coordinate).all() or np.any(np.diff(coordinate) <= 0.0):
            raise ValueError(f"{name} must be finite, increasing, and have length {size}")
    if new_xv[0] < old_xv[0] or new_xv[-1] > old_xv[-1] or new_yv[0] < old_yv[0] or new_yv[-1] > old_yv[-1]:
        raise ValueError("new grid must remain within the old interpolation domain")
    yy, xx = np.meshgrid(new_yv, new_xv, indexing="ij")
    points = np.column_stack((yy.ravel(), xx.ravel()))
    result = np.empty((new_nodes, new_nodes, NVAR))
    for component in range(NVAR):
        values = old[..., component]
        logarithmic = component in (0, 3)
        if logarithmic:
            values = np.log(values)
        interpolator = RegularGridInterpolator(
            (old_yv, old_xv), values, method="linear", bounds_error=True
        )
        interpolated = interpolator(points).reshape(new_nodes, new_nodes)
        result[..., component] = np.exp(interpolated) if logarithmic else interpolated
    if mass_weights is None:
        weights = trapezoidal_node_weights(new_nodes)
    else:
        weights = np.asarray(mass_weights, dtype=float)
        if weights.shape != (new_nodes, new_nodes):
            raise ValueError("mass_weights must match the refined node grid")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("mass_weights must be finite and nonnegative")
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            raise ValueError("mass_weights must have positive total weight")
        weights = weights / total_weight
    current_mass = float(np.sum(weights * result[..., 0]))
    result[..., 0] *= target_mean_density / current_mass
    return validate_planar_state(result)


def solve_lid_continuation(
    case: CavityCase,
    *,
    steps: int,
    problem_factory: Callable[[CavityCase], R26NodeBVP] = R26NodeBVP,
    initial_state: np.ndarray | None = None,
    options: SolveOptions | None = None,
) -> tuple[R26SolveResult, ...]:
    """Run independent stateless solves along a monotone lid-speed ladder."""

    if steps < 1:
        raise ValueError("continuation steps must be positive")
    state = case.equilibrium_state() if initial_state is None else validate_planar_state(initial_state)
    speeds = np.linspace(0.0, case.lid_velocity, steps + 1)[1:]
    results: list[R26SolveResult] = []
    for index, speed in enumerate(speeds, start=1):
        step_case = replace(case, name=f"{case.name}-cont{index:03d}", lid_velocity=float(speed))
        result = solve_r26_bvp(problem_factory(step_case), state, options=options)
        results.append(result)
        if not result.converged:
            break
        state = result.state.copy()
    return tuple(results)


def save_restart(path: str | Path, result: R26SolveResult) -> Path:
    """Write a private NPZ restart with enough metadata to reject wrong cases."""

    output = Path(path)
    metadata = {
        "case_name": result.case.name,
        "nodes": result.case.nodes,
        "kn": result.case.kn,
        "kn_convention": result.case.kn_convention.value,
        "lid_velocity": result.case.lid_velocity,
        "grid_stretch_beta": result.case.grid_stretch_beta,
        "x": result.case.x.tolist(),
        "y": result.case.y.tolist(),
        "wall_temperature": result.case.wall_temperature,
        "converged": result.converged,
        "solver_method": result.solver_method,
    }
    np.savez_compressed(output, state=result.state, metadata=np.asarray(json.dumps(metadata)))
    return output


def load_restart(path: str | Path) -> tuple[np.ndarray, dict[str, object]]:
    """Load and validate a private NPZ restart without hidden solver history."""

    with np.load(Path(path), allow_pickle=False) as archive:
        state = validate_planar_state(np.asarray(archive["state"], dtype=float))
        metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
    if state.ndim != 3 or state.shape[-1] != NVAR:
        raise ValueError("restart contains an invalid R26 state shape")
    return state, metadata


__all__ = [
    "analytic_mass_jacobian_row",
    "EncodedR26Objective",
    "EncodedR26MassContinuityObjective",
    "EncodedR26RawMassContinuityObjective",
    "LogStateTransform",
    "R26StateTransform",
    "R26SolveResult",
    "SolveOptions",
    "interpolate_state_grid",
    "jacobian_sparsity",
    "load_restart",
    "physical_pseudo_transient_matrix",
    "pseudo_transient_diagonal",
    "residual_family_row_scales",
    "save_restart",
    "secant_predict_state",
    "solve_lid_continuation",
    "solve_r26_bvp",
]
