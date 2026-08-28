#!/usr/bin/env python3
"""Pressure-based nonlinear path for the compatible finite-volume R26 BVP.

This module does not copy the legacy Code_Saturne/THOR field equations.  It
uses the independently audited Gu--Emerson R26 residual as the nonlinear
oracle and supplies the numerical ingredients that made the pressure-based
codes robust:

* CUBISTA convection through :func:`r26_fv_backend.thor_fv_bulk_residual`;
* collocated Rhie--Chow face fluxes;
* a finite-volume SIMPLE pressure-correction solve with an exact zero-mean
  gauge and impermeable-wall Neumann flux;
* field-family collision/diffusion diagonals as a Krylov preconditioner; and
* final fail-closed checks against the unscaled R26 residual, held continuity,
  mass, density, and temperature.

The SIMPLE operator is only a nonlinear-solver preconditioner.  It cannot
change the accepted physical root because every outer residual evaluation and
the final gate use the complete R26 boundary-value problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.optimize import OptimizeResult, root
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import bmat, coo_matrix, csc_matrix
from scipy.sparse.linalg import LinearOperator, spilu, splu

from r26_cases import CavityCase
from r26_discretization import R26NodeBVP
from r26_fv_backend import (
    fv_absolute_difference_step,
    impermeable_wall_mass_divergence,
    interior_control_volume_widths,
    rhie_chow_inverse_momentum_diagonal,
    thor_fv_bulk_residual,
    wall_bounded_control_volume_weights,
)
from r26_solver import (
    EncodedR26Objective,
    LogStateTransform,
    R26SolveResult,
    jacobian_sparsity,
)
from r26_state import NVAR


THOR_SOLVER_PROVENANCE: Final[str] = (
    "Gu--Emerson JFM 636 Sec. 5.2 CUBISTA/SIMPLE/Rhie--Chow architecture; "
    "legacy Rana Code_Saturne R13/R26 field-by-field source layout; "
    "audited Python R26 residual retained as the nonlinear and acceptance oracle"
)


@dataclass(frozen=True)
class ThorSolveOptions:
    """Bounded controls for the pressure-preconditioned nonlinear solve."""

    residual_tolerance: float = 1.0e-9
    raw_tolerance: float = 1.0e-8
    held_out_continuity_tolerance: float = 1.0e-8
    mass_tolerance: float = 1.0e-10
    max_iterations: int = 80
    inner_max_iterations: int = 80
    line_search: str = "armijo"
    display: bool = False
    invalid_penalty: float = 1.0e8
    velocity_relaxation: float = 0.7
    pressure_relaxation: float = 0.3
    thermal_relaxation: float = 0.6
    moment_relaxation: float = 0.5
    boundary_relaxation: float = 0.25
    ilu_drop_tolerance: float = 1.0e-5
    ilu_fill_factor: float = 24.0

    def __post_init__(self) -> None:
        tolerances = (
            self.residual_tolerance,
            self.raw_tolerance,
            self.held_out_continuity_tolerance,
            self.mass_tolerance,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in tolerances):
            raise ValueError("THOR tolerances must be finite and positive")
        if self.max_iterations < 1 or self.inner_max_iterations < 1:
            raise ValueError("THOR iteration limits must be positive")
        if self.line_search not in {"armijo", "wolfe"}:
            raise ValueError("THOR line_search must be armijo or wolfe")
        relaxations = (
            self.velocity_relaxation,
            self.pressure_relaxation,
            self.thermal_relaxation,
            self.moment_relaxation,
            self.boundary_relaxation,
        )
        if not all(np.isfinite(value) and 0.0 < value <= 1.0 for value in relaxations):
            raise ValueError("THOR relaxation factors must lie in (0,1]")
        if not np.isfinite(self.ilu_drop_tolerance) or self.ilu_drop_tolerance <= 0.0:
            raise ValueError("ILU drop tolerance must be finite and positive")
        if not np.isfinite(self.ilu_fill_factor) or self.ilu_fill_factor < 1.0:
            raise ValueError("ILU fill factor must be finite and at least one")


@dataclass(frozen=True)
class ThorSolveResult:
    """Strict result plus pressure-preconditioner work diagnostics."""

    solution: R26SolveResult
    raw_acceptance_gate: float
    raw_gate_passed: bool
    pressure_factorizations: int
    preconditioner_applications: int
    frozen_jacobian_residual_evaluations: int
    frozen_jacobian_nonzeros: int
    ilu_available: bool
    provenance: str = THOR_SOLVER_PROVENANCE


def make_thor_problem(case: CavityCase) -> R26NodeBVP:
    """Build the CUBISTA/Rhie--Chow BVP with matching FV mass weights."""

    return R26NodeBVP(
        case,
        bulk_operator=thor_fv_bulk_residual,
        mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
    )


def _row_scales(problem: R26NodeBVP) -> np.ndarray:
    """Reconstruct the exact raw-to-scaled row factors of ``R26NodeBVP``."""

    case = problem.case
    scales = np.empty(problem.shape, dtype=float)
    scales[1:-1, 1:-1] = case.scaling.bulk
    for node in problem.boundary_nodes:
        scales[node.j, node.i, :11] = case.scaling.wall
        scales[node.j, node.i, 11:] = case.scaling.extrapolation
    for j, i in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        scales[j, i] = case.scaling.corner
    scales[problem.mass_j, problem.mass_i, 0] = case.scaling.mass
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise RuntimeError("failed to reconstruct positive R26 row scales")
    return scales


def _pressure_correction_matrix(
    rho: np.ndarray,
    d_cell: np.ndarray | tuple[np.ndarray, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[csc_matrix, np.ndarray]:
    """Assemble ``-div(rho*d*grad)`` plus a zero-mean pressure gauge.

    The operator is assembled in integrated finite-volume form, so every
    interior face contributes an equal and opposite coefficient to its two
    neighbouring control volumes.  Physical-wall flux is exactly zero.  A
    scalar cell coefficient preserves the historical isotropic preconditioner;
    a ``(d_x, d_y)`` pair supplies the component-wise momentum coefficients
    required by a segregated SIMPLE solve.
    """

    ny, nx = rho.shape
    if isinstance(d_cell, tuple):
        if len(d_cell) != 2:
            raise ValueError("anisotropic SIMPLE coefficients require (d_x, d_y)")
        d_x = np.asarray(d_cell[0], dtype=float)
        d_y = np.asarray(d_cell[1], dtype=float)
    else:
        d_x = np.asarray(d_cell, dtype=float)
        d_y = d_x
    if (
        d_x.shape != rho.shape
        or d_y.shape != rho.shape
        or not np.isfinite(d_x).all()
        or not np.isfinite(d_y).all()
        or np.any(d_x <= 0.0)
        or np.any(d_y <= 0.0)
    ):
        raise FloatingPointError(
            "SIMPLE inverse momentum diagonals must match rho and be finite positive"
        )
    ni = nx - 2
    nj = ny - 2
    count = ni * nj
    dx = interior_control_volume_widths(x)
    dy = interior_control_volume_widths(y)
    volumes = dy[:, None] * dx[None, :]

    def index(j: int, i: int) -> int:
        return j * ni + i

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diagonal = np.zeros(count, dtype=float)

    for j in range(nj):
        gj = j + 1
        for i in range(ni - 1):
            gi = i + 1
            west = index(j, i)
            east = index(j, i + 1)
            conductivity = 0.5 * (
                rho[gj, gi] * d_x[gj, gi]
                + rho[gj, gi + 1] * d_x[gj, gi + 1]
            )
            coefficient = conductivity * dy[j] / (x[gi + 1] - x[gi])
            diagonal[west] += coefficient
            diagonal[east] += coefficient
            rows.extend((west, east))
            cols.extend((east, west))
            data.extend((-coefficient, -coefficient))

    for j in range(nj - 1):
        gj = j + 1
        for i in range(ni):
            gi = i + 1
            south = index(j, i)
            north = index(j + 1, i)
            conductivity = 0.5 * (
                rho[gj, gi] * d_y[gj, gi]
                + rho[gj + 1, gi] * d_y[gj + 1, gi]
            )
            coefficient = conductivity * dx[i] / (y[gj + 1] - y[gj])
            diagonal[south] += coefficient
            diagonal[north] += coefficient
            rows.extend((south, north))
            cols.extend((north, south))
            data.extend((-coefficient, -coefficient))

    rows.extend(range(count))
    cols.extend(range(count))
    data.extend(diagonal.tolist())
    operator = coo_matrix((data, (rows, cols)), shape=(count, count)).tocsc()
    gauge = volumes.ravel()
    augmented = bmat(
        (
            (operator, csc_matrix(gauge[:, None])),
            (csc_matrix(gauge[None, :]), csc_matrix((1, 1))),
        ),
        format="csc",
    )
    return augmented, volumes


class SimpleR26Preconditioner(LinearOperator):
    """Field-family inverse with a conservative SIMPLE pressure correction."""

    def __init__(
        self,
        problem: R26NodeBVP,
        transform: LogStateTransform,
        initial_vector: np.ndarray,
        options: ThorSolveOptions,
    ) -> None:
        self.problem = problem
        self.transform = transform
        self.options = options
        self.row_scales = _row_scales(problem)
        self.state = transform.decode(initial_vector)
        self.d_cell = np.empty(problem.shape[:2])
        self._pressure_factor = None
        self._volumes = np.empty((problem.case.nodes - 2,) * 2)
        self.pressure_factorizations = 0
        self.applications = 0
        super().__init__(dtype=np.dtype(float), shape=(problem.unknown_count,) * 2)
        self._rebuild(self.state)

    def _rebuild(self, state: np.ndarray) -> None:
        case = self.problem.case
        mu = case.mu(state[..., 3])
        self.state = state
        self.d_cell = rhie_chow_inverse_momentum_diagonal(mu, case.x, case.y)
        matrix, self._volumes = _pressure_correction_matrix(
            state[..., 0], self.d_cell, case.x, case.y
        )
        self._pressure_factor = splu(matrix)
        self.pressure_factorizations += 1

    def update(self, vector: np.ndarray, residual: np.ndarray) -> None:
        """Refresh coefficients when SciPy accepts a new nonlinear iterate."""

        del residual
        try:
            state = self.transform.decode(vector)
            self._rebuild(state)
        except (FloatingPointError, RuntimeError, ValueError):
            # Retain the last valid factorization.  The nonlinear objective
            # independently rejects the invalid iterate with a finite guard.
            return

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        self.applications += 1
        rhs = np.asarray(vector, dtype=float)
        if rhs.shape != (self.problem.unknown_count,) or not np.isfinite(rhs).all():
            raise ValueError("SIMPLE preconditioner vector has an invalid shape or value")
        raw = rhs.reshape(self.problem.shape) * self.row_scales
        state = self.state
        case = self.problem.case
        mu = case.mu(state[..., 3])
        d_cell = self.d_cell
        laplacian_diagonal = 1.0 / d_cell

        collision = np.zeros_like(state)
        collision[..., 4:6] = 2.0 / (3.0 * mu[..., None])
        collision[..., 6:9] = 1.0 / mu[..., None]
        collision[..., 9:12] = 7.0 / (6.0 * mu[..., None])
        collision[..., 12:16] = 3.0 / (2.0 * mu[..., None])
        collision[..., 16] = 2.0 / (3.0 * mu)
        diagonal = laplacian_diagonal[..., None] + collision
        diagonal[..., 0] = 1.0

        physical = np.zeros_like(state)
        physical[1:-1, 1:-1] = raw[1:-1, 1:-1] / diagonal[1:-1, 1:-1]
        physical[1:-1, 1:-1, 1:3] *= self.options.velocity_relaxation
        physical[1:-1, 1:-1, 3] *= self.options.thermal_relaxation
        physical[1:-1, 1:-1, 4:] *= self.options.moment_relaxation

        rho = state[..., 0]
        velocity_x = 0.5 * (
            physical[:, 1:, 1] + physical[:, :-1, 1]
        )
        velocity_y = 0.5 * (
            physical[1:, :, 2] + physical[:-1, :, 2]
        )
        predicted_continuity = impermeable_wall_mass_divergence(
            0.5 * (rho[:, 1:] + rho[:, :-1]) * velocity_x,
            0.5 * (rho[1:] + rho[:-1]) * velocity_y,
            case.x,
            case.y,
        )
        target = raw[..., 0].copy()
        mass_rhs = float(target[self.problem.mass_j, self.problem.mass_i])
        target[self.problem.mass_j, self.problem.mass_i] = 0.0
        remainder = target[1:-1, 1:-1] - predicted_continuity[1:-1, 1:-1]
        augmented_rhs = np.concatenate(
            ((remainder * self._volumes).ravel(), np.zeros(1))
        )
        if self._pressure_factor is None:
            raise RuntimeError("SIMPLE pressure factorization is unavailable")
        pressure_interior = self._pressure_factor.solve(augmented_rhs)[:-1].reshape(
            self._volumes.shape
        )
        pressure = np.empty(case.x.size * case.y.size, dtype=float).reshape(
            case.y.size, case.x.size
        )
        pressure[1:-1, 1:-1] = pressure_interior
        pressure[0, 1:-1] = pressure[1, 1:-1]
        pressure[-1, 1:-1] = pressure[-2, 1:-1]
        pressure[1:-1, 0] = pressure[1:-1, 1]
        pressure[1:-1, -1] = pressure[1:-1, -2]
        pressure[0, 0] = pressure[1, 1]
        pressure[0, -1] = pressure[1, -2]
        pressure[-1, 0] = pressure[-2, 1]
        pressure[-1, -1] = pressure[-2, -2]
        grad_x = np.gradient(pressure, case.x, axis=1, edge_order=2)
        grad_y = np.gradient(pressure, case.y, axis=0, edge_order=2)
        alpha_p = self.options.pressure_relaxation
        physical[1:-1, 1:-1, 1] -= (
            alpha_p * d_cell[1:-1, 1:-1] * grad_x[1:-1, 1:-1]
        )
        physical[1:-1, 1:-1, 2] -= (
            alpha_p * d_cell[1:-1, 1:-1] * grad_y[1:-1, 1:-1]
        )
        density_correction = alpha_p * pressure / state[..., 3]
        weights = self.problem.mass_weights
        density_correction -= float(np.sum(weights * density_correction))
        density_correction += mass_rhs
        physical[..., 0] = density_correction

        boundary = np.ones(self.problem.shape[:2], dtype=bool)
        boundary[1:-1, 1:-1] = False
        physical[boundary] = self.options.boundary_relaxation * raw[boundary]

        encoded = physical.copy()
        encoded[..., 0] /= state[..., 0]
        encoded[..., 3] /= state[..., 3]
        return encoded.ravel()


class ThorHybridPreconditioner(LinearOperator):
    """One frozen CUBISTA defect Jacobian with a SIMPLE fallback.

    A purely diagonal SIMPLE operator is too weak for the tightly coupled wall
    moment equations.  The legacy pressure-based codes solve each frozen
    convection--diffusion equation before the next outer correction.  Here one
    colored finite-difference linearization plays the same frozen-defect role,
    ILU supplies bounded-memory field coupling, and the conservative SIMPLE
    operator remains the pressure-aware fallback.  The expensive Jacobian is
    built once rather than refactorized at every nonlinear iteration.
    """

    def __init__(
        self,
        problem: R26NodeBVP,
        transform: LogStateTransform,
        objective: EncodedR26Objective,
        initial_vector: np.ndarray,
        options: ThorSolveOptions,
    ) -> None:
        self.simple = SimpleR26Preconditioner(
            problem, transform, initial_vector, options
        )
        self.residual_evaluations = 0

        def sampled(vector: np.ndarray) -> np.ndarray:
            self.residual_evaluations += 1
            return objective(vector)

        self.matrix = approx_derivative(
            sampled,
            initial_vector,
            method="2-point",
            abs_step=fv_absolute_difference_step(initial_vector),
            sparsity=jacobian_sparsity(problem),
        ).tocsc()
        self.ilu = None
        try:
            self.ilu = spilu(
                self.matrix,
                drop_tol=options.ilu_drop_tolerance,
                fill_factor=options.ilu_fill_factor,
            )
        except RuntimeError:
            self.ilu = None
        self.applications = 0
        super().__init__(dtype=np.dtype(float), shape=self.simple.shape)

    def update(self, vector: np.ndarray, residual: np.ndarray) -> None:
        self.simple.update(vector, residual)

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        self.applications += 1
        value = np.asarray(vector, dtype=float)
        if self.ilu is not None:
            correction = self.ilu.solve(value)
            if np.isfinite(correction).all():
                return correction
        return self.simple @ value


def _raw_gate(problem: R26NodeBVP, state: np.ndarray) -> float:
    diagnostics = problem.evaluate(state).diagnostics
    return float(
        max(
            diagnostics.raw_total_linf,
            abs(diagnostics.held_out_continuity),
            abs(diagnostics.mass_error),
        )
    )


def solve_r26_thor_bvp(
    problem: R26NodeBVP,
    initial_state: np.ndarray,
    *,
    options: ThorSolveOptions | None = None,
) -> ThorSolveResult:
    """Solve one fixed CUBISTA-FV R26 case with SIMPLE-preconditioned Krylov."""

    options = ThorSolveOptions() if options is None else options
    transform = LogStateTransform(problem.shape)
    encoded_initial = transform.encode(initial_state)
    objective = EncodedR26Objective(problem, transform, options.invalid_penalty)
    initial_evaluation = problem.evaluate(initial_state)
    initial_diagnostics = initial_evaluation.diagnostics
    initial_raw_gate = _raw_gate(problem, initial_state)
    initial_passed = bool(
        initial_raw_gate <= options.raw_tolerance
        and initial_diagnostics.total_linf <= options.residual_tolerance
        and abs(initial_diagnostics.held_out_continuity)
        <= options.held_out_continuity_tolerance
        and abs(initial_diagnostics.mass_error) <= options.mass_tolerance
        and initial_diagnostics.min_density > 0.0
        and initial_diagnostics.min_temperature > 0.0
    )
    if initial_passed:
        solution = R26SolveResult(
            case=problem.case,
            state=np.asarray(initial_state, dtype=float).copy(),
            encoded_state=encoded_initial,
            diagnostics=initial_diagnostics,
            converged=True,
            scipy_success=True,
            message="initial state satisfies the strict raw THOR/R26 gate",
            iterations=0,
            function_evaluations=1,
            invalid_evaluations=0,
            last_invalid_error=None,
            solver_method="thor-simple-krylov",
        )
        return ThorSolveResult(
            solution=solution,
            raw_acceptance_gate=initial_raw_gate,
            raw_gate_passed=True,
            pressure_factorizations=0,
            preconditioner_applications=0,
            frozen_jacobian_residual_evaluations=0,
            frozen_jacobian_nonzeros=0,
            ilu_available=False,
        )
    preconditioner = ThorHybridPreconditioner(
        problem, transform, objective, encoded_initial, options
    )
    evaluations = 0
    last_vector = encoded_initial.copy()

    def counted(vector: np.ndarray) -> np.ndarray:
        nonlocal evaluations, last_vector
        evaluations += 1
        last_vector = np.asarray(vector, dtype=float).copy()
        return objective(vector)

    try:
        scipy_result: OptimizeResult = root(
            counted,
            encoded_initial,
            method="krylov",
            options={
                "fatol": options.residual_tolerance,
                "maxiter": options.max_iterations,
                "line_search": options.line_search,
                "disp": options.display,
                "jac_options": {
                    "inner_M": preconditioner,
                    "inner_maxiter": options.inner_max_iterations,
                },
            },
        )
    except (FloatingPointError, RuntimeError, ValueError) as exc:
        scipy_result = OptimizeResult(
            x=last_vector,
            success=False,
            message=f"THOR/SIMPLE nonlinear solve stopped fail-closed: {type(exc).__name__}: {exc}",
            nit=0,
        )

    try:
        encoded = np.asarray(scipy_result.x, dtype=float)
        state = transform.decode(encoded)
        evaluation = problem.evaluate(state)
        physical_final = True
    except (FloatingPointError, ValueError):
        encoded = encoded_initial
        state = transform.decode(encoded_initial)
        evaluation = problem.evaluate(state)
        physical_final = False

    diagnostics = evaluation.diagnostics
    raw_gate = _raw_gate(problem, state)
    raw_passed = bool(
        physical_final
        and raw_gate <= options.raw_tolerance
        and abs(diagnostics.mass_error) <= options.mass_tolerance
        and diagnostics.min_density > 0.0
        and diagnostics.min_temperature > 0.0
    )
    converged = bool(
        scipy_result.success
        and raw_passed
        and diagnostics.total_linf <= options.residual_tolerance
        and abs(diagnostics.held_out_continuity)
        <= options.held_out_continuity_tolerance
    )
    message = str(scipy_result.message)
    if scipy_result.success and not converged:
        message += (
            "; optimizer stopped but strict raw R26 gate failed "
            f"(scaled={diagnostics.total_linf:.3e}, raw={raw_gate:.3e})"
        )
    solution = R26SolveResult(
        case=problem.case,
        state=state,
        encoded_state=encoded,
        diagnostics=diagnostics,
        converged=converged,
        scipy_success=bool(scipy_result.success),
        message=message,
        iterations=int(getattr(scipy_result, "nit", 0) or 0),
        function_evaluations=evaluations,
        invalid_evaluations=objective.invalid_evaluations,
        last_invalid_error=objective.last_invalid_error,
        solver_method="thor-simple-krylov",
    )
    return ThorSolveResult(
        solution=solution,
        raw_acceptance_gate=raw_gate,
        raw_gate_passed=raw_passed,
        pressure_factorizations=preconditioner.simple.pressure_factorizations,
        preconditioner_applications=preconditioner.applications,
        frozen_jacobian_residual_evaluations=preconditioner.residual_evaluations,
        frozen_jacobian_nonzeros=int(preconditioner.matrix.nnz),
        ilu_available=preconditioner.ilu is not None,
    )


__all__ = [
    "SimpleR26Preconditioner",
    "ThorHybridPreconditioner",
    "THOR_SOLVER_PROVENANCE",
    "ThorSolveOptions",
    "ThorSolveResult",
    "make_thor_problem",
    "solve_r26_thor_bvp",
]
