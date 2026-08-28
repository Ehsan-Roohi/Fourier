#!/usr/bin/env python3
"""Documented field-by-field reconstruction of Gu--Emerson Sec. 5.2.

This is deliberately *not* described as the unavailable THOR source code.
It executes the printed Gu--Emerson field order and uses the repository's
audited R26 equations as the defect oracle.  Numerical details that are not
printed in the paper are fixed below, reported in every result, and called a
reconstruction:

* one coloured field-family defect matrix per fixed Picard refresh cycle;
* SuperLU sparse direct block solves (LSMR only if a block is singular);
* fixed under-relaxation, with no adaptive continuation or line search;
* the explicitly documented Rhie--Chow coefficient in ``r26_fv_backend``;
* local solution of the complete smooth-wall equations after every sweep;
* the repository's declared bilinear sharp-corner extension.

No global Newton, Krylov, homotopy, pseudo-arclength, clipping, filtering, or
Tikhonov row occurs in this module.  Acceptance is based only on the complete
unscaled physical R26 boundary-value residual, the held continuity equation,
mass, and positivity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Final

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import lsmr, splu

from r26_cases import CavityCase
from r26_discretization import R26NodeBVP, linear_wall_extrapolation
from r26_fv_backend import (
    rhie_chow_inverse_momentum_diagonal,
    thor_fv_bulk_residual,
    wall_bounded_control_volume_weights,
)
from r26_gu_emerson_algorithm import (
    GU_EMERSON_STAGE_ORDER,
    GuEmersonAlgorithmDisclosure,
    GuEmersonSegregatedOperators,
    NumericalControlSource,
    advance_gu_emerson_outer_iteration,
)
from r26_gu_emerson_variables import (
    GuEmersonFields,
    gu_emerson_fields_as_planar17,
    gu_emerson_fields_from_planar17,
    gu_emerson_fields_from_state,
    state_from_gu_emerson_fields,
)
from r26_state import planar_state_to_tensors
from r26_thor_solver import _pressure_correction_matrix
from r26_wall_conditions import (
    WallFreeQuantities,
    WallParameters,
    WallUnknowns,
    extract_face_quantities,
    project_closures,
    solve_wall_face,
    square_wall_frame,
)


GU_EMERSON_RECONSTRUCTION_PROVENANCE: Final[str] = (
    "Gu--Emerson JFM 636 (2009), Sec. 5.2 and Eqs. (48)--(63), for the "
    "published variable split and segregated order; Alves--Oliveira--Pinho "
    "CUBISTA; documented local reconstruction controls in this module"
)

FIELD_SLOTS: Final[dict[str, tuple[int, ...]]] = {
    "velocity": (1, 2),
    "temperature": (3,),
    "g": (6, 7, 8),
    "h": (4, 5),
    "omega": (12, 13, 14, 15),
    "gamma": (9, 10, 11),
    "chi": (16,),
}


@dataclass(frozen=True)
class GuEmersonReconstructionOptions:
    """Fixed non-paper controls for one reproducible reconstruction gate."""

    max_outer_iterations: int = 80
    raw_tolerance: float = 1.0e-8
    scaled_tolerance: float = 1.0e-8
    held_continuity_tolerance: float = 1.0e-8
    mass_tolerance: float = 1.0e-10
    finite_difference_step: float = 2.0e-6
    stencil_radius: int = 3
    matrix_refresh_interval: int = 5
    velocity_relaxation: float = 0.55
    pressure_relaxation: float = 0.25
    temperature_relaxation: float = 0.55
    g_relaxation: float = 0.45
    h_relaxation: float = 0.45
    omega_relaxation: float = 0.40
    gamma_relaxation: float = 0.40
    chi_relaxation: float = 0.40
    wall_relaxation: float = 0.25
    wall_tolerance: float = 1.0e-11
    wall_max_evaluations: int = 600

    def __post_init__(self) -> None:
        if (
            self.max_outer_iterations < 1
            or self.wall_max_evaluations < 1
            or self.matrix_refresh_interval < 1
        ):
            raise ValueError("iteration/evaluation limits must be positive")
        if self.stencil_radius < 1:
            raise ValueError("stencil_radius must be positive")
        positive = (
            self.raw_tolerance,
            self.scaled_tolerance,
            self.held_continuity_tolerance,
            self.mass_tolerance,
            self.finite_difference_step,
            self.wall_tolerance,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("all tolerances and the FD step must be finite and positive")
        relaxations = tuple(self.relaxation(name) for name in FIELD_SLOTS) + (
            self.pressure_relaxation,
            self.wall_relaxation,
        )
        if not all(np.isfinite(value) and 0.0 < value <= 1.0 for value in relaxations):
            raise ValueError("fixed reconstruction relaxations must lie in (0,1]")

    def relaxation(self, stage: str) -> float:
        name = "temperature" if stage == "temperature" else stage
        return float(getattr(self, f"{name}_relaxation"))

    @property
    def disclosure(self) -> GuEmersonAlgorithmDisclosure:
        tag = "documented local reconstruction; not specified by Gu--Emerson"
        controls = {
            "linear_solver": NumericalControlSource(
                "SuperLU sparse direct field blocks; LSMR singular-block fallback",
                tag,
            ),
            "under_relaxation_factors": NumericalControlSource(
                (
                    f"u={self.velocity_relaxation}, p={self.pressure_relaxation}, "
                    f"T={self.temperature_relaxation}, g={self.g_relaxation}, "
                    f"h={self.h_relaxation}, omega={self.omega_relaxation}, "
                    f"gamma={self.gamma_relaxation}, chi={self.chi_relaxation}, "
                    f"wall_correction={self.wall_relaxation}"
                ),
                tag,
            ),
            "source_term_linearisation": NumericalControlSource(
                (
                    "coloured field-family defect matrices refreshed every "
                    f"{self.matrix_refresh_interval} fixed Picard sweeps"
                ),
                tag,
            ),
            "rhie_chow_face_coefficient": NumericalControlSource(
                "inverse central-diffusion momentum diagonal from r26_fv_backend",
                tag,
            ),
            "sharp_corner_treatment": NumericalControlSource(
                "explicit bilinear adjacent-face extension; corners excluded from wall metrics",
                tag,
            ),
            "numerical_convergence_thresholds": NumericalControlSource(
                (
                    f"raw={self.raw_tolerance:g}, scaled={self.scaled_tolerance:g}, "
                    f"held_continuity={self.held_continuity_tolerance:g}, "
                    f"mass={self.mass_tolerance:g}"
                ),
                tag,
            ),
        }
        return GuEmersonAlgorithmDisclosure(controls=controls)


@dataclass(frozen=True)
class GuEmersonSweepRecord:
    outer_iteration: int
    raw_gate: float
    scaled_linf: float
    held_continuity: float
    mass_error: float
    min_density: float
    min_temperature: float
    stage_order: tuple[str, ...]


@dataclass(frozen=True)
class GuEmersonReconstructionResult:
    state: np.ndarray
    fields: GuEmersonFields
    converged: bool
    message: str
    outer_iterations: int
    residual_evaluations: int
    block_factorizations: int
    lsmr_fallbacks: int
    wall_solves: int
    wall_function_evaluations: int
    records: tuple[GuEmersonSweepRecord, ...]
    disclosure: GuEmersonAlgorithmDisclosure
    provenance: str = GU_EMERSON_RECONSTRUCTION_PROVENANCE


def make_gu_emerson_reconstruction_problem(case: CavityCase) -> R26NodeBVP:
    """Return the collocated CUBISTA/Rhie--Chow physical R26 problem."""

    return R26NodeBVP(
        case,
        bulk_operator=thor_fv_bulk_residual,
        mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
    )


def _interior_flat_indices(nodes: int, slots: tuple[int, ...]) -> np.ndarray:
    index = np.arange(nodes * nodes * 17).reshape(nodes, nodes, 17)
    return index[1:-1, 1:-1][..., list(slots)].ravel()


def _local_block_sparsity(nodes: int, components: int, radius: int) -> csc_matrix:
    side = nodes - 2
    count = side * side * components
    rows: list[int] = []
    cols: list[int] = []

    def local_index(j: int, i: int, component: int) -> int:
        return (j * side + i) * components + component

    for j in range(side):
        for i in range(side):
            for output_component in range(components):
                row = local_index(j, i, output_component)
                for jj in range(max(0, j - radius), min(side, j + radius + 1)):
                    for ii in range(max(0, i - radius), min(side, i + radius + 1)):
                        for input_component in range(components):
                            rows.append(row)
                            cols.append(local_index(jj, ii, input_component))
    data = np.ones(len(rows), dtype=bool)
    return coo_matrix((data, (rows, cols)), shape=(count, count)).tocsc()


class _SegregatedReconstructionOperators:
    """State-free callbacks used by the published-order driver."""

    def __init__(self, problem: R26NodeBVP, options: GuEmersonReconstructionOptions) -> None:
        self.problem = problem
        self.options = options
        self.residual_evaluations = 0
        self.block_factorizations = 0
        self.lsmr_fallbacks = 0
        self.wall_solves = 0
        self.wall_function_evaluations = 0
        self.executed_stages: list[str] = []
        self._block_matrices: dict[str, csc_matrix] = {}
        self._block_factors: dict[str, object | None] = {}
        self.completed_sweeps = 0

    def _physical(self, fields: GuEmersonFields) -> np.ndarray:
        mu = self.problem.case.mu(fields.theta)
        return state_from_gu_emerson_fields(
            fields,
            x=self.problem.case.x,
            y=self.problem.case.y,
            mu=mu,
        )

    def _bulk_stage_residual(self, fields: GuEmersonFields, slots: tuple[int, ...]) -> np.ndarray:
        state = self._physical(fields)
        mu = self.problem.case.mu(state[..., 3])
        raw = self.problem._bulk(state, mu)
        self.residual_evaluations += 1
        return (
            raw[1:-1, 1:-1][..., list(slots)]
            / self.problem.case.scaling.bulk[list(slots)]
        ).ravel()

    def _solve_field(self, fields: GuEmersonFields, stage: str) -> GuEmersonFields:
        self.executed_stages.append(stage)
        slots = FIELD_SLOTS[stage]
        packed = gu_emerson_fields_as_planar17(fields)
        full_indices = _interior_flat_indices(self.problem.case.nodes, slots)
        base = packed.ravel()[full_indices].copy()
        logarithmic = stage == "temperature"
        encoded = np.log(base) if logarithmic else base

        def with_vector(vector: np.ndarray) -> GuEmersonFields:
            value = np.exp(vector) if logarithmic else np.asarray(vector, dtype=float)
            candidate = packed.copy().ravel()
            candidate[full_indices] = value
            return gu_emerson_fields_from_planar17(candidate.reshape(packed.shape))

        def objective(vector: np.ndarray) -> np.ndarray:
            return self._bulk_stage_residual(with_vector(vector), slots)

        residual = objective(encoded)
        if not np.isfinite(residual).all():
            raise FloatingPointError(f"{stage} defect contains NaN or infinity")
        if np.max(np.abs(residual), initial=0.0) == 0.0:
            return fields
        matrix = self._block_matrices.get(stage)
        factor = self._block_factors.get(stage)
        if matrix is None:
            sparsity = _local_block_sparsity(
                self.problem.case.nodes, len(slots), self.options.stencil_radius
            )
            matrix = approx_derivative(
                objective,
                encoded,
                method="2-point",
                abs_step=self.options.finite_difference_step * (1.0 + np.abs(encoded)),
                sparsity=sparsity,
            ).tocsc()
            self.block_factorizations += 1
            try:
                factor = splu(matrix)
            except RuntimeError:
                factor = None
            self._block_matrices[stage] = matrix
            self._block_factors[stage] = factor
        rhs = -residual
        if factor is not None:
            correction = factor.solve(rhs)
        else:
            self.lsmr_fallbacks += 1
            correction = lsmr(
                matrix,
                rhs,
                atol=1.0e-11,
                btol=1.0e-11,
                maxiter=max(200, 4 * matrix.shape[0]),
            )[0]
        if not np.isfinite(correction).all():
            raise FloatingPointError(f"{stage} block correction is non-finite")
        updated = encoded + self.options.relaxation(stage) * correction
        return with_vector(updated)

    def solve_velocity(self, fields: GuEmersonFields) -> GuEmersonFields:
        if self.completed_sweeps % self.options.matrix_refresh_interval == 0:
            self._block_matrices.clear()
            self._block_factors.clear()
        return self._solve_field(fields, "velocity")

    def simple_pressure_correction(self, fields: GuEmersonFields) -> GuEmersonFields:
        self.executed_stages.append("simple_pressure_correction")
        case = self.problem.case
        state = self._physical(fields)
        mu = case.mu(state[..., 3])
        continuity = self.problem._bulk(state, mu)[..., 0]
        self.residual_evaluations += 1
        d_cell = rhie_chow_inverse_momentum_diagonal(mu, case.x, case.y)
        matrix, volumes = _pressure_correction_matrix(
            state[..., 0], d_cell, case.x, case.y
        )
        rhs = np.concatenate(((-continuity[1:-1, 1:-1] * volumes).ravel(), (0.0,)))
        pressure_inner = splu(matrix).solve(rhs)[:-1].reshape(volumes.shape)
        pressure = np.empty_like(fields.rho)
        pressure[1:-1, 1:-1] = pressure_inner
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
        alpha = self.options.pressure_relaxation
        velocity = np.asarray(fields.velocity).copy()
        velocity[1:-1, 1:-1, 0] -= alpha * d_cell[1:-1, 1:-1] * grad_x[1:-1, 1:-1]
        velocity[1:-1, 1:-1, 1] -= alpha * d_cell[1:-1, 1:-1] * grad_y[1:-1, 1:-1]
        rho = np.asarray(fields.rho).copy()
        rho[1:-1, 1:-1] += alpha * pressure_inner / fields.theta[1:-1, 1:-1]
        interior_weights = self.problem.mass_weights[1:-1, 1:-1]
        mass_error = float(np.sum(self.problem.mass_weights * rho) - case.mean_density)
        rho[1:-1, 1:-1] -= mass_error / float(np.sum(interior_weights))
        if np.any(rho <= 0.0):
            raise FloatingPointError("SIMPLE density correction produced non-positive rho")
        return replace(fields, rho=rho, velocity=velocity)

    def solve_temperature(self, fields: GuEmersonFields) -> np.ndarray:
        return self._solve_field(fields, "temperature").theta

    def solve_g(self, fields: GuEmersonFields) -> np.ndarray:
        return self._solve_field(fields, "g").g

    def solve_h(self, fields: GuEmersonFields) -> np.ndarray:
        return self._solve_field(fields, "h").h

    def solve_omega(self, fields: GuEmersonFields) -> np.ndarray:
        return self._solve_field(fields, "omega").omega

    def solve_gamma(self, fields: GuEmersonFields) -> np.ndarray:
        return self._solve_field(fields, "gamma").gamma

    def solve_chi(self, fields: GuEmersonFields) -> np.ndarray:
        return self._solve_field(fields, "chi").chi

    def update_wall_boundaries(
        self, physical: np.ndarray, fields: GuEmersonFields
    ) -> GuEmersonFields:
        self.executed_stages.extend(("physical_moment_reconstruction", "wall_boundary_update"))
        case = self.problem.case
        state = np.asarray(physical, dtype=float).copy()
        mu = case.mu(state[..., 3])
        closures = self.problem._closures(state, mu)
        for node in self.problem.boundary_nodes:
            frame = square_wall_frame(node.side)
            near = extract_face_quantities(
                planar_state_to_tensors(state[node.near_j, node.near_i]),
                frame,
                gas_constant=case.gas_constant,
            )[0].as_array()
            nxt = extract_face_quantities(
                planar_state_to_tensors(state[node.next_j, node.next_i]),
                frame,
                gas_constant=case.gas_constant,
            )[0].as_array()
            if node.side in {"left", "right"}:
                coordinates = (case.x[node.i], case.x[node.near_i], case.x[node.next_i])
            else:
                coordinates = (case.y[node.j], case.y[node.near_j], case.y[node.next_j])
            free = WallFreeQuantities.from_array(
                linear_wall_extrapolation(*coordinates, near, nxt)
            )
            local_closure = type(closures)(
                phi=np.asarray(closures.phi[node.j, node.i]),
                psi=np.asarray(closures.psi[node.j, node.i]),
                Omega=np.asarray(closures.Omega[node.j, node.i]),
                equation25_mode=closures.equation25_mode,
                provenance=closures.provenance,
                coefficient_mode=closures.coefficient_mode,
            )
            parameters = WallParameters(
                wall_temperature=case.wall_temperature,
                accommodation=case.accommodation,
                gas_constant=case.gas_constant,
                wall_velocity=case.wall_velocity(node.side),
            )
            _, absolute = extract_face_quantities(
                planar_state_to_tensors(state[node.j, node.i]),
                frame,
                gas_constant=case.gas_constant,
            )
            relative = replace(
                absolute,
                u_t=float(
                    np.dot(
                        state[node.j, node.i, 1:3]
                        - case.wall_velocity(node.side)[:2],
                        frame.tangent[:2],
                    )
                ),
            )
            projected = project_closures(local_closure, frame)
            try:
                result = solve_wall_face(
                    free,
                    projected,
                    frame,
                    parameters,
                    initial=relative,
                    max_nfev=self.options.wall_max_evaluations,
                    tolerance=self.options.wall_tolerance,
                )
            except FloatingPointError:
                result = solve_wall_face(
                    free,
                    projected,
                    frame,
                    parameters,
                    initial=None,
                    max_nfev=self.options.wall_max_evaluations,
                    tolerance=self.options.wall_tolerance,
                )
            self.wall_solves += 1
            self.wall_function_evaluations += result.nfev
            if not result.success:
                raise RuntimeError(
                    f"local {node.side} wall solve failed at ({node.j},{node.i}): "
                    f"{result.message}; scaled={np.max(np.abs(result.scaled_residual)):.3e}"
                )
            current_wall = state[node.j, node.i].copy()
            state[node.j, node.i] = current_wall + self.options.wall_relaxation * (
                result.planar_state - current_wall
            )

        state[0, 0] = state[0, 1] + state[1, 0] - state[1, 1]
        state[0, -1] = state[0, -2] + state[1, -1] - state[1, -2]
        state[-1, 0] = state[-1, 1] + state[-2, 0] - state[-2, 1]
        state[-1, -1] = state[-1, -2] + state[-2, -1] - state[-2, -2]
        if np.any(state[..., 0] <= 0.0) or np.any(state[..., 3] <= 0.0):
            raise FloatingPointError("bilinear corner reconstruction violated positivity")
        self.completed_sweeps += 1
        return gu_emerson_fields_from_state(
            state,
            x=case.x,
            y=case.y,
            mu=case.mu(state[..., 3]),
        )

    @property
    def callbacks(self) -> GuEmersonSegregatedOperators:
        return GuEmersonSegregatedOperators(
            solve_velocity=self.solve_velocity,
            simple_pressure_correction=self.simple_pressure_correction,
            solve_temperature=self.solve_temperature,
            solve_g=self.solve_g,
            solve_h=self.solve_h,
            solve_omega=self.solve_omega,
            solve_gamma=self.solve_gamma,
            solve_chi=self.solve_chi,
            update_wall_boundaries=self.update_wall_boundaries,
        )


def _gate(problem: R26NodeBVP, state: np.ndarray) -> tuple[float, object, bool]:
    diagnostics = problem.evaluate(state).diagnostics
    raw = float(
        max(
            diagnostics.raw_total_linf,
            abs(diagnostics.held_out_continuity),
            abs(diagnostics.mass_error),
        )
    )
    return raw, diagnostics, bool(diagnostics.min_density > 0.0 and diagnostics.min_temperature > 0.0)


def solve_gu_emerson_reconstruction(
    problem: R26NodeBVP,
    initial_state: np.ndarray,
    *,
    options: GuEmersonReconstructionOptions | None = None,
    record_callback: Callable[[GuEmersonSweepRecord, np.ndarray], None] | None = None,
) -> GuEmersonReconstructionResult:
    """Run bounded published-order sweeps and enforce the complete raw gate."""

    options = GuEmersonReconstructionOptions() if options is None else options
    disclosure = options.disclosure
    disclosure.require_production_authorization()
    state = np.asarray(initial_state, dtype=float).copy()
    fields = gu_emerson_fields_from_state(
        state,
        x=problem.case.x,
        y=problem.case.y,
        mu=problem.case.mu(state[..., 3]),
    )
    operators = _SegregatedReconstructionOperators(problem, options)
    records: list[GuEmersonSweepRecord] = []
    converged = False
    message = "bounded Gu--Emerson reconstruction work budget exhausted"

    for outer in range(1, options.max_outer_iterations + 1):
        operators.executed_stages = []
        sweep = advance_gu_emerson_outer_iteration(
            fields,
            operators.callbacks,
            x=problem.case.x,
            y=problem.case.y,
            viscosity=problem.case.mu,
        )
        if tuple(operators.executed_stages) != GU_EMERSON_STAGE_ORDER:
            raise RuntimeError("published Gu--Emerson stage order was not executed exactly")
        fields = sweep.fields
        state = sweep.physical_state
        raw, diagnostics, positive = _gate(problem, state)
        record = GuEmersonSweepRecord(
            outer_iteration=outer,
            raw_gate=raw,
            scaled_linf=diagnostics.total_linf,
            held_continuity=diagnostics.held_out_continuity,
            mass_error=diagnostics.mass_error,
            min_density=diagnostics.min_density,
            min_temperature=diagnostics.min_temperature,
            stage_order=tuple(operators.executed_stages),
        )
        records.append(record)
        if record_callback is not None:
            record_callback(record, state)
        converged = bool(
            positive
            and raw <= options.raw_tolerance
            and diagnostics.total_linf <= options.scaled_tolerance
            and abs(diagnostics.held_out_continuity)
            <= options.held_continuity_tolerance
            and abs(diagnostics.mass_error) <= options.mass_tolerance
        )
        if converged:
            message = "complete raw R26 gate reached by published-order reconstruction"
            break

    return GuEmersonReconstructionResult(
        state=state,
        fields=fields,
        converged=converged,
        message=message,
        outer_iterations=len(records),
        residual_evaluations=operators.residual_evaluations,
        block_factorizations=operators.block_factorizations,
        lsmr_fallbacks=operators.lsmr_fallbacks,
        wall_solves=operators.wall_solves,
        wall_function_evaluations=operators.wall_function_evaluations,
        records=tuple(records),
        disclosure=disclosure,
    )


__all__ = [
    "FIELD_SLOTS",
    "GU_EMERSON_RECONSTRUCTION_PROVENANCE",
    "GuEmersonReconstructionOptions",
    "GuEmersonReconstructionResult",
    "GuEmersonSweepRecord",
    "make_gu_emerson_reconstruction_problem",
    "solve_gu_emerson_reconstruction",
]
