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
from typing import Callable, Final, Literal

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import lsmr, splu

from r26_bulk_equations import (
    closure_derivatives_on_grid,
    gu_emerson_nonlinear_sources,
)
from r26_cases import CavityCase
from r26_discretization import R26NodeBVP, linear_wall_extrapolation
from r26_fv_backend import (
    impermeable_wall_mass_divergence,
    thor_fv_bulk_residual,
    wall_bounded_face_divergence,
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
from r26_tensor_closures import closures_from_tensors, finite_difference_gradients
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

# The supplied Rana Code_Saturne R26 implementation updates the nonlinear
# right-hand-side arrays with these fixed history factors before the
# segregated scalar equations are assembled.  They are not generic field
# under-relaxation factors: keeping the distinction is essential because the
# collision and diffusion contributions remain implicit in the matrix.  See
# ``SRCR26_22nd_NOV/cs_user_modules.f90`` (SHA-256
# d92e0142776d90499e2beea4a8b3b37b590597f66b61f43bb49f58ade73a884b),
# routines ``cs_user_source_sigma``, ``cs_user_source_heatflux``,
# ``cs_user_source_mijk``, ``cs_user_source_rij`` and
# ``cs_user_source_delta``.
RANA_SOURCE_HISTORY_RELAXATION: Final[dict[str, float]] = {
    "g": 1.0e-2,
    "h": 1.0e-2,
    "omega": 5.0e-1,
    "gamma": 5.0e-1,
    "chi": 1.0e-1,
}

# Code_Saturne 5.0.3 owns pressure as an independent variable.  Its steady
# defaults use relxst=0.7 for transported variables and 1-relxst=0.3 for
# pressure (``iniini.f90`` and ``modini.f90`` at tag v5.0.3).  The supplied
# Rana physical-properties routine then updates density at the beginning of a
# time step from total pressure and temperature with the fixed history factor
# below.  It is not a SIMPLE pressure relaxation and must not be applied to
# the pressure increment itself.
CODE_SATURNE_V5_COMMIT: Final[str] = "e17068ce692ad2d90c694d375b7c098043b16969"
CODE_SATURNE_V5_STEADY_FIELD_RELAXATION: Final[float] = 7.0e-1
CODE_SATURNE_V5_STEADY_PRESSURE_RELAXATION: Final[float] = 3.0e-1
RANA_THERMOPHYSICAL_HISTORY_RELAXATION: Final[float] = 2.0e-4

PressureDensityCoupling = Literal[
    "legacy_direct_mass_constrained",
    "code_saturne_v5_lagged_total_pressure_diagnostic",
]


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
    use_rana_source_history: bool = True
    pressure_density_coupling: PressureDensityCoupling = (
        "legacy_direct_mass_constrained"
    )
    density_property_relaxation: float = RANA_THERMOPHYSICAL_HISTORY_RELAXATION

    @classmethod
    def code_saturne_v5_rana_diagnostic(
        cls, **overrides: object
    ) -> "GuEmersonReconstructionOptions":
        """Return a non-authorizing Code_Saturne/Rana carrier diagnostic.

        The pressure/density semantics and steady defaults are visible in the
        official v5.0.3 core and supplied Rana routines.  Missing historical
        face coefficients and case files mean this profile is suitable only
        for mismatch measurement, not production or historical reproduction.
        """

        values: dict[str, object] = {
            "matrix_refresh_interval": 1,
            "velocity_relaxation": CODE_SATURNE_V5_STEADY_FIELD_RELAXATION,
            "pressure_relaxation": CODE_SATURNE_V5_STEADY_PRESSURE_RELAXATION,
            "temperature_relaxation": CODE_SATURNE_V5_STEADY_FIELD_RELAXATION,
            "g_relaxation": CODE_SATURNE_V5_STEADY_FIELD_RELAXATION,
            "h_relaxation": CODE_SATURNE_V5_STEADY_FIELD_RELAXATION,
            "omega_relaxation": CODE_SATURNE_V5_STEADY_FIELD_RELAXATION,
            "gamma_relaxation": CODE_SATURNE_V5_STEADY_FIELD_RELAXATION,
            "chi_relaxation": CODE_SATURNE_V5_STEADY_FIELD_RELAXATION,
            "wall_relaxation": 1.0,
            "use_rana_source_history": True,
            "pressure_density_coupling": (
                "code_saturne_v5_lagged_total_pressure_diagnostic"
            ),
            "density_property_relaxation": (
                RANA_THERMOPHYSICAL_HISTORY_RELAXATION
            ),
        }
        values.update(overrides)
        return cls(**values)

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
        if self.pressure_density_coupling not in (
            "legacy_direct_mass_constrained",
            "code_saturne_v5_lagged_total_pressure_diagnostic",
        ):
            raise ValueError("unknown pressure-density coupling")
        if not (
            np.isfinite(self.density_property_relaxation)
            and 0.0 < self.density_property_relaxation <= 1.0
        ):
            raise ValueError("density property relaxation must lie in (0,1]")
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
        saturne_carrier = (
            self.pressure_density_coupling
            == "code_saturne_v5_lagged_total_pressure_diagnostic"
        )
        relaxation_source = (
            (
                "Code_Saturne v5.0.3 iniini.f90/modini.f90 at commit "
                f"{CODE_SATURNE_V5_COMMIT}; Rana physical-properties source "
                "SHA-256 a01d309692acf26093c65aa4c11453afc07f3f98b7be1bb8f2c1ea7ba2e44d5d"
            )
            if saturne_carrier
            else tag
        )
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
                relaxation_source,
            ),
            "source_term_linearisation": NumericalControlSource(
                (
                    (
                        "Rana Code_Saturne fixed nonlinear-source history factors "
                        "g=0.01, h=0.01, omega=0.5, gamma=0.5, chi=0.1; "
                        "collision/diffusion retained in coloured field matrices"
                    )
                    if self.use_rana_source_history
                    else "Gu--Emerson field defect with no transferred Rana source history"
                ),
                (
                    (
                        "SRCR26_22nd_NOV/cs_user_modules.f90 SHA-256 "
                        "d92e0142776d90499e2beea4a8b3b37b590597f66b61f43bb49f58ade73a884b"
                    )
                    if self.use_rana_source_history
                    else "Gu--Emerson JFM 636 (2009), Eqs. (56)--(63) and Sec. 5.2"
                ),
            ),
            "rhie_chow_face_coefficient": NumericalControlSource(
                (
                    "component-wise inverse diagonal of the velocity block; "
                    "full velocity/flux correction; pressure retained independently and "
                    f"rho lagged by {self.density_property_relaxation:g} from p_total/theta"
                    if saturne_carrier
                    else "component-wise inverse diagonal of the velocity block solved immediately before SIMPLE"
                ),
                (
                    "Code_Saturne v5.0.3 resopv.f90/navstv.f90 at commit "
                    f"{CODE_SATURNE_V5_COMMIT}; supplied Rana cs_user_physical_properties_R26.f90"
                    if saturne_carrier
                    else "Patankar SIMPLE as required by Gu--Emerson JFM 636 (2009), Sec. 5.2"
                ),
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
        self._source_history: np.ndarray | None = None
        self._simple_inverse_momentum_diagonal: tuple[np.ndarray, np.ndarray] | None = None
        self._saturne_total_pressure: np.ndarray | None = None
        self.completed_sweeps = 0

    @property
    def uses_saturne_carrier(self) -> bool:
        return (
            self.options.pressure_density_coupling
            == "code_saturne_v5_lagged_total_pressure_diagnostic"
        )

    def _apply_lagged_density_property(
        self, fields: GuEmersonFields
    ) -> GuEmersonFields:
        """Apply the supplied Rana ``rho <- p_total/(R*T)`` history update.

        Code_Saturne calls the physical-properties hook before the flow solve,
        so a pressure correction produced in one outer iteration affects
        density at the beginning of the next.  Boundary-node density remains
        owned by the complete R26 wall solve later in the printed sweep.
        """

        if not self.uses_saturne_carrier:
            return fields
        rho = np.asarray(fields.rho, dtype=float)
        theta = np.asarray(fields.theta, dtype=float)
        if self._saturne_total_pressure is None:
            self._saturne_total_pressure = rho * theta
            return fields
        target = self._saturne_total_pressure / theta
        alpha = self.options.density_property_relaxation
        updated = rho.copy()
        updated[1:-1, 1:-1] = (
            alpha * target[1:-1, 1:-1]
            + (1.0 - alpha) * rho[1:-1, 1:-1]
        )
        if not np.isfinite(updated).all() or np.any(updated <= 0.0):
            raise FloatingPointError(
                "Code_Saturne/Rana thermophysical density update produced invalid rho"
            )
        return replace(fields, rho=updated)

    def _physical(self, fields: GuEmersonFields) -> np.ndarray:
        mu = self.problem.case.mu(fields.theta)
        return state_from_gu_emerson_fields(
            fields,
            x=self.problem.case.x,
            y=self.problem.case.y,
            mu=mu,
        )

    def _nonlinear_sources(self, state: np.ndarray) -> np.ndarray:
        """Pack the five printed nonlinear source families in planar rows."""

        case = self.problem.case
        mu = case.mu(state[..., 3])
        tensors = planar_state_to_tensors(state)
        gradients = finite_difference_gradients(state, x=case.x, y=case.y, edge_order=2)
        closures = closures_from_tensors(
            tensors,
            gradients,
            mu=mu,
            coefficient_mode=case.r26_closure_mode,
        )
        derivatives = closure_derivatives_on_grid(closures, case.x, case.y, edge_order=2)
        source = gu_emerson_nonlinear_sources(
            tensors,
            gradients,
            closures,
            derivatives,
            mu=mu,
        )
        packed = np.zeros_like(state)
        packed[..., 4] = source.Q[..., 0]
        packed[..., 5] = source.Q[..., 1]
        packed[..., 6] = source.Sigma[..., 0, 0]
        packed[..., 7] = source.Sigma[..., 0, 1]
        packed[..., 8] = source.Sigma[..., 1, 1]
        packed[..., 9] = source.S[..., 0, 0]
        packed[..., 10] = source.S[..., 0, 1]
        packed[..., 11] = source.S[..., 1, 1]
        packed[..., 12] = source.M[..., 0, 0, 0]
        packed[..., 13] = source.M[..., 0, 0, 1]
        packed[..., 14] = source.M[..., 0, 1, 1]
        packed[..., 15] = source.M[..., 1, 1, 1]
        packed[..., 16] = source.N
        if not np.isfinite(packed).all():
            raise FloatingPointError("Rana nonlinear source history contains NaN or infinity")
        return packed

    @staticmethod
    def _face_average(field: np.ndarray, axis: int) -> np.ndarray:
        if axis == 1:
            return 0.5 * (field[:, 1:] + field[:, :-1])
        if axis == 0:
            return 0.5 * (field[1:] + field[:-1])
        raise ValueError("face axis must be 0 or 1")

    def _saturne_pressure_momentum_correction(
        self, state: np.ndarray
    ) -> np.ndarray:
        """Replace ``grad(rho*theta)`` by the independent carrier pressure."""

        if not self.uses_saturne_carrier or self._saturne_total_pressure is None:
            return np.zeros(state.shape[:2] + (2,), dtype=float)
        case = self.problem.case
        thermodynamic = state[..., 0] * state[..., 3]
        delta = self._saturne_total_pressure - thermodynamic
        flux_x = np.zeros((case.nodes, case.nodes - 1, 2), dtype=float)
        flux_y = np.zeros((case.nodes - 1, case.nodes, 2), dtype=float)
        flux_x[..., 0] = self._face_average(delta, 1)
        flux_y[..., 1] = self._face_average(delta, 0)
        wall_x = np.zeros((case.nodes, case.nodes, 2), dtype=float)
        wall_y = np.zeros((case.nodes, case.nodes, 2), dtype=float)
        wall_x[:, 0, 0] = delta[:, 0]
        wall_x[:, -1, 0] = delta[:, -1]
        wall_y[0, :, 1] = delta[0, :]
        wall_y[-1, :, 1] = delta[-1, :]
        return wall_bounded_face_divergence(
            flux_x, flux_y, wall_x, wall_y, case.x, case.y
        )

    def _saturne_predicted_continuity(self, fields: GuEmersonFields) -> np.ndarray:
        """Return the carrier mass imbalance using its stored pressure and d."""

        if self._saturne_total_pressure is None:
            raise RuntimeError("Code_Saturne carrier pressure was not initialized")
        if self._simple_inverse_momentum_diagonal is None:
            raise RuntimeError("Code_Saturne carrier requires the velocity diagonal")
        case = self.problem.case
        rho = np.asarray(fields.rho, dtype=float)
        velocity = np.asarray(fields.velocity, dtype=float)
        d_x, d_y = self._simple_inverse_momentum_diagonal
        pressure = self._saturne_total_pressure
        grad_x = np.gradient(pressure, case.x, axis=1, edge_order=2)
        grad_y = np.gradient(pressure, case.y, axis=0, edge_order=2)
        face_gradient_x = np.diff(pressure, axis=1) / np.diff(case.x)[None, :]
        face_gradient_y = np.diff(pressure, axis=0) / np.diff(case.y)[:, None]
        mass_x = (
            self._face_average(rho, 1) * self._face_average(velocity[..., 0], 1)
            + self._face_average(rho * d_x, 1)
            * (self._face_average(grad_x, 1) - face_gradient_x)
        )
        mass_y = (
            self._face_average(rho, 0) * self._face_average(velocity[..., 1], 0)
            + self._face_average(rho * d_y, 0)
            * (self._face_average(grad_y, 0) - face_gradient_y)
        )
        return impermeable_wall_mass_divergence(
            mass_x, mass_y, case.x, case.y
        )

    def _refresh_source_history(self, fields: GuEmersonFields) -> None:
        if not self.options.use_rana_source_history:
            self._source_history = None
            return
        current = self._nonlinear_sources(self._physical(fields))
        if self._source_history is None:
            self._source_history = np.zeros_like(current)
        for stage in ("g", "h", "omega", "gamma", "chi"):
            slots = FIELD_SLOTS[stage]
            alpha = RANA_SOURCE_HISTORY_RELAXATION[stage]
            self._source_history[..., list(slots)] = (
                alpha * current[..., list(slots)]
                + (1.0 - alpha) * self._source_history[..., list(slots)]
            )

    def _bulk_stage_residual(self, fields: GuEmersonFields, slots: tuple[int, ...]) -> np.ndarray:
        state = self._physical(fields)
        mu = self.problem.case.mu(state[..., 3])
        raw = self.problem._bulk(state, mu)
        self.residual_evaluations += 1
        if self.uses_saturne_carrier and any(slot in (1, 2) for slot in slots):
            raw[..., 1:3] += self._saturne_pressure_momentum_correction(state)
        if self.options.use_rana_source_history and any(slot >= 4 for slot in slots):
            if self._source_history is None:
                raise RuntimeError("nonlinear source history was not initialized")
            # ``steady_r26_bulk_residual`` stores every balance as LHS - RHS.
            # Replacing -S(current) by -S(history) therefore adds the difference
            # S(current)-S(history).  During numerical matrix construction this
            # cancellation keeps the nonlinear source explicit, while the
            # collision and diffusion derivatives remain on the implicit block.
            current_source = self._nonlinear_sources(state)
            raw[..., 4:] += current_source[..., 4:] - self._source_history[..., 4:]
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
        fields = self._apply_lagged_density_property(fields)
        self._refresh_source_history(fields)
        if self.completed_sweeps % self.options.matrix_refresh_interval == 0:
            self._block_matrices.clear()
            self._block_factors.clear()
        updated = self._solve_field(fields, "velocity")
        matrix = self._block_matrices.get("velocity")
        if matrix is None:
            self._simple_inverse_momentum_diagonal = None
        else:
            side = self.problem.case.nodes - 2
            raw_diagonal = matrix.diagonal().reshape(side, side, 2)
            raw_diagonal = raw_diagonal * self.problem.case.scaling.bulk[1:3]
            if not np.isfinite(raw_diagonal).all() or np.any(raw_diagonal <= 0.0):
                raise FloatingPointError(
                    "velocity block has a non-positive SIMPLE momentum diagonal"
                )
            # The velocity stage applies fixed under-relaxation to its linear
            # correction, so the SIMPLE coefficient is alpha_u/a_P rather
            # than 1/a_P.  This is the same under-relaxed momentum equation
            # that produced the provisional velocity.
            inverse = self.options.velocity_relaxation / raw_diagonal
            d_x = np.empty_like(fields.rho)
            d_y = np.empty_like(fields.rho)
            d_x[1:-1, 1:-1] = inverse[..., 0]
            d_y[1:-1, 1:-1] = inverse[..., 1]
            for coefficient in (d_x, d_y):
                coefficient[0, 1:-1] = coefficient[1, 1:-1]
                coefficient[-1, 1:-1] = coefficient[-2, 1:-1]
                coefficient[1:-1, 0] = coefficient[1:-1, 1]
                coefficient[1:-1, -1] = coefficient[1:-1, -2]
                coefficient[0, 0] = coefficient[1, 1]
                coefficient[0, -1] = coefficient[1, -2]
                coefficient[-1, 0] = coefficient[-2, 1]
                coefficient[-1, -1] = coefficient[-2, -2]
            self._simple_inverse_momentum_diagonal = (d_x, d_y)
        return updated

    def simple_pressure_correction(self, fields: GuEmersonFields) -> GuEmersonFields:
        self.executed_stages.append("simple_pressure_correction")
        case = self.problem.case
        state = self._physical(fields)
        mu = case.mu(state[..., 3])
        physical_continuity = self.problem._bulk(state, mu)[..., 0]
        self.residual_evaluations += 1
        continuity = physical_continuity
        if self.uses_saturne_carrier and self._simple_inverse_momentum_diagonal is not None:
            continuity = self._saturne_predicted_continuity(fields)
        continuity_linf = float(
            np.max(np.abs(continuity[1:-1, 1:-1]), initial=0.0)
        )
        # Compatible wall-bounded equilibrium telescopes to zero analytically,
        # but non-dimensional coordinate arithmetic may leave a few ulps.  Do
        # not manufacture a pressure equation when the preceding velocity
        # equation was also exactly stationary and therefore has no matrix.
        roundoff_zero = 64.0 * np.finfo(float).eps
        if continuity_linf <= roundoff_zero:
            return fields
        if self._simple_inverse_momentum_diagonal is None:
            raise RuntimeError(
                "SIMPLE requires the diagonal of the velocity equation solved in stage (i)"
            )
        d_x, d_y = self._simple_inverse_momentum_diagonal
        matrix, volumes = _pressure_correction_matrix(
            state[..., 0], (d_x, d_y), case.x, case.y
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
        velocity = np.asarray(fields.velocity).copy()
        # Canonical SIMPLE applies the full velocity correction obtained from
        # the pressure-correction equation.  Pressure under-relaxation applies
        # to the pressure update, not to this flux correction.
        velocity[1:-1, 1:-1, 0] -= d_x[1:-1, 1:-1] * grad_x[1:-1, 1:-1]
        velocity[1:-1, 1:-1, 1] -= d_y[1:-1, 1:-1] * grad_y[1:-1, 1:-1]
        alpha = self.options.pressure_relaxation
        if self.uses_saturne_carrier:
            if self._saturne_total_pressure is None:
                self._saturne_total_pressure = np.asarray(fields.rho) * np.asarray(
                    fields.theta
                )
            # ``resopv`` corrects face fluxes with the complete increment;
            # ``navstv`` relaxes the stored pressure field separately.  The
            # augmented pressure system already enforces a zero-volume-mean
            # increment, matching the closed-domain pressure gauge.
            self._saturne_total_pressure = (
                self._saturne_total_pressure + alpha * pressure
            )
            if (
                not np.isfinite(self._saturne_total_pressure).all()
                or np.any(self._saturne_total_pressure <= 0.0)
            ):
                raise FloatingPointError(
                    "Code_Saturne total-pressure update produced invalid pressure"
                )
            return replace(fields, velocity=velocity)

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
    "CODE_SATURNE_V5_COMMIT",
    "CODE_SATURNE_V5_STEADY_FIELD_RELAXATION",
    "CODE_SATURNE_V5_STEADY_PRESSURE_RELAXATION",
    "FIELD_SLOTS",
    "GU_EMERSON_RECONSTRUCTION_PROVENANCE",
    "RANA_THERMOPHYSICAL_HISTORY_RELAXATION",
    "GuEmersonReconstructionOptions",
    "GuEmersonReconstructionResult",
    "GuEmersonSweepRecord",
    "make_gu_emerson_reconstruction_problem",
    "solve_gu_emerson_reconstruction",
]
