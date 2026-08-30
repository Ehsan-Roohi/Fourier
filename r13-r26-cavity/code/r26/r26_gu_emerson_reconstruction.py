#!/usr/bin/env python3
"""Documented field-by-field reconstruction of Gu--Emerson Sec. 5.2.

This is deliberately *not* described as the unavailable THOR source code.
It executes the printed Gu--Emerson field order.  Two explicitly named
equation backends are available: the historical physical-balance-defect
reconstruction and a direct finite-volume discretization of equation (63) in
``g,h,omega,gamma,chi``.  Numerical details that are not printed in the paper
are fixed below, reported in every result, and called a reconstruction:

* one coloured field-family defect matrix per fixed Picard refresh cycle;
* SuperLU sparse direct block solves (LSMR only if a block is singular);
* fixed under-relaxation, with no adaptive continuation or line search;
* the explicitly documented Rhie--Chow coefficient in ``r26_fv_backend``;
* local solution of the complete smooth-wall equations after every sweep;
* the repository's declared bilinear sharp-corner extension.

No global Newton, Krylov, homotopy, pseudo-arclength, clipping, filtering, or
Tikhonov row occurs in this module.  Direct equation-(63) acceptance requires
both its transformed finite-volume residual and the complete unscaled physical
R26 boundary-value residual, together with held continuity, mass and positivity.
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
from r26_gu_emerson_transformed_fv import (
    GuEmersonEquation63Consistency,
    GuEmersonEquation63PicardData,
    gu_emerson_equation63_consistency,
    gu_emerson_equation63_picard_data,
    gu_emerson_equation63_picard_residual,
    gu_emerson_equation63_terms,
    gu_emerson_transformed_fv_residual,
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
EquationBackend = Literal[
    "physical-balance-defect",
    "equation63-transformed-fv",
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
    equation_backend: EquationBackend = "physical-balance-defect"
    scalar_block_safeguard: bool = False
    outer_anderson_acceleration: bool = False
    outer_anderson_depth: int = 1
    outer_sweep_safeguard: bool = False
    outer_backtracking_factor: float = 0.5
    outer_minimum_step: float = 1.0 / 128.0
    outer_sufficient_decrease: float = 1.0e-4
    outer_nonmonotone_window: int = 1

    @classmethod
    def asme2009_equation63_source_backed(
        cls, **overrides: object
    ) -> "GuEmersonReconstructionOptions":
        """Return the direct transformed-PDE profile for ASME reproduction.

        The equation backend, field order, CUBISTA, central terms, SIMPLE and
        Rhie--Chow are fixed by Gu--Emerson.  The paper does not print
        relaxation values; the only available source-backed values are the
        steady Code_Saturne v5.0.3 defaults (0.7 fields, 0.3 pressure).  This
        method uses those values without importing Rana source histories or
        its lagged thermophysical-density carrier.
        """

        values: dict[str, object] = {
            "equation_backend": "equation63-transformed-fv",
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
            "use_rana_source_history": False,
            "pressure_density_coupling": "legacy_direct_mass_constrained",
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def asme2009_equation63_safeguarded_n8(
        cls, **overrides: object
    ) -> "GuEmersonReconstructionOptions":
        """Return the bounded N8 development profile for the direct PDE.

        This profile keeps the Code_Saturne field and pressure defaults used
        by :meth:`asme2009_equation63_source_backed`, but applies the existing
        declared local wall Picard factor and a bounded nonmonotone full-sweep
        backtracking safeguard.  Neither control is attributed to Gu--Emerson;
        the profile is an explicitly non-production globalization experiment.
        """

        values: dict[str, object] = {
            "wall_relaxation": 0.25,
            "scalar_block_safeguard": True,
            "outer_anderson_acceleration": True,
            "outer_anderson_depth": 1,
            "outer_sweep_safeguard": True,
            "outer_minimum_step": 1.0 / 4096.0,
            "outer_nonmonotone_window": 10,
            "chi_relaxation": 1.0,
        }
        values.update(overrides)
        return cls.asme2009_equation63_source_backed(**values)

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
        if self.equation_backend not in (
            "physical-balance-defect",
            "equation63-transformed-fv",
        ):
            raise ValueError("unknown Gu--Emerson equation backend")
        if (
            self.equation_backend == "equation63-transformed-fv"
            and self.use_rana_source_history
        ):
            raise ValueError(
                "direct equation-(63) sources cannot be replaced by Rana source history"
            )
        if (
            self.scalar_block_safeguard
            or self.outer_anderson_acceleration
            or self.outer_sweep_safeguard
        ) and (
            self.equation_backend != "equation63-transformed-fv"
            or self.pressure_density_coupling != "legacy_direct_mass_constrained"
        ):
            raise ValueError(
                "equation-(63) safeguards require the direct backend "
                "with mass-constrained density"
            )
        if not isinstance(self.scalar_block_safeguard, bool):
            raise TypeError("scalar_block_safeguard must be boolean")
        if not isinstance(self.outer_anderson_acceleration, bool):
            raise TypeError("outer_anderson_acceleration must be boolean")
        if not isinstance(self.outer_anderson_depth, int) or not (
            1 <= self.outer_anderson_depth <= 8
        ):
            raise ValueError("outer_anderson_depth must be an integer in [1,8]")
        if not isinstance(self.outer_nonmonotone_window, int) or not (
            1 <= self.outer_nonmonotone_window <= 20
        ):
            raise ValueError("outer_nonmonotone_window must be an integer in [1,20]")
        if self.outer_anderson_acceleration and not self.outer_sweep_safeguard:
            raise ValueError(
                "outer Anderson acceleration requires the fail-closed sweep safeguard"
            )
        if not isinstance(self.outer_sweep_safeguard, bool):
            raise TypeError("outer_sweep_safeguard must be boolean")
        if not (
            np.isfinite(self.outer_backtracking_factor)
            and 0.0 < self.outer_backtracking_factor < 1.0
            and np.isfinite(self.outer_minimum_step)
            and 0.0 < self.outer_minimum_step <= 1.0
            and np.isfinite(self.outer_sufficient_decrease)
            and 0.0 <= self.outer_sufficient_decrease < 1.0
        ):
            raise ValueError("outer safeguard controls are outside their admissible ranges")
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
        direct_equation63 = self.equation_backend == "equation63-transformed-fv"
        relaxation_source = (
            (
                "Code_Saturne v5.0.3 iniini.f90/modini.f90 at commit "
                f"{CODE_SATURNE_V5_COMMIT}; Rana physical-properties source "
                "SHA-256 a01d309692acf26093c65aa4c11453afc07f3f98b7be1bb8f2c1ea7ba2e44d5d"
            )
            if saturne_carrier
            else (
                "Code_Saturne v5.0.3 steady defaults at commit "
                f"{CODE_SATURNE_V5_COMMIT}; Gu--Emerson does not print relaxation values"
                if direct_equation63
                else tag
            )
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
                    else (
                        "direct equation-(63) central source identity from audited "
                        "Gu--Emerson physical equations; printed linear collision "
                        "sinks implicit; remaining source, viscosity and mass flux "
                        "frozen within each sequential field block"
                        if direct_equation63
                        else "Gu--Emerson field defect with no transferred Rana source history"
                    )
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
            "outer_sweep_globalization": NumericalControlSource(
                (
                    "enabled bounded nonmonotone full-sweep backtracking: "
                    f"factor={self.outer_backtracking_factor:g}, "
                    f"minimum_step={self.outer_minimum_step:g}, "
                    f"Armijo={self.outer_sufficient_decrease:g}, "
                    f"nonmonotone_window={self.outer_nonmonotone_window}"
                    if self.outer_sweep_safeguard
                    else "disabled"
                ),
                tag,
            ),
            "scalar_block_globalization": NumericalControlSource(
                (
                    "enabled equation-(63) nonlinear-Linf non-increase "
                    f"backtracking: factor={self.outer_backtracking_factor:g}, "
                    f"minimum_step={self.outer_minimum_step:g}"
                    if self.scalar_block_safeguard
                    else "disabled"
                ),
                tag,
            ),
            "outer_fixed_point_acceleration": NumericalControlSource(
                (
                    f"enabled depth-{self.outer_anderson_depth} Anderson "
                    "residual-minimizing affine mixing; "
                    "every candidate remains subject to the outer acceptance safeguard"
                    if self.outer_anderson_acceleration
                    else "disabled"
                ),
                tag,
            ),
        }
        return GuEmersonAlgorithmDisclosure(controls=controls)


@dataclass(frozen=True)
class GuEmersonSweepRecord:
    outer_iteration: int
    raw_gate: float
    transformed_equation63_linf: float
    scaled_linf: float
    held_continuity: float
    mass_error: float
    min_density: float
    min_temperature: float
    stage_order: tuple[str, ...]
    accepted_outer_step: float = 1.0
    backtracking_trials: int = 0
    normalized_acceptance_merit: float = float("nan")
    accepted_block_steps: tuple[tuple[str, float], ...] = ()
    block_backtracking_trials: tuple[tuple[str, int], ...] = ()
    anderson_used: bool = False
    anderson_current_weight: float = 1.0
    physical_point_linf: float = float("nan")
    transport_discretization_linf: float = float("nan")
    source_discretization_linf: float = float("nan")
    equation63_identity_roundoff: float = float("nan")
    physical_point_argmax_slot: int = -1
    transport_discretization_argmax_slot: int = -1
    source_discretization_argmax_slot: int = -1


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
    best_outer_iteration: int
    best_normalized_acceptance_merit: float
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
        self.accepted_block_steps: dict[str, float] = {}
        self.block_backtracking_trials: dict[str, int] = {}
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

    def _bulk_stage_residual(
        self,
        fields: GuEmersonFields,
        slots: tuple[int, ...],
        equation63_picard: GuEmersonEquation63PicardData | None = None,
    ) -> np.ndarray:
        state = self._physical(fields)
        if self.options.equation_backend == "equation63-transformed-fv":
            raw = (
                gu_emerson_transformed_fv_residual(fields, case=self.problem.case)
                if equation63_picard is None
                else gu_emerson_equation63_picard_residual(
                    fields,
                    case=self.problem.case,
                    frozen=equation63_picard,
                )
            )
        else:
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
        equation63_picard = None
        if self.options.equation_backend == "equation63-transformed-fv":
            equation63_picard = gu_emerson_equation63_picard_data(
                fields, case=self.problem.case
            )

        def with_vector(vector: np.ndarray) -> GuEmersonFields:
            value = np.exp(vector) if logarithmic else np.asarray(vector, dtype=float)
            candidate = packed.copy().ravel()
            candidate[full_indices] = value
            return gu_emerson_fields_from_planar17(candidate.reshape(packed.shape))

        def objective(vector: np.ndarray) -> np.ndarray:
            return self._bulk_stage_residual(
                with_vector(vector), slots, equation63_picard
            )

        residual = objective(encoded)
        if not np.isfinite(residual).all():
            raise FloatingPointError(f"{stage} defect contains NaN or infinity")
        if np.max(np.abs(residual), initial=0.0) == 0.0:
            self.accepted_block_steps[stage] = 0.0
            self.block_backtracking_trials[stage] = 0
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
        accepted_step = self.options.relaxation(stage)
        trials = 0
        updated = with_vector(encoded + accepted_step * correction)
        if self.options.scalar_block_safeguard and stage != "velocity":
            baseline_merit = float(
                np.max(
                    np.abs(
                        gu_emerson_transformed_fv_residual(
                            fields, case=self.problem.case
                        )[1:-1, 1:-1]
                    ),
                    initial=0.0,
                )
            )
            candidate_merit = float(
                np.max(
                    np.abs(
                        gu_emerson_transformed_fv_residual(
                            updated, case=self.problem.case
                        )[1:-1, 1:-1]
                    ),
                    initial=0.0,
                )
            )
            while candidate_merit > baseline_merit * (1.0 + 1.0e-12):
                accepted_step *= self.options.outer_backtracking_factor
                trials += 1
                if accepted_step < self.options.outer_minimum_step:
                    accepted_step = 0.0
                    updated = fields
                    break
                updated = with_vector(encoded + accepted_step * correction)
                candidate_merit = float(
                    np.max(
                        np.abs(
                            gu_emerson_transformed_fv_residual(
                                updated, case=self.problem.case
                            )[1:-1, 1:-1]
                        ),
                        initial=0.0,
                    )
                )
        self.accepted_block_steps[stage] = accepted_step
        self.block_backtracking_trials[stage] = trials
        return updated

    def solve_velocity(self, fields: GuEmersonFields) -> GuEmersonFields:
        self.accepted_block_steps = {}
        self.block_backtracking_trials = {}
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
        physical_continuity = (
            gu_emerson_transformed_fv_residual(fields, case=case)[..., 0]
            if self.options.equation_backend == "equation63-transformed-fv"
            else self.problem._bulk(state, mu)[..., 0]
        )
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


def _sweep_metrics(
    problem: R26NodeBVP,
    fields: GuEmersonFields,
    state: np.ndarray,
    options: GuEmersonReconstructionOptions,
) -> tuple[
    float,
    object,
    bool,
    float,
    float,
    GuEmersonEquation63Consistency,
]:
    """Return every acceptance metric and their tolerance-normalized maximum."""

    raw, diagnostics, positive = _gate(problem, state)
    if options.equation_backend == "equation63-transformed-fv":
        terms = gu_emerson_equation63_terms(fields, case=problem.case)
        transformed_linf = float(
            np.max(np.abs(terms.residual[1:-1, 1:-1]), initial=0.0)
        )
        consistency = gu_emerson_equation63_consistency(terms)
    else:
        transformed_linf = float("nan")
        consistency = GuEmersonEquation63Consistency(
            physical_point_linf=float("nan"),
            transport_discretization_linf=float("nan"),
            source_discretization_linf=float("nan"),
            identity_roundoff=float("nan"),
            physical_point_argmax_slot=-1,
            transport_discretization_argmax_slot=-1,
            source_discretization_argmax_slot=-1,
        )
    normalized = [
        raw / options.raw_tolerance,
        diagnostics.total_linf / options.scaled_tolerance,
        abs(diagnostics.held_out_continuity)
        / options.held_continuity_tolerance,
        abs(diagnostics.mass_error) / options.mass_tolerance,
    ]
    if options.equation_backend == "equation63-transformed-fv":
        normalized.append(transformed_linf / options.raw_tolerance)
    merit = float(max(normalized)) if positive else float("inf")
    return raw, diagnostics, positive, transformed_linf, merit, consistency


def _anderson_candidate(
    map_states: list[np.ndarray],
    map_residuals: list[np.ndarray],
    component_scale: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    """Return a residual-minimizing affine mix of fixed-point maps."""

    scale = np.asarray(component_scale, dtype=float)
    if scale.shape != (17,) or np.any(scale <= 0.0):
        raise ValueError("Anderson component scale must contain 17 positive values")
    if len(map_states) != len(map_residuals) or len(map_states) < 2:
        raise ValueError("Anderson mixing requires matching histories of length >= 2")
    residual_columns = [
        (np.asarray(value, dtype=float) / scale).ravel()
        for value in map_residuals
    ]
    reference = residual_columns[-1]
    differences = np.column_stack(
        [value - reference for value in residual_columns[:-1]]
    )
    if not np.isfinite(differences).all() or not np.isfinite(reference).all():
        return np.asarray(map_states[-1], dtype=float), 1.0, False
    coefficients, _, rank, _ = np.linalg.lstsq(
        differences, -reference, rcond=None
    )
    if rank == 0 or not np.isfinite(coefficients).all():
        return np.asarray(map_states[-1], dtype=float), 1.0, False
    weights = np.concatenate((coefficients, (1.0 - float(np.sum(coefficients)),)))
    candidate = np.zeros_like(np.asarray(map_states[-1], dtype=float))
    for weight, mapped in zip(weights, map_states, strict=True):
        candidate += weight * np.asarray(mapped, dtype=float)
    current_weight = float(weights[-1])
    if (
        not np.isfinite(current_weight)
        or not np.isfinite(candidate).all()
        or np.any(candidate[..., 0] <= 0.0)
        or np.any(candidate[..., 3] <= 0.0)
    ):
        return np.asarray(map_states[-1], dtype=float), 1.0, False
    return candidate, current_weight, True


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
    baseline = _sweep_metrics(problem, fields, state, options)
    accepted_merit_history = [baseline[4]]
    best_state = state.copy()
    best_fields = fields
    best_outer_iteration = 0
    best_merit = baseline[4]
    map_state_history: list[np.ndarray] = []
    map_residual_history: list[np.ndarray] = []

    for outer in range(1, options.max_outer_iterations + 1):
        previous_state = state
        previous_fields = fields
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
        current_map_state = sweep.physical_state
        current_map_residual = current_map_state - previous_state
        proposed_fields = sweep.fields
        proposed_state = current_map_state
        anderson_used = False
        anderson_current_weight = 1.0
        if (
            options.outer_anderson_acceleration
            and map_state_history
        ):
            proposed_state, anderson_current_weight, anderson_used = (
                _anderson_candidate(
                    map_state_history + [current_map_state],
                    map_residual_history + [current_map_residual],
                    problem.case.scaling.bulk,
                )
            )
            if anderson_used:
                proposed_fields = gu_emerson_fields_from_state(
                    proposed_state,
                    x=problem.case.x,
                    y=problem.case.y,
                    mu=problem.case.mu(proposed_state[..., 3]),
                )
        accepted_step = 1.0
        backtracking_trials = 0
        rejected = False
        if not options.outer_sweep_safeguard:
            metrics = _sweep_metrics(
                problem, proposed_fields, proposed_state, options
            )
        else:
            baseline_merit = max(
                accepted_merit_history[-options.outer_nonmonotone_window :]
            )
            candidates = [(proposed_state, proposed_fields, anderson_used)]
            if anderson_used:
                candidates.append((current_map_state, sweep.fields, False))
            accepted = False
            for full_step_state, full_step_fields, candidate_uses_anderson in candidates:
                accepted_step = 1.0
                proposed_state = full_step_state
                proposed_fields = full_step_fields
                metrics = _sweep_metrics(
                    problem, proposed_fields, proposed_state, options
                )
                while metrics[4] > baseline_merit * (
                    1.0 - options.outer_sufficient_decrease * accepted_step
                ):
                    accepted_step *= options.outer_backtracking_factor
                    backtracking_trials += 1
                    if accepted_step < options.outer_minimum_step:
                        break
                    proposed_state = previous_state + accepted_step * (
                        full_step_state - previous_state
                    )
                    if (
                        np.any(proposed_state[..., 0] <= 0.0)
                        or np.any(proposed_state[..., 3] <= 0.0)
                    ):
                        metrics = (
                            float("inf"),
                            baseline[1],
                            False,
                            float("inf"),
                            float("inf"),
                            baseline[5],
                        )
                        continue
                    proposed_fields = gu_emerson_fields_from_state(
                        proposed_state,
                        x=problem.case.x,
                        y=problem.case.y,
                        mu=problem.case.mu(proposed_state[..., 3]),
                    )
                    metrics = _sweep_metrics(
                        problem, proposed_fields, proposed_state, options
                    )
                if accepted_step >= options.outer_minimum_step:
                    anderson_used = candidate_uses_anderson
                    if not anderson_used:
                        anderson_current_weight = 1.0
                    accepted = True
                    break
            if not accepted:
                accepted_step = 0.0
                proposed_fields = previous_fields
                proposed_state = previous_state
                metrics = baseline
                rejected = True
                anderson_used = False
                anderson_current_weight = 1.0
                message = (
                    "bounded outer safeguard rejected both the Anderson "
                    "candidate and raw published-order sweep at its minimum step"
                )
        fields = proposed_fields
        state = proposed_state
        raw, diagnostics, positive, transformed_linf, merit, consistency = metrics
        record = GuEmersonSweepRecord(
            outer_iteration=outer,
            raw_gate=raw,
            transformed_equation63_linf=transformed_linf,
            scaled_linf=diagnostics.total_linf,
            held_continuity=diagnostics.held_out_continuity,
            mass_error=diagnostics.mass_error,
            min_density=diagnostics.min_density,
            min_temperature=diagnostics.min_temperature,
            stage_order=tuple(operators.executed_stages),
            accepted_outer_step=accepted_step,
            backtracking_trials=backtracking_trials,
            normalized_acceptance_merit=merit,
            accepted_block_steps=tuple(
                (name, operators.accepted_block_steps.get(name, float("nan")))
                for name in FIELD_SLOTS
            ),
            block_backtracking_trials=tuple(
                (name, operators.block_backtracking_trials.get(name, 0))
                for name in FIELD_SLOTS
            ),
            anderson_used=anderson_used,
            anderson_current_weight=anderson_current_weight,
            physical_point_linf=consistency.physical_point_linf,
            transport_discretization_linf=(
                consistency.transport_discretization_linf
            ),
            source_discretization_linf=(
                consistency.source_discretization_linf
            ),
            equation63_identity_roundoff=consistency.identity_roundoff,
            physical_point_argmax_slot=consistency.physical_point_argmax_slot,
            transport_discretization_argmax_slot=(
                consistency.transport_discretization_argmax_slot
            ),
            source_discretization_argmax_slot=(
                consistency.source_discretization_argmax_slot
            ),
        )
        records.append(record)
        if record_callback is not None:
            record_callback(record, state)
        if not rejected and merit < best_merit:
            best_state = state.copy()
            best_fields = fields
            best_outer_iteration = outer
            best_merit = merit
        converged = bool(
            positive
            and raw <= options.raw_tolerance
            and (
                options.equation_backend != "equation63-transformed-fv"
                or transformed_linf <= options.raw_tolerance
            )
            and diagnostics.total_linf <= options.scaled_tolerance
            and abs(diagnostics.held_out_continuity)
            <= options.held_continuity_tolerance
            and abs(diagnostics.mass_error) <= options.mass_tolerance
        )
        if converged:
            message = (
                "direct equation-(63) and complete raw physical R26 gates reached "
                "by published-order transformed-variable solve"
                if options.equation_backend == "equation63-transformed-fv"
                else "complete raw R26 gate reached by published-order reconstruction"
            )
            break
        if rejected:
            break
        map_state_history.append(current_map_state.copy())
        map_residual_history.append(current_map_residual.copy())
        if len(map_state_history) > options.outer_anderson_depth:
            map_state_history.pop(0)
            map_residual_history.pop(0)
        baseline = metrics
        accepted_merit_history.append(merit)

    if options.outer_sweep_safeguard and not converged:
        state = best_state
        fields = best_fields
        message = (
            f"{message}; returning best accepted sweep {best_outer_iteration}"
        )
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
        best_outer_iteration=best_outer_iteration,
        best_normalized_acceptance_merit=best_merit,
    )


__all__ = [
    "CODE_SATURNE_V5_COMMIT",
    "CODE_SATURNE_V5_STEADY_FIELD_RELAXATION",
    "CODE_SATURNE_V5_STEADY_PRESSURE_RELAXATION",
    "EquationBackend",
    "FIELD_SLOTS",
    "GU_EMERSON_RECONSTRUCTION_PROVENANCE",
    "RANA_THERMOPHYSICAL_HISTORY_RELAXATION",
    "GuEmersonReconstructionOptions",
    "GuEmersonReconstructionResult",
    "GuEmersonSweepRecord",
    "make_gu_emerson_reconstruction_problem",
    "solve_gu_emerson_reconstruction",
]
