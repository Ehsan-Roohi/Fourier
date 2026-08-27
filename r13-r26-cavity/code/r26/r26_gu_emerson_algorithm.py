#!/usr/bin/env python3
"""Published Gu--Emerson R26 segregated iteration contract.

Section 5.2 of Gu & Emerson, JFM 636 (2009), prints a six-stage steady
iteration.  This module makes that order executable without relabelling the
repository's monolithic Krylov path as the paper algorithm.

The paper specifies finite volume, a collocated arrangement, CUBISTA,
SIMPLE, Rhie--Chow, central diffusion/source differences, and the field
order below.  It does *not* publish the linear solver, under-relaxation
factors, source-term implicitness, the Rhie--Chow coefficient, a corner rule,
or numerical convergence thresholds.  Consequently this module is an
algorithm driver and disclosure boundary: concrete operators must be supplied
from a cited implementation, and production authorization fails closed until
all unpublished controls have declared provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Final, Mapping

import numpy as np

from r26_tensor_closures import closure_coefficients
from r26_gu_emerson_variables import (
    GuEmersonFields,
    state_from_gu_emerson_fields,
)


GU_EMERSON_STAGE_ORDER: Final[tuple[str, ...]] = (
    "velocity",
    "simple_pressure_correction",
    "temperature",
    "g",
    "h",
    "omega",
    "gamma",
    "chi",
    "physical_moment_reconstruction",
    "wall_boundary_update",
)

GU_EMERSON_UNPUBLISHED_CONTROLS: Final[tuple[str, ...]] = (
    "linear_solver",
    "under_relaxation_factors",
    "source_term_linearisation",
    "rhie_chow_face_coefficient",
    "sharp_corner_treatment",
    "numerical_convergence_thresholds",
)


@dataclass(frozen=True)
class GuEmersonFieldEquation:
    """One convection--diffusion equation in the printed (56)--(63) system."""

    field: str
    equation: int
    diffusion_multiplier: float


def gu_emerson_field_equations(
    coefficient_mode: str = "jfm2009",
) -> tuple[GuEmersonFieldEquation, ...]:
    """Return the exact equation order and diffusion multipliers of Eq. (63)."""

    coefficients = closure_coefficients(coefficient_mode)
    return (
        GuEmersonFieldEquation("velocity", 56, 1.0),
        GuEmersonFieldEquation("temperature", 57, 2.0 / 5.0),
        GuEmersonFieldEquation("g", 58, 3.0 / 2.0),
        GuEmersonFieldEquation("h", 59, 5.0 / 6.0),
        GuEmersonFieldEquation("omega", 60, coefficients.C1),
        GuEmersonFieldEquation("gamma", 61, 7.0 * coefficients.Y1 / 9.0),
        GuEmersonFieldEquation("chi", 62, 3.0 / 7.0),
    )


@dataclass(frozen=True)
class NumericalControlSource:
    """One non-paper numerical choice and the external source that fixes it."""

    value: str
    provenance: str

    def __post_init__(self) -> None:
        if not self.value.strip() or not self.provenance.strip():
            raise ValueError("a numerical control requires both value and provenance")


@dataclass(frozen=True)
class GuEmersonAlgorithmDisclosure:
    """Machine-checkable distinction between printed and unpublished details."""

    spatial_discretization: str = "collocated finite volume"
    convection: str = "CUBISTA"
    diffusion_and_sources: str = "central difference"
    pressure_velocity_coupling: str = "SIMPLE"
    face_velocity_interpolation: str = "Rhie--Chow"
    controls: Mapping[str, NumericalControlSource] | None = None

    @property
    def unresolved_controls(self) -> tuple[str, ...]:
        declared = set(() if self.controls is None else self.controls)
        return tuple(name for name in GU_EMERSON_UNPUBLISHED_CONTROLS if name not in declared)

    @property
    def production_authorized(self) -> bool:
        return not self.unresolved_controls

    def require_production_authorization(self) -> None:
        unresolved = self.unresolved_controls
        if unresolved:
            raise RuntimeError(
                "Gu--Emerson production run is not reproducible from the paper alone; "
                "missing externally sourced controls: " + ", ".join(unresolved)
            )


VelocitySolve = Callable[[GuEmersonFields], GuEmersonFields]
SimpleCorrection = Callable[[GuEmersonFields], GuEmersonFields]
ScalarOrTensorSolve = Callable[[GuEmersonFields], np.ndarray]
WallUpdate = Callable[[np.ndarray, GuEmersonFields], GuEmersonFields]


@dataclass(frozen=True)
class GuEmersonSegregatedOperators:
    """Concrete field operators consumed in the exact published order.

    Each callback owns only one printed stage.  This makes a monolithic root
    solve, pseudo-arclength continuation, homotopy, or a frozen global
    Jacobian structurally impossible inside this driver.
    """

    solve_velocity: VelocitySolve
    simple_pressure_correction: SimpleCorrection
    solve_temperature: ScalarOrTensorSolve
    solve_g: ScalarOrTensorSolve
    solve_h: ScalarOrTensorSolve
    solve_omega: ScalarOrTensorSolve
    solve_gamma: ScalarOrTensorSolve
    solve_chi: ScalarOrTensorSolve
    update_wall_boundaries: WallUpdate


@dataclass(frozen=True)
class GuEmersonOuterIteration:
    """State returned after one complete printed Gu--Emerson outer iteration."""

    fields: GuEmersonFields
    physical_state: np.ndarray
    stage_order: tuple[str, ...] = GU_EMERSON_STAGE_ORDER


def advance_gu_emerson_outer_iteration(
    fields: GuEmersonFields,
    operators: GuEmersonSegregatedOperators,
    *,
    x: np.ndarray,
    y: np.ndarray,
    viscosity: Callable[[np.ndarray], np.ndarray],
) -> GuEmersonOuterIteration:
    """Execute one outer iteration exactly in the order printed in Sec. 5.2."""

    current = operators.solve_velocity(fields)
    if not isinstance(current, GuEmersonFields):
        raise TypeError("velocity stage must return GuEmersonFields")
    current = operators.simple_pressure_correction(current)
    if not isinstance(current, GuEmersonFields):
        raise TypeError("SIMPLE stage must return GuEmersonFields")

    for name, solve in (
        ("theta", operators.solve_temperature),
        ("g", operators.solve_g),
        ("h", operators.solve_h),
        ("omega", operators.solve_omega),
        ("gamma", operators.solve_gamma),
        ("chi", operators.solve_chi),
    ):
        value = np.asarray(solve(current), dtype=float)
        current = replace(current, **{name: value})

    mu = np.asarray(viscosity(np.asarray(current.theta, dtype=float)), dtype=float)
    physical = state_from_gu_emerson_fields(current, x=x, y=y, mu=mu)
    current = operators.update_wall_boundaries(physical, current)
    if not isinstance(current, GuEmersonFields):
        raise TypeError("wall stage must return GuEmersonFields")
    mu = np.asarray(viscosity(np.asarray(current.theta, dtype=float)), dtype=float)
    physical = state_from_gu_emerson_fields(current, x=x, y=y, mu=mu)
    return GuEmersonOuterIteration(fields=current, physical_state=physical)


__all__ = [
    "GU_EMERSON_STAGE_ORDER",
    "GU_EMERSON_UNPUBLISHED_CONTROLS",
    "GuEmersonAlgorithmDisclosure",
    "GuEmersonFieldEquation",
    "GuEmersonOuterIteration",
    "GuEmersonSegregatedOperators",
    "NumericalControlSource",
    "advance_gu_emerson_outer_iteration",
    "gu_emerson_field_equations",
]
