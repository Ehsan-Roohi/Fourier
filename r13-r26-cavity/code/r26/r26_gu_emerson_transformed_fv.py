#!/usr/bin/env python3
"""Direct finite-volume form of Gu--Emerson equations (56)--(63).

The earlier reconstruction advanced the equation-(48) variables, but formed
each block defect by selecting rows from a discretized physical-moment BVP.
That is a useful algebraic check; it is not the discretization printed in
Gu & Emerson, JFM 636 (2009), section 5.2.

This module discretizes the transformed steady equation

    div(rho*u*Phi) - div(mu/Gamma_Phi * grad(Phi)) = S_Phi

directly for ``Phi=(u,T,g,h,omega,gamma,chi)``.  CUBISTA is used for the
convected transformed field, central face differences for diffusion, the
existing Rhie--Chow mass flux for pressure--velocity coupling, and the same
wall-bounded conservative control volumes as the independent physical R26
gate.  The source is evaluated without transcribing the very long expanded
right-hand sides (58)--(62): at the *continuous, centrally differentiated*
level it is the exact algebraic identity

    S_Phi = [div(rho*u*Phi) - div(mu/Gamma_Phi grad(Phi))]
            - R_physical(Phi),

where ``R_physical`` is the audited pointwise Gu--Emerson R26 equation after
the exact (48)--(55) reconstruction.  No finite-volume physical residual is
used in this identity.  This preserves every printed nonlinear source and
closure contraction while making convection and the principal diffusion
operator a genuine equation-(63) finite-volume discretization.

Only interior entries are balance rows.  Smooth-wall and corner rows remain
owned by ``r26_discretization`` and are independently checked in physical
moments.  The paper does not publish a sharp-corner rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from r26_bulk_equations import bulk_residual_grid
from r26_cases import CavityCase
from r26_fv_backend import (
    compatible_face_fields,
    cubista_face_value,
    impermeable_wall_mass_divergence,
    wall_bounded_face_divergence,
)
from r26_gu_emerson_variables import (
    GuEmersonFields,
    gu_emerson_fields_as_planar17,
    state_from_gu_emerson_fields,
)
from r26_tensor_closures import closure_coefficients, gu_emerson_closures


GU_EMERSON_TRANSFORMED_FV_PROVENANCE: Final[str] = (
    "Gu--Emerson JFM 636 (2009), Eqs. (48)--(63) and Sec. 5.2: direct "
    "transformed-variable FV, CUBISTA convection, central diffusion/source, "
    "SIMPLE/Rhie--Chow mass flux; source identity evaluated from the audited "
    "pointwise physical R26 equations"
)


@dataclass(frozen=True)
class GuEmersonEquation63Terms:
    """Auditable terms of one direct transformed finite-volume evaluation."""

    residual: np.ndarray
    finite_volume_lhs: np.ndarray
    central_point_lhs: np.ndarray
    source: np.ndarray
    physical_point_residual: np.ndarray
    gamma_by_slot: np.ndarray
    provenance: str = GU_EMERSON_TRANSFORMED_FV_PROVENANCE


@dataclass(frozen=True)
class GuEmersonEquation63PicardData:
    """Coefficients held fixed during one segregated field solve.

    Gu--Emerson solve equation (63) as a sequence of convection--diffusion
    equations.  The source, viscosity and mass flux entering one field block
    are therefore evaluated from the latest iterate and are not differentiated
    with respect to that block's unknown.  Rebuilding this object before the
    next field retains the printed sequential coupling.
    """

    explicit_source: np.ndarray
    implicit_sink_by_slot: np.ndarray
    mu: np.ndarray
    gamma_by_slot: np.ndarray
    mass_x: np.ndarray
    mass_y: np.ndarray
    provenance: str = (
        "Gu--Emerson JFM 636 (2009), Sec. 5.2 segregated Picard stage: "
        "central source and transport coefficients evaluated at stage entry"
    )


def equation63_gamma_by_slot(coefficient_mode: str) -> np.ndarray:
    """Return the printed ``Gamma_Phi`` in planar-17 storage order.

    Slot zero is continuity and therefore has no equation-(63) diffusion
    coefficient; it is represented by ``inf`` so accidental division cannot
    introduce a density diffusion term.
    """

    coefficients = closure_coefficients(coefficient_mode)
    gamma = np.full(17, np.inf, dtype=float)
    gamma[1:3] = 1.0
    gamma[3] = 2.0 / 5.0
    gamma[4:6] = 5.0 / 6.0
    gamma[6:9] = 3.0 / 2.0
    gamma[9:12] = 7.0 * coefficients.Y1 / 9.0
    gamma[12:16] = coefficients.C1
    gamma[16] = 3.0 / 7.0
    return gamma


def _coordinates(value: np.ndarray, size: int, name: str) -> np.ndarray:
    coordinate = np.asarray(value, dtype=float)
    if (
        coordinate.shape != (size,)
        or not np.isfinite(coordinate).all()
        or np.any(np.diff(coordinate) <= 0.0)
    ):
        raise ValueError(f"{name} must be finite, increasing, and length {size}")
    return coordinate


def _face_average(field: np.ndarray, axis: int) -> np.ndarray:
    value = np.asarray(field, dtype=float)
    if axis == 1:
        return 0.5 * (value[:, 1:] + value[:, :-1])
    if axis == 0:
        return 0.5 * (value[1:] + value[:-1])
    raise ValueError("face axis must be zero or one")


def _normal_face_gradient(
    field: np.ndarray, coordinate: np.ndarray, *, axis: int
) -> np.ndarray:
    value = np.asarray(field, dtype=float)
    trailing = (1,) * (value.ndim - 2)
    if axis == 1:
        spacing = np.diff(coordinate).reshape((1, coordinate.size - 1) + trailing)
        return (value[:, 1:] - value[:, :-1]) / spacing
    if axis == 0:
        spacing = np.diff(coordinate).reshape((coordinate.size - 1, 1) + trailing)
        return (value[1:] - value[:-1]) / spacing
    raise ValueError("face axis must be zero or one")


def _central_point_lhs(
    packed: np.ndarray,
    physical_point_residual: np.ndarray,
    mu: np.ndarray,
    gamma: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Evaluate the differential left side of equation (63) centrally."""

    rho = packed[..., 0]
    velocity = packed[..., 1:3]
    mass = physical_point_residual[..., 0]
    result = np.zeros_like(packed)
    result[..., 0] = mass
    for slot in range(1, packed.shape[-1]):
        field = packed[..., slot]
        d_dx = np.gradient(field, x, axis=1, edge_order=2)
        d_dy = np.gradient(field, y, axis=0, edge_order=2)
        convection = field * mass + rho * (
            velocity[..., 0] * d_dx + velocity[..., 1] * d_dy
        )
        coefficient = mu / gamma[slot]
        diffusion = np.gradient(coefficient * d_dx, x, axis=1, edge_order=2)
        diffusion += np.gradient(coefficient * d_dy, y, axis=0, edge_order=2)
        result[..., slot] = convection - diffusion
    return result


def _finite_volume_lhs(
    packed: np.ndarray,
    mu: np.ndarray,
    gamma: np.ndarray,
    mass_x: np.ndarray,
    mass_y: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Discretize equation-(63) transition/convection/diffusion terms."""

    result = np.zeros_like(packed)
    result[..., 0] = impermeable_wall_mass_divergence(mass_x, mass_y, x, y)
    for slot in range(1, packed.shape[-1]):
        field = packed[..., slot]
        face_x = cubista_face_value(field, mass_x, axis=1)
        face_y = cubista_face_value(field, mass_y, axis=0)
        coefficient_x = _face_average(mu / gamma[slot], 1)
        coefficient_y = _face_average(mu / gamma[slot], 0)
        flux_x = mass_x * face_x - coefficient_x * _normal_face_gradient(
            field, x, axis=1
        )
        flux_y = mass_y * face_y - coefficient_y * _normal_face_gradient(
            field, y, axis=0
        )

        d_dx = np.gradient(field, x, axis=1, edge_order=2)
        d_dy = np.gradient(field, y, axis=0, edge_order=2)
        wall_x = np.zeros_like(field)
        wall_y = np.zeros_like(field)
        # Impermeability makes the transformed convective wall flux zero.
        # The principal diffusion flux uses the same second-order one-sided
        # wall derivative that supplies the central source evaluation.
        wall_x[:, 0] = -(mu[:, 0] / gamma[slot]) * d_dx[:, 0]
        wall_x[:, -1] = -(mu[:, -1] / gamma[slot]) * d_dx[:, -1]
        wall_y[0, :] = -(mu[0, :] / gamma[slot]) * d_dy[0, :]
        wall_y[-1, :] = -(mu[-1, :] / gamma[slot]) * d_dy[-1, :]
        result[..., slot] = wall_bounded_face_divergence(
            flux_x, flux_y, wall_x, wall_y, x, y
        )
    return result


def gu_emerson_equation63_terms(
    fields: GuEmersonFields,
    *,
    case: CavityCase,
) -> GuEmersonEquation63Terms:
    """Return every term of the direct transformed finite-volume equations."""

    x = _coordinates(case.x, case.nodes, "x")
    y = _coordinates(case.y, case.nodes, "y")
    packed = gu_emerson_fields_as_planar17(fields)
    if packed.shape != (case.nodes, case.nodes, 17):
        raise ValueError(
            "transformed fields must match the square case grid and planar-17 layout"
        )
    if np.any(packed[..., 0] <= 0.0) or np.any(packed[..., 3] <= 0.0):
        raise FloatingPointError("transformed FV requires positive rho and theta")
    mu = np.asarray(case.mu(packed[..., 3]), dtype=float)
    physical = state_from_gu_emerson_fields(fields, x=x, y=y, mu=mu)
    physical_point = bulk_residual_grid(
        physical,
        x=x,
        y=y,
        mu=mu,
        case=case,
        edge_order=2,
    )
    gamma = equation63_gamma_by_slot(case.r26_closure_mode)
    central_lhs = _central_point_lhs(
        packed, physical_point, mu, gamma, x, y
    )
    source = central_lhs - physical_point

    closures = gu_emerson_closures(
        physical,
        x=x,
        y=y,
        mu=mu,
        edge_order=2,
        coefficient_mode=case.r26_closure_mode,
    )
    faces = compatible_face_fields(physical, x, y, mu, closures)
    finite_volume_lhs = _finite_volume_lhs(
        packed, mu, gamma, faces.mass_x, faces.mass_y, x, y
    )
    residual = finite_volume_lhs - source
    # Boundary entries are not transformed balance rows.  Keeping them zero
    # prevents a caller from mistaking the diagnostic source values for WBCs.
    residual[[0, -1], :, :] = 0.0
    residual[:, [0, -1], :] = 0.0
    finite_volume_lhs[[0, -1], :, :] = 0.0
    finite_volume_lhs[:, [0, -1], :] = 0.0
    if not all(
        np.isfinite(value).all()
        for value in (residual, finite_volume_lhs, central_lhs, source, physical_point)
    ):
        raise FloatingPointError("equation-(63) evaluation produced NaN or infinity")
    return GuEmersonEquation63Terms(
        residual=residual,
        finite_volume_lhs=finite_volume_lhs,
        central_point_lhs=central_lhs,
        source=source,
        physical_point_residual=physical_point,
        gamma_by_slot=gamma,
    )


def gu_emerson_equation63_picard_data(
    fields: GuEmersonFields,
    *,
    case: CavityCase,
) -> GuEmersonEquation63PicardData:
    """Freeze one stage's equation-(63) source and transport coefficients."""

    terms = gu_emerson_equation63_terms(fields, case=case)
    mu = np.asarray(case.mu(fields.theta), dtype=float)
    physical = state_from_gu_emerson_fields(
        fields,
        x=case.x,
        y=case.y,
        mu=mu,
    )
    closures = gu_emerson_closures(
        physical,
        x=case.x,
        y=case.y,
        mu=mu,
        edge_order=2,
        coefficient_mode=case.r26_closure_mode,
    )
    faces = compatible_face_fields(physical, case.x, case.y, mu, closures)
    packed = gu_emerson_fields_as_planar17(fields)
    pressure = packed[..., 0] * packed[..., 3]
    collision = pressure * packed[..., 0] / mu
    sink = np.zeros_like(packed)
    sink[..., 6:9] = collision[..., None]
    sink[..., 4:6] = (2.0 / 3.0 * collision)[..., None]
    sink[..., 12:16] = (3.0 / 2.0 * collision)[..., None]
    sink[..., 9:12] = (7.0 / 6.0 * collision)[..., None]
    sink[..., 16] = 2.0 / 3.0 * collision
    # Equations (58)--(62) print ``-a*(p/mu)*rho*Phi`` on the
    # right-hand side.  Move that dissipative linear term to the implicit
    # diagonal; all remaining source terms stay at their stage-entry values.
    explicit_source = np.asarray(terms.source, dtype=float) + sink * packed
    return GuEmersonEquation63PicardData(
        explicit_source=explicit_source.copy(),
        implicit_sink_by_slot=sink,
        mu=mu.copy(),
        gamma_by_slot=np.asarray(terms.gamma_by_slot, dtype=float).copy(),
        mass_x=np.asarray(faces.mass_x, dtype=float).copy(),
        mass_y=np.asarray(faces.mass_y, dtype=float).copy(),
    )


def gu_emerson_equation63_picard_residual(
    fields: GuEmersonFields,
    *,
    case: CavityCase,
    frozen: GuEmersonEquation63PicardData,
) -> np.ndarray:
    """Evaluate one equation-(63) block with its right-hand side frozen."""

    packed = gu_emerson_fields_as_planar17(fields)
    expected = (case.nodes, case.nodes, 17)
    if packed.shape != expected or frozen.explicit_source.shape != expected:
        raise ValueError("Picard data and transformed fields must match the case grid")
    if np.any(packed[..., 0] <= 0.0) or np.any(packed[..., 3] <= 0.0):
        raise FloatingPointError("transformed FV requires positive rho and theta")
    finite_volume_lhs = _finite_volume_lhs(
        packed,
        frozen.mu,
        frozen.gamma_by_slot,
        frozen.mass_x,
        frozen.mass_y,
        case.x,
        case.y,
    )
    residual = (
        finite_volume_lhs
        + frozen.implicit_sink_by_slot * packed
        - frozen.explicit_source
    )
    residual[[0, -1], :, :] = 0.0
    residual[:, [0, -1], :] = 0.0
    if not np.isfinite(residual).all():
        raise FloatingPointError("equation-(63) Picard residual is non-finite")
    return residual


def gu_emerson_transformed_fv_residual(
    fields: GuEmersonFields,
    *,
    case: CavityCase,
) -> np.ndarray:
    """Return the direct equation-(63) finite-volume residual."""

    return gu_emerson_equation63_terms(fields, case=case).residual


__all__ = [
    "GU_EMERSON_TRANSFORMED_FV_PROVENANCE",
    "GuEmersonEquation63PicardData",
    "GuEmersonEquation63Terms",
    "equation63_gamma_by_slot",
    "gu_emerson_equation63_picard_data",
    "gu_emerson_equation63_picard_residual",
    "gu_emerson_equation63_terms",
    "gu_emerson_transformed_fv_residual",
]
