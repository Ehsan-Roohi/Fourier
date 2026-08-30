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
gate.  The source is evaluated without transcribing every expanded tensor
component: cell-centred transformed and physical fluxes are subtracted before
central face interpolation.  This is algebraically the printed right-hand
side of equations (56)--(62), while avoiding the difference of two nonlinear
CUBISTA reconstructions.  The remaining collision and nonlinear terms are
evaluated directly from ``Sigma,Q,M,S,N``.  This path does not call the
physical BVP residual.

Only interior entries are balance rows.  Smooth-wall and corner rows remain
owned by ``r26_discretization`` and are independently checked in physical
moments.  The paper does not publish a sharp-corner rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from r26_bulk_equations import (
    R26BulkResidual,
    bulk_residual_grid,
    closure_derivatives_on_grid,
    gu_emerson_nonlinear_sources,
)
from r26_cases import CavityCase
from r26_fv_backend import (
    CompatibleFaceFields,
    compatible_face_fields,
    compatible_wall_fluxes,
    cubista_face_value,
    wall_bounded_face_divergence,
)
from r26_gu_emerson_variables import (
    GuEmersonFields,
    gu_emerson_fields_as_planar17,
    state_from_gu_emerson_fields,
)
from r26_state import planar_state_to_tensors
from r26_tensor_closures import (
    R26Closures,
    closure_coefficients,
    finite_difference_gradients,
    gu_emerson_closures,
    stf2_project,
    stf3_project,
)


GU_EMERSON_TRANSFORMED_FV_PROVENANCE: Final[str] = (
    "Gu--Emerson JFM 636 (2009), Eqs. (48)--(63) and Sec. 5.2: direct "
    "transformed-variable FV, CUBISTA convection, direct central source "
    "fluxes for Eqs. (56)--(62), and SIMPLE/Rhie--Chow mass flux"
)


@dataclass(frozen=True)
class GuEmersonEquation63Terms:
    """Auditable terms of one direct transformed finite-volume evaluation."""

    residual: np.ndarray
    finite_volume_lhs: np.ndarray
    central_point_lhs: np.ndarray
    source: np.ndarray
    source_transport: np.ndarray
    physical_local_terms: np.ndarray
    compatible_source: np.ndarray
    physical_point_residual: np.ndarray
    gamma_by_slot: np.ndarray
    provenance: str = GU_EMERSON_TRANSFORMED_FV_PROVENANCE


@dataclass(frozen=True)
class GuEmersonEquation63Consistency:
    """Interior decomposition of the transformed finite-volume residual.

    The direct discretization satisfies

    ``R_63 = R_physical,point + (L_FV - L_central)``
    ``- (S_FV - S_central)``.

    The last two terms are the transport and source discretization defects,
    not reconstruction errors.  Keeping both separate exposes their balance
    and prevents a small transformed norm from hiding either contribution.
    """

    physical_point_linf: float
    transport_discretization_linf: float
    source_discretization_linf: float
    identity_roundoff: float
    transformed_argmax_slot: int
    physical_point_argmax_slot: int
    transport_discretization_argmax_slot: int
    source_discretization_argmax_slot: int


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


def _finite_volume_fluxes(
    packed: np.ndarray,
    mu: np.ndarray,
    gamma: np.ndarray,
    mass_x: np.ndarray,
    mass_y: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return internal and physical-wall fluxes on equation-(63)'s left side."""

    flux_x = np.zeros((packed.shape[0], packed.shape[1] - 1, 17), dtype=float)
    flux_y = np.zeros((packed.shape[0] - 1, packed.shape[1], 17), dtype=float)
    wall_x = np.zeros_like(packed)
    wall_y = np.zeros_like(packed)
    flux_x[..., 0] = mass_x
    flux_y[..., 0] = mass_y
    for slot in range(1, packed.shape[-1]):
        field = packed[..., slot]
        face_x = cubista_face_value(field, mass_x, axis=1)
        face_y = cubista_face_value(field, mass_y, axis=0)
        coefficient_x = _face_average(mu / gamma[slot], 1)
        coefficient_y = _face_average(mu / gamma[slot], 0)
        flux_x[..., slot] = mass_x * face_x - coefficient_x * _normal_face_gradient(
            field, x, axis=1
        )
        flux_y[..., slot] = mass_y * face_y - coefficient_y * _normal_face_gradient(
            field, y, axis=0
        )

        d_dx = np.gradient(field, x, axis=1, edge_order=2)
        d_dy = np.gradient(field, y, axis=0, edge_order=2)
        # Impermeability makes the transformed convective wall flux zero.
        # The principal diffusion flux uses the same second-order one-sided
        # wall derivative used by the compatible printed RHS flux.
        wall_x[:, 0, slot] = -(mu[:, 0] / gamma[slot]) * d_dx[:, 0]
        wall_x[:, -1, slot] = -(mu[:, -1] / gamma[slot]) * d_dx[:, -1]
        wall_y[0, :, slot] = -(mu[0, :] / gamma[slot]) * d_dy[0, :]
        wall_y[-1, :, slot] = -(mu[-1, :] / gamma[slot]) * d_dy[-1, :]
    return flux_x, flux_y, wall_x, wall_y


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

    flux_x, flux_y, wall_x, wall_y = _finite_volume_fluxes(
        packed, mu, gamma, mass_x, mass_y, x, y
    )
    return wall_bounded_face_divergence(
        flux_x, flux_y, wall_x, wall_y, x, y
    )


def _pack_balance_families(
    *,
    mass: np.ndarray,
    momentum: np.ndarray,
    theta: np.ndarray,
    stress: np.ndarray,
    heat: np.ndarray,
    m: np.ndarray,
    R: np.ndarray,
    Delta: np.ndarray,
) -> np.ndarray:
    """Pack scalar/vector/STF balance families in planar-17 row order."""

    return R26BulkResidual(
        mass=mass,
        momentum=momentum,
        theta=theta,
        heat_flux=heat,
        stress=stress,
        R=R,
        m=m,
        Delta=Delta,
    ).as_planar17(atol=3.0e-10)


def _physical_fv_fluxes(
    physical: np.ndarray,
    closures: R26Closures,
    faces: CompatibleFaceFields,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the conservative physical transport fluxes in planar-17 order."""

    tensors = planar_state_to_tensors(physical)
    rho = np.asarray(tensors.rho)
    velocity = np.asarray(tensors.velocity)
    theta = np.asarray(tensors.theta)
    sigma = np.asarray(tensors.sigma)
    q = np.asarray(tensors.heat_flux)
    mm = np.asarray(tensors.m)
    rr = np.asarray(tensors.R)
    delta = np.asarray(tensors.Delta)

    def transported(field: np.ndarray, axis: int) -> np.ndarray:
        mass = faces.mass_x if axis == 1 else faces.mass_y
        return cubista_face_value(field, mass, axis=axis)

    pressure = rho * theta
    pressure_x = _face_average(pressure, 1)
    pressure_y = _face_average(pressure, 0)
    momentum_x = faces.mass_x[..., None] * transported(velocity, 1)
    momentum_y = faces.mass_y[..., None] * transported(velocity, 0)
    momentum_x += faces.sigma_x[..., :, 0]
    momentum_y += faces.sigma_y[..., :, 1]
    momentum_x[..., 0] += pressure_x
    momentum_y[..., 1] += pressure_y
    theta_x = (
        faces.mass_x * transported(theta, 1)
        + 2.0 / 3.0 * faces.q_x[..., 0]
    )
    theta_y = (
        faces.mass_y * transported(theta, 0)
        + 2.0 / 3.0 * faces.q_y[..., 1]
    )
    stress_x = (
        faces.velocity_x[..., 0, None, None] * transported(sigma, 1)
        + faces.m_x[..., :, :, 0]
    )
    stress_y = (
        faces.velocity_y[..., 1, None, None] * transported(sigma, 0)
        + faces.m_y[..., :, :, 1]
    )
    heat_x = (
        faces.velocity_x[..., 0, None] * transported(q, 1)
        + 0.5 * faces.R_x[..., :, 0]
    )
    heat_y = (
        faces.velocity_y[..., 1, None] * transported(q, 0)
        + 0.5 * faces.R_y[..., :, 1]
    )
    m_x = (
        faces.velocity_x[..., 0, None, None, None] * transported(mm, 1)
        + faces.phi_x[..., :, :, :, 0]
    )
    m_y = (
        faces.velocity_y[..., 1, None, None, None] * transported(mm, 0)
        + faces.phi_y[..., :, :, :, 1]
    )
    R_x = (
        faces.velocity_x[..., 0, None, None] * transported(rr, 1)
        + faces.psi_x[..., :, :, 0]
    )
    R_y = (
        faces.velocity_y[..., 1, None, None] * transported(rr, 0)
        + faces.psi_y[..., :, :, 1]
    )
    Delta_x = (
        faces.velocity_x[..., 0] * transported(delta, 1)
        + faces.Omega_x[..., 0]
    )
    Delta_y = (
        faces.velocity_y[..., 1] * transported(delta, 0)
        + faces.Omega_y[..., 1]
    )
    flux_x = _pack_balance_families(
        mass=faces.mass_x,
        momentum=momentum_x,
        theta=theta_x,
        stress=stress_x,
        heat=heat_x,
        m=m_x,
        R=R_x,
        Delta=Delta_x,
    )
    flux_y = _pack_balance_families(
        mass=faces.mass_y,
        momentum=momentum_y,
        theta=theta_y,
        stress=stress_y,
        heat=heat_y,
        m=m_y,
        R=R_y,
        Delta=Delta_y,
    )
    walls = compatible_wall_fluxes(physical, closures)
    wall_x = _pack_balance_families(
        mass=walls.mass_x,
        momentum=walls.momentum_x,
        theta=walls.theta_x,
        stress=walls.stress_x,
        heat=walls.heat_x,
        m=walls.m_x,
        R=walls.R_x,
        Delta=walls.Delta_x,
    )
    wall_y = _pack_balance_families(
        mass=walls.mass_y,
        momentum=walls.momentum_y,
        theta=walls.theta_y,
        stress=walls.stress_y,
        heat=walls.heat_y,
        m=walls.m_y,
        R=walls.R_y,
        Delta=walls.Delta_y,
    )
    return flux_x, flux_y, wall_x, wall_y


def _physical_local_terms(
    physical: np.ndarray,
    mu: np.ndarray,
    closures: R26Closures,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Return the printed non-transport terms of the eight physical balances."""

    tensors = planar_state_to_tensors(physical)
    gradients = finite_difference_gradients(
        physical, x=x, y=y, edge_order=2
    )
    derivatives = closure_derivatives_on_grid(
        closures, x, y, edge_order=2
    )
    nonlinear = gu_emerson_nonlinear_sources(
        tensors, gradients, closures, derivatives, mu=mu
    )
    rho = np.asarray(tensors.rho)
    theta = np.asarray(tensors.theta)
    sigma = np.asarray(tensors.sigma)
    q = np.asarray(tensors.heat_flux)
    mm = np.asarray(tensors.m)
    rr = np.asarray(tensors.R)
    delta = np.asarray(tensors.Delta)
    gu = np.asarray(gradients.velocity)
    grho = np.asarray(gradients.rho)
    gtheta = np.asarray(gradients.theta)
    gq = np.asarray(gradients.heat_flux)
    gs = np.asarray(gradients.sigma)
    pressure = rho * theta
    div_u = np.einsum("...ii->...", gu)
    sigma_grad_u = np.einsum("...ij,...ij->...", sigma, gu)
    grad_sigma_over_rho = np.moveaxis(gs, -3, -1) / rho[..., None, None, None]
    grad_sigma_over_rho -= (
        np.einsum("...ij,...k->...ijk", sigma, grho)
        / rho[..., None, None, None] ** 2
    )
    grad_q_over_rho = np.swapaxes(gq, -2, -1) / rho[..., None, None]
    grad_q_over_rho -= (
        np.einsum("...i,...j->...ij", q, grho)
        / rho[..., None, None] ** 2
    )
    div_q = np.einsum("...ii->...", gq)
    div_q_over_rho = (
        div_q / rho
        - np.einsum("...i,...i->...", q, grho) / rho**2
    )
    zero = np.zeros_like(rho)
    momentum = np.zeros(rho.shape + (3,), dtype=float)
    theta_local = 2.0 / 3.0 * (pressure * div_u + sigma_grad_u)
    stress_local = pressure[..., None, None] / mu[..., None, None] * sigma
    stress_local += 2.0 * pressure[..., None, None] * stf2_project(gu)
    stress_local -= nonlinear.Sigma
    heat_local = 2.0 * pressure[..., None] / (3.0 * mu[..., None]) * q
    heat_local += 5.0 / 2.0 * pressure[..., None] * gtheta
    heat_local -= nonlinear.Q
    m_local = 3.0 * pressure[..., None, None, None] / (
        2.0 * mu[..., None, None, None]
    ) * mm
    m_local += 3.0 * pressure[..., None, None, None] * stf3_project(
        grad_sigma_over_rho
    )
    m_local -= nonlinear.M
    R_local = 7.0 * pressure[..., None, None] / (
        6.0 * mu[..., None, None]
    ) * rr
    R_local += 28.0 / 5.0 * pressure[..., None, None] * stf2_project(
        grad_q_over_rho
    )
    R_local -= nonlinear.S
    Delta_local = 2.0 * pressure / (3.0 * mu) * delta
    Delta_local += 8.0 * pressure * div_q_over_rho
    Delta_local -= nonlinear.N
    return _pack_balance_families(
        mass=zero,
        momentum=momentum,
        theta=theta_local,
        stress=stress_local,
        heat=heat_local,
        m=m_local,
        R=R_local,
        Delta=Delta_local,
    )


def _central_equation63_source_fluxes(
    physical: np.ndarray,
    packed: np.ndarray,
    mu: np.ndarray,
    gamma: np.ndarray,
    closures: R26Closures,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centrally interpolated source fluxes for equations (56)--(62).

    At a cell centre, transformed flux minus physical-moment flux is exactly
    the conservative part of the printed right-hand side.  Interpolating that
    difference centrally implements section 5.2 without introducing CUBISTA
    into a source term.
    """

    tensors = planar_state_to_tensors(physical)
    rho = np.asarray(tensors.rho)
    velocity = np.asarray(tensors.velocity)
    theta = np.asarray(tensors.theta)
    sigma = np.asarray(tensors.sigma)
    q = np.asarray(tensors.heat_flux)
    mm = np.asarray(tensors.m)
    rr = np.asarray(tensors.R)
    delta = np.asarray(tensors.Delta)
    pressure = rho * theta
    d_dx = np.gradient(packed, x, axis=1, edge_order=2)
    d_dy = np.gradient(packed, y, axis=0, edge_order=2)
    diffusion = mu[..., None] / gamma
    transformed_x = (
        rho[..., None] * velocity[..., 0, None] * packed - diffusion * d_dx
    )
    transformed_y = (
        rho[..., None] * velocity[..., 1, None] * packed - diffusion * d_dy
    )
    transformed_x[..., 0] = rho * velocity[..., 0]
    transformed_y[..., 0] = rho * velocity[..., 1]

    momentum_x = rho[..., None] * velocity[..., 0, None] * velocity
    momentum_y = rho[..., None] * velocity[..., 1, None] * velocity
    momentum_x += sigma[..., :, 0]
    momentum_y += sigma[..., :, 1]
    momentum_x[..., 0] += pressure
    momentum_y[..., 1] += pressure
    physical_x = _pack_balance_families(
        mass=rho * velocity[..., 0],
        momentum=momentum_x,
        theta=rho * velocity[..., 0] * theta + 2.0 / 3.0 * q[..., 0],
        stress=velocity[..., 0, None, None] * sigma + mm[..., :, :, 0],
        heat=velocity[..., 0, None] * q + 0.5 * rr[..., :, 0],
        m=(
            velocity[..., 0, None, None, None] * mm
            + np.asarray(closures.phi)[..., :, :, :, 0]
        ),
        R=(
            velocity[..., 0, None, None] * rr
            + np.asarray(closures.psi)[..., :, :, 0]
        ),
        Delta=velocity[..., 0] * delta + np.asarray(closures.Omega)[..., 0],
    )
    physical_y = _pack_balance_families(
        mass=rho * velocity[..., 1],
        momentum=momentum_y,
        theta=rho * velocity[..., 1] * theta + 2.0 / 3.0 * q[..., 1],
        stress=velocity[..., 1, None, None] * sigma + mm[..., :, :, 1],
        heat=velocity[..., 1, None] * q + 0.5 * rr[..., :, 1],
        m=(
            velocity[..., 1, None, None, None] * mm
            + np.asarray(closures.phi)[..., :, :, :, 1]
        ),
        R=(
            velocity[..., 1, None, None] * rr
            + np.asarray(closures.psi)[..., :, :, 1]
        ),
        Delta=velocity[..., 1] * delta + np.asarray(closures.Omega)[..., 1],
    )
    source_x = transformed_x - physical_x
    source_y = transformed_y - physical_y
    return (
        _face_average(source_x, 1),
        _face_average(source_y, 0),
        source_x,
        source_y,
    )


def _equation63_source(
    physical: np.ndarray,
    packed: np.ndarray,
    mu: np.ndarray,
    gamma: np.ndarray,
    closures: R26Closures,
    transformed_fluxes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    faces: CompatibleFaceFields,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the central source and its compatible-flux audit baseline."""

    physical_fluxes = _physical_fv_fluxes(physical, closures, faces)
    compatible_fluxes = tuple(
        transformed - physical_flux
        for transformed, physical_flux in zip(
            transformed_fluxes, physical_fluxes, strict=True
        )
    )
    compatible_transport = wall_bounded_face_divergence(
        *compatible_fluxes, x, y
    )
    central_fluxes = _central_equation63_source_fluxes(
        physical, packed, mu, gamma, closures, x, y
    )
    source_transport = wall_bounded_face_divergence(
        *central_fluxes, x, y
    )
    physical_local_terms = _physical_local_terms(
        physical, mu, closures, x, y
    )
    source = source_transport - physical_local_terms
    compatible_source = compatible_transport - physical_local_terms
    source[[0, -1], :, :] = 0.0
    source[:, [0, -1], :] = 0.0
    source_transport[[0, -1], :, :] = 0.0
    source_transport[:, [0, -1], :] = 0.0
    physical_local_terms[[0, -1], :, :] = 0.0
    physical_local_terms[:, [0, -1], :] = 0.0
    compatible_source[[0, -1], :, :] = 0.0
    compatible_source[:, [0, -1], :] = 0.0
    return source, source_transport, physical_local_terms, compatible_source


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
    closures = gu_emerson_closures(
        physical,
        x=x,
        y=y,
        mu=mu,
        edge_order=2,
        coefficient_mode=case.r26_closure_mode,
    )
    faces = compatible_face_fields(physical, x, y, mu, closures)
    transformed_fluxes = _finite_volume_fluxes(
        packed, mu, gamma, faces.mass_x, faces.mass_y, x, y
    )
    finite_volume_lhs = wall_bounded_face_divergence(
        *transformed_fluxes, x, y
    )
    (
        source,
        source_transport,
        physical_local_terms,
        compatible_source,
    ) = _equation63_source(
        physical,
        packed,
        mu,
        gamma,
        closures,
        transformed_fluxes,
        faces,
        x,
        y,
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
        for value in (
            residual,
            finite_volume_lhs,
            central_lhs,
            source,
            source_transport,
            physical_local_terms,
            compatible_source,
            physical_point,
        )
    ):
        raise FloatingPointError("equation-(63) evaluation produced NaN or infinity")
    return GuEmersonEquation63Terms(
        residual=residual,
        finite_volume_lhs=finite_volume_lhs,
        central_point_lhs=central_lhs,
        source=source,
        source_transport=source_transport,
        physical_local_terms=physical_local_terms,
        compatible_source=compatible_source,
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


def gu_emerson_equation63_consistency(
    terms: GuEmersonEquation63Terms,
) -> GuEmersonEquation63Consistency:
    """Audit the discrete/central decomposition on interior balance rows."""

    interior = np.s_[1:-1, 1:-1]
    physical = np.asarray(terms.physical_point_residual)[interior]
    transport_discretization = (
        np.asarray(terms.finite_volume_lhs)
        - np.asarray(terms.central_point_lhs)
    )[interior]
    central_source = (
        np.asarray(terms.central_point_lhs)
        - np.asarray(terms.physical_point_residual)
    )
    source_discretization = (
        np.asarray(terms.source) - central_source
    )[interior]
    closure = (
        np.asarray(terms.residual)[interior]
        - physical
        - transport_discretization
        + source_discretization
    )
    physical_by_slot = np.max(np.abs(physical), axis=(0, 1))
    transport_by_slot = np.max(
        np.abs(transport_discretization), axis=(0, 1)
    )
    source_by_slot = np.max(np.abs(source_discretization), axis=(0, 1))
    transformed_by_slot = np.max(
        np.abs(np.asarray(terms.residual)[interior]), axis=(0, 1)
    )
    return GuEmersonEquation63Consistency(
        physical_point_linf=float(np.max(physical_by_slot, initial=0.0)),
        transport_discretization_linf=float(
            np.max(transport_by_slot, initial=0.0)
        ),
        source_discretization_linf=float(
            np.max(source_by_slot, initial=0.0)
        ),
        identity_roundoff=float(np.max(np.abs(closure), initial=0.0)),
        transformed_argmax_slot=int(np.argmax(transformed_by_slot)),
        physical_point_argmax_slot=int(np.argmax(physical_by_slot)),
        transport_discretization_argmax_slot=int(
            np.argmax(transport_by_slot)
        ),
        source_discretization_argmax_slot=int(np.argmax(source_by_slot)),
    )


__all__ = [
    "GU_EMERSON_TRANSFORMED_FV_PROVENANCE",
    "GuEmersonEquation63Consistency",
    "GuEmersonEquation63PicardData",
    "GuEmersonEquation63Terms",
    "equation63_gamma_by_slot",
    "gu_emerson_equation63_consistency",
    "gu_emerson_equation63_picard_data",
    "gu_emerson_equation63_picard_residual",
    "gu_emerson_equation63_terms",
    "gu_emerson_transformed_fv_residual",
]
