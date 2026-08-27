#!/usr/bin/env python3
"""Compatible collocated finite-volume transport for the private R26 model.

Gu & Emerson (JFM 636, 2009), section 5, specify a collocated finite-volume
method, central diffusion/source differences, CUBISTA convection, SIMPLE, and
Rhie--Chow interpolation.  They do not print their face coefficient formula.

This verification backend implements the documented transport needed by both
the monolithic steady solver and the pressure-based THOR-style path:

* arithmetic central face convection for backward compatibility, plus an
  explicitly selectable bounded CUBISTA normalized-variable interpolation;
* compatible face-gradient/face-divergence transport using the gradient and
  non-gradient moment split of equations (48)--(55), with ``Omega_G`` limited
  to the printed equation (27) ``grad(Delta/rho)`` term;
* a Rhie--Chow normal face velocity using the central-diffusion momentum
  diagonal implied by equations (56), (63), ``Gamma_u=1``;
* one common wall-bounded control-volume geometry for all 17 balances: every
  interior face is shared, normal mass flux is exactly zero at an impermeable
  wall, and the remaining physical wall fluxes are evaluated from the current
  wall-node moments and closures.

The latter coefficient is an explicit implementation convention:

``d_P = 1 / [mu_P (a_x,P + a_y,P)]``

where ``a`` is the positive diagonal magnitude of the standard nonuniform
three-point second derivative.  On a uniform grid this reduces exactly to
``1/[2 mu_P (1/dx**2 + 1/dy**2)]``.  It is arithmetically interpolated to a
face.  The correction is

``u_n,f = linear(u_n)_f + d_f[linear(grad p)_f.n - (p_N-p_P)/delta]``.

It vanishes for a linear pressure field and is not an artificial filter.  The
raw physical equations and all wall rows remain independently available as
acceptance gates.  Sharp-corner policy remains owned by ``r26_discretization``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from r26_bulk_equations import (
    R26BulkResidual,
    bulk_residual_grid,
    closure_derivatives_on_grid,
    steady_r26_bulk_residual,
)
from r26_state import planar_state_to_tensors, validate_planar_state
from r26_tensor_closures import (
    R26Closures,
    closure_coefficients,
    closures_from_tensors,
    finite_difference_gradients,
    stf2_project,
    stf3_project,
    stf4_project,
)


FV_PROVENANCE: Final[str] = (
    "Gu--Emerson (2009) Eqs. (27), (48)-(63), collocated finite volume and "
    "Rhie--Chow per Sec. 5.2; arithmetic central convection and explicit "
    "nonuniform three-point momentum-diagonal convention; one common physical-wall "
    "control-volume geometry for all 17 balances"
)

DEFAULT_FV_FD_STEP_SCALE: Final[float] = 2.0e-6

CUBISTA_PROVENANCE: Final[str] = (
    "Alves--Oliveira--Pinho CUBISTA normalized-variable interpolation; "
    "Gu--Emerson JFM 636 Sec. 5.2 pressure-based R26 transport"
)


def fv_absolute_difference_step(
    encoded_state: np.ndarray,
    step_scale: float = DEFAULT_FV_FD_STEP_SCALE,
) -> np.ndarray:
    """Return an absolute finite-difference step with a unit magnitude floor.

    SciPy interprets an explicitly supplied ``rel_step`` as proportional to
    ``abs(x)``.  That is unsafe for R26 continuation because valid high-order
    moments are often 1e-8 or smaller, making the perturbation round to zero
    and producing a false Jacobian rank loss.  This audited convention keeps
    the requested relative scaling for large coordinates while guaranteeing
    an absolute floor for small ones.
    """

    value = np.asarray(encoded_state, dtype=float)
    scale = float(step_scale)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError("encoded_state must be a nonempty finite vector")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("finite-difference step scale must be finite and positive")
    return scale * (1.0 + np.abs(value))


@dataclass(frozen=True)
class CompatibleFaceFields:
    """Face values used by the conservative R26 transport divergences."""

    velocity_x: np.ndarray
    velocity_y: np.ndarray
    mass_x: np.ndarray
    mass_y: np.ndarray
    sigma_x: np.ndarray
    sigma_y: np.ndarray
    q_x: np.ndarray
    q_y: np.ndarray
    m_x: np.ndarray
    m_y: np.ndarray
    R_x: np.ndarray
    R_y: np.ndarray
    phi_x: np.ndarray
    phi_y: np.ndarray
    psi_x: np.ndarray
    psi_y: np.ndarray
    Omega_x: np.ndarray
    Omega_y: np.ndarray


@dataclass(frozen=True)
class CompatibleWallFluxes:
    """Coordinate fluxes on physical walls for the eight balance families.

    The arrays retain the complete node-grid leading shape for simple and
    auditable indexing, but only the two x-boundaries of each ``*_x`` array
    and the two y-boundaries of each ``*_y`` array are consumed.  Normal
    convection is identically zero because no penetration is a prescribed
    physical boundary flux, rather than a wall residual that may be nonzero
    during an intermediate nonlinear iteration.
    """

    mass_x: np.ndarray
    mass_y: np.ndarray
    momentum_x: np.ndarray
    momentum_y: np.ndarray
    theta_x: np.ndarray
    theta_y: np.ndarray
    stress_x: np.ndarray
    stress_y: np.ndarray
    heat_x: np.ndarray
    heat_y: np.ndarray
    m_x: np.ndarray
    m_y: np.ndarray
    R_x: np.ndarray
    R_y: np.ndarray
    Delta_x: np.ndarray
    Delta_y: np.ndarray


def _coordinates(coordinate: np.ndarray, size: int, name: str) -> np.ndarray:
    value = np.asarray(coordinate, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all() or np.any(np.diff(value) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing with length {size}")
    return value


def _second_derivative_diagonal(coordinate: np.ndarray) -> np.ndarray:
    """Positive diagonal magnitude of a nonuniform three-point Laplacian.

    At interior node ``i`` this is
    ``2/(h_w+h_e) * (1/h_w + 1/h_e)``.  Boundary nodes do not own momentum
    balances, so their Rhie--Chow coefficient inherits the nearest interior
    diagonal.  The definition is continuous with the uniform-grid formula
    and avoids introducing an arbitrary local-spacing proxy on stretched
    grids.
    """

    value = np.asarray(coordinate, dtype=float)
    if value.ndim != 1 or value.size < 3:
        raise ValueError("coordinate needs at least three nodes")
    spacing = np.diff(value)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
        raise ValueError("coordinate must be finite and strictly increasing")
    result = np.empty_like(value)
    west = spacing[:-1]
    east = spacing[1:]
    result[1:-1] = 2.0 / (west + east) * (1.0 / west + 1.0 / east)
    result[0] = result[1]
    result[-1] = result[-2]
    return result


def rhie_chow_inverse_momentum_diagonal(
    mu: float | np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Return the audited cell coefficient used by Rhie--Chow/SIMPLE.

    The coefficient is the inverse positive central-diffusion momentum
    diagonal.  Exposing the same implementation to the nonlinear
    preconditioner prevents the face interpolation and pressure-correction
    equation from silently using different discrete operators.
    """

    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    viscosity = np.broadcast_to(np.asarray(mu, dtype=float), (yy.size, xx.size))
    if not np.isfinite(viscosity).all() or np.any(viscosity <= 0.0):
        raise FloatingPointError("mu must be finite and positive")
    return 1.0 / (
        viscosity
        * (
            _second_derivative_diagonal(xx)[None, :]
            + _second_derivative_diagonal(yy)[:, None]
        )
    )


def _face_average(field: np.ndarray, axis: int) -> np.ndarray:
    a = np.asarray(field, dtype=float)
    if axis == 1:
        return 0.5 * (a[:, 1:] + a[:, :-1])
    if axis == 0:
        return 0.5 * (a[1:] + a[:-1])
    raise ValueError("face axis must be 0 or 1")


def cubista_face_value(
    field: np.ndarray,
    face_flux: np.ndarray,
    *,
    axis: int,
) -> np.ndarray:
    """Return bounded CUBISTA values on faces normal to ``axis``.

    ``face_flux`` is scalar and fixes the upwind direction.  The transported
    ``field`` may have arbitrary trailing tensor dimensions.  On a face where
    a remote-upwind node is unavailable, or where the normalized-variable
    denominator is numerically zero, the method fails safely to first-order
    upwind.  This is the standard deferred-convection boundary treatment; it
    never extrapolates an unavailable ghost value.
    """

    value = np.asarray(field, dtype=float)
    flux = np.asarray(face_flux, dtype=float)
    if value.ndim < 2 or axis not in (0, 1):
        raise ValueError("CUBISTA field needs two grid axes and axis 0 or 1")
    expected = list(value.shape[:2])
    expected[axis] -= 1
    if flux.shape != tuple(expected):
        raise ValueError("face_flux shape is incompatible with the transported field")
    if not np.isfinite(value).all() or not np.isfinite(flux).all():
        raise FloatingPointError("CUBISTA inputs must be finite")

    moved = np.moveaxis(value, axis, 0)
    flux_moved = np.moveaxis(flux, axis, 0)
    n = moved.shape[0]
    if n < 3:
        raise ValueError("CUBISTA requires at least three nodes along the face axis")

    left = moved[:-1]
    right = moved[1:]
    positive = flux_moved >= 0.0
    trailing = (1,) * (value.ndim - 2)
    positive_expanded = positive.reshape(positive.shape + trailing)
    upwind = np.where(positive_expanded, left, right)
    downwind = np.where(positive_expanded, right, left)

    remote_positive = np.concatenate((moved[:1], moved[:-2]), axis=0)
    remote_negative = np.concatenate((moved[2:], moved[-1:]), axis=0)
    remote = np.where(positive_expanded, remote_positive, remote_negative)

    denominator = downwind - remote
    scale = np.maximum.reduce((np.abs(upwind), np.abs(downwind), np.abs(remote)))
    regular = np.abs(denominator) > 32.0 * np.finfo(float).eps * np.maximum(1.0, scale)
    normalized = np.divide(
        upwind - remote,
        denominator,
        out=np.zeros_like(upwind),
        where=regular,
    )
    bounded = (normalized > 0.0) & (normalized < 1.0)
    face_normalized = normalized.copy()
    first = bounded & (normalized < 3.0 / 8.0)
    second = bounded & (normalized >= 3.0 / 8.0) & (normalized <= 3.0 / 4.0)
    third = bounded & (normalized > 3.0 / 4.0)
    face_normalized[first] = 7.0 / 4.0 * normalized[first]
    face_normalized[second] = 3.0 / 4.0 * normalized[second] + 3.0 / 8.0
    face_normalized[third] = 1.0 / 4.0 * normalized[third] + 3.0 / 4.0
    result = np.where(
        regular & bounded,
        remote + face_normalized * denominator,
        upwind,
    )

    boundary_positive = np.zeros_like(positive)
    boundary_positive[0] = True
    boundary_negative = np.zeros_like(positive)
    boundary_negative[-1] = True
    unavailable = (positive & boundary_positive) | ((~positive) & boundary_negative)
    unavailable_expanded = unavailable.reshape(unavailable.shape + trailing)
    result = np.where(unavailable_expanded, upwind, result)
    return np.moveaxis(result, 0, axis)


def _face_gradient(field: np.ndarray, x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray:
    """Full derivative-first gradient on x- or y-normal internal faces."""

    a = np.asarray(field, dtype=float)
    trailing = (1,) * (a.ndim - 2)
    if axis == 1:
        dx = np.diff(x).reshape((1, x.size - 1) + trailing)
        normal = (a[:, 1:] - a[:, :-1]) / dx
        tangential_cell = np.gradient(a, y, axis=0, edge_order=2)
        tangential = _face_average(tangential_cell, 1)
        zero = np.zeros_like(normal)
        return np.stack((normal, tangential, zero), axis=2)
    if axis == 0:
        dy = np.diff(y).reshape((y.size - 1, 1) + trailing)
        normal = (a[1:] - a[:-1]) / dy
        tangential_cell = np.gradient(a, x, axis=1, edge_order=2)
        tangential = _face_average(tangential_cell, 0)
        zero = np.zeros_like(normal)
        return np.stack((tangential, normal, zero), axis=2)
    raise ValueError("face axis must be 0 or 1")


def _midpoint_face_divergence(
    flux_x: np.ndarray,
    flux_y: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Legacy midpoint divergence retained only for independent tests.

    This operator places the first/last normal faces halfway between a wall
    node and its adjacent interior node.  It is not used by the accepted
    compatible-FV residual because those faces are not physical boundaries.
    """

    fx = np.asarray(flux_x, dtype=float)
    fy = np.asarray(flux_y, dtype=float)
    ny, nx_minus_one = fx.shape[:2]
    if fy.shape[:2] != (ny - 1, nx_minus_one + 1) or fx.shape[2:] != fy.shape[2:]:
        raise ValueError("x/y face flux arrays have incompatible shapes")
    nx = nx_minus_one + 1
    output = np.zeros((ny, nx) + fx.shape[2:], dtype=float)
    xwidth = (0.5 * (x[2:] - x[:-2])).reshape((1, nx - 2) + (1,) * (fx.ndim - 2))
    ywidth = (0.5 * (y[2:] - y[:-2])).reshape((ny - 2, 1) + (1,) * (fy.ndim - 2))
    output[:, 1:-1] += (fx[:, 1:] - fx[:, :-1]) / xwidth
    output[1:-1] += (fy[1:] - fy[:-1]) / ywidth
    return output


def interior_control_volume_widths(coordinate: np.ndarray) -> np.ndarray:
    """Return widths of wall-bounded control volumes at interior nodes.

    Wall nodes carry boundary conditions rather than balance equations.  The
    first and last interior control volumes therefore extend to the physical
    wall, while all other faces lie halfway between adjacent interior nodes.
    These widths cover the full physical interval exactly and are positive on
    every strictly increasing node grid.
    """

    value = np.asarray(coordinate, dtype=float)
    if value.ndim != 1 or value.size < 4:
        raise ValueError("coordinate must contain two walls and at least two interior nodes")
    if not np.isfinite(value).all() or np.any(np.diff(value) <= 0.0):
        raise ValueError("coordinate must be finite and strictly increasing")
    interior = value[1:-1]
    faces = np.concatenate(
        (
            value[:1],
            0.5 * (interior[:-1] + interior[1:]),
            value[-1:],
        )
    )
    widths = np.diff(faces)
    if widths.shape != interior.shape or np.any(widths <= 0.0):
        raise RuntimeError("failed to construct positive interior control-volume widths")
    return widths


def wall_bounded_control_volume_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return normalized node-array weights for the interior FV volumes.

    Wall nodes carry boundary equations and therefore have zero volume.  The
    positive interior weights are the tensor product of the same wall-bounded
    widths used in every compatible-FV balance.  Their sum is one on a unit
    square and is normalized by physical domain area on a general rectangle.
    """

    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    dx = interior_control_volume_widths(xx)
    dy = interior_control_volume_widths(yy)
    weights = np.zeros((yy.size, xx.size), dtype=float)
    area = float((xx[-1] - xx[0]) * (yy[-1] - yy[0]))
    weights[1:-1, 1:-1] = dy[:, None] * dx[None, :] / area
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise RuntimeError("failed to construct finite nonnegative FV weights")
    if not np.isclose(float(np.sum(weights)), 1.0, rtol=2.0e-14, atol=2.0e-14):
        raise RuntimeError("wall-bounded FV weights do not cover the domain")
    return weights


def wall_bounded_face_divergence(
    flux_x: np.ndarray,
    flux_y: np.ndarray,
    wall_flux_x: np.ndarray,
    wall_flux_y: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Conservative divergence on the common physical-wall control volumes.

    ``flux_x`` and ``flux_y`` hold midpoint flux densities between every pair
    of adjacent nodes.  Wall-to-first-interior midpoint entries are excluded:
    the first and last interior control volumes extend to the physical walls,
    where ``wall_flux_x``/``wall_flux_y`` provide the coordinate flux.  All
    other faces are shared by exactly two neighbouring control volumes.
    """

    fx = np.asarray(flux_x, dtype=float)
    fy = np.asarray(flux_y, dtype=float)
    wx = np.asarray(wall_flux_x, dtype=float)
    wy = np.asarray(wall_flux_y, dtype=float)
    ny, nx_minus_one = fx.shape[:2]
    nx = nx_minus_one + 1
    trailing = fx.shape[2:]
    if fy.shape[:2] != (ny - 1, nx) or fy.shape[2:] != trailing:
        raise ValueError("x/y face flux arrays have incompatible shapes")
    if wx.shape != (ny, nx) + trailing or wy.shape != (ny, nx) + trailing:
        raise ValueError("physical wall flux arrays have incompatible shapes")
    if not all(np.isfinite(value).all() for value in (fx, fy, wx, wy)):
        raise FloatingPointError("face and wall fluxes must be finite")
    xx = _coordinates(x, nx, "x")
    yy = _coordinates(y, ny, "y")
    dx = interior_control_volume_widths(xx)
    dy = interior_control_volume_widths(yy)

    bounded_x = np.empty((ny - 2, nx - 1) + trailing, dtype=float)
    bounded_y = np.empty((ny - 1, nx - 2) + trailing, dtype=float)
    bounded_x[:, 0] = wx[1:-1, 0]
    bounded_x[:, -1] = wx[1:-1, -1]
    bounded_x[:, 1:-1] = fx[1:-1, 1:-1]
    bounded_y[0] = wy[0, 1:-1]
    bounded_y[-1] = wy[-1, 1:-1]
    bounded_y[1:-1] = fy[1:-1, 1:-1]

    dx_shape = (1, nx - 2) + (1,) * len(trailing)
    dy_shape = (ny - 2, 1) + (1,) * len(trailing)
    output = np.zeros((ny, nx) + trailing, dtype=float)
    output[1:-1, 1:-1] = (
        (bounded_x[:, 1:] - bounded_x[:, :-1]) / dx.reshape(dx_shape)
        + (bounded_y[1:] - bounded_y[:-1]) / dy.reshape(dy_shape)
    )
    return output


def impermeable_wall_mass_divergence(
    mass_x: np.ndarray,
    mass_y: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Return a globally conservative interior continuity divergence.

    ``mass_x`` and ``mass_y`` contain flux densities between every pair of
    adjacent grid nodes.  Because the wall nodes are boundary nodes, not
    control-volume centres, the wall-to-first-interior interpolants are not
    physical boundary fluxes and are deliberately excluded.  The boundary
    face flux is exactly zero for the impermeable cavity.  Each remaining
    interior face appears in its two neighbouring balances with opposite
    signs, so the area-weighted sum of all interior continuity rows is an
    algebraic zero (up to roundoff) for every state.

    The returned array has the complete node-grid shape and zero boundary
    entries; only its interior is a physical balance row.
    """

    fx = np.asarray(mass_x, dtype=float)
    fy = np.asarray(mass_y, dtype=float)
    if fx.ndim != 2 or fy.ndim != 2:
        raise ValueError("mass face fluxes must be two-dimensional scalar fields")
    ny, nx_minus_one = fx.shape
    nx = nx_minus_one + 1
    if fy.shape != (ny - 1, nx):
        raise ValueError("x/y mass face flux arrays have incompatible shapes")
    zero = np.zeros((ny, nx), dtype=float)
    return wall_bounded_face_divergence(fx, fy, zero, zero, x, y)


def _mu_expand(mu: np.ndarray, rank: int) -> np.ndarray:
    return mu[(Ellipsis,) + (None,) * rank]


def _quotient_gradient_cell(field: np.ndarray, rho: np.ndarray, gradient: np.ndarray, grho: np.ndarray) -> np.ndarray:
    rank = field.ndim - rho.ndim
    return gradient / _mu_expand(rho, rank + 1) - np.expand_dims(field, axis=2) * _mu_expand(grho, rank) / _mu_expand(rho, rank + 1) ** 2


def compatible_face_fields(
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    closures: R26Closures,
) -> CompatibleFaceFields:
    tensors = planar_state_to_tensors(state)
    gradients = finite_difference_gradients(state, x=x, y=y, edge_order=2)
    rho = np.asarray(tensors.rho)
    velocity = np.asarray(tensors.velocity)
    theta = np.asarray(tensors.theta)
    sigma = np.asarray(tensors.sigma)
    q = np.asarray(tensors.heat_flux)
    mm = np.asarray(tensors.m)
    rr = np.asarray(tensors.R)
    delta = np.asarray(tensors.Delta)

    sigma_g_cell = -2.0 * _mu_expand(mu, 2) * stf2_project(gradients.velocity)
    q_g_cell = -15.0 / 4.0 * _mu_expand(mu, 1) * gradients.theta

    sigma_over_rho = sigma / _mu_expand(rho, 2)
    q_over_rho = q / _mu_expand(rho, 1)
    m_over_rho = mm / _mu_expand(rho, 3)
    R_over_rho = rr / _mu_expand(rho, 2)
    delta_over_rho = delta / rho

    grad_sigma_ratio_cell = _quotient_gradient_cell(sigma, rho, gradients.sigma, gradients.rho)
    grad_q_ratio_cell = _quotient_gradient_cell(q, rho, gradients.heat_flux, gradients.rho)
    grad_m_ratio_cell = _quotient_gradient_cell(mm, rho, gradients.m, gradients.rho)
    grad_R_ratio_cell = _quotient_gradient_cell(rr, rho, gradients.R, gradients.rho)
    grad_delta_ratio_cell = gradients.Delta / rho[..., None] - delta[..., None] * gradients.rho / rho[..., None] ** 2

    coefficients = closure_coefficients(closures.coefficient_mode)
    m_g_cell = -2.0 * _mu_expand(mu, 3) * stf3_project(np.moveaxis(grad_sigma_ratio_cell, 2, -1))
    R_g_cell = -24.0 / 5.0 * _mu_expand(mu, 2) * stf2_project(np.swapaxes(grad_q_ratio_cell, 2, 3))
    phi_g_cell = -4.0 / coefficients.C1 * _mu_expand(mu, 4) * stf4_project(np.moveaxis(grad_m_ratio_cell, 2, -1))
    psi_g_cell = -27.0 / (7.0 * coefficients.Y1) * _mu_expand(mu, 3) * stf3_project(np.moveaxis(grad_R_ratio_cell, 2, -1))
    # Equation (27) defines only the Delta-gradient term as Omega_G.  The
    # ``-4 mu div(R/rho)`` term belongs to Omega_R in equation (26) and must
    # therefore remain in the arithmetically interpolated NGTM remainder.
    Omega_g_cell = -7.0 / 3.0 * _mu_expand(mu, 1) * grad_delta_ratio_cell

    def build(axis: int) -> tuple[np.ndarray, ...]:
        muf = _face_average(mu, axis)
        grad_u = _face_gradient(velocity, x, y, axis)
        sigma_g = -2.0 * _mu_expand(muf, 2) * stf2_project(grad_u)
        sigma_face = sigma_g + _face_average(sigma - sigma_g_cell, axis)

        grad_theta = _face_gradient(theta, x, y, axis)
        q_g = -15.0 / 4.0 * _mu_expand(muf, 1) * grad_theta
        q_face = q_g + _face_average(q - q_g_cell, axis)

        grad_sigma_ratio = _face_gradient(sigma_over_rho, x, y, axis)
        m_g = -2.0 * _mu_expand(muf, 3) * stf3_project(np.moveaxis(grad_sigma_ratio, 2, -1))
        m_face = m_g + _face_average(mm - m_g_cell, axis)

        grad_q_ratio = _face_gradient(q_over_rho, x, y, axis)
        R_g = -24.0 / 5.0 * _mu_expand(muf, 2) * stf2_project(np.swapaxes(grad_q_ratio, 2, 3))
        R_face = R_g + _face_average(rr - R_g_cell, axis)

        grad_m_ratio = _face_gradient(m_over_rho, x, y, axis)
        phi_g = -4.0 / coefficients.C1 * _mu_expand(muf, 4) * stf4_project(np.moveaxis(grad_m_ratio, 2, -1))
        phi_face = phi_g + _face_average(np.asarray(closures.phi) - phi_g_cell, axis)

        grad_R_ratio = _face_gradient(R_over_rho, x, y, axis)
        psi_g = -27.0 / (7.0 * coefficients.Y1) * _mu_expand(muf, 3) * stf3_project(np.moveaxis(grad_R_ratio, 2, -1))
        psi_face = psi_g + _face_average(np.asarray(closures.psi) - psi_g_cell, axis)

        grad_delta_ratio = _face_gradient(delta_over_rho, x, y, axis)
        Omega_g = -7.0 / 3.0 * _mu_expand(muf, 1) * grad_delta_ratio
        Omega_face = Omega_g + _face_average(np.asarray(closures.Omega) - Omega_g_cell, axis)
        return sigma_face, q_face, m_face, R_face, phi_face, psi_face, Omega_face

    sigma_x, q_x, m_x, R_x, phi_x, psi_x, Omega_x = build(1)
    sigma_y, q_y, m_y, R_y, phi_y, psi_y, Omega_y = build(0)

    pressure = rho * theta
    grad_p_x = np.gradient(pressure, x, axis=1, edge_order=2)
    grad_p_y = np.gradient(pressure, y, axis=0, edge_order=2)
    d_cell = rhie_chow_inverse_momentum_diagonal(mu, x, y)
    velocity_x = _face_average(velocity, 1)
    velocity_y = _face_average(velocity, 0)
    dx_face = np.diff(x)[None, :]
    dy_face = np.diff(y)[:, None]
    correction_x = _face_average(d_cell, 1) * (
        _face_average(grad_p_x, 1) - (pressure[:, 1:] - pressure[:, :-1]) / dx_face
    )
    correction_y = _face_average(d_cell, 0) * (
        _face_average(grad_p_y, 0) - (pressure[1:] - pressure[:-1]) / dy_face
    )
    velocity_x[..., 0] += correction_x
    velocity_y[..., 1] += correction_y
    mass_x = _face_average(rho, 1) * velocity_x[..., 0]
    mass_y = _face_average(rho, 0) * velocity_y[..., 1]
    return CompatibleFaceFields(
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        mass_x=mass_x,
        mass_y=mass_y,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        q_x=q_x,
        q_y=q_y,
        m_x=m_x,
        m_y=m_y,
        R_x=R_x,
        R_y=R_y,
        phi_x=phi_x,
        phi_y=phi_y,
        psi_x=psi_x,
        psi_y=psi_y,
        Omega_x=Omega_x,
        Omega_y=Omega_y,
    )


def compatible_wall_fluxes(
    state: np.ndarray,
    closures: R26Closures,
) -> CompatibleWallFluxes:
    """Return physical coordinate fluxes on the four impermeable walls.

    These are the conservative fluxes appearing in Gu--Emerson equations
    (2)--(4), (7)--(10), and (16)--(18), evaluated from the current wall-node
    moments and ``phi/psi/Omega`` closures.  Normal convective transport is
    exactly zero at the physical boundary.  The independent wall residuals
    still enforce the complete 11 Maxwell relations and no penetration.
    """

    tensors = planar_state_to_tensors(state)
    rho = np.asarray(tensors.rho, dtype=float)
    theta = np.asarray(tensors.theta, dtype=float)
    pressure = rho * theta
    sigma = np.asarray(tensors.sigma, dtype=float)
    q = np.asarray(tensors.heat_flux, dtype=float)
    mm = np.asarray(tensors.m, dtype=float)
    rr = np.asarray(tensors.R, dtype=float)
    phi = np.asarray(closures.phi, dtype=float)
    psi = np.asarray(closures.psi, dtype=float)
    Omega = np.asarray(closures.Omega, dtype=float)

    mass_x = np.zeros_like(rho)
    mass_y = np.zeros_like(rho)
    momentum_x = sigma[..., :, 0].copy()
    momentum_y = sigma[..., :, 1].copy()
    momentum_x[..., 0] += pressure
    momentum_y[..., 1] += pressure
    theta_x = 2.0 / 3.0 * q[..., 0]
    theta_y = 2.0 / 3.0 * q[..., 1]
    stress_x = mm[..., :, :, 0]
    stress_y = mm[..., :, :, 1]
    heat_x = 0.5 * rr[..., :, 0]
    heat_y = 0.5 * rr[..., :, 1]
    m_x = phi[..., :, :, :, 0]
    m_y = phi[..., :, :, :, 1]
    R_x = psi[..., :, :, 0]
    R_y = psi[..., :, :, 1]
    Delta_x = Omega[..., 0]
    Delta_y = Omega[..., 1]
    result = CompatibleWallFluxes(
        mass_x=mass_x,
        mass_y=mass_y,
        momentum_x=momentum_x,
        momentum_y=momentum_y,
        theta_x=theta_x,
        theta_y=theta_y,
        stress_x=stress_x,
        stress_y=stress_y,
        heat_x=heat_x,
        heat_y=heat_y,
        m_x=m_x,
        m_y=m_y,
        R_x=R_x,
        R_y=R_y,
        Delta_x=Delta_x,
        Delta_y=Delta_y,
    )
    if not all(np.isfinite(np.asarray(value)).all() for value in result.__dict__.values()):
        raise FloatingPointError("physical wall fluxes contain NaN or infinity")
    return result


def _replace_interior(raw: np.ndarray, old_transport: np.ndarray, new_transport: np.ndarray) -> np.ndarray:
    result = np.asarray(raw, dtype=float).copy()
    result[1:-1, 1:-1] += new_transport[1:-1, 1:-1] - old_transport[1:-1, 1:-1]
    return result


def compatible_fv_bulk_residual(
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: float | np.ndarray = 1.0,
    body_force: np.ndarray | None = None,
    *,
    edge_order: int = 2,
    case: object | None = None,
    convection_scheme: str = "central",
) -> np.ndarray:
    """Return planar-17 R26 rows with compatible FV transport divergences.

    ``case`` selects the closure coefficient set.  ``convection_scheme`` is
    either the historical arithmetic ``central`` interpolation or bounded
    ``cubista``.  Pressure coupling always uses the same stated Rhie--Chow
    normal face flux, independent of the transported-value interpolation.
    """

    if edge_order != 2:
        raise ValueError("compatible FV backend currently requires edge_order=2")
    scheme = str(convection_scheme).lower()
    if scheme not in {"central", "cubista"}:
        raise ValueError("convection_scheme must be central or cubista")
    u = validate_planar_state(state)
    if u.ndim != 3 or min(u.shape[:2]) < 5:
        raise ValueError("state must have shape (ny,nx,17) with at least 5x5 nodes")
    yy = _coordinates(y, u.shape[0], "y")
    xx = _coordinates(x, u.shape[1], "x")
    muv = np.broadcast_to(np.asarray(mu, dtype=float), u.shape[:2])
    if not np.isfinite(muv).all() or np.any(muv <= 0.0):
        raise FloatingPointError("mu must be finite and positive")

    tensors = planar_state_to_tensors(u)
    gradients = finite_difference_gradients(u, x=xx, y=yy, edge_order=2)
    coefficient_mode = "jfm2009" if case is None else str(
        getattr(case, "r26_closure_mode", "jfm2009")
    )
    closures = closures_from_tensors(
        tensors,
        gradients,
        mu=muv,
        coefficient_mode=coefficient_mode,
    )
    closure_derivatives = closure_derivatives_on_grid(closures, xx, yy, edge_order=2)

    acceleration = None
    if body_force is not None:
        force = np.asarray(body_force, dtype=float)
        if force.shape[-1:] == (2,):
            full = np.zeros(force.shape[:-1] + (3,))
            full[..., :2] = force
            force = full
        force = np.broadcast_to(force, u.shape[:2] + (3,))
        acceleration = force / np.asarray(tensors.rho)[..., None]
    raw = steady_r26_bulk_residual(
        tensors, gradients, closures, closure_derivatives, mu=muv, acceleration=acceleration
    )
    faces = compatible_face_fields(u, xx, yy, muv, closures)
    walls = compatible_wall_fluxes(u, closures)

    def transported(field: np.ndarray, axis: int) -> np.ndarray:
        if scheme == "central":
            return _face_average(field, axis)
        flux = faces.mass_x if axis == 1 else faces.mass_y
        return cubista_face_value(field, flux, axis=axis)

    rho = np.asarray(tensors.rho)
    velocity = np.asarray(tensors.velocity)
    theta = np.asarray(tensors.theta)
    sigma = np.asarray(tensors.sigma)
    q = np.asarray(tensors.heat_flux)
    mm = np.asarray(tensors.m)
    rr = np.asarray(tensors.R)
    delta = np.asarray(tensors.Delta)
    gu = np.asarray(gradients.velocity)
    grho = np.asarray(gradients.rho)
    gtheta = np.asarray(gradients.theta)
    gs = np.asarray(gradients.sigma)
    gq = np.asarray(gradients.heat_flux)
    gm = np.asarray(gradients.m)
    gr = np.asarray(gradients.R)
    gd = np.asarray(gradients.Delta)
    div_u = np.einsum("...ii->...", gu)

    old_mass = np.einsum("...i,...i->...", velocity, grho) + rho * div_u
    new_mass = wall_bounded_face_divergence(
        faces.mass_x,
        faces.mass_y,
        walls.mass_x,
        walls.mass_y,
        xx,
        yy,
    )

    old_momentum = (
        np.einsum("...i,...j,...j->...i", velocity, velocity, grho)
        + rho[..., None] * np.einsum("...j,...ji->...i", velocity, gu)
        + rho[..., None] * velocity * div_u[..., None]
        + np.einsum("...jij->...i", gs)
        + theta[..., None] * grho
        + rho[..., None] * gtheta
    )
    pressure = rho * theta
    pressure_x = _face_average(pressure, 1)
    pressure_y = _face_average(pressure, 0)
    # Preserve the historical compatible-FV operator bit for bit: its central
    # momentum flux transports the Rhie--Chow face velocity itself.  Only the
    # explicitly selected CUBISTA path replaces that transported value.
    transported_velocity_x = (
        faces.velocity_x if scheme == "central" else transported(velocity, 1)
    )
    transported_velocity_y = (
        faces.velocity_y if scheme == "central" else transported(velocity, 0)
    )
    momentum_flux_x = faces.mass_x[..., None] * transported_velocity_x + faces.sigma_x[..., :, 0]
    momentum_flux_y = faces.mass_y[..., None] * transported_velocity_y + faces.sigma_y[..., :, 1]
    momentum_flux_x[..., 0] += pressure_x
    momentum_flux_y[..., 1] += pressure_y
    new_momentum = wall_bounded_face_divergence(
        momentum_flux_x,
        momentum_flux_y,
        walls.momentum_x,
        walls.momentum_y,
        xx,
        yy,
    )

    old_theta = theta * old_mass + rho * np.einsum("...i,...i->...", velocity, gtheta)
    old_theta += 2.0 / 3.0 * np.einsum("...ii->...", gq)
    theta_flux_x = faces.mass_x * transported(theta, 1) + 2.0 / 3.0 * faces.q_x[..., 0]
    theta_flux_y = faces.mass_y * transported(theta, 0) + 2.0 / 3.0 * faces.q_y[..., 1]
    new_theta = wall_bounded_face_divergence(
        theta_flux_x,
        theta_flux_y,
        walls.theta_x,
        walls.theta_y,
        xx,
        yy,
    )

    old_stress = sigma * div_u[..., None, None] + np.einsum("...k,...kij->...ij", velocity, gs)
    old_stress += np.einsum("...kijk->...ij", gm)
    stress_flux_x = faces.velocity_x[..., 0, None, None] * transported(sigma, 1) + faces.m_x[..., :, :, 0]
    stress_flux_y = faces.velocity_y[..., 1, None, None] * transported(sigma, 0) + faces.m_y[..., :, :, 1]
    new_stress = wall_bounded_face_divergence(
        stress_flux_x,
        stress_flux_y,
        walls.stress_x,
        walls.stress_y,
        xx,
        yy,
    )

    old_heat = q * div_u[..., None] + np.einsum("...j,...ji->...i", velocity, gq)
    old_heat += 0.5 * np.einsum("...jij->...i", gr)
    heat_flux_x = faces.velocity_x[..., 0, None] * transported(q, 1) + 0.5 * faces.R_x[..., :, 0]
    heat_flux_y = faces.velocity_y[..., 1, None] * transported(q, 0) + 0.5 * faces.R_y[..., :, 1]
    new_heat = wall_bounded_face_divergence(
        heat_flux_x,
        heat_flux_y,
        walls.heat_x,
        walls.heat_y,
        xx,
        yy,
    )

    old_m = mm * div_u[..., None, None, None] + np.einsum("...l,...lijk->...ijk", velocity, gm)
    old_m += np.asarray(closure_derivatives.div_phi)
    m_flux_x = faces.velocity_x[..., 0, None, None, None] * transported(mm, 1) + faces.phi_x[..., :, :, :, 0]
    m_flux_y = faces.velocity_y[..., 1, None, None, None] * transported(mm, 0) + faces.phi_y[..., :, :, :, 1]
    new_m = wall_bounded_face_divergence(
        m_flux_x,
        m_flux_y,
        walls.m_x,
        walls.m_y,
        xx,
        yy,
    )

    old_R = rr * div_u[..., None, None] + np.einsum("...k,...kij->...ij", velocity, gr)
    old_R += np.asarray(closure_derivatives.div_psi)
    R_flux_x = faces.velocity_x[..., 0, None, None] * transported(rr, 1) + faces.psi_x[..., :, :, 0]
    R_flux_y = faces.velocity_y[..., 1, None, None] * transported(rr, 0) + faces.psi_y[..., :, :, 1]
    new_R = wall_bounded_face_divergence(
        R_flux_x,
        R_flux_y,
        walls.R_x,
        walls.R_y,
        xx,
        yy,
    )

    old_delta = delta * div_u + np.einsum("...i,...i->...", velocity, gd)
    old_delta += np.asarray(closure_derivatives.div_Omega)
    delta_flux_x = faces.velocity_x[..., 0] * transported(delta, 1) + faces.Omega_x[..., 0]
    delta_flux_y = faces.velocity_y[..., 1] * transported(delta, 0) + faces.Omega_y[..., 1]
    new_delta = wall_bounded_face_divergence(
        delta_flux_x,
        delta_flux_y,
        walls.Delta_x,
        walls.Delta_y,
        xx,
        yy,
    )

    corrected = R26BulkResidual(
        mass=_replace_interior(raw.mass, old_mass, new_mass),
        momentum=_replace_interior(raw.momentum, old_momentum, new_momentum),
        theta=_replace_interior(raw.theta, old_theta, new_theta),
        stress=_replace_interior(raw.stress, old_stress, new_stress),
        heat_flux=_replace_interior(raw.heat_flux, old_heat, new_heat),
        m=_replace_interior(raw.m, old_m, new_m),
        R=_replace_interior(raw.R, old_R, new_R),
        Delta=_replace_interior(raw.Delta, old_delta, new_delta),
        provenance=(
            FV_PROVENANCE
            if scheme == "central"
            else f"{FV_PROVENANCE}; {CUBISTA_PROVENANCE}"
        ),
    )
    planar = corrected.as_planar17()
    if not np.isfinite(planar).all():
        raise FloatingPointError("compatible FV R26 residual contains NaN or infinity")
    return planar


def thor_fv_bulk_residual(
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: float | np.ndarray = 1.0,
    body_force: np.ndarray | None = None,
    *,
    edge_order: int = 2,
    case: object | None = None,
) -> np.ndarray:
    """THOR-style compatible R26 transport with bounded CUBISTA convection."""

    return compatible_fv_bulk_residual(
        state,
        x,
        y,
        mu,
        body_force,
        edge_order=edge_order,
        case=case,
        convection_scheme="cubista",
    )


__all__ = [
    "CompatibleFaceFields",
    "CompatibleWallFluxes",
    "CUBISTA_PROVENANCE",
    "DEFAULT_FV_FD_STEP_SCALE",
    "FV_PROVENANCE",
    "compatible_face_fields",
    "compatible_fv_bulk_residual",
    "compatible_wall_fluxes",
    "cubista_face_value",
    "fv_absolute_difference_step",
    "impermeable_wall_mass_divergence",
    "interior_control_volume_widths",
    "rhie_chow_inverse_momentum_diagonal",
    "thor_fv_bulk_residual",
    "wall_bounded_control_volume_weights",
    "wall_bounded_face_divergence",
]
