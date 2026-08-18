#!/usr/bin/env python3
"""Checkerboard-free global-polynomial derivative backend for private R26.

The original prototype used the three-point collocated ``numpy.gradient``
operator.  Its centred interior symbol vanishes at the alternating-grid
frequency, and the assembled equilibrium cavity Jacobian consequently has
discrete null modes.  This module supplies a monolithic alternative: the
unique polynomial differentiation matrix on all wall-inclusive nodes.

This is *not* advertised as Gu--Emerson's finite-volume/SIMPLE algorithm.
It is a method-of-lines collocation discretization of the same audited bulk
and wall equations.  No constraint, penalty, filtering, or regularization is
added to remove a nullspace.  The derivative is exact for every polynomial of
degree at most ``N-1``; in particular, it has no nonconstant nodal null vector.
For the deliberately small verification grids used here, global polynomial
collocation is a useful independent discretization.  Equispaced high-order
collocation is not recommended for production grid refinement because of the
Runge/conditioning problem; a later production backend should use
Chebyshev--Lobatto nodes or a compatible finite-volume/Rhie--Chow scheme.
"""

from __future__ import annotations

import numpy as np

from r26_bulk_equations import (
    R26ClosureDerivatives,
    _grid_force_density,
    steady_r26_bulk_residual,
)
from r26_state import planar_state_to_tensors, validate_planar_state
from r26_tensor_closures import (
    R26Closures,
    R26Gradients,
    closures_from_tensors,
    resolve_closure_mode,
)


def polynomial_first_derivative_matrix(nodes: np.ndarray) -> np.ndarray:
    """Return the barycentric first-derivative matrix on distinct nodes.

    If ``p`` is sampled at ``N`` nodes and has degree at most ``N-1``, then
    ``D @ p(nodes)`` equals ``p'(nodes)`` to roundoff.  Rows sum to zero by
    construction, so constants differentiate exactly to zero.
    """

    x = np.asarray(nodes, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("nodes must be a finite one-dimensional vector")
    separation = x[:, None] - x[None, :]
    if np.any((separation == 0.0) & ~np.eye(x.size, dtype=bool)):
        raise ValueError("differentiation nodes must be distinct")

    # Barycentric interpolation weights w_i = prod_{k != i}(x_i-x_k)^-1.
    work = separation.copy()
    np.fill_diagonal(work, 1.0)
    weights = 1.0 / np.prod(work, axis=1)
    derivative = (weights[None, :] / weights[:, None]) / work
    np.fill_diagonal(derivative, 0.0)
    np.fill_diagonal(derivative, -np.sum(derivative, axis=1))
    return derivative


def differentiate_axis(field: np.ndarray, matrix: np.ndarray, axis: int) -> np.ndarray:
    """Apply a one-dimensional differentiation matrix along ``axis``."""

    value = np.asarray(field, dtype=float)
    derivative = np.asarray(matrix, dtype=float)
    axis = int(np.core.numeric.normalize_axis_index(axis, value.ndim))
    if derivative.shape != (value.shape[axis], value.shape[axis]):
        raise ValueError("derivative matrix shape does not match selected axis")
    moved = np.moveaxis(value, axis, 0)
    result = np.tensordot(derivative, moved, axes=((1,), (0,)))
    return np.moveaxis(result, 0, axis)


def spectral_gradients(
    state: np.ndarray, *, x: np.ndarray, y: np.ndarray
) -> R26Gradients:
    """Differentiate a planar 17-state grid with global collocation matrices."""

    u = validate_planar_state(state)
    if u.ndim != 3:
        raise ValueError(f"state grid must have shape (ny,nx,17), got {u.shape}")
    dx = polynomial_first_derivative_matrix(x)
    dy = polynomial_first_derivative_matrix(y)
    tensors = planar_state_to_tensors(u)

    def gradient(field: np.ndarray, tensor_rank: int) -> np.ndarray:
        ddx = differentiate_axis(field, dx, axis=1)
        ddy = differentiate_axis(field, dy, axis=0)
        ddz = np.zeros_like(field)
        return np.stack((ddx, ddy, ddz), axis=-(tensor_rank + 1))

    return R26Gradients(
        rho=gradient(tensors.rho, 0),
        velocity=gradient(tensors.velocity, 1),
        theta=gradient(tensors.theta, 0),
        heat_flux=gradient(tensors.heat_flux, 1),
        sigma=gradient(tensors.sigma, 2),
        R=gradient(tensors.R, 2),
        m=gradient(tensors.m, 3),
        Delta=gradient(tensors.Delta, 0),
    )


def spectral_gu_emerson_closures(
    state: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    case: object | None = None,
    coefficient_mode: str | None = None,
) -> R26Closures:
    """Evaluate Gu--Emerson closures with the same spectral state gradient."""

    u = validate_planar_state(state)
    tensors = planar_state_to_tensors(u)
    gradients = spectral_gradients(u, x=x, y=y)
    return closures_from_tensors(
        tensors,
        gradients,
        mu=mu,
        coefficient_mode=resolve_closure_mode(
            case=case,
            coefficient_mode=coefficient_mode,
        ),
    )


def spectral_closure_derivatives(
    closures: R26Closures, *, x: np.ndarray, y: np.ndarray
) -> R26ClosureDerivatives:
    """Differentiate and contract phi/psi/Omega with spectral matrices."""

    dx = polynomial_first_derivative_matrix(x)
    dy = polynomial_first_derivative_matrix(y)

    def planar_gradient(field: np.ndarray) -> np.ndarray:
        return np.stack(
            (
                differentiate_axis(field, dx, axis=1),
                differentiate_axis(field, dy, axis=0),
                np.zeros_like(field),
            ),
            axis=2,
        )

    # gphi[..., d,i,j,k,l] and divergence d_l phi[i,j,k,l].
    gphi = planar_gradient(np.asarray(closures.phi, dtype=float))
    gpsi = planar_gradient(np.asarray(closures.psi, dtype=float))
    gomega = planar_gradient(np.asarray(closures.Omega, dtype=float))
    return R26ClosureDerivatives(
        div_phi=np.einsum("...l ijkl->...ijk", gphi, optimize=True),
        div_psi=np.einsum("...k ijk->...ij", gpsi, optimize=True),
        grad_Omega=gomega,
    )


def spectral_bulk_residual_grid(
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    body_force: np.ndarray | None = None,
    *,
    case: object | None = None,
    coefficient_mode: str | None = None,
) -> np.ndarray:
    """Evaluate the steady R26 bulk residual using global collocation."""

    u = validate_planar_state(state)
    if u.ndim != 3:
        raise ValueError(f"state grid must have shape (ny,nx,17), got {u.shape}")
    tensors = planar_state_to_tensors(u)
    gradients = spectral_gradients(u, x=x, y=y)
    closures = closures_from_tensors(
        tensors,
        gradients,
        mu=mu,
        coefficient_mode=resolve_closure_mode(
            case=case,
            coefficient_mode=coefficient_mode,
        ),
    )
    closure_derivatives = spectral_closure_derivatives(closures, x=x, y=y)
    force_density = _grid_force_density(body_force, u.shape[:2])
    acceleration = None
    if force_density is not None:
        acceleration = force_density / np.asarray(tensors.rho)[..., None]
    return steady_r26_bulk_residual(
        tensors,
        gradients,
        closures,
        closure_derivatives,
        mu=mu,
        acceleration=acceleration,
    ).as_planar17()


__all__ = [
    "differentiate_axis",
    "polynomial_first_derivative_matrix",
    "spectral_bulk_residual_grid",
    "spectral_closure_derivatives",
    "spectral_gradients",
    "spectral_gu_emerson_closures",
]
