#!/usr/bin/env python3
"""Even/odd moment-compatible finite-difference backend for planar R26.

Moment systems naturally alternate tensor rank: spatial differentiation maps
an even-rank moment equation to an odd-rank one and vice versa.  Applying the
same centred collocated derivative in both directions makes the composed
operator ``D*D`` blind to the Nyquist/checkerboard mode.  A conventional
staggered discretization instead uses a forward difference from even to odd
locations and the paired backward difference from odd to even locations.

This module preserves that algebraic pairing while retaining the existing
17-component array API:

* even moments ``rho,theta,sigma,R,Delta`` use ``D_plus``;
* odd moments ``u,q,m`` use ``D_minus``;
* even closure ``phi`` uses ``D_plus`` in its divergence; and
* odd closures ``psi,Omega`` use ``D_minus``.

The construction adds no filtering, penalty, artificial constraint, or
regularization.  It is first-order at the present prototype level, exactly
differentiates constants and linear fields, and the paired second derivative
has no alternating-grid kernel.  It should be viewed as a checkerboard audit
backend; a production implementation should place the odd moments on their
actual staggered faces and use an SBP/FV boundary closure.
"""

from __future__ import annotations

import numpy as np

from r26_bulk_equations import (
    R26ClosureDerivatives,
    _grid_force_density,
    steady_r26_bulk_residual,
)
from r26_spectral_backend import (
    differentiate_axis,
    polynomial_first_derivative_matrix,
)
from r26_state import planar_state_to_tensors, validate_planar_state
from r26_tensor_closures import (
    R26Closures,
    R26Gradients,
    closures_from_tensors,
    resolve_closure_mode,
)


def paired_first_derivative_matrices(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return forward ``D_plus`` and backward ``D_minus`` nodal differences.

    The final/initial rows respectively use the only inward first difference.
    On all true interior rows the pair has opposite bias.  Both matrices have
    precisely the constants as their one-dimensional nullspace.
    """

    x = np.asarray(nodes, dtype=float)
    if x.ndim != 1 or x.size < 3 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0.0):
        raise ValueError("nodes must be a strictly increasing finite vector of length >= 3")
    n = x.size
    plus = np.zeros((n, n))
    minus = np.zeros((n, n))
    for i in range(n - 1):
        inv = 1.0 / (x[i + 1] - x[i])
        plus[i, i] = -inv
        plus[i, i + 1] = inv
    plus[-1, -2:] = np.asarray((-1.0, 1.0)) / (x[-1] - x[-2])
    for i in range(1, n):
        inv = 1.0 / (x[i] - x[i - 1])
        minus[i, i - 1] = -inv
        minus[i, i] = inv
    minus[0, :2] = np.asarray((-1.0, 1.0)) / (x[1] - x[0])
    return plus, minus


def oriented_second_order_matrix(nodes: np.ndarray, orientation: str = "forward") -> np.ndarray:
    """Return a three-point one-sided, second-order differentiation matrix.

    ``forward`` uses ``(i,i+1,i+2)`` wherever possible and the final
    three-node polynomial at the last two rows.  ``backward`` is its exact
    reflected counterpart.  Local polynomial weights also make this valid on
    nonuniform nodes.  The two orientations must be used as a reported
    sensitivity pair; neither is silently privileged as a physical direction.
    """

    x = np.asarray(nodes, dtype=float)
    if x.ndim != 1 or x.size < 3 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0.0):
        raise ValueError("nodes must be a strictly increasing finite vector of length >= 3")
    key = orientation.lower()
    if key not in {"forward", "backward", "wall", "alternating"}:
        raise ValueError(
            "orientation must be 'forward', 'backward', 'wall', or 'alternating'"
        )
    n = x.size
    result = np.zeros((n, n))
    for i in range(n):
        if key == "alternating":
            midpoint = 0.5 * (n - 1)
            if i == midpoint:
                start = i - 1
            else:
                reflected_i = i if i < midpoint else n - 1 - i
                lower_choice = "backward" if reflected_i % 2 == 0 else "forward"
                choice = lower_choice if i < midpoint else (
                    "forward" if lower_choice == "backward" else "backward"
                )
                start = min(i, n - 3) if choice == "forward" else max(0, i - 2)
        elif key == "wall":
            midpoint = 0.5 * (n - 1)
            if n % 2 == 0 and i == n // 2 - 1:
                # Couple the two half grids with a reflected crossing pair.
                start = i
            elif n % 2 == 0 and i == n // 2:
                start = i - 2
            elif i < midpoint:
                start = max(0, i - 2)
            elif i > midpoint:
                start = min(i, n - 3)
            else:
                # The unique reflection-antisymmetric three-point row.
                start = i - 1
        elif key == "forward":
            start = min(i, n - 3)
        else:
            start = max(0, i - 2)
        indices = np.arange(start, start + 3)
        local = polynomial_first_derivative_matrix(x[indices])
        local_row = int(np.where(indices == i)[0][0])
        result[i, indices] = local[local_row]
    return result


def _apply_gradient(
    field: np.ndarray,
    *,
    dx: np.ndarray,
    dy: np.ndarray,
    tensor_rank: int,
) -> np.ndarray:
    return np.stack(
        (
            differentiate_axis(field, dx, axis=1),
            differentiate_axis(field, dy, axis=0),
            np.zeros_like(field),
        ),
        axis=-(tensor_rank + 1),
    )


def staggered_gradients(state: np.ndarray, *, x: np.ndarray, y: np.ndarray) -> R26Gradients:
    """Compute rank-parity compatible gradients of the 17-state grid."""

    u = validate_planar_state(state)
    if u.ndim != 3:
        raise ValueError(f"state grid must have shape (ny,nx,17), got {u.shape}")
    dx_plus, dx_minus = paired_first_derivative_matrices(x)
    dy_plus, dy_minus = paired_first_derivative_matrices(y)
    t = planar_state_to_tensors(u)

    def even(field: np.ndarray, rank: int) -> np.ndarray:
        return _apply_gradient(field, dx=dx_plus, dy=dy_plus, tensor_rank=rank)

    def odd(field: np.ndarray, rank: int) -> np.ndarray:
        return _apply_gradient(field, dx=dx_minus, dy=dy_minus, tensor_rank=rank)

    return R26Gradients(
        rho=even(t.rho, 0),
        velocity=odd(t.velocity, 1),
        theta=even(t.theta, 0),
        heat_flux=odd(t.heat_flux, 1),
        sigma=even(t.sigma, 2),
        R=even(t.R, 2),
        m=odd(t.m, 3),
        Delta=even(t.Delta, 0),
    )


def staggered_gu_emerson_closures(
    state: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    case: object | None = None,
    coefficient_mode: str | None = None,
) -> R26Closures:
    u = validate_planar_state(state)
    return closures_from_tensors(
        planar_state_to_tensors(u),
        staggered_gradients(u, x=x, y=y),
        mu=mu,
        coefficient_mode=resolve_closure_mode(
            case=case,
            coefficient_mode=coefficient_mode,
        ),
    )


def staggered_closure_derivatives(
    closures: R26Closures, *, x: np.ndarray, y: np.ndarray
) -> R26ClosureDerivatives:
    """Use even-to-odd D+ for phi and odd-to-even D- for psi/Omega."""

    dx_plus, dx_minus = paired_first_derivative_matrices(x)
    dy_plus, dy_minus = paired_first_derivative_matrices(y)

    def gradient(field: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        return np.stack(
            (
                differentiate_axis(field, dx, axis=1),
                differentiate_axis(field, dy, axis=0),
                np.zeros_like(field),
            ),
            axis=2,
        )

    gphi = gradient(np.asarray(closures.phi), dx_plus, dy_plus)
    gpsi = gradient(np.asarray(closures.psi), dx_minus, dy_minus)
    gomega = gradient(np.asarray(closures.Omega), dx_minus, dy_minus)
    return R26ClosureDerivatives(
        div_phi=np.einsum("...lijkl->...ijk", gphi, optimize=True),
        div_psi=np.einsum("...kijk->...ij", gpsi, optimize=True),
        grad_Omega=gomega,
    )


def staggered_bulk_residual_grid(
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    body_force: np.ndarray | None = None,
    *,
    case: object | None = None,
    coefficient_mode: str | None = None,
) -> np.ndarray:
    """Evaluate the full steady R26 bulk residual with paired derivatives."""

    u = validate_planar_state(state)
    t = planar_state_to_tensors(u)
    gradients = staggered_gradients(u, x=x, y=y)
    closures = closures_from_tensors(
        t,
        gradients,
        mu=mu,
        coefficient_mode=resolve_closure_mode(
            case=case,
            coefficient_mode=coefficient_mode,
        ),
    )
    derivatives = staggered_closure_derivatives(closures, x=x, y=y)
    force_density = _grid_force_density(body_force, u.shape[:2])
    acceleration = None
    if force_density is not None:
        acceleration = force_density / np.asarray(t.rho)[..., None]
    return steady_r26_bulk_residual(
        t,
        gradients,
        closures,
        derivatives,
        mu=mu,
        acceleration=acceleration,
    ).as_planar17()


def oriented_forward_gradients(
    state: np.ndarray, *, x: np.ndarray, y: np.ndarray
) -> R26Gradients:
    """Consistent one-sided audit gradient using D+ for every moment.

    This deliberately orientation-dependent operator is exposed for a paired
    forward/reverse sensitivity audit.  It is consistent and checkerboard-
    free but is not the preferred production discretization.
    """

    u = validate_planar_state(state)
    dx, _ = paired_first_derivative_matrices(x)
    dy, _ = paired_first_derivative_matrices(y)
    t = planar_state_to_tensors(u)

    def gradient(field: np.ndarray, rank: int) -> np.ndarray:
        return _apply_gradient(field, dx=dx, dy=dy, tensor_rank=rank)

    return R26Gradients(
        rho=gradient(t.rho, 0),
        velocity=gradient(t.velocity, 1),
        theta=gradient(t.theta, 0),
        heat_flux=gradient(t.heat_flux, 1),
        sigma=gradient(t.sigma, 2),
        R=gradient(t.R, 2),
        m=gradient(t.m, 3),
        Delta=gradient(t.Delta, 0),
    )


def oriented_forward_gu_emerson_closures(
    state: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    case: object | None = None,
    coefficient_mode: str | None = None,
) -> R26Closures:
    u = validate_planar_state(state)
    return closures_from_tensors(
        planar_state_to_tensors(u),
        oriented_forward_gradients(u, x=x, y=y),
        mu=mu,
        coefficient_mode=resolve_closure_mode(
            case=case,
            coefficient_mode=coefficient_mode,
        ),
    )


def oriented_forward_closure_derivatives(
    closures: R26Closures, *, x: np.ndarray, y: np.ndarray
) -> R26ClosureDerivatives:
    dx, _ = paired_first_derivative_matrices(x)
    dy, _ = paired_first_derivative_matrices(y)

    def gradient(field: np.ndarray) -> np.ndarray:
        return np.stack(
            (
                differentiate_axis(field, dx, axis=1),
                differentiate_axis(field, dy, axis=0),
                np.zeros_like(field),
            ),
            axis=2,
        )

    gphi = gradient(np.asarray(closures.phi))
    gpsi = gradient(np.asarray(closures.psi))
    gomega = gradient(np.asarray(closures.Omega))
    return R26ClosureDerivatives(
        div_phi=np.einsum("...lijkl->...ijk", gphi, optimize=True),
        div_psi=np.einsum("...kijk->...ij", gpsi, optimize=True),
        grad_Omega=gomega,
    )


def oriented_forward_bulk_residual_grid(
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    body_force: np.ndarray | None = None,
    *,
    case: object | None = None,
    coefficient_mode: str | None = None,
) -> np.ndarray:
    u = validate_planar_state(state)
    t = planar_state_to_tensors(u)
    gradients = oriented_forward_gradients(u, x=x, y=y)
    closures = closures_from_tensors(
        t,
        gradients,
        mu=mu,
        coefficient_mode=resolve_closure_mode(
            case=case,
            coefficient_mode=coefficient_mode,
        ),
    )
    derivatives = oriented_forward_closure_derivatives(closures, x=x, y=y)
    force_density = _grid_force_density(body_force, u.shape[:2])
    acceleration = None
    if force_density is not None:
        acceleration = force_density / np.asarray(t.rho)[..., None]
    return steady_r26_bulk_residual(
        t,
        gradients,
        closures,
        derivatives,
        mu=mu,
        acceleration=acceleration,
    ).as_planar17()


def oriented_second_order_gradients(
    state: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    orientation: str = "forward",
    orientation_x: str | None = None,
    orientation_y: str | None = None,
) -> R26Gradients:
    """Apply the declared O(h^2) one-sided orientation to every moment."""

    u = validate_planar_state(state)
    ox = orientation if orientation_x is None else orientation_x
    oy = orientation if orientation_y is None else orientation_y
    dx = oriented_second_order_matrix(x, ox)
    dy = oriented_second_order_matrix(y, oy)
    t = planar_state_to_tensors(u)

    def gradient(field: np.ndarray, rank: int) -> np.ndarray:
        return _apply_gradient(field, dx=dx, dy=dy, tensor_rank=rank)

    return R26Gradients(
        rho=gradient(t.rho, 0),
        velocity=gradient(t.velocity, 1),
        theta=gradient(t.theta, 0),
        heat_flux=gradient(t.heat_flux, 1),
        sigma=gradient(t.sigma, 2),
        R=gradient(t.R, 2),
        m=gradient(t.m, 3),
        Delta=gradient(t.Delta, 0),
    )


def _oriented_second_order_closures(
    state: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    orientation_x: str,
    orientation_y: str,
    case: object | None = None,
    coefficient_mode: str | None = None,
) -> R26Closures:
    u = validate_planar_state(state)
    gradients = oriented_second_order_gradients(
        u, x=x, y=y, orientation_x=orientation_x, orientation_y=orientation_y
    )
    return closures_from_tensors(
        planar_state_to_tensors(u),
        gradients,
        mu=mu,
        coefficient_mode=resolve_closure_mode(
            case=case,
            coefficient_mode=coefficient_mode,
        ),
    )


def _oriented_second_order_closure_derivatives(
    closures: R26Closures,
    *,
    x: np.ndarray,
    y: np.ndarray,
    orientation_x: str,
    orientation_y: str,
) -> R26ClosureDerivatives:
    dx = oriented_second_order_matrix(x, orientation_x)
    dy = oriented_second_order_matrix(y, orientation_y)

    def gradient(field: np.ndarray) -> np.ndarray:
        return np.stack(
            (
                differentiate_axis(field, dx, axis=1),
                differentiate_axis(field, dy, axis=0),
                np.zeros_like(field),
            ),
            axis=2,
        )

    gphi = gradient(np.asarray(closures.phi))
    gpsi = gradient(np.asarray(closures.psi))
    gomega = gradient(np.asarray(closures.Omega))
    return R26ClosureDerivatives(
        div_phi=np.einsum("...lijkl->...ijk", gphi, optimize=True),
        div_psi=np.einsum("...kijk->...ij", gpsi, optimize=True),
        grad_Omega=gomega,
    )


def _oriented_second_order_bulk(
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    body_force: np.ndarray | None,
    *,
    orientation_x: str,
    orientation_y: str,
    case: object | None = None,
    coefficient_mode: str | None = None,
) -> np.ndarray:
    u = validate_planar_state(state)
    t = planar_state_to_tensors(u)
    gradients = oriented_second_order_gradients(
        u, x=x, y=y, orientation_x=orientation_x, orientation_y=orientation_y
    )
    closures = closures_from_tensors(
        t,
        gradients,
        mu=mu,
        coefficient_mode=resolve_closure_mode(
            case=case,
            coefficient_mode=coefficient_mode,
        ),
    )
    derivatives = _oriented_second_order_closure_derivatives(
        closures, x=x, y=y, orientation_x=orientation_x, orientation_y=orientation_y
    )
    force_density = _grid_force_density(body_force, u.shape[:2])
    acceleration = None
    if force_density is not None:
        acceleration = force_density / np.asarray(t.rho)[..., None]
    return steady_r26_bulk_residual(
        t,
        gradients,
        closures,
        derivatives,
        mu=mu,
        acceleration=acceleration,
    ).as_planar17()


def forward_second_order_gu_emerson_closures(
    state: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    case: object | None = None,
    coefficient_mode: str | None = None,
) -> R26Closures:
    return _oriented_second_order_closures(
        state,
        x=x,
        y=y,
        mu=mu,
        orientation_x="forward",
        orientation_y="forward",
        case=case,
        coefficient_mode=coefficient_mode,
    )


def backward_second_order_gu_emerson_closures(
    state: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    case: object | None = None,
    coefficient_mode: str | None = None,
) -> R26Closures:
    return _oriented_second_order_closures(
        state,
        x=x,
        y=y,
        mu=mu,
        orientation_x="backward",
        orientation_y="backward",
        case=case,
        coefficient_mode=coefficient_mode,
    )


def forward_second_order_bulk_residual_grid(
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    body_force: np.ndarray | None = None,
    *,
    case: object | None = None,
    coefficient_mode: str | None = None,
) -> np.ndarray:
    return _oriented_second_order_bulk(
        state,
        x,
        y,
        mu,
        body_force,
        orientation_x="forward",
        orientation_y="forward",
        case=case,
        coefficient_mode=coefficient_mode,
    )


def backward_second_order_bulk_residual_grid(
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    body_force: np.ndarray | None = None,
    *,
    case: object | None = None,
    coefficient_mode: str | None = None,
) -> np.ndarray:
    return _oriented_second_order_bulk(
        state,
        x,
        y,
        mu,
        body_force,
        orientation_x="backward",
        orientation_y="backward",
        case=case,
        coefficient_mode=coefficient_mode,
    )


def make_oriented_second_order_operators(
    orientation_x: str, orientation_y: str
) -> tuple[object, object]:
    """Return BVP-compatible bulk/closure callables for one x/y bias pair."""

    # Validate eagerly.
    oriented_second_order_matrix(np.asarray((0.0, 0.5, 1.0)), orientation_x)
    oriented_second_order_matrix(np.asarray((0.0, 0.5, 1.0)), orientation_y)

    def closure_operator(
        state: np.ndarray,
        *,
        x: np.ndarray,
        y: np.ndarray,
        mu: np.ndarray,
        case: object | None = None,
        coefficient_mode: str | None = None,
    ) -> R26Closures:
        return _oriented_second_order_closures(
            state,
            x=x,
            y=y,
            mu=mu,
            orientation_x=orientation_x,
            orientation_y=orientation_y,
            case=case,
            coefficient_mode=coefficient_mode,
        )

    def bulk_operator(
        state: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        mu: np.ndarray,
        body_force: np.ndarray | None = None,
        *,
        case: object | None = None,
        coefficient_mode: str | None = None,
    ) -> np.ndarray:
        return _oriented_second_order_bulk(
            state,
            x,
            y,
            mu,
            body_force,
            orientation_x=orientation_x,
            orientation_y=orientation_y,
            case=case,
            coefficient_mode=coefficient_mode,
        )

    closure_operator.__name__ = f"{orientation_x}_{orientation_y}_o2_closures"
    bulk_operator.__name__ = f"{orientation_x}_{orientation_y}_o2_bulk"
    return bulk_operator, closure_operator


__all__ = [
    "backward_second_order_bulk_residual_grid",
    "backward_second_order_gu_emerson_closures",
    "forward_second_order_bulk_residual_grid",
    "forward_second_order_gu_emerson_closures",
    "make_oriented_second_order_operators",
    "oriented_forward_bulk_residual_grid",
    "oriented_forward_closure_derivatives",
    "oriented_forward_gradients",
    "oriented_forward_gu_emerson_closures",
    "oriented_second_order_gradients",
    "oriented_second_order_matrix",
    "paired_first_derivative_matrices",
    "staggered_bulk_residual_grid",
    "staggered_closure_derivatives",
    "staggered_gradients",
    "staggered_gu_emerson_closures",
]
