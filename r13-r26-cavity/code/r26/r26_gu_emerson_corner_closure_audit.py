#!/usr/bin/env python3
"""N8 diagnostic that decouples smooth-wall closures from sharp corners.

The square-cavity corner values are the repository's declared bilinear,
non-paper extension.  The default global finite-difference closure gradient
uses those corner values in the tangential derivative at the first and last
smooth node of every wall.  This module provides a deliberately separate
diagnostic operator: only those eight tangential derivatives are replaced by
an O(2) one-sided derivative through three *smooth-wall* nodes.  Normal wall
derivatives, all bulk derivatives, the printed R26 closures, wall equations,
and transformed equation-(63) residual are unchanged.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields

import numpy as np

from r26_cases import CavityCase
from r26_discretization import R26NodeBVP
from r26_fv_backend import thor_fv_bulk_residual, wall_bounded_control_volume_weights
from r26_state import planar_state_to_tensors, validate_planar_state
from r26_tensor_closures import (
    R26Closures,
    R26Gradients,
    closures_from_tensors,
    finite_difference_gradients,
)


def _smooth_wall_tangential_derivative(
    values: np.ndarray,
    coordinates: np.ndarray,
    *,
    axis: int,
    upper: bool,
) -> np.ndarray:
    """Return the endpoint derivative on three non-corner wall nodes."""

    coordinate = np.asarray(coordinates, dtype=float)
    selection = slice(-4, -1) if upper else slice(1, 4)
    derivative = np.gradient(
        np.asarray(values)[selection],
        coordinate[selection],
        axis=axis,
        edge_order=2,
    )
    return np.asarray(derivative[-1 if upper else 0])


def _replace_corner_adjacent_tangential_derivatives(
    gradient: np.ndarray,
    values: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Replace the eight wall-adjacent tangential derivative entries."""

    result = np.asarray(gradient, dtype=float).copy()
    source = np.asarray(values, dtype=float)
    ny, nx = source.shape[:2]
    for i in (0, nx - 1):
        result[1, i, 1] = _smooth_wall_tangential_derivative(
            source[:, i], y, axis=0, upper=False
        )
        result[ny - 2, i, 1] = _smooth_wall_tangential_derivative(
            source[:, i], y, axis=0, upper=True
        )
    for j in (0, ny - 1):
        result[j, 1, 0] = _smooth_wall_tangential_derivative(
            source[j], x, axis=0, upper=False
        )
        result[j, nx - 2, 0] = _smooth_wall_tangential_derivative(
            source[j], x, axis=0, upper=True
        )
    return result


def corner_excluding_wall_gradients(
    state: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
) -> R26Gradients:
    """Differentiate R26 tensors without corner values in smooth-wall tangents."""

    value = validate_planar_state(state)
    if value.ndim != 3 or min(value.shape[:2]) < 5:
        raise ValueError("corner-excluding closure gradients require a grid of at least 5x5")
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.shape != (value.shape[1],) or yy.shape != (value.shape[0],):
        raise ValueError("x and y coordinates must match the state grid")
    if np.any(np.diff(xx) <= 0.0) or np.any(np.diff(yy) <= 0.0):
        raise ValueError("x and y coordinates must be strictly increasing")

    tensors = planar_state_to_tensors(value)
    baseline = finite_difference_gradients(value, x=xx, y=yy, edge_order=2)
    corrected: dict[str, np.ndarray] = {}
    for item in dataclass_fields(R26Gradients):
        name = item.name
        corrected[name] = _replace_corner_adjacent_tangential_derivatives(
            getattr(baseline, name), getattr(tensors, name), x=xx, y=yy
        )
    return R26Gradients(**corrected)


def corner_excluding_wall_closures(
    state: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    case: CavityCase,
) -> R26Closures:
    """Evaluate the printed closures with the diagnostic wall gradients."""

    value = validate_planar_state(state)
    return closures_from_tensors(
        planar_state_to_tensors(value),
        corner_excluding_wall_gradients(value, x=x, y=y),
        mu=mu,
        coefficient_mode=case.r26_closure_mode,
    )


def make_corner_excluding_gu_emerson_problem(case: CavityCase) -> R26NodeBVP:
    """Build a diagnostic physical objective; never used by the production factory."""

    return R26NodeBVP(
        case,
        bulk_operator=thor_fv_bulk_residual,
        closure_operator=corner_excluding_wall_closures,
        mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
    )


__all__ = [
    "corner_excluding_wall_closures",
    "corner_excluding_wall_gradients",
    "make_corner_excluding_gu_emerson_problem",
]
