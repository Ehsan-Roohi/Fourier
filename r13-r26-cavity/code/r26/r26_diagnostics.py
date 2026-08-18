#!/usr/bin/env python3
"""Fail-closed diagnostics for the private planar Python R26 solver.

These routines separate algebraic/physical admissibility checks from
comparison with an external R13 or DSMC reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from r26_state import STATE_INDEX, planar_state_to_tensors, validate_planar_state


@dataclass(frozen=True)
class PhysicalDiagnostics:
    finite: bool
    rho_min: float
    temperature_min: float
    pressure_min: float
    realizability_eigenvalue_min: float
    realizability_negative_count: int
    mass_mean: float
    mass_error: float


def realizability_matrix(state: np.ndarray) -> np.ndarray:
    """Return Gu--Emerson Eq. (35)'s 3x3 validity matrix.

    With ``R=1`` in the nondimensional solver,
    ``V = I + sigma/p - [2 q q/(3 p^2 T)]/[1+Delta/(6 p T)]``.
    Non-negative eigenvalues are necessary, but not sufficient, for moment
    realizability.
    """

    u = validate_planar_state(state)
    tensors = planar_state_to_tensors(u)
    rho = np.asarray(tensors.rho)
    temperature = np.asarray(tensors.theta)
    pressure = rho * temperature
    denominator = 1.0 + np.asarray(tensors.Delta) / (6.0 * pressure * temperature)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0.0):
        raise FloatingPointError("Eq. (35) denominator is non-positive")
    return (
        np.eye(3)
        + np.asarray(tensors.sigma) / pressure[..., None, None]
        - 2.0
        * np.einsum("...i,...j->...ij", tensors.heat_flux, tensors.heat_flux)
        / (3.0 * pressure[..., None, None] ** 2 * temperature[..., None, None])
        / denominator[..., None, None]
    )


def physical_diagnostics(state: np.ndarray, *, target_mean_density: float = 1.0) -> PhysicalDiagnostics:
    u = validate_planar_state(state)
    pressure = u[..., STATE_INDEX["rho"]] * u[..., STATE_INDEX["theta"]]
    eigenvalues = np.linalg.eigvalsh(realizability_matrix(u))
    mean_density = float(np.mean(u[..., STATE_INDEX["rho"]]))
    return PhysicalDiagnostics(
        finite=bool(np.isfinite(u).all()),
        rho_min=float(np.min(u[..., STATE_INDEX["rho"]])),
        temperature_min=float(np.min(u[..., STATE_INDEX["theta"]])),
        pressure_min=float(np.min(pressure)),
        realizability_eigenvalue_min=float(np.min(eigenvalues)),
        realizability_negative_count=int(np.count_nonzero(eigenvalues < -1.0e-10)),
        mass_mean=mean_density,
        mass_error=mean_density - float(target_mean_density),
    )


def rana_global_metrics(
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    lid_velocity: float,
) -> dict[str, float | str]:
    """Evaluate Rana Eq. (30) on a wall-inclusive node grid."""

    u = validate_planar_state(state)
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    if u.ndim != 3 or u.shape[:2] != (yv.size, xv.size):
        raise ValueError("state/grid shape mismatch")
    if xv.size < 3 or yv.size < 3 or np.any(np.diff(xv) <= 0.0) or np.any(np.diff(yv) <= 0.0):
        raise ValueError("x and y must be increasing node coordinates")
    speed = abs(float(lid_velocity))
    if speed <= 0.0:
        raise ValueError("lid velocity must be nonzero")
    sigma_integral = float(np.trapezoid(u[-1, :, STATE_INDEX["sigma_xy"]], xv))
    center_velocity = np.asarray(
        [np.interp(0.5, xv, row[:, STATE_INDEX["vx"]]) for row in u], dtype=float
    )
    reduction = np.sqrt(2.0) / speed
    return {
        "D": abs(reduction * sigma_integral),
        "D_signed": reduction * sigma_integral,
        "D_sigma_over_p0_signed": sigma_integral,
        "D_reduced_stress_factor": float(reduction),
        "G": float(np.trapezoid(np.abs(center_velocity), yv) / speed),
        "provenance": "Rana Eq. (30), wall-inclusive node-grid trapezoid",
    }


def relative_grid_change(coarse: np.ndarray, fine_on_coarse: np.ndarray) -> np.ndarray:
    """Per-component RMS relative change with a symmetric safe denominator."""

    a = validate_planar_state(coarse)
    b = validate_planar_state(fine_on_coarse)
    if a.shape != b.shape:
        raise ValueError("coarse and restricted-fine states must have the same shape")
    axes = tuple(range(a.ndim - 1))
    numerator = np.sqrt(np.mean((b - a) ** 2, axis=axes))
    scale = np.maximum(np.sqrt(np.mean(a**2, axis=axes)), np.sqrt(np.mean(b**2, axis=axes)))
    return numerator / np.maximum(scale, 1.0e-14)


__all__ = [
    "PhysicalDiagnostics",
    "physical_diagnostics",
    "rana_global_metrics",
    "realizability_matrix",
    "relative_grid_change",
]
