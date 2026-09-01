#!/usr/bin/env python3
"""Gu--Emerson gradient/non-gradient R26 variable transformation.

Gu & Emerson, JFM 636 (2009), equations (48)--(55), do not solve the
physical moments ``sigma, q, m, R, Delta`` as the primary finite-volume
unknowns.  They split each moment into a printed gradient contribution and a
non-gradient contribution::

    sigma = sigma_G + rho*g       q     = q_G     + rho*h
    m     = m_G     + rho*omega   R     = R_G     + rho*gamma
    Delta = Delta_G + rho*chi

This module implements that change of variables only.  It deliberately does
not choose a linear solver, relaxation factor, source-term linearisation,
Rhie--Chow coefficient, or corner rule; those implementation details are not
printed in the paper and do not belong in an exact algebraic mapping.

The scalar ``theta`` used by this repository is ``R*T``.  Therefore the
printed Fourier contribution ``-(15/4) R mu grad(T)`` is represented as
``-(15/4) mu grad(theta)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from r26_cases import CavityCase
from r26_state import (
    NVAR,
    StateTensors,
    planar_state_to_tensors,
    tensors_to_planar_state,
    validate_planar_state,
)
from r26_tensor_closures import stf2_project, stf3_project


GU_EMERSON_VARIABLE_PROVENANCE: Final[str] = (
    "Gu--Emerson, JFM 636 (2009), equations (15) and (48)--(55); "
    "theta=R*T"
)


@dataclass(frozen=True)
class GuEmersonFields:
    """The cell fields advanced by the segregated Gu--Emerson algorithm.

    ``g`` and ``gamma`` are full three-dimensional STF rank-2 tensors;
    ``omega`` is a full STF rank-3 tensor.  A planar calculation retains full
    2D3V tensor structure and enforces z-reflection parity when converting
    back to the public 17-component state.
    """

    rho: np.ndarray
    velocity: np.ndarray
    theta: np.ndarray
    g: np.ndarray
    h: np.ndarray
    omega: np.ndarray
    gamma: np.ndarray
    chi: np.ndarray
    provenance: str = GU_EMERSON_VARIABLE_PROVENANCE


def _coordinates(coordinate: np.ndarray, size: int, name: str) -> np.ndarray:
    value = np.asarray(coordinate, dtype=float)
    if (
        value.shape != (size,)
        or not np.isfinite(value).all()
        or np.any(np.diff(value) <= 0.0)
    ):
        raise ValueError(f"{name} must be finite, strictly increasing, and length {size}")
    return value


def _positive_grid(value: float | np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    try:
        result = np.broadcast_to(np.asarray(value, dtype=float), shape)
    except ValueError as exc:
        raise ValueError(f"{name} cannot broadcast to grid shape {shape}") from exc
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise FloatingPointError(f"{name} must be finite and positive")
    return result


def _grid_gradient(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    tensor_rank: int,
) -> np.ndarray:
    """Derivative-first 2D3V gradient with the repository's O(2) stencil."""

    value = np.asarray(field, dtype=float)
    if value.ndim != 2 + tensor_rank or value.shape[:2] != (y.size, x.size):
        raise ValueError(
            f"rank-{tensor_rank} field must start with {(y.size, x.size)}, got {value.shape}"
        )
    if value.shape[2:] != (3,) * tensor_rank:
        raise ValueError(f"rank-{tensor_rank} field must end in {(3,) * tensor_rank}")
    if not np.isfinite(value).all():
        raise FloatingPointError("field contains NaN or infinity")
    if min(x.size, y.size) < 3:
        raise ValueError("Gu--Emerson gradient reconstruction requires at least 3x3 cells")
    d_dx = np.gradient(value, x, axis=1, edge_order=2)
    d_dy = np.gradient(value, y, axis=0, edge_order=2)
    d_dz = np.zeros_like(value)
    return np.stack((d_dx, d_dy, d_dz), axis=2)


def _validate_fields(fields: GuEmersonFields, x: np.ndarray, y: np.ndarray) -> GuEmersonFields:
    shape = (y.size, x.size)
    expected = {
        "rho": shape,
        "velocity": shape + (3,),
        "theta": shape,
        "g": shape + (3, 3),
        "h": shape + (3,),
        "omega": shape + (3, 3, 3),
        "gamma": shape + (3, 3),
        "chi": shape,
    }
    for name, expected_shape in expected.items():
        value = np.asarray(getattr(fields, name), dtype=float)
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}")
        if not np.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN or infinity")
    if np.any(np.asarray(fields.rho) <= 0.0) or np.any(np.asarray(fields.theta) <= 0.0):
        raise FloatingPointError("Gu--Emerson fields require positive rho and theta")

    atol = 2.0e-11
    for name in ("g", "gamma"):
        value = np.asarray(getattr(fields, name), dtype=float)
        if np.max(np.abs(value - np.swapaxes(value, -1, -2)), initial=0.0) > atol:
            raise ValueError(f"{name} must be symmetric")
        if np.max(np.abs(np.trace(value, axis1=-2, axis2=-1)), initial=0.0) > atol:
            raise ValueError(f"{name} must be trace free")
    omega = np.asarray(fields.omega, dtype=float)
    if np.max(np.abs(omega - stf3_project(omega)), initial=0.0) > atol:
        raise ValueError("omega must be symmetric and trace free")
    return fields


def gradient_parts_from_primitive_and_moments(
    *,
    rho: np.ndarray,
    velocity: np.ndarray,
    theta: np.ndarray,
    sigma: np.ndarray,
    heat_flux: np.ndarray,
    mu: float | np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the five printed gradient contributions in equations (49)--(52).

    The ``m``, ``R`` and ``Delta`` contributions use the quotient fields
    ``sigma/rho`` and ``q/rho`` exactly as printed in equation (15).  No
    product-rule approximation or frozen-density substitution is used.
    """

    rho_value = np.asarray(rho, dtype=float)
    theta_value = np.asarray(theta, dtype=float)
    shape = rho_value.shape
    if shape != (len(y), len(x)) or theta_value.shape != shape:
        raise ValueError("rho and theta must match the supplied two-dimensional grid")
    if np.any(rho_value <= 0.0) or np.any(theta_value <= 0.0):
        raise FloatingPointError("rho and theta must be positive")
    viscosity = _positive_grid(mu, shape, "viscosity mu")
    grad_velocity = _grid_gradient(velocity, x, y, tensor_rank=1)
    grad_theta = _grid_gradient(theta_value, x, y, tensor_rank=0)
    sigma_gradient = -2.0 * viscosity[..., None, None] * stf2_project(grad_velocity)
    q_gradient = -15.0 / 4.0 * viscosity[..., None] * grad_theta

    sigma_over_rho = np.asarray(sigma, dtype=float) / rho_value[..., None, None]
    q_over_rho = np.asarray(heat_flux, dtype=float) / rho_value[..., None]
    grad_sigma_over_rho = _grid_gradient(sigma_over_rho, x, y, tensor_rank=2)
    grad_q_over_rho = _grid_gradient(q_over_rho, x, y, tensor_rank=1)
    m_gradient = -2.0 * viscosity[..., None, None, None] * stf3_project(
        grad_sigma_over_rho
    )
    R_gradient = -24.0 / 5.0 * viscosity[..., None, None] * stf2_project(
        grad_q_over_rho
    )
    Delta_gradient = -12.0 * viscosity * np.einsum("...ii->...", grad_q_over_rho)
    return sigma_gradient, q_gradient, m_gradient, R_gradient, Delta_gradient


def gu_emerson_fields_from_state(
    state: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    mu: float | np.ndarray,
) -> GuEmersonFields:
    """Map a physical planar R26 state to the equation-(48) primary fields."""

    tensors = planar_state_to_tensors(state)
    ny, nx = np.asarray(tensors.rho).shape
    xx = _coordinates(x, nx, "x")
    yy = _coordinates(y, ny, "y")
    parts = gradient_parts_from_primitive_and_moments(
        rho=tensors.rho,
        velocity=tensors.velocity,
        theta=tensors.theta,
        sigma=tensors.sigma,
        heat_flux=tensors.heat_flux,
        mu=mu,
        x=xx,
        y=yy,
    )
    sigma_gradient, q_gradient, m_gradient, R_gradient, Delta_gradient = parts
    rho = np.asarray(tensors.rho, dtype=float)
    result = GuEmersonFields(
        rho=rho.copy(),
        velocity=np.asarray(tensors.velocity, dtype=float).copy(),
        theta=np.asarray(tensors.theta, dtype=float).copy(),
        g=(np.asarray(tensors.sigma) - sigma_gradient) / rho[..., None, None],
        h=(np.asarray(tensors.heat_flux) - q_gradient) / rho[..., None],
        omega=(np.asarray(tensors.m) - m_gradient) / rho[..., None, None, None],
        gamma=(np.asarray(tensors.R) - R_gradient) / rho[..., None, None],
        chi=(np.asarray(tensors.Delta) - Delta_gradient) / rho,
    )
    return _validate_fields(result, xx, yy)


def state_from_gu_emerson_fields(
    fields: GuEmersonFields,
    *,
    x: np.ndarray,
    y: np.ndarray,
    mu: float | np.ndarray,
) -> np.ndarray:
    """Reconstruct ``sigma,q,m,R,Delta`` sequentially from equations (48)--(55)."""

    rho = np.asarray(fields.rho, dtype=float)
    ny, nx = rho.shape
    xx = _coordinates(x, nx, "x")
    yy = _coordinates(y, ny, "y")
    _validate_fields(fields, xx, yy)
    viscosity = _positive_grid(mu, rho.shape, "viscosity mu")
    velocity = np.asarray(fields.velocity, dtype=float)
    theta = np.asarray(fields.theta, dtype=float)

    grad_velocity = _grid_gradient(velocity, xx, yy, tensor_rank=1)
    grad_theta = _grid_gradient(theta, xx, yy, tensor_rank=0)
    sigma = (
        -2.0 * viscosity[..., None, None] * stf2_project(grad_velocity)
        + rho[..., None, None] * np.asarray(fields.g)
    )
    heat_flux = (
        -15.0 / 4.0 * viscosity[..., None] * grad_theta
        + rho[..., None] * np.asarray(fields.h)
    )

    grad_sigma_over_rho = _grid_gradient(
        sigma / rho[..., None, None], xx, yy, tensor_rank=2
    )
    grad_q_over_rho = _grid_gradient(
        heat_flux / rho[..., None], xx, yy, tensor_rank=1
    )
    m = (
        -2.0 * viscosity[..., None, None, None] * stf3_project(grad_sigma_over_rho)
        + rho[..., None, None, None] * np.asarray(fields.omega)
    )
    R = (
        -24.0 / 5.0 * viscosity[..., None, None] * stf2_project(grad_q_over_rho)
        + rho[..., None, None] * np.asarray(fields.gamma)
    )
    Delta = (
        -12.0 * viscosity * np.einsum("...ii->...", grad_q_over_rho)
        + rho * np.asarray(fields.chi)
    )
    return tensors_to_planar_state(
        StateTensors(
            rho=rho,
            velocity=velocity,
            theta=theta,
            heat_flux=heat_flux,
            sigma=sigma,
            R=R,
            m=m,
            Delta=Delta,
        ),
        atol=3.0e-10,
    )


def gu_emerson_fields_as_planar17(fields: GuEmersonFields) -> np.ndarray:
    """Pack the 17 independent transformed unknowns in physical-state order.

    Slots occupied by ``q,sigma,R,m,Delta`` in the physical state hold
    ``h,g,gamma,omega,chi`` respectively.  This ordering is only a storage
    mapping; it does not identify the non-gradient fields with the physical
    moments.
    """

    return tensors_to_planar_state(
        StateTensors(
            rho=np.asarray(fields.rho),
            velocity=np.asarray(fields.velocity),
            theta=np.asarray(fields.theta),
            heat_flux=np.asarray(fields.h),
            sigma=np.asarray(fields.g),
            R=np.asarray(fields.gamma),
            m=np.asarray(fields.omega),
            Delta=np.asarray(fields.chi),
        )
    )


def gu_emerson_fields_from_planar17(packed: np.ndarray) -> GuEmersonFields:
    """Inverse of :func:`gu_emerson_fields_as_planar17`."""

    tensors = planar_state_to_tensors(packed)
    return GuEmersonFields(
        rho=np.asarray(tensors.rho).copy(),
        velocity=np.asarray(tensors.velocity).copy(),
        theta=np.asarray(tensors.theta).copy(),
        g=np.asarray(tensors.sigma).copy(),
        h=np.asarray(tensors.heat_flux).copy(),
        omega=np.asarray(tensors.m).copy(),
        gamma=np.asarray(tensors.R).copy(),
        chi=np.asarray(tensors.Delta).copy(),
    )


@dataclass(frozen=True)
class GuEmersonLogStateTransform:
    """Newton coordinates for the printed Gu--Emerson primary variables.

    Density and temperature retain the repository's positivity-preserving
    logarithms.  The other fifteen slots store ``u,g,h,omega,gamma,chi`` in
    planar-17 order.  Decoding reconstructs the physical moments through
    equations (48)--(55), so a physical R26 residual can be solved without
    treating those moments as the nonlinear unknowns.

    The physical pseudo-time chain rule is non-diagonal because the printed
    reconstruction contains gradients.  The nonlinear solver therefore uses
    a radius-two colored sparse matrix for that map instead of a false
    diagonal approximation.
    """

    case: CavityCase
    maximum_log_magnitude: float = 50.0
    supports_physical_pseudo_transient: bool = True
    physical_pseudo_transient_stencil_radius: int = 2

    def __post_init__(self) -> None:
        if self.maximum_log_magnitude <= 0.0:
            raise ValueError("maximum log magnitude must be positive")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.case.nodes, self.case.nodes, NVAR)

    def encode(self, state: np.ndarray) -> np.ndarray:
        physical = validate_planar_state(np.asarray(state, dtype=float))
        if physical.shape != self.shape:
            raise ValueError(f"state shape must be {self.shape}")
        fields = gu_emerson_fields_from_state(
            physical,
            x=self.case.x,
            y=self.case.y,
            mu=self.case.mu(physical[..., 3]),
        )
        encoded = gu_emerson_fields_as_planar17(fields)
        encoded[..., 0] = np.log(encoded[..., 0])
        encoded[..., 3] = np.log(encoded[..., 3])
        return encoded.ravel()

    def decode(self, vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=float)
        if value.shape != (int(np.prod(self.shape)),) or not np.isfinite(value).all():
            raise ValueError("encoded transformed state has incorrect shape or non-finite values")
        packed = value.reshape(self.shape).copy()
        logs = packed[..., (0, 3)]
        if np.max(np.abs(logs), initial=0.0) > self.maximum_log_magnitude:
            raise FloatingPointError(
                "rho/T log coordinate exceeded the transformed solver domain"
            )
        packed[..., 0] = np.exp(packed[..., 0])
        packed[..., 3] = np.exp(packed[..., 3])
        fields = gu_emerson_fields_from_planar17(packed)
        return validate_planar_state(
            state_from_gu_emerson_fields(
                fields,
                x=self.case.x,
                y=self.case.y,
                mu=self.case.mu(fields.theta),
            )
        )

    def least_squares_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.shape, -np.inf)
        upper = np.full(self.shape, np.inf)
        lower[..., 0] = lower[..., 3] = -self.maximum_log_magnitude
        upper[..., 0] = upper[..., 3] = self.maximum_log_magnitude
        return lower.ravel(), upper.ravel()


__all__ = [
    "GU_EMERSON_VARIABLE_PROVENANCE",
    "GuEmersonFields",
    "GuEmersonLogStateTransform",
    "gradient_parts_from_primitive_and_moments",
    "gu_emerson_fields_as_planar17",
    "gu_emerson_fields_from_planar17",
    "gu_emerson_fields_from_state",
    "state_from_gu_emerson_fields",
]
