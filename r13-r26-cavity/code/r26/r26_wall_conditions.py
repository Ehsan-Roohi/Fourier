#!/usr/bin/env python3
"""Gu--Emerson planar R26 smooth-wall conditions.

This module implements equations (32)--(34) and Appendix C, (C1)--(C8),
of Gu & Emerson, J. Fluid Mech. 636 (2009).  A wall face owns its own fixed
geometric inward normal and tangent.  Nothing in this module stores temporal
wall history, relaxes a boundary value in pseudo-time, or assigns a shared
corner node to one of two faces.

The physical wall unknowns are

``(u_t,T,sigma_tt,sigma_nn,q_t,m_ttt,m_nnt,R_tt,R_nn,Delta)``.

The six quantities extrapolated from the gas are

``(p,sigma_nt,q_n,m_nnn,m_ntt,R_nt)``.

For the local nonlinear solve, ``T`` and the effective pressure ``p_alpha``
are represented logarithmically.  Equation (34) is then used to recover
``sigma_nn``.  This is an algebraically equivalent parametrization that
guarantees the two positivity conditions required by the wall formulae.

All quantities are dimensional unless the caller supplies one completely
consistent nondimensional system (usually ``gas_constant=1``).  The packed
state follows the bulk-equation convention ``theta=R*T``; the wall solver's
``WallUnknowns.temperature`` is the thermodynamic temperature ``T``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.optimize import least_squares

from r26_state import StateTensors, planar_state_to_tensors, tensors_to_planar_state
from r26_tensor_closures import R26Closures


WALL_EQUATION_ORDER: Final[tuple[str, ...]] = (
    "no_penetration",
    "slip_32",
    "temperature_jump_33",
    "C1_sigma_tt",
    "C2_sigma_nn",
    "C3_q_t",
    "C4_m_ttt",
    "C5_m_nnt",
    "C6_R_tt",
    "C7_R_nn",
    "C8_Delta",
)


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise FloatingPointError(f"{name} must be finite")
    return result


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite three-vector")
    return result


@dataclass(frozen=True)
class WallFrame:
    """A fixed planar wall frame; ``normal`` points from wall into gas."""

    normal: np.ndarray
    tangent: np.ndarray

    def __post_init__(self) -> None:
        normal = _finite_vector(self.normal, "wall normal")
        tangent = _finite_vector(self.tangent, "wall tangent")
        if abs(normal[2]) > 5.0e-14 or abs(tangent[2]) > 5.0e-14:
            raise ValueError("the planar 2D3V wall normal and tangent must have zero z component")
        if not np.isclose(np.linalg.norm(normal), 1.0, rtol=0.0, atol=5.0e-13):
            raise ValueError("wall normal must already be a unit vector")
        if not np.isclose(np.linalg.norm(tangent), 1.0, rtol=0.0, atol=5.0e-13):
            raise ValueError("wall tangent must already be a unit vector")
        if not np.isclose(np.dot(normal, tangent), 0.0, rtol=0.0, atol=5.0e-13):
            raise ValueError("wall normal and tangent must be orthogonal")
        object.__setattr__(self, "normal", normal.copy())
        object.__setattr__(self, "tangent", tangent.copy())

    @property
    def spanwise(self) -> np.ndarray:
        """Return the unit z direction, independent of local handedness."""

        return np.asarray((0.0, 0.0, 1.0))


def square_wall_frame(side: str) -> WallFrame:
    """Return the declared fixed geometric frame for one square-cavity side."""

    frames = {
        "left": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "right": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "bottom": ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        "top": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    }
    try:
        normal, tangent = frames[side.lower()]
    except KeyError as exc:
        raise ValueError("side must be left, right, bottom, or top") from exc
    return WallFrame(np.asarray(normal), np.asarray(tangent))


@dataclass(frozen=True)
class WallFreeQuantities:
    """The six face quantities supplied by one-sided gas extrapolation."""

    pressure: float
    sigma_nt: float
    q_n: float
    m_nnn: float
    m_ntt: float
    R_nt: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        if self.pressure <= 0.0:
            raise FloatingPointError("extrapolated thermodynamic pressure must be positive")

    def as_array(self) -> np.ndarray:
        return np.asarray(tuple(getattr(self, name) for name in self.__dataclass_fields__))

    @classmethod
    def from_array(cls, value: np.ndarray) -> "WallFreeQuantities":
        array = np.asarray(value, dtype=float)
        if array.shape != (6,):
            raise ValueError("free wall data must contain six values")
        return cls(*array)


@dataclass(frozen=True)
class WallUnknowns:
    """The ten coupled physical wall unknowns."""

    u_t: float
    temperature: float
    sigma_tt: float
    sigma_nn: float
    q_t: float
    m_ttt: float
    m_nnt: float
    R_tt: float
    R_nn: float
    Delta: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        if self.temperature <= 0.0:
            raise FloatingPointError("wall gas temperature must be positive")


@dataclass(frozen=True)
class ProjectedClosures:
    """Only the higher-closure components appearing in the wall equations."""

    phi_nnnn: float
    phi_nntt: float
    phi_nttt: float
    phi_nnnt: float
    phi_tttt: float
    psi_nnt: float
    psi_ttt: float
    psi_ntt: float
    psi_nnn: float
    Omega_n: float
    Omega_t: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))

    @classmethod
    def zeros(cls) -> "ProjectedClosures":
        return cls(*(0.0 for _ in range(11)))


@dataclass(frozen=True)
class WallParameters:
    """Material and wall data for one smooth face."""

    wall_temperature: float
    accommodation: float = 1.0
    gas_constant: float = 1.0
    wall_velocity: np.ndarray | None = None
    positivity_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        wall_temperature = _finite_scalar(self.wall_temperature, "wall temperature")
        accommodation = _finite_scalar(self.accommodation, "accommodation")
        gas_constant = _finite_scalar(self.gas_constant, "gas constant")
        floor = _finite_scalar(self.positivity_floor, "positivity floor")
        if wall_temperature <= 0.0 or gas_constant <= 0.0 or floor <= 0.0:
            raise ValueError("wall temperature, gas constant, and positivity floor must be positive")
        if accommodation <= 0.0 or accommodation > 1.0:
            raise ValueError("accommodation must lie in (0,1]")
        velocity = np.zeros(3) if self.wall_velocity is None else _finite_vector(
            self.wall_velocity, "wall velocity"
        )
        if abs(velocity[2]) > 5.0e-14:
            raise ValueError("planar wall velocity must have zero z component")
        object.__setattr__(self, "wall_temperature", wall_temperature)
        object.__setattr__(self, "accommodation", accommodation)
        object.__setattr__(self, "gas_constant", gas_constant)
        object.__setattr__(self, "wall_velocity", velocity.copy())
        object.__setattr__(self, "positivity_floor", floor)

    @property
    def A(self) -> float:
        return (2.0 - self.accommodation) / self.accommodation


@dataclass(frozen=True)
class WallSolveResult:
    """Diagnostics and reconstructed state from a local ten-equation solve."""

    success: bool
    unknowns: WallUnknowns
    effective_pressure: float
    state: StateTensors
    planar_state: np.ndarray
    residual: np.ndarray
    scaled_residual: np.ndarray
    nfev: int
    cost: float
    message: str


def _contract(vector_list: tuple[np.ndarray, ...], tensor: np.ndarray) -> float:
    result = np.asarray(tensor, dtype=float)
    if result.shape != (3,) * len(vector_list) or not np.isfinite(result).all():
        raise ValueError(f"rank-{len(vector_list)} closure tensor has invalid shape or values")
    for vector in vector_list:
        result = np.tensordot(_finite_vector(vector, "projection vector"), result, axes=(0, 0))
    return float(result)


def project_closures(closures: R26Closures, frame: WallFrame) -> ProjectedClosures:
    """Project full Cartesian ``phi,psi,Omega`` onto a fixed wall frame."""

    phi = np.asarray(closures.phi, dtype=float)
    psi = np.asarray(closures.psi, dtype=float)
    omega = np.asarray(closures.Omega, dtype=float)
    if phi.shape != (3, 3, 3, 3) or psi.shape != (3, 3, 3) or omega.shape != (3,):
        raise ValueError("project_closures expects one point, without leading grid dimensions")
    n, t = frame.normal, frame.tangent
    return ProjectedClosures(
        phi_nnnn=_contract((n, n, n, n), phi),
        phi_nntt=_contract((n, n, t, t), phi),
        phi_nttt=_contract((n, t, t, t), phi),
        phi_nnnt=_contract((n, n, n, t), phi),
        phi_tttt=_contract((t, t, t, t), phi),
        psi_nnt=_contract((n, n, t), psi),
        psi_ttt=_contract((t, t, t), psi),
        psi_ntt=_contract((n, t, t), psi),
        psi_nnn=_contract((n, n, n), psi),
        Omega_n=float(np.dot(omega, n)),
        Omega_t=float(np.dot(omega, t)),
    )


def effective_pressure(
    free: WallFreeQuantities,
    unknowns: WallUnknowns,
    closure: ProjectedClosures,
    parameters: WallParameters,
) -> float:
    """Evaluate Gu--Emerson equation (34), ``p_alpha``."""

    thermal = parameters.gas_constant * unknowns.temperature
    return float(
        free.pressure
        + 0.5 * unknowns.sigma_nn
        - (30.0 * unknowns.R_nn + 7.0 * unknowns.Delta) / (840.0 * thermal)
        - closure.phi_nnnn / (24.0 * thermal)
    )


def coupled_wall_residual(
    unknowns: WallUnknowns,
    free: WallFreeQuantities,
    closure: ProjectedClosures,
    parameters: WallParameters,
) -> np.ndarray:
    """Return residuals for (32), (33), and (C1)--(C8), in that order."""

    T = unknowns.temperature
    thermal = parameters.gas_constant * T
    thermal_wall = parameters.gas_constant * parameters.wall_temperature
    if thermal <= parameters.positivity_floor:
        raise FloatingPointError("wall thermal energy is non-positive")
    palpha = effective_pressure(free, unknowns, closure, parameters)
    if not np.isfinite(palpha) or palpha <= parameters.positivity_floor:
        raise FloatingPointError("effective wall pressure p_alpha is non-positive")

    sqrt_thermal = np.sqrt(thermal)
    half_range = np.sqrt(np.pi * thermal / 2.0)
    uhat = unknowns.u_t / sqrt_thermal
    That = parameters.wall_temperature / T
    A = parameters.A

    slip_rhs = (
        -A * half_range * free.sigma_nt / palpha
        - (5.0 * unknowns.m_nnt + 2.0 * unknowns.q_t) / (10.0 * palpha)
        + (9.0 * closure.Omega_t + 70.0 * closure.psi_nnt)
        / (2520.0 * palpha * thermal)
    )
    temperature_rhs = (
        -A * half_range * free.q_n / (2.0 * palpha)
        - thermal * unknowns.sigma_nn / (4.0 * palpha)
        + unknowns.u_t**2 / 4.0
        - (75.0 * unknowns.R_nn + 28.0 * unknowns.Delta) / (840.0 * palpha)
        + closure.phi_nnnn / (24.0 * palpha)
    )
    c1_rhs = (
        -A * half_range * (5.0 * free.m_ntt + 2.0 * free.q_n) / (5.0 * thermal)
        + palpha * (uhat**2 + That - 1.0)
        - (unknowns.R_tt + unknowns.R_nn) / (14.0 * thermal)
        - unknowns.Delta / (30.0 * thermal)
        - closure.phi_nntt / (2.0 * thermal)
    )
    c2_rhs = (
        -A * half_range * (5.0 * free.m_nnn + 6.0 * free.q_n) / (10.0 * thermal)
        + palpha * (That - 1.0)
        - unknowns.R_nn / (7.0 * thermal)
        - unknowns.Delta / (30.0 * thermal)
        - closure.phi_nnnn / (6.0 * thermal)
    )
    c3_rhs = (
        -(5.0 / 18.0) * A * half_range * (7.0 * free.sigma_nt + free.R_nt / thermal)
        - (5.0 / 18.0)
        * uhat
        * palpha
        * sqrt_thermal
        * (uhat**2 + 6.0 * That)
        - 10.0 * unknowns.m_nnt / 9.0
        - 5.0 * closure.psi_nnt / (81.0 * thermal)
        - closure.Omega_t / (56.0 * thermal)
    )
    c4_rhs = (
        -A
        * half_range
        * (3.0 * free.sigma_nt + 3.0 * free.R_nt / (7.0 * thermal) + closure.phi_nttt / thermal)
        - palpha * uhat * sqrt_thermal * (uhat**2 + 3.0 * That)
        - 1.5 * unknowns.m_nnt
        - 9.0 * unknowns.q_t / 5.0
        - 9.0 * closure.Omega_t / (280.0 * thermal)
        - (2.0 * closure.psi_ttt + 3.0 * closure.psi_nnt) / (36.0 * thermal)
    )
    c5_rhs = (
        -A
        * half_range
        * (free.sigma_nt + free.R_nt / (7.0 * thermal) + closure.phi_nnnt / (3.0 * thermal))
        - 2.0 * unknowns.q_t / 5.0
        - 2.0 * That * uhat * palpha * sqrt_thermal / 3.0
        - closure.psi_nnt / (18.0 * thermal)
        - closure.Omega_t / (140.0 * thermal)
    )
    c6_rhs = (
        -A
        * half_range
        * (
            28.0 * free.q_n / 15.0
            + 14.0 * free.m_ntt / 3.0
            + closure.Omega_n / (15.0 * thermal)
            + 14.0 * closure.psi_ntt / (27.0 * thermal)
        )
        + 7.0
        * palpha
        * thermal
        * (uhat**4 + 6.0 * That * uhat**2 + 3.0 * That**2 - 3.0)
        / 9.0
        - 14.0 * thermal * unknowns.sigma_tt / 3.0
        - unknowns.R_nn / 3.0
        - 14.0 * unknowns.Delta / 45.0
        - 7.0 * (closure.phi_tttt + 3.0 * closure.phi_nntt) / 9.0
    )
    c7_rhs = (
        -A
        * half_range
        * (
            21.0 * free.q_n / 8.0
            + 35.0 * free.m_nnn / 16.0
            + 35.0 * closure.psi_nnn / (144.0 * thermal)
            + 3.0 * closure.Omega_n / (32.0 * thermal)
        )
        + 7.0 * palpha * thermal * (That**2 - 1.0) / 4.0
        - 7.0 * thermal * unknowns.sigma_nn / 2.0
        - 7.0 * unknowns.Delta / 30.0
        - 7.0 * closure.phi_nnnn / 6.0
    )
    c8_rhs = (
        -(35.0 / 4.0)
        * A
        * half_range
        * (free.q_n + closure.Omega_n / (28.0 * thermal))
        - (5.0 / 4.0)
        * palpha
        * thermal
        * (6.0 - 6.0 * That**2 - uhat**4 / 4.0 - 3.0 * uhat**2 * That)
        - 15.0 * thermal * unknowns.sigma_nn / 4.0
        - 15.0 * unknowns.R_nn / 8.0
        + 35.0 * closure.phi_nnnn / 48.0
    )

    residual = np.asarray(
        (
            unknowns.u_t - slip_rhs,
            thermal - thermal_wall - temperature_rhs,
            unknowns.sigma_tt - c1_rhs,
            unknowns.sigma_nn - c2_rhs,
            unknowns.q_t - c3_rhs,
            unknowns.m_ttt - c4_rhs,
            unknowns.m_nnt - c5_rhs,
            unknowns.R_tt - c6_rhs,
            unknowns.R_nn - c7_rhs,
            unknowns.Delta - c8_rhs,
        )
    )
    if not np.isfinite(residual).all():
        raise FloatingPointError("R26 wall residual produced NaN or infinity")
    return residual


def wall_residual_scales(
    unknowns: WallUnknowns, free: WallFreeQuantities, parameters: WallParameters
) -> np.ndarray:
    """Return dimensional scales for the ten coupled residual families."""

    thermal = parameters.gas_constant * unknowns.temperature
    speed = np.sqrt(thermal)
    pressure = max(abs(free.pressure), parameters.positivity_floor)
    return np.asarray(
        (
            speed,
            thermal,
            pressure,
            pressure,
            pressure * speed,
            pressure * speed,
            pressure * speed,
            pressure * thermal,
            pressure * thermal,
            pressure * thermal,
        )
    )


def extract_face_quantities(
    tensors: StateTensors,
    frame: WallFrame,
    *,
    gas_constant: float = 1.0,
) -> tuple[WallFreeQuantities, WallUnknowns]:
    """Project a single full state into the six free and ten solved variables."""

    rho = _finite_scalar(np.asarray(tensors.rho), "rho")
    thermal = _finite_scalar(np.asarray(tensors.theta), "theta=R*T")
    gas_constant = _finite_scalar(gas_constant, "gas constant")
    if rho <= 0.0 or thermal <= 0.0 or gas_constant <= 0.0:
        raise FloatingPointError("rho, theta=R*T, and gas constant must be positive")
    temperature = thermal / gas_constant
    velocity = np.asarray(tensors.velocity, dtype=float)
    q = np.asarray(tensors.heat_flux, dtype=float)
    sigma = np.asarray(tensors.sigma, dtype=float)
    rr = np.asarray(tensors.R, dtype=float)
    mm = np.asarray(tensors.m, dtype=float)
    if velocity.shape != (3,) or q.shape != (3,) or sigma.shape != (3, 3):
        raise ValueError("extract_face_quantities expects one state point")
    n, t = frame.normal, frame.tangent
    free = WallFreeQuantities(
        pressure=rho * thermal,
        sigma_nt=_contract((n, t), sigma),
        q_n=float(np.dot(q, n)),
        m_nnn=_contract((n, n, n), mm),
        m_ntt=_contract((n, t, t), mm),
        R_nt=_contract((n, t), rr),
    )
    unknowns = WallUnknowns(
        u_t=float(np.dot(velocity, t)),
        temperature=temperature,
        sigma_tt=_contract((t, t), sigma),
        sigma_nn=_contract((n, n), sigma),
        q_t=float(np.dot(q, t)),
        m_ttt=_contract((t, t, t), mm),
        m_nnt=_contract((n, n, t), mm),
        R_tt=_contract((t, t), rr),
        R_nn=_contract((n, n), rr),
        Delta=_finite_scalar(np.asarray(tensors.Delta), "Delta"),
    )
    return free, unknowns


def _symmetric_rank3_component(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (
        np.einsum("i,j,k->ijk", a, b, c)
        + np.einsum("i,j,k->ijk", a, c, b)
        + np.einsum("i,j,k->ijk", b, a, c)
        + np.einsum("i,j,k->ijk", b, c, a)
        + np.einsum("i,j,k->ijk", c, a, b)
        + np.einsum("i,j,k->ijk", c, b, a)
    ) / 6.0


def reconstruct_wall_tensors(
    free: WallFreeQuantities,
    unknowns: WallUnknowns,
    frame: WallFrame,
    parameters: WallParameters,
) -> StateTensors:
    """Reconstruct one planar 17-moment wall state from face-frame values."""

    n, t, z = frame.normal, frame.tangent, frame.spanwise
    thermal = parameters.gas_constant * unknowns.temperature
    rho = free.pressure / thermal
    if rho <= parameters.positivity_floor:
        raise FloatingPointError("reconstructed wall density is non-positive")
    velocity = parameters.wall_velocity + unknowns.u_t * t
    q = free.q_n * n + unknowns.q_t * t

    sigma_zz = -unknowns.sigma_nn - unknowns.sigma_tt
    sigma = (
        unknowns.sigma_nn * np.outer(n, n)
        + unknowns.sigma_tt * np.outer(t, t)
        + free.sigma_nt * (np.outer(n, t) + np.outer(t, n))
        + sigma_zz * np.outer(z, z)
    )
    R_zz = -unknowns.R_nn - unknowns.R_tt
    rr = (
        unknowns.R_nn * np.outer(n, n)
        + unknowns.R_tt * np.outer(t, t)
        + free.R_nt * (np.outer(n, t) + np.outer(t, n))
        + R_zz * np.outer(z, z)
    )

    m_nzz = -free.m_nnn - free.m_ntt
    m_tzz = -unknowns.m_nnt - unknowns.m_ttt
    mm = (
        free.m_nnn * np.einsum("i,j,k->ijk", n, n, n)
        + 3.0 * unknowns.m_nnt * _symmetric_rank3_component(n, n, t)
        + 3.0 * free.m_ntt * _symmetric_rank3_component(n, t, t)
        + unknowns.m_ttt * np.einsum("i,j,k->ijk", t, t, t)
        + 3.0 * m_nzz * _symmetric_rank3_component(n, z, z)
        + 3.0 * m_tzz * _symmetric_rank3_component(t, z, z)
    )
    return StateTensors(
        rho=np.asarray(rho),
        velocity=velocity,
        theta=np.asarray(thermal),
        heat_flux=q,
        sigma=sigma,
        R=rr,
        m=mm,
        Delta=np.asarray(unknowns.Delta),
    )


def smooth_wall_residual_from_tensors(
    tensors: StateTensors,
    closures: R26Closures,
    frame: WallFrame,
    parameters: WallParameters,
) -> np.ndarray:
    """Return all 11 smooth-wall residuals, including no penetration."""

    free, unknowns_absolute = extract_face_quantities(
        tensors, frame, gas_constant=parameters.gas_constant
    )
    relative_velocity = np.asarray(tensors.velocity) - parameters.wall_velocity
    no_penetration = float(np.dot(relative_velocity, frame.normal))
    relative_unknowns = WallUnknowns(
        u_t=float(np.dot(relative_velocity, frame.tangent)),
        temperature=unknowns_absolute.temperature,
        sigma_tt=unknowns_absolute.sigma_tt,
        sigma_nn=unknowns_absolute.sigma_nn,
        q_t=unknowns_absolute.q_t,
        m_ttt=unknowns_absolute.m_ttt,
        m_nnt=unknowns_absolute.m_nnt,
        R_tt=unknowns_absolute.R_tt,
        R_nn=unknowns_absolute.R_nn,
        Delta=unknowns_absolute.Delta,
    )
    projected = project_closures(closures, frame)
    return np.concatenate(
        (np.asarray((no_penetration,)), coupled_wall_residual(relative_unknowns, free, projected, parameters))
    )


def wall_residual(
    state_at_wall: np.ndarray,
    closures_at_wall: R26Closures,
    normal: np.ndarray,
    tangent: np.ndarray,
    wall_velocity: np.ndarray,
    wall_temperature: float,
    *,
    alpha: float = 1.0,
    gas_constant: float = 1.0,
) -> np.ndarray:
    """Solver-friendly wrapper returning the ordered 11-vector residual."""

    frame = WallFrame(normal, tangent)
    parameters = WallParameters(
        wall_temperature=wall_temperature,
        accommodation=alpha,
        gas_constant=gas_constant,
        wall_velocity=wall_velocity,
    )
    return smooth_wall_residual_from_tensors(
        planar_state_to_tensors(np.asarray(state_at_wall, dtype=float)),
        closures_at_wall,
        frame,
        parameters,
    )


def free_extrapolation_values(
    state: np.ndarray,
    normal: np.ndarray,
    tangent: np.ndarray,
    *,
    gas_constant: float = 1.0,
) -> np.ndarray:
    """Return ``(p,sigma_nt,q_n,m_nnn,m_ntt,R_nt)`` at one state point."""

    free, _ = extract_face_quantities(
        planar_state_to_tensors(np.asarray(state, dtype=float)),
        WallFrame(normal, tangent),
        gas_constant=gas_constant,
    )
    return free.as_array()


def extrapolate_face_free_quantities(
    near_state: np.ndarray,
    next_state: np.ndarray,
    frame: WallFrame,
    *,
    gas_constant: float = 1.0,
    wall_to_near_over_spacing: float = 0.5,
) -> WallFreeQuantities:
    """Linearly extrapolate the six free quantities to one cell-centred face.

    For a uniform cell-centred mesh, the wall is half a cell outward from the
    first centre and the default is ``1.5*near - 0.5*next``.  The function
    returns data for one face only; a corner cell therefore receives two
    independent face reconstructions rather than one shared corner state.
    """

    ratio = _finite_scalar(wall_to_near_over_spacing, "wall/centre distance ratio")
    if ratio < 0.0:
        raise ValueError("wall/centre distance ratio must be non-negative")
    near_free, _ = extract_face_quantities(
        planar_state_to_tensors(np.asarray(near_state, dtype=float)), frame, gas_constant=gas_constant
    )
    next_free, _ = extract_face_quantities(
        planar_state_to_tensors(np.asarray(next_state, dtype=float)), frame, gas_constant=gas_constant
    )
    values = (1.0 + ratio) * near_free.as_array() - ratio * next_free.as_array()
    return WallFreeQuantities.from_array(values)


def _decode_solver_variables(
    vector: np.ndarray,
    free: WallFreeQuantities,
    closure: ProjectedClosures,
    parameters: WallParameters,
) -> tuple[WallUnknowns, float]:
    x = np.asarray(vector, dtype=float)
    if x.shape != (10,) or not np.isfinite(x).all():
        raise ValueError("local wall solver vector must contain ten finite values")
    temperature = float(np.exp(x[1]))
    palpha = float(np.exp(x[3]))
    thermal = parameters.gas_constant * temperature
    sigma_nn = 2.0 * (
        palpha
        - free.pressure
        + (30.0 * x[8] + 7.0 * x[9]) / (840.0 * thermal)
        + closure.phi_nnnn / (24.0 * thermal)
    )
    return (
        WallUnknowns(
            u_t=x[0],
            temperature=temperature,
            sigma_tt=x[2],
            sigma_nn=sigma_nn,
            q_t=x[4],
            m_ttt=x[5],
            m_nnt=x[6],
            R_tt=x[7],
            R_nn=x[8],
            Delta=x[9],
        ),
        palpha,
    )


def _encode_solver_variables(
    unknowns: WallUnknowns,
    free: WallFreeQuantities,
    closure: ProjectedClosures,
    parameters: WallParameters,
) -> np.ndarray:
    palpha = effective_pressure(free, unknowns, closure, parameters)
    if palpha <= parameters.positivity_floor:
        raise FloatingPointError("initial wall state has non-positive p_alpha")
    return np.asarray(
        (
            unknowns.u_t,
            np.log(unknowns.temperature),
            unknowns.sigma_tt,
            np.log(palpha),
            unknowns.q_t,
            unknowns.m_ttt,
            unknowns.m_nnt,
            unknowns.R_tt,
            unknowns.R_nn,
            unknowns.Delta,
        )
    )


def solve_wall_face(
    free: WallFreeQuantities,
    closure: ProjectedClosures,
    frame: WallFrame,
    parameters: WallParameters,
    *,
    initial: WallUnknowns | None = None,
    max_nfev: int = 600,
    tolerance: float = 1.0e-11,
) -> WallSolveResult:
    """Solve the ten coupled smooth-face wall equations without time history."""

    if initial is None:
        initial = WallUnknowns(
            u_t=0.0,
            temperature=parameters.wall_temperature,
            sigma_tt=0.0,
            sigma_nn=0.0,
            q_t=0.0,
            m_ttt=0.0,
            m_nnt=0.0,
            R_tt=0.0,
            R_nn=0.0,
            Delta=0.0,
        )
        if effective_pressure(free, initial, closure, parameters) <= parameters.positivity_floor:
            # Only initialize p_alpha, not a physical wall value.  The local
            # solve remains governed by the exact ten wall equations.
            thermal = parameters.gas_constant * initial.temperature
            sigma_nn = 2.0 * (
                max(free.pressure, 100.0 * parameters.positivity_floor)
                - free.pressure
                + closure.phi_nnnn / (24.0 * thermal)
            )
            initial = WallUnknowns(**{**initial.__dict__, "sigma_nn": sigma_nn})
    x0 = _encode_solver_variables(initial, free, closure, parameters)

    def objective(vector: np.ndarray) -> np.ndarray:
        unknowns, _ = _decode_solver_variables(vector, free, closure, parameters)
        return coupled_wall_residual(unknowns, free, closure, parameters) / wall_residual_scales(
            unknowns, free, parameters
        )

    result = least_squares(
        objective,
        x0,
        method="trf",
        jac="3-point",
        x_scale="jac",
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
        max_nfev=int(max_nfev),
    )
    unknowns, palpha = _decode_solver_variables(result.x, free, closure, parameters)
    raw = coupled_wall_residual(unknowns, free, closure, parameters)
    scaled = raw / wall_residual_scales(unknowns, free, parameters)
    tensors = reconstruct_wall_tensors(free, unknowns, frame, parameters)
    planar = tensors_to_planar_state(tensors)
    converged = bool(result.success and np.max(np.abs(scaled)) <= max(100.0 * tolerance, 1.0e-9))
    return WallSolveResult(
        success=converged,
        unknowns=unknowns,
        effective_pressure=palpha,
        state=tensors,
        planar_state=planar,
        residual=raw,
        scaled_residual=scaled,
        nfev=int(result.nfev),
        cost=float(result.cost),
        message=str(result.message),
    )


__all__ = [
    "ProjectedClosures",
    "WALL_EQUATION_ORDER",
    "WallFrame",
    "WallFreeQuantities",
    "WallParameters",
    "WallSolveResult",
    "WallUnknowns",
    "coupled_wall_residual",
    "effective_pressure",
    "extract_face_quantities",
    "extrapolate_face_free_quantities",
    "free_extrapolation_values",
    "project_closures",
    "reconstruct_wall_tensors",
    "smooth_wall_residual_from_tensors",
    "solve_wall_face",
    "square_wall_frame",
    "wall_residual",
    "wall_residual_scales",
]
