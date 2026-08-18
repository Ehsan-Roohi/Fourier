#!/usr/bin/env python3
"""17-state coefficient matrices transcribed from Rana's supplied old code.

The arrays use the state order and quasilinear structure associated with
Appendix A of Rana, Torrilhon & Struchtrup, JCP 236 (2013) 169--186.  They are
an exact clean-room transcription of the supplied implementation, not a claim
that every old-code term equals the final printed paper.  The private MATLAB
source used for the independent cross-check is intentionally not distributed.
"""

from __future__ import annotations

import numpy as np


NVAR = 17


def _state(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    if u.shape != (NVAR,):
        raise ValueError(f"state must have shape {(NVAR,)}, got {u.shape}")
    if not np.isfinite(u).all() or u[0] <= 0.0 or u[3] <= 0.0:
        raise FloatingPointError("R13 state must be finite with positive rho and theta")
    return u


def flux_x(u: np.ndarray) -> np.ndarray:
    u = _state(u)
    return np.asarray([
        [u[1], u[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [u[3], u[0]*u[1], 0, u[0], 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, u[0]*u[1], 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, u[6] + u[0]*u[3], u[7], (3*u[0]*u[1])/2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-((u[6]*u[3])/u[0]), u[12] + (16*u[4])/5, u[13] + (2*u[5])/5, (5*u[6])/2 + (5*u[0]*u[3])/2, u[1], 0, -(u[6]/u[0]) + u[3], -(u[7]/u[0]), 0, 1/2, 0, 0, 0, 0, 0, 0, 1/6],
        [-((u[7]*u[3])/u[0]), u[13] + (7*u[5])/5, u[14] + (7*u[4])/5, (5*u[7])/2, 0, u[1], -(u[7]/u[0]), -(u[8]/u[0]) + u[3], 0, 0, 1/2, 0, 0, 0, 0, 0, 0],
        [0, (7*u[6])/3 + (4*u[0]*u[3])/3, -((2*u[7])/3), 0, 8/15, 0, u[1], 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 2*u[7], u[6] + u[0]*u[3], 0, 0, 2/5, 0, u[1], 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, -((2*u[6])/3) + u[8] - (2*u[0]*u[3])/3, (4*u[7])/3, 0, -(4/15), 0, 0, 0, u[1], 0, 0, 0, 0, 0, 1, 0, 0],
        [-(2*u[4]*u[3])/(3*u[0]), 0, 0, -(2*u[4])/3, (2*u[3])/3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-u[3]*u[5]/(2*u[0]), 0, 0, -u[5]/2, 0, u[3]/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [u[3]*u[4]/(3*u[0]), 0, 0, u[4]/3, -u[3]/3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-(3*u[3]*u[6])/(5*u[0]), 0, 0, -(3*u[6])/5, 0, 0, (3*u[3])/5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-(8*u[3]*u[7])/(15*u[0]), 0, 0, -(8*u[7])/15, 0, 0, 0, (8*u[3])/15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [(((2*u[6]*u[3])/15)-u[8]*u[3]/3)/u[0], 0, 0, (((2*u[6])/15)-u[8]/3), 0, 0, -((2*u[3])/15), 0, u[3]/3, 0, 0, 0, 0, 0, 0, 0, 0],
        [((2*u[3]*u[7])/5)/(u[0]), 0, 0, ((2*u[7])/5), 0, 0, 0, -((2*u[3])/5), 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-8*u[4]*u[3]/u[0], 0, 0, -8*u[4], 8*u[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)


def flux_y(u: np.ndarray) -> np.ndarray:
    u = _state(u)
    return np.asarray([
        [u[2], 0, u[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, u[0]*u[2], 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [u[3], 0, u[0]*u[2], u[0], 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, u[7], u[8] + u[0]*u[3], (3*u[0]*u[2])/2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-((u[7]*u[3])/u[0]), u[13] + (7*u[5])/5, u[14] + (7*u[4])/5, (5*u[7])/2, u[2], 0, 0, -(u[6]/u[0]) + u[3], -(u[7]/u[0]), 0, 1/2, 0, 0, 0, 0, 0, 0],
        [-((u[8]*u[3])/u[0]), u[14] + (2*u[4])/5, u[15] + (16*u[5])/5, (5*u[8])/2 + (5*u[0]*u[3])/2, 0, u[2], 0, -(u[7]/u[0]), -(u[8]/u[0]) + u[3], 0, 0, 1/2, 0, 0, 0, 0, 1/6],
        [0, (4*u[7])/ 3, u[6] - (2*u[8])/3 - (2*u[0]*u[3])/3, 0, 0, -(4/15), u[2], 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, u[8] + u[0]*u[3], 2*u[7], 0, 2/5, 0, 0, u[2], 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, -((2*u[7])/3), (7*u[8])/3 + (4*u[0]*u[3])/3, 0, 0, 8/15, 0, 0, u[2], 0, 0, 0, 0, 0, 0, 1, 0],
        [u[3]*u[5]/(3*u[0]), 0, 0, u[5]/(3), 0, -u[3]/3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-u[3]*u[4]/(2*u[0]), 0, 0, -u[4]/(2), u[3]/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-(2*u[5]*u[3])/(3*u[0]), 0, 0, -(2*u[5])/(3), 0, (2*u[3])/3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [((2*u[3]*u[7])/5)/(u[0]), 0, 0, ((2*u[7])/5), 0, 0, 0, -((2*u[3])/5), 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [(((2*u[8]*u[3])/15)-u[6]*u[3]/3)/u[0], 0, 0, (((2*u[8])/15)-u[6]/3), 0, 0, u[3]/3, 0, -((2*u[3])/15), 0, 0, 0, 0, 0, 0, 0, 0],
        [-(8*u[3]*u[7])/(15*u[0]), 0, 0, -(8*u[7])/(15), 0, 0, 0, (8*u[3])/15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-(3*u[3]*u[8])/(5*u[0]), 0, 0, -(3*u[8])/(5), 0, 0, 0, 0, (3*u[3])/5, 0, 0, 0, 0, 0, 0, 0, 0],
        [-8*u[5]*u[3]/u[0], 0, 0, -8*u[5], 0, 8*u[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)


def production(u: np.ndarray, kn: float, *, rb: float = 1.0, ra: float = 1.0, ma: float = 1.0) -> np.ndarray:
    u = _state(u)
    if kn <= 0.0:
        raise ValueError("kn must be positive")
    matrix = np.asarray([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, (2*1)/3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, (2*1)/3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -((16*u[4]*rb*1)/(45*u[0]*u[3])), (8*u[5]*rb*1)/(45*u[0]*u[3]), -((5*(-(4/7)+(24*ra)/7)*u[6]*1)/(72*u[0]))+(5*(-(4/7)+(24*ra)/7)*u[8]*1)/(72*u[0]), -((5*(-(4/7)+(24*ra)/7)*u[7]*1)/(72*u[0])), (5*(-(4/7)+(24*ra)/7)*u[6]*1)/(72*u[0])+(5*(-(4/7)+(24*ra)/7)*u[8]*1)/(36*u[0]), (5*1)/24, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -((5*(-(4/7)+(24*ra)/7)*u[7]*1)/(48*u[0])), -((5*(-(4/7)+(24*ra)/7)*u[6]*1)/(48*u[0]))-(5*(-(4/7)+(24*ra)/7)*u[8]*1)/(48*u[0]), -((5*(-(4/7)+(24*ra)/7)*u[7]*1)/(48*u[0])), 0, (5*1)/24, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, (8*u[4]*rb*1)/(45*u[0]*u[3]), -((16*u[5]*rb*1)/(45*u[0]*u[3])), (5*(-(4/7)+(24*ra)/7)*u[6]*1)/(36*u[0])+(5*(-(4/7)+(24*ra)/7)*u[8]*1)/(72*u[0]), -((5*(-(4/7)+(24*ra)/7)*u[7]*1)/(72*u[0])), (5*(-(4/7)+(24*ra)/7)*u[6]*1)/(72*u[0])-(5*(-(4/7)+(24*ra)/7)*u[8]*1)/(72*u[0]), 0, 0, (5*1)/24, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -((3*(4/5+(8*ma)/15)*u[6]*1)/(20*u[0]*u[3])), ((4/5+(8*ma)/15)*u[7]*1)/(10*u[0]*u[3]), -((3*(4/5+(8*ma)/15)*u[4]*1)/(20*u[0]*u[3])), ((4/5+(8*ma)/15)*u[5]*1)/(10*u[0]*u[3]), 0, 0, 0, 0, 1/2, 0, 0, 0, 0],
        [0, 0, 0, 0, -((2*(4/5+(8*ma)/15)*u[7]*1)/(15*u[0]*u[3])), -(((4/5+(8*ma)/15)*u[6]*1)/(12*u[0]*u[3]))+((4/5+(8*ma)/15)*u[8]*1)/(30*u[0]*u[3]), -(((4/5+(8*ma)/15)*u[5]*1)/(12*u[0]*u[3])), -((2*(4/5+(8*ma)/15)*u[4]*1)/(15*u[0]*u[3])), ((4/5+(8*ma)/15)*u[5]*1)/(30*u[0]*u[3]), 0, 0, 0, 0, 1/2, 0, 0, 0],
        [0, 0, 0, 0, ((4/5+(8*ma)/15)*u[6]*1)/(30*u[0]*u[3])-((4/5+(8*ma)/15)*u[8]*1)/(12*u[0]*u[3]), -((2*(4/5+(8*ma)/15)*u[7]*1)/(15*u[0]*u[3])), ((4/5+(8*ma)/15)*u[4]*1)/(30*u[0]*u[3]), -((2*(4/5+(8*ma)/15)*u[5]*1)/(15*u[0]*u[3])), -(((4/5+(8*ma)/15)*u[4]*1)/(12*u[0]*u[3])), 0, 0, 0, 0, 0, 1/2, 0, 0],
        [0, 0, 0, 0, ((4/5+(8*ma)/15)*u[7]*1)/(10*u[0]*u[3]), -((3*(4/5+(8*ma)/15)*u[8]*1)/(20*u[0]*u[3])), 0, ((4/5+(8*ma)/15)*u[4]*1)/(10*u[0]*u[3]), -((3*(4/5+(8*ma)/15)*u[5]*1)/(20*u[0]*u[3])), 0, 0, 0, 0, 0, 0, 1/2, 0],
        [0, 0, 0, 0, -112*u[4]/(15*u[0]*u[3]), -112*u[5]/(15*u[0]*u[3]), -20*u[6]/(3*u[0])-10*u[8]/(3*u[0]), -20*u[7]/(3*u[0]), -20*u[8]/(3*u[0])-10*u[6]/(3*u[0]), 0, 0, 0, 0, 0, 0, 0, 2/3],
    ], dtype=float)
    return u[0] * u[3] / (kn * np.sqrt(u[3])) * matrix


def _effective_pressure(u: np.ndarray, axis: str, mode: str) -> float:
    if axis == "x":
        normal_stress, tangential_stress = u[6], u[8]
        normal_r, tangential_r = u[9], u[11]
    elif axis == "y":
        normal_stress, tangential_stress = u[8], u[6]
        normal_r, tangential_r = u[11], u[9]
    else:
        raise ValueError("axis must be x or y")
    if mode == "legacy-normal":
        stress, regularized = normal_stress, normal_r
    elif mode == "paper-tangential":
        stress, regularized = tangential_stress, tangential_r
    else:
        raise ValueError("pressure mode must be legacy-normal or paper-tangential")
    return float(
        u[0] * u[3] + stress / 2.0
        - regularized / (28.0 * u[3]) - u[16] / (120.0 * u[3])
    )


def wall_matrix(u: np.ndarray, *, axis: str, normal: int, accommodation: float = 1.0, pressure_mode: str = "legacy-normal") -> np.ndarray:
    u = _state(u)
    if normal not in (-1, 1):
        raise ValueError("normal must be -1 or +1")
    if not 0.0 < accommodation <= 1.0:
        raise ValueError("accommodation must be in (0, 1]")
    chi = accommodation
    effective_pressure = _effective_pressure(u, axis, pressure_mode)
    if effective_pressure <= 0.0:
        raise FloatingPointError("non-positive effective wall pressure")
    fac = np.sqrt(2.0 / (np.pi * u[3])) * chi / (2.0 - chi)
    if axis == "x":
        return np.asarray([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -2*fac*normal*effective_pressure, 0, 0, -((fac*normal*u[3])/2), 0, 0, -((5*fac*normal)/28), 0, 0, 0, 0, 0, 0, -((fac*normal))/15],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -fac*normal*effective_pressure, 0, 0, -((fac*normal)/5), 0, 0, 0, 0, 0, 0, 0, -((fac*normal)/2), 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 7*fac*normal*effective_pressure*u[3], 0, 0, -((11*fac*normal*u[3])/5), 0, 0, 0, 0, 0, 0, 0, -((fac*normal*u[3])/2), 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, (2/5)*fac*normal*effective_pressure, 0, 0, -(7*fac*normal*u[3])/5, 0, 0, -(fac*normal)/14, 0, 0, 0, 0, 0, 0, fac*normal/75],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, -(1/5)*fac*normal*effective_pressure, 0, 0, (fac*normal*u[3])/5, 0, -fac*normal*u[3], 0, 0, -((fac*normal)/14), 0, 0, 0, 0, -fac*normal/150],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=float)
    if axis == "y":
        return np.asarray([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -2*fac*normal*effective_pressure, 0, 0, 0, 0, -((fac*normal*u[3])/2), 0, 0, -((5*fac*normal)/28), 0, 0, 0, 0, -((fac*normal)/15)],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -fac*normal*effective_pressure, 0, 0, -((fac*normal)/5), 0, 0, 0, 0, 0, 0, 0, 0, 0, -((fac*normal)/2), 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 7*fac*normal*effective_pressure*u[3], 0, 0, -((11*fac*normal*u[3])/5), 0, 0, 0, 0, 0, 0, 0, 0, 0, -((fac*normal*u[3])/2), 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, -(1/5)*fac*normal*effective_pressure, 0, 0, -fac*normal*u[3], 0, (fac*normal*u[3])/5, -((fac*normal)/14), 0, 0, 0, 0, 0, 0, -fac*normal/150],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, (2/5)*fac*normal*effective_pressure, 0, 0, 0, 0, -(7*fac*normal*u[3])/5, 0, 0, -(fac*normal)/14, 0, 0, 0, 0, fac*normal/75],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=float)
    raise ValueError("axis must be x or y")


def wall_data(u: np.ndarray, *, axis: str, normal: int, wall_velocity: float, wall_temperature: float = 1.0, accommodation: float = 1.0, pressure_mode: str = "legacy-normal") -> np.ndarray:
    u = _state(u)
    if normal not in (-1, 1):
        raise ValueError("normal must be -1 or +1")
    if not 0.0 < accommodation <= 1.0:
        raise ValueError("accommodation must be in (0, 1]")
    chi = accommodation
    effective_pressure = _effective_pressure(u, axis, pressure_mode)
    if effective_pressure <= 0.0:
        raise FloatingPointError("non-positive effective wall pressure")
    fac = np.sqrt(2.0 / (np.pi * u[3])) * chi / (2.0 - chi)
    if axis == "x":
        return np.asarray([
        [0, 0, 0, 0, 2*fac*normal*effective_pressure*wall_temperature+(fac*normal*effective_pressure*(u[2]-wall_velocity)**2)/2, 0, 0, fac*normal*effective_pressure*wall_velocity, 0, 0, -fac*normal*effective_pressure*u[3]*(wall_velocity)+6*fac*normal*effective_pressure*((0*u[3]-wall_temperature)*u[2]+(u[3]-wall_temperature)*(-wall_velocity))-(fac*normal*effective_pressure*(u[2]-wall_velocity)**3), 0, -2/5*fac*normal*effective_pressure*wall_temperature-3*(fac*normal*effective_pressure*(u[2]-wall_velocity)**2)/5, 0, 1/5*fac*normal*effective_pressure*wall_temperature+4*(fac*normal*effective_pressure*(u[2]-wall_velocity)**2)/5, 0, 0],
    ], dtype=float).reshape(NVAR)
    if axis == "y":
        return np.asarray([
        [0, 0, 0, 0, 0, 2*fac*normal*effective_pressure*wall_temperature+(fac*normal*effective_pressure*(u[1]-wall_velocity)**2)/2, 0, fac*normal*effective_pressure*wall_velocity, 0, 0, -fac*normal*effective_pressure*u[3]*(wall_velocity)+6*fac*normal*effective_pressure*((0*u[3]-wall_temperature)*u[1]+(u[3]-wall_temperature)*(-wall_velocity))-(fac*normal*effective_pressure*(u[1]-wall_velocity)**3), 0, 0, 1/5*fac*normal*effective_pressure*wall_temperature+4*(fac*normal*effective_pressure*(u[1]-wall_velocity)**2)/5, 0, -2/5*fac*normal*effective_pressure*wall_temperature-3*(fac*normal*effective_pressure*(u[1]-wall_velocity)**2)/5, 0],
    ], dtype=float).reshape(NVAR)
    raise ValueError("axis must be x or y")
