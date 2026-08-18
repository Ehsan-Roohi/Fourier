#!/usr/bin/env python3
"""Audited R13 production operator for the 2013 lid-cavity formulation.

This module is deliberately separate from the manuscript tree.  It encodes
the *reduced* production matrix printed in Appendix A of

    Rana, Torrilhon & Struchtrup, JCP 236 (2013) 169-186,
    doi:10.1016/j.jcp.2012.11.023.

The paper states that the bulk equations are for Maxwell molecules (section
2.1).  With the paper's reference Knudsen number

    Kn_Rana = mu_0 / (rho_0 sqrt(theta_0) L)

and a Maxwell-molecule viscosity law mu/mu_0 = theta/theta_0, the dimensional
relaxation factor p/mu becomes ``rho / Kn_Rana`` after nondimensionalisation.
The public ``production_maxwell`` function therefore multiplies the printed
matrix by rho/Kn_Rana.

Important audit caveat
----------------------
Equation (11) and Appendix A literally display ``P(U) U / Kn`` with no rho
prefactor.  The rho/Kn factor used here follows instead from equations (3)
and (4), the stated Maxwell model, and the stated nondimensionalisation.  Both
interpretations are exposed explicitly below; an external reference result
is required before calling either branch a reproduction of the authors'
numerical data.  No coefficient is fitted to DSMC or to an existing R13
state.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import numpy as np


NVAR = 17

# The state list printed below Eq. (11) places m_xyy before m_xxy.  That list
# conflicts with two independent pieces of the same Appendix: (i) the x-flux
# matrix places the divergence partner of sigma_xy in slot 13, and (ii) the
# x-wall matrix applies the m_ssn condition in slot 14.  Those two matrices,
# the surrounding boundary-condition text, and the archived solver therefore
# all use the conventional executable order below.  We record both orders so
# that no implicit permutation is possible.
STATE_ORDER_PRINTED_EQ11 = (
    "rho",
    "vx",
    "vy",
    "theta",
    "qx",
    "qy",
    "sigma_xx",
    "sigma_xy",
    "sigma_yy",
    "R_xx",
    "R_xy",
    "R_yy",
    "m_xxx",
    "m_xyy",
    "m_xxy",
    "m_yyy",
    "Delta",
)

STATE_ORDER_SOLVER = (
    "rho",
    "vx",
    "vy",
    "theta",
    "qx",
    "qy",
    "sigma_xx",
    "sigma_xy",
    "sigma_yy",
    "R_xx",
    "R_xy",
    "R_yy",
    "m_xxx",
    "m_xxy",
    "m_xyy",
    "m_yyy",
    "Delta",
)


def _validated_state(u: Sequence[object], *, exact: bool) -> list[object]:
    values = list(u)
    if len(values) != NVAR:
        raise ValueError(f"state must contain {NVAR} entries")
    if exact:
        # Exact Fraction arithmetic is used by the symbolic tests.  Avoid a
        # float conversion here, since that would defeat the test's purpose.
        if values[0] <= 0 or values[3] <= 0:
            raise FloatingPointError("rho and theta must be positive")
        return values
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or array[0] <= 0.0 or array[3] <= 0.0:
        raise FloatingPointError("state must be finite with positive rho and theta")
    return array.tolist()


def appendix_a_reduced_production_matrix(
    u: Sequence[object], *, exact: bool = False
) -> np.ndarray:
    """Return the reduced 17-by-17 matrix printed in Appendix A.

    ``exact=True`` keeps all rational arithmetic as ``fractions.Fraction``
    objects.  This is used to test the transcription without floating-point
    tolerances.
    """

    u = _validated_state(u, exact=exact)
    rho, theta = u[0], u[3]
    qx, qy = u[4], u[5]
    sxx, sxy, syy = u[6], u[7], u[8]

    def f(numerator: int, denominator: int = 1) -> object:
        return Fraction(numerator, denominator) if exact else numerator / denominator

    zero = Fraction(0, 1) if exact else 0.0
    m: list[list[object]] = [[zero for _ in range(NVAR)] for _ in range(NVAR)]

    # q_i and sigma_ij relaxation rows.
    m[4][4] = f(2, 3)
    m[5][5] = f(2, 3)
    m[6][6] = f(1)
    m[7][7] = f(1)
    m[8][8] = f(1)

    # R_xx, R_xy and R_yy rows.
    m[9][4] = -f(16, 45) * qx / (theta * rho)
    m[9][5] = f(8, 45) * qy / (theta * rho)
    m[9][6] = -f(25, 126) * (sxx - syy) / rho
    m[9][7] = -f(25, 126) * sxy / rho
    m[9][8] = f(25, 126) * (sxx + 2 * syy) / rho
    m[9][9] = f(5, 24)

    m[10][4] = -f(4, 15) * qy / (theta * rho)
    m[10][5] = -f(4, 15) * qx / (theta * rho)
    m[10][6] = -f(25, 84) * sxy / rho
    m[10][7] = -f(25, 84) * (sxx + syy) / rho
    m[10][8] = -f(25, 84) * sxy / rho
    m[10][10] = f(5, 24)

    m[11][4] = f(8, 45) * qx / (theta * rho)
    m[11][5] = -f(16, 45) * qy / (theta * rho)
    m[11][6] = f(25, 126) * (2 * sxx + syy) / rho
    m[11][7] = -f(25, 126) * sxy / rho
    m[11][8] = f(25, 126) * (sxx - syy) / rho
    m[11][11] = f(5, 24)

    # Third-order STF rows in the executable A/X-matrix order:
    # m_xxx, m_xxy, m_xyy, m_yyy.  The isolated state list below Eq. (11)
    # swaps the two mixed labels, but using that list as a permutation would
    # make the flux and wall matrices internally inconsistent.
    m[12][4] = -f(6, 25) * sxx / (theta * rho)
    m[12][5] = f(4, 25) * sxy / (theta * rho)
    m[12][6] = -f(4, 25) * qx / (theta * rho)
    m[12][7] = f(8, 75) * qy / (theta * rho)
    m[12][12] = f(1, 2)

    m[13][4] = -f(16, 75) * sxy / (theta * rho)
    m[13][5] = -f(2, 75) * (5 * sxx - 2 * syy) / (theta * rho)
    m[13][6] = -f(4, 45) * qy / (theta * rho)
    m[13][7] = -f(32, 225) * qx / (theta * rho)
    m[13][8] = f(8, 225) * qy / (theta * rho)
    m[13][13] = f(1, 2)

    m[14][4] = f(2, 75) * (2 * sxx - 5 * syy) / (theta * rho)
    m[14][5] = -f(16, 75) * sxy / (theta * rho)
    m[14][6] = f(8, 225) * qx / (theta * rho)
    m[14][7] = -f(32, 225) * qy / (theta * rho)
    m[14][8] = -f(4, 45) * qx / (theta * rho)
    m[14][14] = f(1, 2)

    m[15][4] = f(4, 25) * sxy / (theta * rho)
    m[15][5] = -f(6, 25) * syy / (theta * rho)
    m[15][7] = f(8, 75) * qx / (theta * rho)
    m[15][8] = -f(4, 25) * qy / (theta * rho)
    m[15][15] = f(1, 2)

    # Delta row.
    m[16][4] = -f(112, 15) * qx / (theta * rho)
    m[16][5] = -f(112, 15) * qy / (theta * rho)
    m[16][6] = -f(10, 3) * (2 * sxx + syy) / rho
    m[16][7] = -f(20, 3) * sxy / rho
    m[16][8] = -f(10, 3) * (sxx + 2 * syy) / rho
    m[16][16] = f(2, 3)

    return np.asarray(m, dtype=object if exact else float)


def maxwell_collision_prefactor(u: Sequence[object], kn_rana: float) -> float:
    """Return rho/Kn_Rana for mu/mu_0 = theta/theta_0."""

    values = _validated_state(u, exact=False)
    if not np.isfinite(kn_rana) or kn_rana <= 0.0:
        raise ValueError("kn_rana must be finite and positive")
    return float(values[0]) / float(kn_rana)


def legacy_sqrt_temperature_prefactor(
    u: Sequence[object], kn_rana: float
) -> float:
    """Return the prefactor found in the archived source, for audit only."""

    values = _validated_state(u, exact=False)
    if not np.isfinite(kn_rana) or kn_rana <= 0.0:
        raise ValueError("kn_rana must be finite and positive")
    return float(values[0]) * np.sqrt(float(values[3])) / float(kn_rana)


def production_maxwell(u: Sequence[object], kn_rana: float, **_: object) -> np.ndarray:
    """Return the collision-consistent Maxwell production operator."""

    matrix = appendix_a_reduced_production_matrix(u)
    return maxwell_collision_prefactor(u, kn_rana) * matrix


def production_appendix_literal(
    u: Sequence[object], kn_rana: float, **_: object
) -> np.ndarray:
    """Return the literal Eq. (11)+Appendix-A operator, P(U)/Kn.

    This branch is provided for an explicit audit of the paper's printed
    matrix form.  It is not selected by ``production_maxwell``.
    """

    if not np.isfinite(kn_rana) or kn_rana <= 0.0:
        raise ValueError("kn_rana must be finite and positive")
    return appendix_a_reduced_production_matrix(u) / float(kn_rana)


def kn_rana_from_gu(kn_gu: float) -> float:
    """Convert the Gu convention used by R26/DSMC to the Rana convention."""

    if not np.isfinite(kn_gu) or kn_gu <= 0.0:
        raise ValueError("kn_gu must be finite and positive")
    return float(np.sqrt(2.0 / np.pi) * kn_gu)


def kn_gu_from_rana(kn_rana: float) -> float:
    """Convert the Rana convention to the Gu convention."""

    if not np.isfinite(kn_rana) or kn_rana <= 0.0:
        raise ValueError("kn_rana must be finite and positive")
    return float(np.sqrt(np.pi / 2.0) * kn_rana)


# Deliberately no ``production = production_maxwell`` alias is exported.  A
# caller must go through r13_maxwell_adapter.install_candidate(), which checks
# the archived solver's state/flux/wall ordering before replacing its global
# production function.
