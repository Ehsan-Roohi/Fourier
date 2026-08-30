#!/usr/bin/env python3
"""Private R26 cavity case definitions and explicit transport conventions.

The published Rana and Gu--Emerson Knudsen numbers do not use the same
normalization.  This module keeps that distinction in the type carrying a
case; there is deliberately no unlabelled ``mu = Kn`` shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Final

import numpy as np

from r26_state import NVAR


SQRT_2_OVER_PI: Final[float] = float(np.sqrt(2.0 / np.pi))


@dataclass(frozen=True)
class GuASME2009CavityContract:
    """Source-locked dimensional data for the published driven cavity.

    Gu, John, Tang & Emerson, ASME MNHMT2009-18236, explicitly print every
    value below except ``length_m``.  The paper fixes ``Kn=lambda/L`` so the
    nondimensional solution does not require a dimensional ``L``.  The
    50-micrometre length is retained only as the project's declared THOR
    comparison scale and is never attributed to the ASME paper.
    """

    gas: str = "argon"
    reference_temperature_K: float = 273.0
    reference_viscosity_Pa_s: float = 21.25e-6
    sutherland_temperature_K: float = 144.0
    gas_constant_J_kg_K: float = 208.0
    accommodation: float = 1.0
    published_grid_points: int = 100
    published_lid_speeds_m_s: tuple[float, float] = (10.0, 100.0)
    published_knudsen_numbers: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5)
    project_comparison_length_m: float = 5.0e-5
    length_provenance: str = (
        "project THOR benchmark declaration; the ASME paper leaves L symbolic"
    )
    paper: str = (
        "Gu--John--Tang--Emerson, ASME MNHMT2009-18236, Eqs. (35)--(36)"
    )


GU_ASME2009_CAVITY_CONTRACT: Final[GuASME2009CavityContract] = (
    GuASME2009CavityContract()
)


class KnudsenConvention(str, Enum):
    """Supported definitions of the input Knudsen number."""

    RANA = "rana_mu_over_rho_cL"
    GU_MEAN_FREE_PATH = "gu_lambda_over_L"


class ViscosityKind(str, Enum):
    POWER_LAW = "power_law"
    GU_SUTHERLAND = "gu_sutherland"


@dataclass(frozen=True)
class ViscosityModel:
    """A nondimensional viscosity law normalized at ``theta=1``.

    ``GU_SUTHERLAND`` stores the dimensional provenance values but evaluates
    a ratio normalized at the case reference temperature.  Consequently the
    equilibrium viscosity remains exactly the value implied by ``Kn`` and its
    declared convention.
    """

    kind: ViscosityKind = ViscosityKind.POWER_LAW
    exponent: float = 1.0
    reference_temperature_K: float = 273.0
    sutherland_temperature_K: float = 144.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.exponent):
            raise ValueError("viscosity exponent must be finite")
        if self.reference_temperature_K <= 0.0:
            raise ValueError("reference temperature must be positive")
        if self.sutherland_temperature_K < 0.0:
            raise ValueError("Sutherland temperature cannot be negative")

    @classmethod
    def power_law(cls, exponent: float = 1.0) -> "ViscosityModel":
        return cls(kind=ViscosityKind.POWER_LAW, exponent=float(exponent))

    @classmethod
    def gu_sutherland(
        cls, *, reference_temperature_K: float = 300.0, sutherland_temperature_K: float = 144.0
    ) -> "ViscosityModel":
        return cls(
            kind=ViscosityKind.GU_SUTHERLAND,
            exponent=1.5,
            reference_temperature_K=float(reference_temperature_K),
            sutherland_temperature_K=float(sutherland_temperature_K),
        )

    def ratio(self, theta: np.ndarray | float) -> np.ndarray:
        """Return ``mu(theta)/mu(theta=1)`` for positive nondimensional T."""

        value = np.asarray(theta, dtype=float)
        if not np.isfinite(value).all() or np.any(value <= 0.0):
            raise FloatingPointError("viscosity requires finite positive temperature")
        if self.kind is ViscosityKind.POWER_LAW:
            return value**self.exponent
        if self.kind is ViscosityKind.GU_SUTHERLAND:
            sstar = self.sutherland_temperature_K / self.reference_temperature_K
            return value**1.5 * (1.0 + sstar) / (value + sstar)
        raise ValueError(f"unsupported viscosity kind {self.kind!r}")


def equilibrium_mu_star(kn: float, convention: KnudsenConvention) -> float:
    """Convert a declared published Kn to the generic nondimensional mu.

    With ``c0=sqrt(R*T0)`` and ``p0=rho0*R*T0``, Rana's convention gives
    ``mu*=Kn``.  Gu's ``lambda=(mu/p)*sqrt(pi*R*T/2)`` gives
    ``mu*=Kn*sqrt(2/pi)`` at equilibrium.
    """

    kn_value = float(kn)
    if not np.isfinite(kn_value) or kn_value <= 0.0:
        raise ValueError("Kn must be finite and positive")
    if convention is KnudsenConvention.RANA:
        return kn_value
    if convention is KnudsenConvention.GU_MEAN_FREE_PATH:
        return kn_value * SQRT_2_OVER_PI
    raise ValueError(f"unsupported Kn convention {convention!r}")


@dataclass(frozen=True)
class ResidualScaling:
    """Explicit scales for the five residual families of the node BVP."""

    bulk: np.ndarray = field(default_factory=lambda: np.ones(NVAR))
    wall: np.ndarray = field(default_factory=lambda: np.ones(11))
    extrapolation: np.ndarray = field(default_factory=lambda: np.ones(6))
    corner: np.ndarray = field(default_factory=lambda: np.ones(NVAR))
    mass: float = 1.0

    def __post_init__(self) -> None:
        expected = (("bulk", NVAR), ("wall", 11), ("extrapolation", 6), ("corner", NVAR))
        for name, size in expected:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all() or np.any(value <= 0.0):
                raise ValueError(f"{name} residual scale must contain {size} positive values")
            object.__setattr__(self, name, value.copy())
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("mass residual scale must be positive")


def collision_balanced_scaling(mu_equilibrium: float) -> ResidualScaling:
    """Balance moment rows by their equilibrium collision frequencies.

    This changes only nonlinear algebraic conditioning.  The unscaled raw
    residual is retained in every final diagnostic.
    """

    mu = float(mu_equilibrium)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("equilibrium viscosity must be positive")
    bulk = np.ones(NVAR)
    bulk[4:6] = max(1.0, 2.0 / (3.0 * mu))
    bulk[6:9] = max(1.0, 1.0 / mu)
    bulk[9:12] = max(1.0, 7.0 / (6.0 * mu))
    bulk[12:16] = max(1.0, 3.0 / (2.0 * mu))
    bulk[16] = max(1.0, 2.0 / (3.0 * mu))
    return ResidualScaling(bulk=bulk)


@dataclass(frozen=True)
class CavityCase:
    """A square, isothermal, diffuse-wall cavity in nondimensional units."""

    name: str
    nodes: int
    kn: float
    kn_convention: KnudsenConvention
    lid_velocity: float
    wall_temperature: float = 1.0
    mean_density: float = 1.0
    gas_constant: float = 1.0
    accommodation: float = 1.0
    viscosity: ViscosityModel = field(default_factory=ViscosityModel)
    scaling: ResidualScaling = field(default_factory=ResidualScaling)
    length: float = 1.0
    grid_stretch_beta: float = 0.0
    r26_closure_mode: str = "jfm2009"
    provenance: str = "private R26 verification case"

    def __post_init__(self) -> None:
        if self.nodes < 5:
            raise ValueError("node-grid R26 BVP needs at least five nodes per direction")
        scalar_positive = (self.kn, self.wall_temperature, self.mean_density, self.gas_constant, self.length)
        if not all(np.isfinite(v) and v > 0.0 for v in scalar_positive):
            raise ValueError("Kn, wall temperature, density, R, and length must be positive")
        if not np.isfinite(self.lid_velocity):
            raise ValueError("lid velocity must be finite")
        if not np.isfinite(self.grid_stretch_beta) or self.grid_stretch_beta < 0.0:
            raise ValueError("grid_stretch_beta must be finite and nonnegative")
        if self.r26_closure_mode not in {"jfm2009", "asme2009-cavity"}:
            raise ValueError("unsupported R26 closure coefficient/equation mode")
        if not (0.0 < self.accommodation <= 1.0):
            raise ValueError("accommodation must lie in (0,1]")

    def _axis(self) -> np.ndarray:
        """Return a symmetric wall-clustered node axis.

        ``beta=0`` is exactly the original uniform grid.  Positive ``beta``
        uses a symmetric tanh map, preserving the geometric centre while
        placing progressively more nodes in both Knudsen layers.  The grid
        convention is stored in the immutable case metadata so a restart
        cannot silently change geometry.
        """

        s = np.linspace(0.0, 1.0, self.nodes)
        beta = float(self.grid_stretch_beta)
        if beta == 0.0:
            return self.length * s
        return self.length * 0.5 * (
            1.0 + np.tanh(beta * (2.0 * s - 1.0)) / np.tanh(beta)
        )

    @property
    def x(self) -> np.ndarray:
        return self._axis()

    @property
    def y(self) -> np.ndarray:
        return self._axis()

    @property
    def mu_equilibrium(self) -> float:
        return equilibrium_mu_star(self.kn, self.kn_convention)

    def mu(self, theta: np.ndarray | float) -> np.ndarray:
        return self.mu_equilibrium * self.viscosity.ratio(theta)

    def wall_velocity(self, side: str) -> np.ndarray:
        velocity = np.zeros(3)
        if side.lower() == "top":
            velocity[0] = self.lid_velocity
        return velocity

    def equilibrium_state(self) -> np.ndarray:
        state = np.zeros((self.nodes, self.nodes, NVAR))
        state[..., 0] = self.mean_density
        state[..., 3] = self.wall_temperature
        return state

    def with_lid_velocity(self, value: float, *, suffix: str | None = None) -> "CavityCase":
        name = self.name if suffix is None else f"{self.name}-{suffix}"
        return replace(self, name=name, lid_velocity=float(value))


def rana_first_case(
    nodes: int,
    *,
    viscosity_exponent: float = 1.0,
    lid_speed_m_per_s: float = 50.0,
    gas_constant_si: float = 208.0,
    temperature_K: float = 273.0,
    grid_stretch_beta: float = 0.0,
) -> CavityCase:
    """Rana et al. first cavity definition, for an R26 predictive run.

    Rana (2013) reports R13, not R26, so its published D/G are comparison
    context rather than R26 validation targets.
    """

    lid_star = lid_speed_m_per_s / np.sqrt(gas_constant_si * temperature_K)
    kn = 0.010
    return CavityCase(
        name=f"rana-first-r26-N{nodes}",
        nodes=nodes,
        kn=kn,
        kn_convention=KnudsenConvention.RANA,
        lid_velocity=float(lid_star),
        viscosity=ViscosityModel.power_law(viscosity_exponent),
        scaling=collision_balanced_scaling(equilibrium_mu_star(kn, KnudsenConvention.RANA)),
        grid_stretch_beta=float(grid_stretch_beta),
        provenance="Rana--Torrilhon--Struchtrup 2013 first cavity; new R26 prediction",
    )


def rana_john_case(
    nodes: int,
    *,
    kn: float,
    lid_speed_m_per_s: float = 50.0,
    gas_constant_si: float = 208.0,
    temperature_K: float = 273.0,
    grid_stretch_beta: float = 0.0,
    closure_mode: str = "jfm2009",
) -> CavityCase:
    """Rana Fig. 3 / John-2010 50 m/s comparison conditions.

    The target ``Kn`` is deliberately explicit.  In particular, the common
    comparison values 0.0798 and 0.3989 are *not* converted with the Gu
    mean-free-path factor.  This prevents a roughly 20% viscosity shift from
    masquerading as an R13/R26 model difference.

    The default closure coefficients are the final Gu--Emerson JFM-2009 R26
    values (``C2`` and ``Y2`` positive, ``Y1=1.698``).  The preliminary ASME
    cavity paper printed a different complete set and remains available only
    through an explicit ``closure_mode='asme2009-cavity'`` request.  Keeping
    these modes explicit prevents a preliminary coefficient set from being
    silently used as the final R26 model in the Rana/John kinetic comparison.
    """

    kn_value = float(kn)
    lid_star = lid_speed_m_per_s / np.sqrt(gas_constant_si * temperature_K)
    return CavityCase(
        name=f"rana-john-Kn{kn_value:g}-U{lid_speed_m_per_s:g}-N{nodes}",
        nodes=nodes,
        kn=kn_value,
        kn_convention=KnudsenConvention.RANA,
        lid_velocity=float(lid_star),
        viscosity=ViscosityModel.gu_sutherland(
            reference_temperature_K=temperature_K,
            sutherland_temperature_K=144.0,
        ),
        scaling=collision_balanced_scaling(
            equilibrium_mu_star(kn_value, KnudsenConvention.RANA)
        ),
        grid_stretch_beta=float(grid_stretch_beta),
        r26_closure_mode=str(closure_mode),
        provenance=(
            "Rana JCP 2013 Fig.3 conditions and John et al. NHT-B 2010 DSMC "
            "context at 50 m/s, expressed in Rana Kn; R26 closure mode is "
            f"{closure_mode!r} and viscosity follows Sutherland; "
            "cross-paper validation candidate"
        ),
    )


def gu_asme2009_cavity_case(
    nodes: int,
    *,
    kn: float,
    lid_speed_m_per_s: float = 10.0,
    gas_constant_si: float = 208.0,
    wall_temperature_K: float = 273.0,
    grid_stretch_beta: float = 0.0,
) -> CavityCase:
    """Direct Gu--John--Tang--Emerson ASME HT2009-88293 cavity case.

    The paper uses ``Kn=lambda/L`` with ``lambda=(mu/p)*sqrt(pi*R*T/2)``,
    argon ``R=208``, ``T0=273 K``, ``S=144 K``, diffuse walls, and reports
    10 and 100 m/s lids for Kn=0.05,0.1,0.2,0.5.  Unsupported parameter
    values remain possible for sensitivity studies but are named explicitly.
    """

    contract = GU_ASME2009_CAVITY_CONTRACT
    kn_value = float(kn)
    lid_star = lid_speed_m_per_s / np.sqrt(gas_constant_si * wall_temperature_K)
    mu_eq = equilibrium_mu_star(kn_value, KnudsenConvention.GU_MEAN_FREE_PATH)
    return CavityCase(
        name=f"gu-asme2009-Kn{kn_value:g}-U{lid_speed_m_per_s:g}-N{nodes}",
        nodes=nodes,
        kn=kn_value,
        kn_convention=KnudsenConvention.GU_MEAN_FREE_PATH,
        lid_velocity=float(lid_star),
        viscosity=ViscosityModel.gu_sutherland(
            reference_temperature_K=wall_temperature_K,
            sutherland_temperature_K=contract.sutherland_temperature_K,
        ),
        scaling=collision_balanced_scaling(mu_eq),
        grid_stretch_beta=float(grid_stretch_beta),
        r26_closure_mode="asme2009-cavity",
        provenance=(
            "Gu--John--Tang--Emerson ASME HT2009-88293 driven cavity; "
            "Gu lambda/L Kn, source-locked ASME closure constants, "
            "Sutherland mu0=21.25e-6 Pa s at 273 K, S=144 K; "
            f"requested grid={nodes} points (paper grid={contract.published_grid_points})"
        ),
    )


def gu_asme2009_published_cavity_case(
    *,
    kn: float,
    lid_speed_m_per_s: float = 10.0,
) -> CavityCase:
    """Return the literal 100x100 ASME-2009 published cavity configuration.

    This constructor rejects off-paper Knudsen numbers and lid speeds.  The
    lower-grid constructor above remains available only for numerical gates.
    """

    contract = GU_ASME2009_CAVITY_CONTRACT
    kn_value = float(kn)
    speed = float(lid_speed_m_per_s)
    if kn_value not in contract.published_knudsen_numbers:
        raise ValueError("published ASME cavity Kn must be 0.05, 0.1, 0.2, or 0.5")
    if speed not in contract.published_lid_speeds_m_s:
        raise ValueError("published ASME cavity lid speed must be 10 or 100 m/s")
    return gu_asme2009_cavity_case(
        contract.published_grid_points,
        kn=kn_value,
        lid_speed_m_per_s=speed,
        gas_constant_si=contract.gas_constant_J_kg_K,
        wall_temperature_K=contract.reference_temperature_K,
        grid_stretch_beta=0.0,
    )


def gu_jfm_cavity_case(
    nodes: int,
    *,
    kn: float = 0.05,
    lid_speed_m_per_s: float = 100.0,
    gas_constant_si: float = 208.0,
    wall_temperature_K: float = 300.0,
    grid_stretch_beta: float = 0.0,
) -> CavityCase:
    """Gu/John driven argon cavity with the Gu mean-free-path Kn."""

    lid_star = lid_speed_m_per_s / np.sqrt(gas_constant_si * wall_temperature_K)
    mu_eq = equilibrium_mu_star(float(kn), KnudsenConvention.GU_MEAN_FREE_PATH)
    return CavityCase(
        name=f"gu-jfm-Kn{kn:g}-U{lid_speed_m_per_s:g}-N{nodes}",
        nodes=nodes,
        kn=float(kn),
        kn_convention=KnudsenConvention.GU_MEAN_FREE_PATH,
        lid_velocity=float(lid_star),
        viscosity=ViscosityModel.gu_sutherland(
            reference_temperature_K=wall_temperature_K,
            sutherland_temperature_K=144.0,
        ),
        scaling=collision_balanced_scaling(mu_eq),
        grid_stretch_beta=float(grid_stretch_beta),
        provenance=(
            "Gu--Emerson JFM 636 R26 / Gu--John--Tang--Emerson cavity; "
            "Sutherland ratio normalized at initial wall temperature"
        ),
    )


def jfm_observability_cavity_case(
    nodes: int,
    *,
    kn: float = 0.05,
    lid_speed_m_per_s: float = 100.0,
    gas_constant_si: float = 208.0,
    wall_temperature_K: float = 300.0,
    viscosity_exponent: float = 0.81,
    grid_stretch_beta: float = 0.0,
) -> CavityCase:
    """Exact nondimensional target used by the anti-Fourier DSMC manuscript.

    The manuscript declares ``Kn=lambda/L`` on the equilibrium VHS
    mean-free-path basis, monatomic argon with ``R=208 J/(kg K)``,
    ``T_w=300 K``, a ``100 m/s`` lid, and the VHS viscosity exponent
    ``omega=0.81``.  The complete R26 coefficients remain the final
    ``jfm2009`` set; this factory does not silently substitute the preliminary
    ASME cavity coefficients.
    """

    kn_value = float(kn)
    lid_star = lid_speed_m_per_s / np.sqrt(gas_constant_si * wall_temperature_K)
    mu_eq = equilibrium_mu_star(kn_value, KnudsenConvention.GU_MEAN_FREE_PATH)
    return CavityCase(
        name=f"jfm-observability-Kn{kn_value:g}-U{lid_speed_m_per_s:g}-N{nodes}",
        nodes=nodes,
        kn=kn_value,
        kn_convention=KnudsenConvention.GU_MEAN_FREE_PATH,
        lid_velocity=float(lid_star),
        viscosity=ViscosityModel.power_law(float(viscosity_exponent)),
        scaling=collision_balanced_scaling(mu_eq),
        grid_stretch_beta=float(grid_stretch_beta),
        r26_closure_mode="jfm2009",
        provenance=(
            "JFM-2026-1451 anti-Fourier DSMC target: argon VHS omega=0.81, "
            "Gu equilibrium mean-free-path Kn, Tw=300 K, diffuse walls, "
            "final JFM-2009 R26 closure coefficients"
        ),
    )


def jfm_maxwell_cavity_case(
    nodes: int,
    *,
    kn: float,
    lid_speed_m_per_s: float = 100.0,
    gas_constant_si: float = 208.0,
    wall_temperature_K: float = 300.0,
    grid_stretch_beta: float = 0.0,
) -> CavityCase:
    """Source-locked pure-Maxwell target for matched R26/DSMC comparisons.

    The nonlinear bulk equations, closure coefficients and smooth-wall
    conditions are the final Gu--Emerson JFM-2009 Maxwell-molecule model.
    Maxwell molecules require ``mu/mu0 = theta``; consequently the viscosity
    exponent is fixed to one and is intentionally not a caller parameter.
    The wall accommodation is fixed to one to match fully diffuse DSMC walls.
    """

    kn_value = float(kn)
    lid_star = lid_speed_m_per_s / np.sqrt(gas_constant_si * wall_temperature_K)
    mu_eq = equilibrium_mu_star(kn_value, KnudsenConvention.GU_MEAN_FREE_PATH)
    return CavityCase(
        name=f"jfm-maxwell-Kn{kn_value:g}-U{lid_speed_m_per_s:g}-N{nodes}",
        nodes=nodes,
        kn=kn_value,
        kn_convention=KnudsenConvention.GU_MEAN_FREE_PATH,
        lid_velocity=float(lid_star),
        wall_temperature=1.0,
        accommodation=1.0,
        viscosity=ViscosityModel.power_law(1.0),
        scaling=collision_balanced_scaling(mu_eq),
        grid_stretch_beta=float(grid_stretch_beta),
        r26_closure_mode="jfm2009",
        provenance=(
            "Pure Maxwell-molecule matched target: Gu--Emerson JFM-2009 "
            "nonlinear R26 equations and closure coefficients, mu/mu0=T/T0, "
            "Gu equilibrium mean-free-path Kn, Tw=300 K, fully diffuse walls"
        ),
    )


__all__ = [
    "CavityCase",
    "KnudsenConvention",
    "ResidualScaling",
    "SQRT_2_OVER_PI",
    "ViscosityKind",
    "ViscosityModel",
    "equilibrium_mu_star",
    "collision_balanced_scaling",
    "gu_jfm_cavity_case",
    "jfm_maxwell_cavity_case",
    "jfm_observability_cavity_case",
    "gu_asme2009_cavity_case",
    "rana_first_case",
    "rana_john_case",
]
