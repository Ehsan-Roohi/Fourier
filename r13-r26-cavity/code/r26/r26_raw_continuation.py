#!/usr/bin/env python3
"""Fail-closed acceptance and step control for private R26 continuation.

The nonlinear solver is free to scale equations internally, but a state is
accepted here only from the original unscaled physical rows, the displaced
continuity equation, the independent mass equation, positivity, a nontrivial
lid response, and a full-column-rank final Jacobian.  In particular,
``diagnostics.total_linf`` (the case-scaled residual) is intentionally never
used by :func:`strict_raw_acceptance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np


class _RawDiagnostics(Protocol):
    raw_total_linf: float
    raw_bulk_linf: float
    raw_wall_linf: float
    raw_extrapolation_linf: float
    raw_corner_linf: float
    held_out_continuity: float
    mass_error: float
    min_density: float
    min_temperature: float


@dataclass(frozen=True)
class RawAcceptanceDecision:
    """Auditable result of the strict, unscaled acceptance gate."""

    accepted: bool
    failed_checks: tuple[str, ...]
    raw_total_linf: float
    held_out_continuity_abs: float
    mass_error_abs: float
    interior_velocity_linf_over_lid: float
    jacobian_rank: int
    unknown_count: int
    equation_count: int


def strict_raw_acceptance(
    diagnostics: _RawDiagnostics,
    response_diagnostics: Mapping[str, float | bool],
    *,
    optimizer_success: bool,
    jacobian_rank: int,
    unknown_count: int,
    equation_count: int,
    raw_tolerance: float,
    held_tolerance: float,
    mass_tolerance: float,
    minimum_response_ratio: float,
) -> RawAcceptanceDecision:
    """Evaluate every fail-closed acceptance requirement.

    The individual raw-family values are checked as well as their reported
    maximum.  This deliberately makes a future diagnostics regression fail
    closed instead of allowing a stale ``raw_total_linf`` field to hide one
    family.  The expected algebra is rectangular: all physical equations plus
    mass, hence exactly ``unknown_count + 1`` rows.
    """

    tolerances = (raw_tolerance, held_tolerance, mass_tolerance)
    if not all(np.isfinite(value) and value > 0.0 for value in tolerances):
        raise ValueError("all acceptance tolerances must be finite and positive")
    if not np.isfinite(minimum_response_ratio) or minimum_response_ratio < 0.0:
        raise ValueError("minimum_response_ratio must be finite and nonnegative")
    if unknown_count < 1 or equation_count < 1 or jacobian_rank < 0:
        raise ValueError("equation/rank counts must be nonnegative and nonempty")

    raw_families = {
        "raw_total_linf": float(diagnostics.raw_total_linf),
        "raw_bulk_linf": float(diagnostics.raw_bulk_linf),
        "raw_wall_linf": float(diagnostics.raw_wall_linf),
        "raw_extrapolation_linf": float(diagnostics.raw_extrapolation_linf),
        "raw_corner_linf": float(diagnostics.raw_corner_linf),
    }
    held = abs(float(diagnostics.held_out_continuity))
    mass = abs(float(diagnostics.mass_error))
    minimum_density = float(diagnostics.min_density)
    minimum_temperature = float(diagnostics.min_temperature)
    response_ratio = float(
        response_diagnostics.get("interior_velocity_linf_over_lid", np.nan)
    )

    failed: list[str] = []
    if not optimizer_success:
        failed.append("optimizer_success")
    for name, value in raw_families.items():
        if not np.isfinite(value) or value > raw_tolerance:
            failed.append(name)
    if not np.isfinite(held) or held > held_tolerance:
        failed.append("held_out_continuity")
    if not np.isfinite(mass) or mass > mass_tolerance:
        failed.append("mass_error")
    if not np.isfinite(minimum_density) or minimum_density <= 0.0:
        failed.append("positive_density")
    if not np.isfinite(minimum_temperature) or minimum_temperature <= 0.0:
        failed.append("positive_temperature")
    if not np.isfinite(response_ratio) or response_ratio < minimum_response_ratio:
        failed.append("interior_lid_response")
    if equation_count != unknown_count + 1:
        failed.append("all_physical_plus_mass_equation_count")
    if jacobian_rank != unknown_count:
        failed.append("full_column_rank")

    return RawAcceptanceDecision(
        accepted=not failed,
        failed_checks=tuple(failed),
        raw_total_linf=raw_families["raw_total_linf"],
        held_out_continuity_abs=held,
        mass_error_abs=mass,
        interior_velocity_linf_over_lid=response_ratio,
        jacobian_rank=int(jacobian_rank),
        unknown_count=int(unknown_count),
        equation_count=int(equation_count),
    )


def next_continuation_step(
    *,
    attempted_step: float,
    accepted: bool,
    growth_factor: float,
    minimum_step: float,
    maximum_step: float,
) -> float:
    """Grow an accepted step or halve a rejected one, with declared bounds."""

    values = (attempted_step, growth_factor, minimum_step, maximum_step)
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("continuation step controls must be finite and positive")
    if growth_factor < 1.0:
        raise ValueError("growth_factor must be at least one")
    if minimum_step > maximum_step:
        raise ValueError("minimum_step cannot exceed maximum_step")
    candidate = attempted_step * growth_factor if accepted else attempted_step * 0.5
    return float(min(maximum_step, max(minimum_step, candidate)))


__all__ = [
    "RawAcceptanceDecision",
    "next_continuation_step",
    "strict_raw_acceptance",
]
