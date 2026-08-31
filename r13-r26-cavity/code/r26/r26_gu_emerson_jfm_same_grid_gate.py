#!/usr/bin/env python3
"""Fail-closed JFM Maxwell N28-to-N28 transformed-variable gate.

The immutable accepted state is a root of the repository's historical
compatible central finite-volume operator.  This gate changes variables on
that *same* 28-by-28 grid, reconstructs the physical moments, and requires the
transformed equation-(63) balance to be algebraically identical to the same
compatible physical operator.  It neither solves nor accepts a higher grid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any

import numpy as np
from scipy.sparse.csgraph import structural_rank

from r26_cases import CavityCase, KnudsenConvention, ViscosityKind
from r26_discretization import R26NodeBVP
from r26_fv_backend import (
    compatible_fv_bulk_residual,
    wall_bounded_control_volume_weights,
)
from r26_gu_emerson_transformed_fv import (
    gu_emerson_compatible_transformed_fv_residual,
)
from r26_gu_emerson_variables import (
    gu_emerson_fields_as_planar17,
    gu_emerson_fields_from_planar17,
    gu_emerson_fields_from_state,
    state_from_gu_emerson_fields,
)
from r26_solver import jacobian_sparsity
from r26_thor_audit import compare_cross_solver_profiles, state_sha256
from r26_validation import global_balance_diagnostics


JFM_N28_REFERENCE_STATE_SHA256 = (
    "e1fb8c5696351f0409c3a7cf984bfd4c99a25dbc79f82bd944655cfa21467ff4"
)
JFM_N28_REFERENCE_CASE = "jfm-maxwell-Kn0.2-U100-N28"


@dataclass(frozen=True)
class JFMSameGridThresholds:
    raw_tolerance: float = 1.0e-8
    compatibility_linf: float = 5.0e-12
    physical_roundtrip_linf: float = 5.0e-11
    transformed_storage_roundtrip_linf: float = 5.0e-15
    maximum_profile_nrms: float = 0.05
    maximum_line_nrms: float = 0.15
    maximum_DG_relative_difference: float = 0.02
    conservation_tolerance: float = 1.0e-7
    wall_normal_velocity_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class JFMSameGridGateResult:
    record: dict[str, Any]
    candidate_state: np.ndarray
    transformed_state: np.ndarray


def _raw_gate(diagnostics: object) -> float:
    return float(
        max(
            diagnostics.raw_total_linf,
            abs(diagnostics.held_out_continuity),
            abs(diagnostics.mass_error),
        )
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"JFM N28 same-grid contract failed: {message}")


def require_jfm_n28_case(case: CavityCase) -> None:
    """Reject any ASME, VHS, stretched, off-grid or reduced-speed case."""

    expected_lid = 100.0 / math.sqrt(208.0 * 300.0)
    _require(case.name == JFM_N28_REFERENCE_CASE, "wrong case name")
    _require(case.nodes == 28, "grid is not N28")
    _require(math.isclose(case.kn, 0.2, rel_tol=0.0, abs_tol=2.0e-15), "wrong Kn")
    _require(
        case.kn_convention is KnudsenConvention.GU_MEAN_FREE_PATH,
        "wrong Kn convention",
    )
    _require(
        math.isclose(case.lid_velocity, expected_lid, rel_tol=0.0, abs_tol=2.0e-14),
        "wrong dimensional lid-speed mapping",
    )
    _require(case.wall_temperature == 1.0, "wrong nondimensional wall temperature")
    _require(case.accommodation == 1.0, "wall is not fully diffuse")
    _require(case.grid_stretch_beta == 0.0, "reference grid is not uniform")
    _require(case.r26_closure_mode == "jfm2009", "wrong closure coefficients")
    _require(case.viscosity.kind is ViscosityKind.POWER_LAW, "wrong viscosity law")
    _require(case.viscosity.exponent == 1.0, "Maxwell viscosity exponent is not one")
    _require("Pure Maxwell-molecule" in case.provenance, "Maxwell provenance missing")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    # bool is an int subclass, so preserve JSON booleans before integer handling.
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Enum):
        return value.value
    return value


def run_jfm_n28_same_grid_gate(
    reference_state: np.ndarray,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    *,
    case: CavityCase,
    source_commit: str,
    expected_reference_sha256: str = JFM_N28_REFERENCE_STATE_SHA256,
    thresholds: JFMSameGridThresholds | None = None,
) -> JFMSameGridGateResult:
    """Transform, reconstruct and independently gate the frozen JFM N28 root."""

    thresholds = JFMSameGridThresholds() if thresholds is None else thresholds
    require_jfm_n28_case(case)
    _require(
        len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit),
        "source commit is not an immutable lowercase SHA",
    )
    state = np.asarray(reference_state, dtype=float)
    x = np.asarray(reference_x, dtype=float)
    y = np.asarray(reference_y, dtype=float)
    _require(state.shape == (28, 28, 17), "reference state shape mismatch")
    _require(np.isfinite(state).all(), "reference state is non-finite")
    _require(np.array_equal(x, case.x), "reference x coordinates changed")
    _require(np.array_equal(y, case.y), "reference y coordinates changed")
    reference_digest = state_sha256(state)
    _require(
        reference_digest == expected_reference_sha256,
        "frozen reference state SHA-256 mismatch",
    )

    problem = R26NodeBVP(
        case,
        bulk_operator=compatible_fv_bulk_residual,
        mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
    )
    reference_evaluation = problem.evaluate(state)
    reference_raw = _raw_gate(reference_evaluation.diagnostics)

    fields = gu_emerson_fields_from_state(
        state,
        x=case.x,
        y=case.y,
        mu=case.mu(state[..., 3]),
    )
    packed = gu_emerson_fields_as_planar17(fields)
    unpacked = gu_emerson_fields_from_planar17(packed)
    repacked = gu_emerson_fields_as_planar17(unpacked)
    storage_roundtrip = float(np.max(np.abs(repacked - packed), initial=0.0))
    candidate = state_from_gu_emerson_fields(
        unpacked,
        x=case.x,
        y=case.y,
        mu=case.mu(unpacked.theta),
    )
    physical_roundtrip = float(np.max(np.abs(candidate - state), initial=0.0))
    candidate_evaluation = problem.evaluate(candidate)
    candidate_raw = _raw_gate(candidate_evaluation.diagnostics)

    transformed_residual = gu_emerson_compatible_transformed_fv_residual(
        unpacked,
        case=case,
        convection_scheme="central",
    )
    physical_bulk = compatible_fv_bulk_residual(
        candidate,
        case.x,
        case.y,
        case.mu(candidate[..., 3]),
        case=case,
        convection_scheme="central",
    )
    interior = np.s_[1:-1, 1:-1]
    transformed_linf = float(
        np.max(np.abs(transformed_residual[interior]), initial=0.0)
    )
    compatibility_error = float(
        np.max(
            np.abs(
                transformed_residual[interior] - physical_bulk[interior]
            ),
            initial=0.0,
        )
    )
    boundary_zero = bool(
        np.array_equal(transformed_residual[0], np.zeros_like(transformed_residual[0]))
        and np.array_equal(transformed_residual[-1], np.zeros_like(transformed_residual[-1]))
        and np.array_equal(transformed_residual[:, 0], np.zeros_like(transformed_residual[:, 0]))
        and np.array_equal(transformed_residual[:, -1], np.zeros_like(transformed_residual[:, -1]))
    )

    reference_balance = global_balance_diagnostics(state, case)
    candidate_balance = global_balance_diagnostics(candidate, case)
    comparison = compare_cross_solver_profiles(
        state,
        x,
        y,
        candidate,
        case.x,
        case.y,
        lid_velocity=case.lid_velocity,
    )
    pattern = jacobian_sparsity(problem)
    rank = int(structural_rank(pattern))
    unknown_count = problem.unknown_count

    diagnostics = candidate_evaluation.diagnostics
    conservation_passed = bool(
        float(candidate_balance["wall_effective_pressure_min"]) > 0.0
        and float(candidate_balance["momentum_boundary_flux_linf"])
        <= thresholds.conservation_tolerance
        and abs(float(candidate_balance["internal_energy_balance_error"]))
        <= thresholds.conservation_tolerance
        and float(candidate_balance["wall_normal_velocity_linf"])
        <= thresholds.wall_normal_velocity_tolerance
    )
    comparison_passed = bool(
        float(comparison["maximum_normalized_rms_difference"])
        <= thresholds.maximum_profile_nrms
        and float(comparison["maximum_line_normalized_rms_difference"])
        <= thresholds.maximum_line_nrms
        and float(comparison["D_relative_difference"])
        <= thresholds.maximum_DG_relative_difference
        and float(comparison["G_relative_difference"])
        <= thresholds.maximum_DG_relative_difference
    )
    passed = bool(
        reference_raw <= thresholds.raw_tolerance
        and candidate_raw <= thresholds.raw_tolerance
        and transformed_linf <= thresholds.raw_tolerance
        and compatibility_error <= thresholds.compatibility_linf
        and physical_roundtrip <= thresholds.physical_roundtrip_linf
        and storage_roundtrip <= thresholds.transformed_storage_roundtrip_linf
        and boundary_zero
        and diagnostics.min_density > 0.0
        and diagnostics.min_temperature > 0.0
        and conservation_passed
        and comparison_passed
        and rank == unknown_count
    )

    record = {
        "status": (
            "R26_GU_EMERSON_JFM_N28_SAME_GRID_GATE_PASSED"
            if passed
            else "R26_GU_EMERSON_JFM_N28_SAME_GRID_GATE_FAILED"
        ),
        "source_commit": source_commit,
        "case": asdict(case),
        "reference": {
            "case_name": JFM_N28_REFERENCE_CASE,
            "state_sha256": reference_digest,
            "required_state_sha256": expected_reference_sha256,
            "independent_raw_gate": reference_raw,
            "diagnostics": asdict(reference_evaluation.diagnostics),
            "global_balances": reference_balance,
        },
        "candidate": {
            "state_sha256": state_sha256(candidate),
            "independent_raw_gate": candidate_raw,
            "transformed_interior_linf": transformed_linf,
            "transformed_vs_compatible_physical_linf": compatibility_error,
            "transformed_boundary_rows_exactly_zero": boundary_zero,
            "physical_roundtrip_linf": physical_roundtrip,
            "transformed_storage_roundtrip_linf": storage_roundtrip,
            "diagnostics": asdict(candidate_evaluation.diagnostics),
            "global_balances": candidate_balance,
        },
        "structural_rank": {
            "unknown_count": unknown_count,
            "rank": rank,
            "full_rank": rank == unknown_count,
            "jacobian_pattern_nonzeros": int(pattern.nnz),
        },
        "same_grid_comparison": comparison,
        "thresholds": asdict(thresholds),
        "conservation_passed": conservation_passed,
        "comparison_passed": comparison_passed,
        "same_grid_gate_passed": passed,
        "same_grid_only": True,
        "higher_grid_run_attempted": False,
        "candidate_accepted": passed,
        "production_accepted": False,
        "n32_authorized": passed,
        "n40_authorized": False,
        "n44_authorized": False,
        "next_required_stage": (
            "run one bounded JFM-Maxwell transformed N32 candidate and gate it"
            if passed
            else "stop; inspect the failed N28 same-grid metric before any higher grid"
        ),
    }
    return JFMSameGridGateResult(
        record=jsonable(record),
        candidate_state=candidate,
        transformed_state=packed,
    )


__all__ = [
    "JFM_N28_REFERENCE_CASE",
    "JFM_N28_REFERENCE_STATE_SHA256",
    "JFMSameGridGateResult",
    "JFMSameGridThresholds",
    "jsonable",
    "require_jfm_n28_case",
    "run_jfm_n28_same_grid_gate",
]
