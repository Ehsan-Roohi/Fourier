from __future__ import annotations

from types import SimpleNamespace

from r26_raw_continuation import next_continuation_step, strict_raw_acceptance


def _diagnostics(*, raw_total: float = 5.0e-9) -> SimpleNamespace:
    return SimpleNamespace(
        # A tiny scaled residual is intentionally present to prove that it is
        # irrelevant to the strict gate.
        total_linf=1.0e-14,
        raw_total_linf=raw_total,
        raw_bulk_linf=raw_total,
        raw_wall_linf=2.0e-9,
        raw_extrapolation_linf=3.0e-9,
        raw_corner_linf=1.0e-12,
        held_out_continuity=4.0e-10,
        mass_error=2.0e-14,
        min_density=0.9,
        min_temperature=0.8,
    )


def _decision(diagnostics: SimpleNamespace):
    return strict_raw_acceptance(
        diagnostics,
        {"interior_velocity_linf_over_lid": 0.17},
        optimizer_success=True,
        jacobian_rank=425,
        unknown_count=425,
        equation_count=426,
        raw_tolerance=1.0e-7,
        held_tolerance=1.0e-7,
        mass_tolerance=1.0e-10,
        minimum_response_ratio=1.0e-4,
    )


def test_scaled_only_small_residual_can_never_pass_raw_acceptance() -> None:
    decision = _decision(_diagnostics(raw_total=2.0e-5))
    assert not decision.accepted
    assert "raw_total_linf" in decision.failed_checks
    assert "raw_bulk_linf" in decision.failed_checks


def test_raw_gate_requires_all_physical_plus_mass_and_full_column_rank() -> None:
    diagnostics = _diagnostics()
    accepted = _decision(diagnostics)
    assert accepted.accepted

    wrong_rows = strict_raw_acceptance(
        diagnostics,
        {"interior_velocity_linf_over_lid": 0.17},
        optimizer_success=True,
        jacobian_rank=425,
        unknown_count=425,
        equation_count=425,
        raw_tolerance=1.0e-7,
        held_tolerance=1.0e-7,
        mass_tolerance=1.0e-10,
        minimum_response_ratio=1.0e-4,
    )
    assert not wrong_rows.accepted
    assert "all_physical_plus_mass_equation_count" in wrong_rows.failed_checks

    rank_deficient = strict_raw_acceptance(
        diagnostics,
        {"interior_velocity_linf_over_lid": 0.17},
        optimizer_success=True,
        jacobian_rank=424,
        unknown_count=425,
        equation_count=426,
        raw_tolerance=1.0e-7,
        held_tolerance=1.0e-7,
        mass_tolerance=1.0e-10,
        minimum_response_ratio=1.0e-4,
    )
    assert not rank_deficient.accepted
    assert "full_column_rank" in rank_deficient.failed_checks


def test_adaptive_step_halves_rejections_and_grows_acceptances() -> None:
    rejected = next_continuation_step(
        attempted_step=0.01,
        accepted=False,
        growth_factor=1.5,
        minimum_step=1.0e-4,
        maximum_step=0.02,
    )
    accepted = next_continuation_step(
        attempted_step=rejected,
        accepted=True,
        growth_factor=1.5,
        minimum_step=1.0e-4,
        maximum_step=0.02,
    )
    assert rejected == 0.005
    assert accepted == 0.0075
