from __future__ import annotations

from dataclasses import replace

import numpy as np

from r26_cases import (
    GU_ASME2009_CAVITY_CONTRACT,
    gu_asme2009_cavity_case,
    gu_asme2009_published_cavity_case,
)
from r26_discretization import R26NodeBVP
from r26_gu_emerson_reconstruction import (
    GuEmersonReconstructionOptions,
    _SegregatedReconstructionOperators,
    make_gu_emerson_reconstruction_problem,
    solve_gu_emerson_reconstruction,
)
from r26_gu_emerson_transformed_fv import (
    equation63_gamma_by_slot,
    gu_emerson_equation63_consistency,
    gu_emerson_equation63_picard_data,
    gu_emerson_equation63_picard_residual,
    gu_emerson_equation63_terms,
)
from r26_gu_emerson_variables import gu_emerson_fields_from_state


def _smooth_asme_fields(nodes: int = 7):
    case = gu_asme2009_cavity_case(
        nodes,
        kn=0.1,
        lid_speed_m_per_s=10.0,
        gas_constant_si=208.0,
        wall_temperature_K=273.0,
        grid_stretch_beta=0.0,
    )
    y, x = np.meshgrid(case.y, case.x, indexing="ij")
    state = case.equilibrium_state()
    envelope = np.sin(np.pi * x) * np.sin(np.pi * y)
    state[..., 0] += 3.0e-3 * envelope
    state[..., 1] = 2.0e-3 * envelope
    state[..., 2] = -1.0e-3 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)
    state[..., 3] += 2.0e-3 * np.sin(np.pi * x) * np.sin(2.0 * np.pi * y)
    for slot in range(4, 17):
        state[..., slot] = 2.0e-5 / (slot + 1.0) * envelope
    fields = gu_emerson_fields_from_state(
        state, x=case.x, y=case.y, mu=case.mu(state[..., 3])
    )
    return case, fields


def test_asme2009_published_case_is_literal_and_off_paper_values_fail() -> None:
    contract = GU_ASME2009_CAVITY_CONTRACT
    assert contract.reference_temperature_K == 273.0
    assert contract.reference_viscosity_Pa_s == 21.25e-6
    assert contract.sutherland_temperature_K == 144.0
    assert contract.gas_constant_J_kg_K == 208.0
    assert contract.accommodation == 1.0
    assert contract.published_grid_points == 100
    assert "ASME paper leaves L symbolic" in contract.length_provenance
    case = gu_asme2009_published_cavity_case(kn=0.2, lid_speed_m_per_s=100.0)
    assert case.nodes == 100
    assert case.r26_closure_mode == "asme2009-cavity"
    assert case.wall_temperature == 1.0
    assert case.grid_stretch_beta == 0.0
    assert np.isclose(case.lid_velocity, 100.0 / np.sqrt(208.0 * 273.0))
    for arguments in (
        {"kn": 0.3, "lid_speed_m_per_s": 100.0},
        {"kn": 0.2, "lid_speed_m_per_s": 50.0},
    ):
        try:
            gu_asme2009_published_cavity_case(**arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("off-paper ASME case must be rejected")


def test_asme_equation63_uses_every_printed_diffusion_multiplier() -> None:
    gamma = equation63_gamma_by_slot("asme2009-cavity")
    assert np.isinf(gamma[0])
    np.testing.assert_allclose(gamma[1:3], 1.0, rtol=0.0, atol=0.0)
    assert gamma[3] == 2.0 / 5.0
    np.testing.assert_allclose(gamma[4:6], 5.0 / 6.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(gamma[6:9], 3.0 / 2.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(gamma[9:12], 7.0 * 1.82 / 9.0)
    np.testing.assert_allclose(gamma[12:16], 2.097)
    assert gamma[16] == 3.0 / 7.0


def test_equation63_source_identity_is_exact_for_all_transformed_rows() -> None:
    case, fields = _smooth_asme_fields()
    terms = gu_emerson_equation63_terms(fields, case=case)
    reconstructed = terms.central_point_lhs - terms.source
    np.testing.assert_allclose(
        reconstructed,
        terms.physical_point_residual,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert np.max(np.abs(terms.source[..., 1:])) > 0.0
    assert np.max(np.abs(terms.finite_volume_lhs[1:-1, 1:-1, 1:])) > 0.0
    assert np.array_equal(terms.residual[0], np.zeros_like(terms.residual[0]))
    assert np.array_equal(terms.residual[-1], np.zeros_like(terms.residual[-1]))
    consistency = gu_emerson_equation63_consistency(terms)
    assert consistency.identity_roundoff < 2.0e-15
    assert consistency.physical_point_linf > 0.0
    assert consistency.transport_discretization_linf > 0.0
    assert 0 <= consistency.physical_point_argmax_slot < 17
    assert 0 <= consistency.transport_discretization_argmax_slot < 17


def test_picard_stage_matches_nonlinear_residual_at_freeze_point() -> None:
    case, fields = _smooth_asme_fields()
    terms = gu_emerson_equation63_terms(fields, case=case)
    frozen = gu_emerson_equation63_picard_data(fields, case=case)
    residual = gu_emerson_equation63_picard_residual(
        fields, case=case, frozen=frozen
    )
    np.testing.assert_allclose(residual, terms.residual, rtol=0.0, atol=2.0e-15)


def test_picard_stage_holds_source_and_transport_data_fixed() -> None:
    case, fields = _smooth_asme_fields()
    frozen = gu_emerson_equation63_picard_data(fields, case=case)
    source_before = frozen.explicit_source.copy()
    sink_before = frozen.implicit_sink_by_slot.copy()
    mass_x_before = frozen.mass_x.copy()
    mass_y_before = frozen.mass_y.copy()
    changed_velocity = np.asarray(fields.velocity).copy()
    changed_velocity[2:-2, 2:-2, 0] += 1.0e-4
    changed = replace(fields, velocity=changed_velocity)
    residual = gu_emerson_equation63_picard_residual(
        changed, case=case, frozen=frozen
    )
    assert np.isfinite(residual).all()
    assert np.array_equal(frozen.explicit_source, source_before)
    assert np.array_equal(frozen.implicit_sink_by_slot, sink_before)
    assert np.array_equal(frozen.mass_x, mass_x_before)
    assert np.array_equal(frozen.mass_y, mass_y_before)
    assert np.max(np.abs(residual[..., 1:3])) > 0.0


def test_printed_collision_sinks_are_implicit_only_for_high_moments() -> None:
    case, fields = _smooth_asme_fields()
    frozen = gu_emerson_equation63_picard_data(fields, case=case)
    sink = frozen.implicit_sink_by_slot
    assert np.array_equal(sink[..., :4], np.zeros_like(sink[..., :4]))
    assert np.all(sink[..., 4:17] > 0.0)
    ratio = sink / np.where(sink[..., 6:9].mean(axis=-1, keepdims=True) > 0.0,
                            sink[..., 6:9].mean(axis=-1, keepdims=True), 1.0)
    np.testing.assert_allclose(ratio[..., 4:6], 2.0 / 3.0)
    np.testing.assert_allclose(ratio[..., 6:9], 1.0)
    np.testing.assert_allclose(ratio[..., 9:12], 7.0 / 6.0)
    np.testing.assert_allclose(ratio[..., 12:16], 3.0 / 2.0)
    np.testing.assert_allclose(ratio[..., 16], 2.0 / 3.0)


def test_equation63_equilibrium_is_an_exact_conservative_fixed_point() -> None:
    case = gu_asme2009_cavity_case(8, kn=0.1, lid_speed_m_per_s=10.0)
    state = case.equilibrium_state()
    fields = gu_emerson_fields_from_state(
        state, x=case.x, y=case.y, mu=case.mu(state[..., 3])
    )
    terms = gu_emerson_equation63_terms(fields, case=case)
    assert np.max(np.abs(terms.residual)) < 5.0e-14
    weights = make_gu_emerson_reconstruction_problem(case).mass_weights
    assert abs(float(np.sum(weights * terms.residual[..., 0]))) < 2.0e-15


def test_direct_equation63_stage_does_not_call_physical_fv_defect() -> None:
    case, fields = _smooth_asme_fields(5)

    def forbidden_bulk(*args, **kwargs):
        raise AssertionError("physical finite-volume defect was called")

    problem = R26NodeBVP(case, bulk_operator=forbidden_bulk)
    options = GuEmersonReconstructionOptions.asme2009_equation63_source_backed(
        max_outer_iterations=1
    )
    operators = _SegregatedReconstructionOperators(problem, options)
    residual = operators._bulk_stage_residual(fields, (1, 2))
    assert residual.shape == (2 * (case.nodes - 2) ** 2,)
    assert np.isfinite(residual).all()


def test_direct_equation63_rejects_rana_source_history_mixing() -> None:
    try:
        GuEmersonReconstructionOptions(
            equation_backend="equation63-transformed-fv",
            use_rana_source_history=True,
        )
    except ValueError as exc:
        assert "cannot be replaced" in str(exc)
    else:
        raise AssertionError("Rana source history must not enter direct Eq. (63)")


def test_direct_equation63_executes_the_complete_published_order() -> None:
    case = replace(
        gu_asme2009_cavity_case(5, kn=0.1, lid_speed_m_per_s=10.0),
        lid_velocity=0.0,
    )
    result = solve_gu_emerson_reconstruction(
        make_gu_emerson_reconstruction_problem(case),
        case.equilibrium_state(),
        options=GuEmersonReconstructionOptions.asme2009_equation63_source_backed(
            max_outer_iterations=1
        ),
    )
    assert result.converged
    assert result.records[0].raw_gate == 0.0
    assert result.records[0].transformed_equation63_linf < 5.0e-14
    assert result.records[0].physical_point_linf < 5.0e-14
    assert result.records[0].transport_discretization_linf < 5.0e-14
    assert result.records[0].equation63_identity_roundoff < 5.0e-14
    assert result.records[0].stage_order == (
        "velocity",
        "simple_pressure_correction",
        "temperature",
        "g",
        "h",
        "omega",
        "gamma",
        "chi",
        "physical_moment_reconstruction",
        "wall_boundary_update",
    )
