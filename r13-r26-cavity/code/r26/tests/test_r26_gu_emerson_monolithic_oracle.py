from dataclasses import replace

import numpy as np

from r26_cases import gu_asme2009_cavity_case
from r26_gu_emerson_monolithic_oracle import (
    EncodedGuEmersonMonolithicObjective,
    GuEmersonColoredNewtonOracleOptions,
    GuEmersonMonolithicOracleOptions,
    GuEmersonPhysicsBlockPreconditioner,
    solve_gu_emerson_colored_newton_oracle,
    solve_gu_emerson_monolithic_oracle,
)
from r26_gu_emerson_reconstruction import make_gu_emerson_reconstruction_problem
from r26_gu_emerson_variables import (
    gu_emerson_fields_as_planar17,
    gu_emerson_fields_from_planar17,
    gu_emerson_fields_from_state,
)
from r26_solver import LogStateTransform


def test_monolithic_objective_is_exact_at_stationary_equilibrium() -> None:
    case = replace(
        gu_asme2009_cavity_case(5, kn=0.1, lid_speed_m_per_s=10.0),
        lid_velocity=0.0,
    )
    state = case.equilibrium_state()
    fields = gu_emerson_fields_from_state(
        state, x=case.x, y=case.y, mu=case.mu(state[..., 3])
    )
    packed = gu_emerson_fields_as_planar17(fields)
    problem = make_gu_emerson_reconstruction_problem(case)
    transform = LogStateTransform(problem.shape)
    objective = EncodedGuEmersonMonolithicObjective(problem, transform, 1.0e8)
    assert np.max(np.abs(objective(transform.encode(packed)))) == 0.0


def test_monolithic_oracle_converges_without_work_at_equilibrium() -> None:
    case = replace(
        gu_asme2009_cavity_case(5, kn=0.1, lid_speed_m_per_s=10.0),
        lid_velocity=0.0,
    )
    state = case.equilibrium_state()
    fields = gu_emerson_fields_from_state(
        state, x=case.x, y=case.y, mu=case.mu(state[..., 3])
    )
    result = solve_gu_emerson_monolithic_oracle(
        make_gu_emerson_reconstruction_problem(case),
        fields,
        options=GuEmersonMonolithicOracleOptions(max_outer_iterations=1),
    )
    assert result.transformed_root_passed
    assert result.complete_physical_gate_passed
    assert result.objective_linf == 0.0

    preconditioned = solve_gu_emerson_monolithic_oracle(
        make_gu_emerson_reconstruction_problem(case),
        fields,
        options=GuEmersonMonolithicOracleOptions(
            max_outer_iterations=1, use_physics_block_preconditioner=True
        ),
    )
    assert preconditioned.objective_linf == 0.0
    assert preconditioned.preconditioner_block_factorizations == 0

    colored = solve_gu_emerson_colored_newton_oracle(
        make_gu_emerson_reconstruction_problem(case),
        fields,
        options=GuEmersonColoredNewtonOracleOptions(max_jacobian_builds=1),
    )
    assert colored.converged
    assert colored.objective_linf == 0.0
    assert colored.jacobian_builds == 0


def test_physics_block_preconditioner_is_finite_and_owns_only_interior_fields() -> None:
    case = replace(
        gu_asme2009_cavity_case(5, kn=0.1, lid_speed_m_per_s=10.0),
        lid_velocity=0.0,
    )
    state = case.equilibrium_state()
    fields = gu_emerson_fields_from_state(
        state, x=case.x, y=case.y, mu=case.mu(state[..., 3])
    )
    packed = gu_emerson_fields_as_planar17(fields)
    for slot in range(1, packed.shape[-1]):
        packed[1:-1, 1:-1, slot] += slot * 1.0e-5
    fields = gu_emerson_fields_from_planar17(packed)
    preconditioner = GuEmersonPhysicsBlockPreconditioner(
        make_gu_emerson_reconstruction_problem(case), fields
    )
    trial = np.linspace(-1.0, 1.0, state.size)
    result = preconditioner @ trial
    assert np.isfinite(result).all()
    assert preconditioner.block_factorizations == 7
    packed_trial = trial.reshape(state.shape)
    packed_result = result.reshape(state.shape)
    np.testing.assert_array_equal(packed_result[..., 0], packed_trial[..., 0])
    np.testing.assert_array_equal(packed_result[0], packed_trial[0])
    np.testing.assert_array_equal(packed_result[-1], packed_trial[-1])
    np.testing.assert_array_equal(packed_result[:, 0], packed_trial[:, 0])
    np.testing.assert_array_equal(packed_result[:, -1], packed_trial[:, -1])


def test_monolithic_oracle_blocks_grids_above_n8() -> None:
    case = gu_asme2009_cavity_case(9, kn=0.1, lid_speed_m_per_s=10.0)
    state = case.equilibrium_state()
    fields = gu_emerson_fields_from_state(
        state, x=case.x, y=case.y, mu=case.mu(state[..., 3])
    )
    try:
        solve_gu_emerson_monolithic_oracle(
            make_gu_emerson_reconstruction_problem(case), fields
        )
    except ValueError as exc:
        assert "restricted to N8" in str(exc)
    else:
        raise AssertionError("oracle must remain blocked above N8")

    try:
        solve_gu_emerson_colored_newton_oracle(
            make_gu_emerson_reconstruction_problem(case), fields
        )
    except ValueError as exc:
        assert "restricted to N8" in str(exc)
    else:
        raise AssertionError("colored Newton oracle must remain blocked above N8")
