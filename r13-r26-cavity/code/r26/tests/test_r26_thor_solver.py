from __future__ import annotations

import numpy as np

from r26_cases import gu_asme2009_cavity_case, rana_first_case
from r26_fv_backend import cubista_face_value, thor_fv_bulk_residual
from r26_solver import LogStateTransform
from r26_thor_solver import (
    SimpleR26Preconditioner,
    ThorSolveOptions,
    make_thor_problem,
    solve_r26_thor_bvp,
)


def test_cubista_uses_all_three_normalized_variable_branches() -> None:
    expected = {0.2: 0.35, 0.5: 0.75, 0.9: 0.975}
    for upwind, face_value in expected.items():
        field = np.asarray(((0.0, upwind, 1.0, 1.2, 1.4),))
        result = cubista_face_value(field, np.ones((1, 4)), axis=1)
        assert result[0, 0] == field[0, 0]
        assert np.isclose(result[0, 1], face_value, rtol=0.0, atol=2.0e-15)
        assert np.all(result >= np.minimum(field[:, :-1], field[:, 1:]))
        assert np.all(result <= np.maximum(field[:, :-1], field[:, 1:]))


def test_cubista_is_directional_and_tensor_shape_safe() -> None:
    scalar = np.asarray(((1.4, 1.2, 1.0, 0.5, 0.0),))
    tensor = np.stack((scalar, -2.0 * scalar), axis=-1)
    flux = -np.ones((1, 4))
    scalar_faces = cubista_face_value(scalar, flux, axis=1)
    tensor_faces = cubista_face_value(tensor, flux, axis=1)
    np.testing.assert_allclose(tensor_faces[..., 0], scalar_faces)
    np.testing.assert_allclose(tensor_faces[..., 1], -2.0 * scalar_faces)
    assert scalar_faces[0, -1] == scalar[0, -1]


def test_thor_cubista_bulk_preserves_exact_equilibrium() -> None:
    case = gu_asme2009_cavity_case(5, kn=0.2, lid_speed_m_per_s=100.0)
    state = case.equilibrium_state()
    residual = thor_fv_bulk_residual(
        state,
        case.x,
        case.y,
        case.mu(state[..., 3]),
        case=case,
    )
    assert np.array_equal(residual, np.zeros_like(state))


def test_simple_preconditioner_honours_the_independent_mass_rhs() -> None:
    case = rana_first_case(5).with_lid_velocity(0.0)
    state = case.equilibrium_state()
    problem = make_thor_problem(case)
    transform = LogStateTransform(problem.shape)
    preconditioner = SimpleR26Preconditioner(
        problem,
        transform,
        transform.encode(state),
        ThorSolveOptions(),
    )
    scaled_rhs = np.zeros(problem.shape)
    requested_mass_change = 0.125
    scaled_rhs[problem.mass_j, problem.mass_i, 0] = requested_mass_change
    encoded_correction = (preconditioner @ scaled_rhs.ravel()).reshape(problem.shape)
    physical_density_correction = state[..., 0] * encoded_correction[..., 0]
    assert np.isclose(
        np.sum(problem.mass_weights * physical_density_correction),
        requested_mass_change,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_stationary_equilibrium_exits_before_building_a_frozen_jacobian() -> None:
    case = rana_first_case(5).with_lid_velocity(0.0)
    result = solve_r26_thor_bvp(make_thor_problem(case), case.equilibrium_state())
    assert result.solution.converged
    assert result.raw_acceptance_gate == 0.0
    assert result.frozen_jacobian_residual_evaluations == 0
    assert result.frozen_jacobian_nonzeros == 0
    assert not result.ilu_available


def test_thor_controls_are_bounded() -> None:
    for kwargs in (
        {"pressure_relaxation": 0.0},
        {"moment_relaxation": 1.01},
        {"ilu_drop_tolerance": 0.0},
        {"ilu_fill_factor": 0.5},
    ):
        try:
            ThorSolveOptions(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid THOR controls were accepted: {kwargs}")
