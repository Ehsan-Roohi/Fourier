from __future__ import annotations

import numpy as np

from r26_state import planar_state_to_tensors, rotate_tensors, tensors_to_planar_state
from r26_tensor_closures import R26Closures, stf3_project, stf4_project
from r26_wall_conditions import (
    ProjectedClosures,
    WALL_EQUATION_ORDER,
    WallFrame,
    WallFreeQuantities,
    WallParameters,
    WallUnknowns,
    coupled_wall_residual,
    effective_pressure,
    extract_face_quantities,
    extrapolate_face_free_quantities,
    free_extrapolation_values,
    project_closures,
    reconstruct_wall_tensors,
    smooth_wall_residual_from_tensors,
    solve_wall_face,
    square_wall_frame,
    wall_residual,
)


def _zero_closures() -> R26Closures:
    return R26Closures(phi=np.zeros((3, 3, 3, 3)), psi=np.zeros((3, 3, 3)), Omega=np.zeros(3))


def _equilibrium_state() -> np.ndarray:
    state = np.zeros(17)
    state[0] = 1.0
    state[3] = 1.0
    return state


def _planar_random_closures(seed: int = 260932) -> R26Closures:
    rng = np.random.default_rng(seed)
    raw4 = rng.normal(scale=0.004, size=(3, 3, 3, 3))
    raw3 = rng.normal(scale=0.004, size=(3, 3, 3))
    for index in np.ndindex(raw4.shape):
        if sum(component == 2 for component in index) % 2:
            raw4[index] = 0.0
    for index in np.ndindex(raw3.shape):
        if sum(component == 2 for component in index) % 2:
            raw3[index] = 0.0
    return R26Closures(
        phi=stf4_project(raw4),
        psi=stf3_project(raw3),
        Omega=np.asarray((0.003, -0.002, 0.0)),
    )


def _rotate_closures(closure: R26Closures, q: np.ndarray) -> R26Closures:
    return R26Closures(
        phi=np.einsum("ai,bj,ck,dl,ijkl->abcd", q, q, q, q, closure.phi),
        psi=np.einsum("ai,bj,ck,ijk->abc", q, q, q, closure.psi),
        Omega=np.einsum("ai,i->a", q, closure.Omega),
    )


def test_equilibrium_stationary_diffuse_wall_is_exact_for_all_four_normals() -> None:
    state = _equilibrium_state()
    closure = _zero_closures()
    for side in ("left", "right", "bottom", "top"):
        frame = square_wall_frame(side)
        residual = wall_residual(
            state,
            closure,
            frame.normal,
            frame.tangent,
            np.zeros(3),
            1.0,
            alpha=1.0,
        )
        assert residual.shape == (11,)
        assert np.array_equal(residual, np.zeros(11))
    assert WALL_EQUATION_ORDER[0] == "no_penetration"
    assert WALL_EQUATION_ORDER[-1] == "C8_Delta"


def test_nonunit_gas_constant_preserves_bulk_theta_equals_RT_convention() -> None:
    gas_constant = 3.0
    temperature = 2.0
    state = np.zeros(17)
    state[0] = 1.7
    state[3] = gas_constant * temperature
    closure = _zero_closures()
    for side in ("left", "right", "bottom", "top"):
        frame = square_wall_frame(side)
        residual = wall_residual(
            state,
            closure,
            frame.normal,
            frame.tangent,
            np.zeros(3),
            temperature,
            alpha=1.0,
            gas_constant=gas_constant,
        )
        assert np.array_equal(residual, np.zeros(11))

        free, unknowns = extract_face_quantities(
            planar_state_to_tensors(state),
            frame,
            gas_constant=gas_constant,
        )
        assert free.pressure == state[0] * state[3]
        assert unknowns.temperature == temperature
        rebuilt = reconstruct_wall_tensors(
            free,
            unknowns,
            frame,
            WallParameters(
                wall_temperature=temperature,
                gas_constant=gas_constant,
            ),
        )
        assert float(rebuilt.theta) == state[3]


def test_square_wall_frames_apply_all_inward_normal_signs_to_free_moments() -> None:
    state = np.asarray(
        (
            1.0,
            0.1,
            0.2,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
        )
    )
    expected = {
        "left": ((5.0, 2.0, 10.0, 12.0, 8.0), (0.2, 6.0, 4.0, 3.0, 13.0, 11.0, 9.0, 7.0)),
        "right": ((-5.0, -2.0, -10.0, -12.0, -8.0), (0.2, 6.0, 4.0, 3.0, 13.0, 11.0, 9.0, 7.0)),
        "bottom": ((5.0, 3.0, 13.0, 11.0, 8.0), (0.1, 4.0, 6.0, 2.0, 10.0, 12.0, 7.0, 9.0)),
        "top": ((-5.0, -3.0, -13.0, -11.0, -8.0), (0.1, 4.0, 6.0, 2.0, 10.0, 12.0, 7.0, 9.0)),
    }
    tensors = planar_state_to_tensors(state)
    for side, (free_expected, unknown_expected) in expected.items():
        free, unknowns = extract_face_quantities(tensors, square_wall_frame(side))
        assert np.array_equal(free.as_array()[1:], np.asarray(free_expected))
        actual_unknowns = np.asarray(
            (
                unknowns.u_t,
                unknowns.sigma_tt,
                unknowns.sigma_nn,
                unknowns.q_t,
                unknowns.m_ttt,
                unknowns.m_nnt,
                unknowns.R_tt,
                unknowns.R_nn,
            )
        )
        assert np.array_equal(actual_unknowns, np.asarray(unknown_expected))


def test_local_solver_recovers_equilibrium_and_positive_palpha() -> None:
    free = WallFreeQuantities(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    parameters = WallParameters(wall_temperature=1.0, accommodation=1.0)
    for side in ("left", "right", "bottom", "top"):
        result = solve_wall_face(free, ProjectedClosures.zeros(), square_wall_frame(side), parameters)
        assert result.success, result.message
        assert result.effective_pressure > 0.0
        assert np.max(np.abs(result.residual)) < 2.0e-12
        assert np.allclose(result.planar_state, _equilibrium_state(), rtol=0.0, atol=2.0e-12)


def test_reconstruction_round_trip_preserves_six_free_and_ten_unknown_values() -> None:
    frame = square_wall_frame("top")
    free = WallFreeQuantities(1.14, -0.017, 0.013, -0.009, 0.006, 0.011)
    unknowns = WallUnknowns(
        u_t=0.12,
        temperature=0.94,
        sigma_tt=0.021,
        sigma_nn=-0.018,
        q_t=-0.014,
        m_ttt=0.008,
        m_nnt=-0.005,
        R_tt=0.019,
        R_nn=-0.016,
        Delta=0.027,
    )
    parameters = WallParameters(
        wall_temperature=1.0,
        gas_constant=1.0,
        wall_velocity=np.asarray((0.2, 0.0, 0.0)),
    )
    tensors = reconstruct_wall_tensors(free, unknowns, frame, parameters)
    packed = tensors_to_planar_state(tensors)
    recovered_free, recovered_unknowns = extract_face_quantities(
        planar_state_to_tensors(packed), frame, gas_constant=1.0
    )
    assert np.allclose(recovered_free.as_array(), free.as_array(), rtol=0.0, atol=2.0e-14)
    assert np.isclose(recovered_unknowns.temperature, unknowns.temperature, atol=2.0e-14)
    for name in ("sigma_tt", "sigma_nn", "q_t", "m_ttt", "m_nnt", "R_tt", "R_nn", "Delta"):
        assert np.isclose(getattr(recovered_unknowns, name), getattr(unknowns, name), atol=3.0e-14)
    # extract_face_quantities reports absolute velocity; the wall residual
    # subtracts the prescribed wall speed before using equation (32).
    assert np.isclose(recovered_unknowns.u_t, unknowns.u_t + 0.2, atol=2.0e-14)


def test_mild_manufactured_face_solve_satisfies_all_ten_coupled_equations() -> None:
    frame = WallFrame(np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 1.0, 0.0)))
    full = _planar_random_closures()
    projected = project_closures(full, frame)
    free = WallFreeQuantities(
        pressure=1.03,
        sigma_nt=0.006,
        q_n=-0.004,
        m_nnn=0.0015,
        m_ntt=-0.001,
        R_nt=0.002,
    )
    parameters = WallParameters(
        wall_temperature=0.98,
        accommodation=0.83,
        wall_velocity=np.asarray((0.0, 0.025, 0.0)),
    )
    result = solve_wall_face(free, projected, frame, parameters, tolerance=3.0e-12)
    assert result.success, (result.message, result.scaled_residual)
    assert np.isfinite(result.planar_state).all()
    assert result.planar_state[0] > 0.0 and result.planar_state[3] > 0.0
    assert result.effective_pressure > 0.0
    assert np.max(np.abs(result.scaled_residual)) < 1.0e-9
    full_residual = smooth_wall_residual_from_tensors(result.state, full, frame, parameters)
    assert abs(full_residual[0]) < 3.0e-14
    assert np.max(np.abs(full_residual[1:] / np.maximum(1.0, np.abs(result.residual)))) < 2.0e-9


def test_rotation_of_face_state_and_closures_preserves_all_wall_residuals() -> None:
    frame = WallFrame(np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 1.0, 0.0)))
    closure = _planar_random_closures(9901)
    projected = project_closures(closure, frame)
    free = WallFreeQuantities(1.0, 0.004, -0.003, 0.001, -0.0012, 0.0018)
    parameters = WallParameters(
        wall_temperature=1.02,
        accommodation=1.0,
        wall_velocity=np.asarray((0.0, 0.03, 0.0)),
    )
    solved = solve_wall_face(free, projected, frame, parameters)
    assert solved.success
    residual = smooth_wall_residual_from_tensors(solved.state, closure, frame, parameters)

    transformations = (
        np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        np.diag((-1.0, 1.0, 1.0)),
        np.diag((1.0, -1.0, 1.0)),
    )
    for q in transformations:
        rotated_frame = WallFrame(q @ frame.normal, q @ frame.tangent)
        rotated_parameters = WallParameters(
            wall_temperature=parameters.wall_temperature,
            accommodation=parameters.accommodation,
            wall_velocity=q @ parameters.wall_velocity,
        )
        rotated_residual = smooth_wall_residual_from_tensors(
            rotate_tensors(solved.state, q),
            _rotate_closures(closure, q),
            rotated_frame,
            rotated_parameters,
        )
        assert np.allclose(rotated_residual, residual, rtol=2.0e-11, atol=2.0e-12)


def test_diffuse_alpha_one_half_range_terms_have_the_printed_signs() -> None:
    sq = np.sqrt(np.pi / 2.0)
    parameters = WallParameters(wall_temperature=1.0, accommodation=1.0)
    unknowns = WallUnknowns(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    shear = coupled_wall_residual(
        unknowns,
        WallFreeQuantities(1.0, 0.01, 0.0, 0.0, 0.0, 0.0),
        ProjectedClosures.zeros(),
        parameters,
    )
    assert np.isclose(shear[0], 0.01 * sq, rtol=0.0, atol=2.0e-15)
    assert np.isclose(shear[4], 0.01 * (35.0 / 18.0) * sq, rtol=0.0, atol=2.0e-15)
    assert shear[5] > 0.0 and shear[6] > 0.0

    normal_heat = coupled_wall_residual(
        unknowns,
        WallFreeQuantities(1.0, 0.0, 0.01, 0.0, 0.0, 0.0),
        ProjectedClosures.zeros(),
        parameters,
    )
    for index in (1, 2, 3, 7, 8, 9):
        assert normal_heat[index] > 0.0


def test_face_extrapolation_is_separate_and_uses_cell_centred_linear_formula() -> None:
    near = _equilibrium_state()
    farther = _equilibrium_state()
    near[0] = 1.1
    farther[0] = 0.9
    near[7], farther[7] = 0.04, 0.02
    frame = square_wall_frame("left")
    extracted = free_extrapolation_values(near, frame.normal, frame.tangent)
    extrapolated = extrapolate_face_free_quantities(near, farther, frame)
    assert np.isclose(extracted[0], 1.1)
    assert np.isclose(extrapolated.pressure, 1.2)
    assert np.isclose(extrapolated.sigma_nt, 0.05)


def test_nonpositive_effective_pressure_is_rejected_before_division() -> None:
    free = WallFreeQuantities(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    parameters = WallParameters(wall_temperature=1.0)
    invalid = WallUnknowns(0.0, 1.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert effective_pressure(free, invalid, ProjectedClosures.zeros(), parameters) < 0.0
    try:
        coupled_wall_residual(invalid, free, ProjectedClosures.zeros(), parameters)
    except FloatingPointError as exc:
        assert "p_alpha" in str(exc)
    else:
        raise AssertionError("negative p_alpha was not rejected")


def test_effective_pressure_is_literal_equation_34() -> None:
    free = WallFreeQuantities(1.7, 0.0, 0.0, 0.0, 0.0, 0.0)
    unknowns = WallUnknowns(0.0, 2.0, 0.0, 0.13, 0.0, 0.0, 0.0, 0.0, -0.21, 0.17)
    closure = ProjectedClosures(0.09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    parameters = WallParameters(wall_temperature=2.0, gas_constant=3.0)
    expected = 1.7 + 0.13 / 2.0 - (30.0 * -0.21 + 7.0 * 0.17) / (840.0 * 6.0) - 0.09 / (24.0 * 6.0)
    assert np.isclose(
        effective_pressure(free, unknowns, closure, parameters), expected, rtol=0.0, atol=2.0e-16
    )
