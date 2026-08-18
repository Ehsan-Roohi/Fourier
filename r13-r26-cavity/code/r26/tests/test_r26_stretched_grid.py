from __future__ import annotations

import numpy as np

from r26_bulk_equations import bulk_residual_grid
from r26_cases import (
    KnudsenConvention,
    SQRT_2_OVER_PI,
    jfm_observability_cavity_case,
    rana_first_case,
    rana_john_case,
)
from r26_discretization import R26NodeBVP, linear_wall_extrapolation
from r26_face_collocation import R26FaceCollocationBVP
from r26_fv_backend import compatible_face_fields, compatible_fv_bulk_residual
from r26_spectral_backend import (
    spectral_bulk_residual_grid,
    spectral_gu_emerson_closures,
)
from r26_solver import interpolate_state_grid
from r26_staggered_backend import (
    backward_second_order_bulk_residual_grid,
    backward_second_order_gu_emerson_closures,
    forward_second_order_bulk_residual_grid,
    forward_second_order_gu_emerson_closures,
    make_oriented_second_order_operators,
    oriented_forward_bulk_residual_grid,
    oriented_forward_gu_emerson_closures,
    staggered_bulk_residual_grid,
    staggered_gu_emerson_closures,
)
from r26_state import NVAR, planar_state_to_tensors
from r26_tensor_closures import (
    closure_coefficients,
    closures_from_tensors,
    finite_difference_gradients,
)
from r26_validation import global_balance_diagnostics, leading_r13_nsf_diagnostics


def _nontrivial_state(case) -> np.ndarray:
    state = case.equilibrium_state()
    yy, xx = np.meshgrid(case.y, case.x, indexing="ij")
    for component in range(1, NVAR):
        if component == 3:
            state[..., component] += 0.01 * xx * yy
        else:
            state[..., component] = 0.002 * (component + 1) * (
                xx + 0.3 * yy + 0.2 * xx * yy
            )
    return state


def test_tanh_grid_is_symmetric_monotone_and_uniform_at_zero_beta() -> None:
    uniform = rana_first_case(11, grid_stretch_beta=0.0)
    assert np.array_equal(uniform.x, np.linspace(0.0, 1.0, 11))

    stretched = rana_first_case(11, grid_stretch_beta=2.5)
    assert np.all(np.diff(stretched.x) > 0.0)
    assert np.allclose(stretched.x + stretched.x[::-1], 1.0, rtol=0.0, atol=3.0e-16)
    assert stretched.x[1] - stretched.x[0] < uniform.x[1] - uniform.x[0]
    assert np.array_equal(stretched.x, stretched.y)


def test_rana_john_case_keeps_rana_kn_convention_and_physical_lid_speed() -> None:
    case = rana_john_case(9, kn=0.0798, grid_stretch_beta=2.0)
    assert case.kn_convention is KnudsenConvention.RANA
    assert case.mu_equilibrium == 0.0798
    assert np.isclose(case.lid_velocity, 50.0 / np.sqrt(208.0 * 273.0))
    assert case.r26_closure_mode == "jfm2009"
    preliminary = rana_john_case(
        9,
        kn=0.0798,
        grid_stretch_beta=2.0,
        closure_mode="asme2009-cavity",
    )
    assert preliminary.r26_closure_mode == "asme2009-cavity"
    assert case.viscosity.kind.value == "gu_sutherland"


def test_jfm_observability_case_matches_manuscript_nondimensionalization() -> None:
    case = jfm_observability_cavity_case(9, grid_stretch_beta=1.25)
    assert case.kn == 0.05
    assert case.kn_convention is KnudsenConvention.GU_MEAN_FREE_PATH
    assert np.isclose(case.mu_equilibrium, 0.05 * SQRT_2_OVER_PI)
    assert np.isclose(case.lid_velocity, 100.0 / np.sqrt(208.0 * 300.0))
    assert case.viscosity.kind.value == "power_law"
    assert case.viscosity.exponent == 0.81
    assert case.r26_closure_mode == "jfm2009"


def test_asme_cavity_closure_mode_is_complete_and_source_locked() -> None:
    jfm = closure_coefficients("jfm2009")
    asme = closure_coefficients("asme2009-cavity")
    assert (jfm.C1, jfm.C2, jfm.Y1, jfm.Y2, jfm.Y3) == (
        2.097,
        0.291,
        1.698,
        1.203,
        0.854,
    )
    assert (asme.C1, asme.C2, asme.Y1, asme.Y2, asme.Y3) == (
        2.097,
        -0.291,
        1.82,
        -1.203,
        0.854,
    )

    case = rana_first_case(5)
    state = _nontrivial_state(case)
    tensors = planar_state_to_tensors(state)
    gradients = finite_difference_gradients(state, x=case.x, y=case.y)
    c_jfm = closures_from_tensors(
        tensors, gradients, mu=case.mu(state[..., 3]), coefficient_mode="jfm2009"
    )
    c_asme = closures_from_tensors(
        tensors,
        gradients,
        mu=case.mu(state[..., 3]),
        coefficient_mode="asme2009-cavity",
    )
    assert c_jfm.coefficient_mode == "jfm2009"
    assert c_asme.coefficient_mode == "asme2009-cavity"
    assert np.max(np.abs(c_jfm.phi - c_asme.phi)) > 1.0e-8
    assert np.max(np.abs(c_jfm.psi - c_asme.psi)) > 1.0e-8


def test_complete_closure_mode_propagates_through_every_grid_backend() -> None:
    jfm = rana_john_case(5, kn=0.0798, closure_mode="jfm2009")
    asme = rana_john_case(5, kn=0.0798, closure_mode="asme2009-cavity")
    state = _nontrivial_state(jfm)
    mu = jfm.mu(state[..., 3])

    closure_operators = (
        spectral_gu_emerson_closures,
        staggered_gu_emerson_closures,
        oriented_forward_gu_emerson_closures,
        forward_second_order_gu_emerson_closures,
        backward_second_order_gu_emerson_closures,
    )
    _, factory_closure = make_oriented_second_order_operators("forward", "backward")
    closure_operators += (factory_closure,)
    for operator in closure_operators:
        closure_jfm = operator(state, x=jfm.x, y=jfm.y, mu=mu, case=jfm)
        closure_asme = operator(state, x=asme.x, y=asme.y, mu=mu, case=asme)
        assert closure_jfm.coefficient_mode == "jfm2009"
        assert closure_asme.coefficient_mode == "asme2009-cavity"
        assert np.max(np.abs(closure_jfm.phi - closure_asme.phi)) > 1.0e-8
        assert np.max(np.abs(closure_jfm.psi - closure_asme.psi)) > 1.0e-8

    bulk_operators = (
        bulk_residual_grid,
        compatible_fv_bulk_residual,
        spectral_bulk_residual_grid,
        staggered_bulk_residual_grid,
        oriented_forward_bulk_residual_grid,
        forward_second_order_bulk_residual_grid,
        backward_second_order_bulk_residual_grid,
    )
    factory_bulk, _ = make_oriented_second_order_operators("backward", "forward")
    bulk_operators += (factory_bulk,)
    for operator in bulk_operators:
        residual_jfm = operator(state, jfm.x, jfm.y, mu, case=jfm)
        residual_asme = operator(state, asme.x, asme.y, mu, case=asme)
        assert np.max(np.abs(residual_jfm - residual_asme)) > 1.0e-9

    # The default node BVP must pass the mode into its independently evaluated
    # interior bulk pipeline, not only into the wall-closure grid.
    node_jfm = R26NodeBVP(jfm).evaluate(state).unscaled_residual
    node_asme = R26NodeBVP(asme).evaluate(state).unscaled_residual
    assert np.max(np.abs(node_jfm[1:-1, 1:-1] - node_asme[1:-1, 1:-1])) > 1.0e-9

    # The face-augmented backend constructs five separate closure blocks.
    # All five must use the same case mode.
    face_jfm = R26FaceCollocationBVP(jfm)
    face_asme = R26FaceCollocationBVP(asme)
    face_state = face_jfm.equilibrium_state()
    face_state.cells[...] = state
    residual_jfm = face_jfm.evaluate(face_state).flat
    residual_asme = face_asme.evaluate(face_state).flat
    assert np.max(np.abs(residual_jfm - residual_asme)) > 1.0e-9


def test_nonuniform_wall_extrapolation_is_exact_for_affine_profiles() -> None:
    wall, near, nxt = 0.0, 0.037, 0.121
    slope = np.asarray((2.0, -3.0, 0.25))
    intercept = np.asarray((-0.4, 0.8, 1.2))
    near_value = intercept + slope * near
    next_value = intercept + slope * nxt
    extrapolated = linear_wall_extrapolation(wall, near, nxt, near_value, next_value)
    assert np.allclose(extrapolated, intercept, rtol=0.0, atol=2.0e-16)

    # The same formula must also work when the coordinate points from the
    # right wall toward decreasing x.
    wall, near, nxt = 1.0, 0.963, 0.879
    near_value = intercept + slope * near
    next_value = intercept + slope * nxt
    extrapolated = linear_wall_extrapolation(wall, near, nxt, near_value, next_value)
    assert np.allclose(extrapolated, intercept + slope, rtol=0.0, atol=1.0e-15)


def test_stretched_grid_equilibrium_and_linear_pressure_rhie_chow() -> None:
    case = rana_first_case(9, grid_stretch_beta=2.5)
    equilibrium = case.equilibrium_state()
    mu = case.mu(equilibrium[..., 3])
    residual = compatible_fv_bulk_residual(equilibrium, case.x, case.y, mu)
    assert np.max(np.abs(residual)) < 5.0e-14

    state = equilibrium.copy()
    state[..., 0] = 1.0 + 0.01 * case.x[None, :]
    mu = case.mu(state[..., 3])
    tensors = planar_state_to_tensors(state)
    gradients = finite_difference_gradients(state, x=case.x, y=case.y)
    closures = closures_from_tensors(tensors, gradients, mu=mu)
    faces = compatible_face_fields(state, case.x, case.y, mu, closures)
    assert np.max(np.abs(faces.velocity_x)) < 2.0e-15
    assert np.max(np.abs(faces.velocity_y)) < 2.0e-15


def test_restart_interpolation_uses_physical_stretched_coordinates() -> None:
    old_case = rana_first_case(7, grid_stretch_beta=1.4)
    new_case = rana_first_case(13, grid_stretch_beta=2.5)
    state = old_case.equilibrium_state()
    yy, xx = np.meshgrid(old_case.y, old_case.x, indexing="ij")
    state[..., 1] = 0.7 + 0.2 * xx - 0.3 * yy
    refined = interpolate_state_grid(
        state,
        new_case.nodes,
        old_x=old_case.x,
        old_y=old_case.y,
        new_x=new_case.x,
        new_y=new_case.y,
    )
    yy_new, xx_new = np.meshgrid(new_case.y, new_case.x, indexing="ij")
    assert np.allclose(refined[..., 1], 0.7 + 0.2 * xx_new - 0.3 * yy_new, atol=3.0e-16)
    assert np.allclose(refined[..., 0], np.ones((13, 13)), rtol=0.0, atol=5.0e-16)


def test_independent_global_and_asymptotic_diagnostics_at_equilibrium() -> None:
    case = rana_first_case(9, grid_stretch_beta=2.0)
    state = case.equilibrium_state()
    balances = global_balance_diagnostics(state, case)
    assert balances["momentum_boundary_flux_linf"] == 0.0
    assert balances["internal_energy_balance_error"] == 0.0
    assert balances["wall_effective_pressure_min"] == 1.0
    assert balances["D_smooth_common_cv"] == 0.0

    limit = leading_r13_nsf_diagnostics(state, case)
    for name in (
        "sigma_vs_NSF",
        "q_vs_NSF",
        "m_vs_leading_R13",
        "R_vs_leading_R13",
        "Delta_vs_leading_R13",
    ):
        assert limit[name]["defect_rms"] < 3.0e-16
