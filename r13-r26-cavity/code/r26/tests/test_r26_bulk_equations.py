from __future__ import annotations

from itertools import permutations

import numpy as np

from r26_bulk_equations import (
    R26ClosureDerivatives,
    bulk_residual_grid,
    closure_derivatives_on_grid,
    gu_emerson_nonlinear_sources,
    rotate_bulk_residual,
    rotate_closure_derivatives,
    rotate_closures,
    steady_r26_bulk_residual,
)
from r26_state import StateTensors, planar_state_to_tensors, rotate_tensors
from r26_tensor_closures import (
    R26Closures,
    R26Gradients,
    closures_from_tensors,
    finite_difference_gradients,
    rotate_gradients,
    stf2_project,
    stf3_project,
    stf4_project,
)


def _zero_gradients(leading: tuple[int, ...] = ()) -> R26Gradients:
    return R26Gradients(
        rho=np.zeros(leading + (3,)),
        velocity=np.zeros(leading + (3, 3)),
        theta=np.zeros(leading + (3,)),
        heat_flux=np.zeros(leading + (3, 3)),
        sigma=np.zeros(leading + (3, 3, 3)),
        R=np.zeros(leading + (3, 3, 3)),
        m=np.zeros(leading + (3, 3, 3, 3)),
        Delta=np.zeros(leading + (3,)),
    )


def _zero_closures(leading: tuple[int, ...] = ()) -> R26Closures:
    return R26Closures(
        phi=np.zeros(leading + (3, 3, 3, 3)),
        psi=np.zeros(leading + (3, 3, 3)),
        Omega=np.zeros(leading + (3,)),
    )


def _zero_closure_derivatives(leading: tuple[int, ...] = ()) -> R26ClosureDerivatives:
    return R26ClosureDerivatives(
        div_phi=np.zeros(leading + (3, 3, 3)),
        div_psi=np.zeros(leading + (3, 3)),
        grad_Omega=np.zeros(leading + (3, 3)),
    )


def _equilibrium_point() -> StateTensors:
    return StateTensors(
        rho=np.asarray(1.0),
        velocity=np.zeros(3),
        theta=np.asarray(1.0),
        heat_flux=np.zeros(3),
        sigma=np.zeros((3, 3)),
        R=np.zeros((3, 3)),
        m=np.zeros((3, 3, 3)),
        Delta=np.asarray(0.0),
    )


def _oracle_stf2(raw: np.ndarray) -> np.ndarray:
    out = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            out[i, j] = 0.5 * (raw[i, j] + raw[j, i])
    trace = sum(out[i, i] for i in range(3))
    for i in range(3):
        out[i, i] -= trace / 3.0
    return out


def _oracle_stf3(raw: np.ndarray) -> np.ndarray:
    sym = np.zeros((3, 3, 3), dtype=float)
    orderings = tuple(permutations(range(3)))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                index = (i, j, k)
                sym[i, j, k] = sum(raw[tuple(index[p] for p in order)] for order in orderings) / 6.0
    trace = np.zeros(3)
    for k in range(3):
        trace[k] = sum(sym[i, i, k] for i in range(3))
    out = sym.copy()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                out[i, j, k] -= (
                    (1.0 if i == j else 0.0) * trace[k]
                    + (1.0 if i == k else 0.0) * trace[j]
                    + (1.0 if j == k else 0.0) * trace[i]
                ) / 5.0
    return out


def _random_full_inputs(
    seed: int,
) -> tuple[StateTensors, R26Gradients, R26Closures, R26ClosureDerivatives, float]:
    rng = np.random.default_rng(seed)
    tensors = StateTensors(
        rho=np.asarray(1.1 + 0.15 * rng.random()),
        velocity=0.09 * rng.normal(size=3),
        theta=np.asarray(0.9 + 0.2 * rng.random()),
        heat_flux=0.035 * rng.normal(size=3),
        sigma=0.045 * stf2_project(rng.normal(size=(3, 3))),
        R=0.03 * stf2_project(rng.normal(size=(3, 3))),
        m=0.02 * stf3_project(rng.normal(size=(3, 3, 3))),
        Delta=np.asarray(0.02 * rng.normal()),
    )
    gradients = R26Gradients(
        rho=0.03 * rng.normal(size=3),
        velocity=0.06 * rng.normal(size=(3, 3)),
        theta=0.025 * rng.normal(size=3),
        heat_flux=0.03 * rng.normal(size=(3, 3)),
        sigma=0.035 * stf2_project(rng.normal(size=(3, 3, 3))),
        R=0.025 * stf2_project(rng.normal(size=(3, 3, 3))),
        m=0.018 * stf3_project(rng.normal(size=(3, 3, 3, 3))),
        Delta=0.012 * rng.normal(size=3),
    )
    closures = R26Closures(
        phi=0.016 * stf4_project(rng.normal(size=(3, 3, 3, 3))),
        psi=0.014 * stf3_project(rng.normal(size=(3, 3, 3))),
        Omega=0.013 * rng.normal(size=3),
    )
    derivatives = R26ClosureDerivatives(
        div_phi=0.011 * stf3_project(rng.normal(size=(3, 3, 3))),
        div_psi=0.01 * stf2_project(rng.normal(size=(3, 3))),
        grad_Omega=0.012 * rng.normal(size=(3, 3)),
    )
    return tensors, gradients, closures, derivatives, 0.78 + 0.1 * rng.random()


def _oracle_M_S_N(
    t: StateTensors,
    g: R26Gradients,
    c: R26Closures,
    d: R26ClosureDerivatives,
    mu: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    rho = float(t.rho)
    theta = float(t.theta)
    pressure = rho * theta
    q = np.asarray(t.heat_flux)
    sigma = np.asarray(t.sigma)
    rr = np.asarray(t.R)
    mm = np.asarray(t.m)
    gu = np.asarray(g.velocity)
    gtheta = np.asarray(g.theta)
    gq = np.asarray(g.heat_flux)
    gs = np.asarray(g.sigma)
    gr = np.asarray(g.R)
    gm = np.asarray(g.m)
    phi = np.asarray(c.phi)
    grad_omega = np.asarray(d.grad_Omega)
    gradp = theta * np.asarray(g.rho) + rho * gtheta
    divu = sum(gu[i, i] for i in range(3))
    divq = sum(gq[i, i] for i in range(3))
    divsigma = np.asarray([sum(gs[l, k, l] for l in range(3)) for k in range(3)])
    divm = np.asarray([[sum(gm[k, i, j, k] for k in range(3)) for j in range(3)] for i in range(3)])
    sigma_grad_u = sum(sigma[i, j] * gu[i, j] for i in range(3) for j in range(3))

    m1 = np.zeros((3, 3, 3))
    m2 = np.zeros_like(m1)
    m3 = np.zeros_like(m1)
    m4 = np.zeros_like(m1)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                m1[i, j, k] = sigma[i, j] * divsigma[k] / rho
                m2[i, j, k] = q[i] * gu[k, j]
                m3[i, j, k] = sum(mm[l, i, j] * gu[l, k] for l in range(3))
                m4[i, j, k] = gr[k, i, j]
    source_m = (
        3.0 * _oracle_stf3(m1)
        - 12.0 / 5.0 * _oracle_stf3(m2)
        - 3.0 * _oracle_stf3(m3)
        - 3.0 / 7.0 * _oracle_stf3(m4)
    )

    raw_sigma_square = np.zeros((3, 3))
    raw_q_gradtheta = np.zeros((3, 3))
    raw_q_divsigma = np.zeros((3, 3))
    raw_sigma_deformation = np.zeros((3, 3))
    raw_r_deformation = np.zeros((3, 3))
    phi_deformation = np.zeros((3, 3))
    m_gradtheta = np.zeros((3, 3))
    m_force = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            raw_sigma_square[i, j] = sum(sigma[k, i] * sigma[j, k] for k in range(3))
            raw_q_gradtheta[i, j] = q[i] * gtheta[j]
            raw_q_divsigma[i, j] = q[i] * divsigma[j]
            raw_sigma_deformation[i, j] = sum(
                sigma[k, i] * (gu[j, k] + gu[k, j]) for k in range(3)
            )
            raw_r_deformation[i, j] = (
                6.0 / 7.0 * rr[i, j] * divu
                + 4.0 / 5.0 * sum(rr[k, i] * gu[j, k] for k in range(3))
                + 2.0 * sum(rr[k, i] * gu[k, j] for k in range(3))
            )
            phi_deformation[i, j] = sum(
                phi[i, j, k, l] * gu[l, k] for k in range(3) for l in range(3)
            )
            m_gradtheta[i, j] = sum(mm[i, j, k] * gtheta[k] for k in range(3))
            m_force[i, j] = sum(mm[i, j, k] * (gradp[k] + divsigma[k]) for k in range(3))
    source_s = (
        -2.0 * pressure / (3.0 * mu * rho) * _oracle_stf2(raw_sigma_square)
        - 28.0 / 5.0 * _oracle_stf2(raw_q_gradtheta)
        + 28.0 / (5.0 * rho) * _oracle_stf2(raw_q_divsigma)
        + 14.0 / (3.0 * rho) * sigma * (divq + sigma_grad_u)
        - 4.0 * theta * _oracle_stf2(raw_sigma_deformation)
        + 8.0 / 3.0 * theta * sigma * divu
        - 2.0 * theta * divm
        - 9.0 * m_gradtheta
        - 2.0 * phi_deformation
        + 2.0 / rho * m_force
        - _oracle_stf2(raw_r_deformation)
        - 14.0 / 15.0 * float(t.Delta) * _oracle_stf2(gu)
        - 2.0 / 5.0 * _oracle_stf2(grad_omega)
    )

    sigma_square = sum(sigma[i, j] ** 2 for i in range(3) for j in range(3))
    fourth = sum(
        (2.0 * theta * sigma[i, j] + rr[i, j]) * gu[j, i]
        for i in range(3)
        for j in range(3)
    )
    source_n = (
        -2.0 * pressure / (3.0 * mu * rho) * sigma_square
        - 4.0 * fourth
        + 8.0 / rho * sum(q[i] * divsigma[i] for i in range(3))
        - 20.0 * sum(q[i] * gtheta[i] for i in range(3))
        - 4.0 / 3.0 * float(t.Delta) * divu
    )
    return source_m, source_s, source_n


def test_uniform_equilibrium_has_exactly_zero_point_residual() -> None:
    residual = steady_r26_bulk_residual(
        _equilibrium_point(),
        _zero_gradients(),
        _zero_closures(),
        _zero_closure_derivatives(),
        mu=1.0,
    )
    for name in ("mass", "momentum", "theta", "stress", "heat_flux", "m", "R", "Delta"):
        value = np.asarray(getattr(residual, name))
        assert np.array_equal(value, np.zeros_like(value))


def test_randomized_M_S_N_match_independent_component_loop_oracle() -> None:
    for seed in (2601, 2602, 2603, 2604):
        tensors, gradients, closures, derivatives, mu = _random_full_inputs(seed)
        actual = gu_emerson_nonlinear_sources(
            tensors, gradients, closures, derivatives, mu=mu
        )
        expected_m, expected_s, expected_n = _oracle_M_S_N(
            tensors, gradients, closures, derivatives, mu
        )
        assert np.allclose(actual.M, expected_m, rtol=3.0e-13, atol=3.0e-13)
        assert np.allclose(actual.S, expected_s, rtol=5.0e-13, atol=5.0e-13)
        assert np.allclose(actual.N, expected_n, rtol=3.0e-13, atol=3.0e-13)


def test_equation10_delta_and_m_terms_are_outside_minus_two_fifths_bracket() -> None:
    """Freeze the two easily mis-parenthesized terms in printed Eq. (10)."""

    tensors = _equilibrium_point()
    raw_m = np.zeros((3, 3, 3))
    raw_m[0, 0, 1] = 0.07
    raw_m[0, 1, 1] = -0.04
    m = stf3_project(raw_m)
    tensors = StateTensors(
        rho=tensors.rho,
        velocity=tensors.velocity,
        theta=tensors.theta,
        heat_flux=tensors.heat_flux,
        sigma=tensors.sigma,
        R=tensors.R,
        m=m,
        Delta=tensors.Delta,
    )
    velocity_gradient = np.asarray(
        [[0.03, -0.02, 0.0], [0.05, -0.01, 0.0], [0.0, 0.0, -0.02]]
    )
    delta_gradient = np.asarray((0.12, -0.06, 0.0))
    gradients = _zero_gradients()
    gradients = R26Gradients(
        rho=gradients.rho,
        velocity=velocity_gradient,
        theta=gradients.theta,
        heat_flux=gradients.heat_flux,
        sigma=gradients.sigma,
        R=gradients.R,
        m=gradients.m,
        Delta=delta_gradient,
    )
    actual = gu_emerson_nonlinear_sources(
        tensors,
        gradients,
        _zero_closures(),
        _zero_closure_derivatives(),
        mu=0.8,
    ).Q
    expected = -delta_gradient / 6.0 - np.einsum(
        "ijk,kj->i", m, velocity_gradient
    )
    assert np.allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)


def test_full_bulk_residual_is_rotation_and_reflection_covariant() -> None:
    transforms = (
        np.diag([-1.0, 1.0, 1.0]),
        np.asarray(
            [
                [0.36, -0.48, 0.80],
                [0.80, 0.60, 0.00],
                [-0.48, 0.64, 0.60],
            ]
        ),
    )
    for orthogonal in transforms:
        tensors, gradients, closures, derivatives, mu = _random_full_inputs(2613)
        original = steady_r26_bulk_residual(
            tensors, gradients, closures, derivatives, mu=mu
        )
        transformed = steady_r26_bulk_residual(
            rotate_tensors(tensors, orthogonal),
            rotate_gradients(gradients, orthogonal),
            rotate_closures(closures, orthogonal),
            rotate_closure_derivatives(derivatives, orthogonal),
            mu=mu,
        )
        expected = rotate_bulk_residual(original, orthogonal)
        for name in ("mass", "momentum", "theta", "stress", "heat_flux", "m", "R", "Delta"):
            assert np.allclose(
                getattr(transformed, name),
                getattr(expected, name),
                rtol=2.0e-11,
                atol=8.0e-13,
            ), name


def test_last_eight_planar_rows_are_R_m_Delta_balance_equations() -> None:
    div_psi = np.asarray(
        [[0.31, -0.07, 0.0], [-0.07, -0.18, 0.0], [0.0, 0.0, -0.13]]
    )
    div_phi = np.zeros((3, 3, 3))
    div_phi[0, 0, 0] = 0.21
    for index in set(permutations((0, 0, 1))):
        div_phi[index] = -0.04
    for index in set(permutations((0, 1, 1))):
        div_phi[index] = 0.06
    div_phi[1, 1, 1] = -0.09
    div_phi[0, 2, 2] = -div_phi[0, 0, 0] - div_phi[0, 1, 1]
    for index in set(permutations((0, 2, 2))):
        div_phi[index] = div_phi[0, 2, 2]
    div_phi[1, 2, 2] = -div_phi[0, 0, 1] - div_phi[1, 1, 1]
    for index in set(permutations((1, 2, 2))):
        div_phi[index] = div_phi[1, 2, 2]
    # Isotropic grad(Omega) has nonzero divergence but zero STF part, so this
    # isolates the Delta flux row from the grad(Omega) term in the R balance.
    grad_omega = 0.03 * np.eye(3)
    derivatives = R26ClosureDerivatives(
        div_phi=div_phi,
        div_psi=div_psi,
        grad_Omega=grad_omega,
    )
    packed = steady_r26_bulk_residual(
        _equilibrium_point(),
        _zero_gradients(),
        _zero_closures(),
        derivatives,
        mu=1.0,
    ).as_planar17()
    assert np.array_equal(packed[:9], np.zeros(9))
    expected_last_eight = np.asarray(
        [
            div_psi[0, 0],
            div_psi[0, 1],
            div_psi[1, 1],
            div_phi[0, 0, 0],
            div_phi[0, 0, 1],
            div_phi[0, 1, 1],
            div_phi[1, 1, 1],
            np.trace(grad_omega),
        ]
    )
    assert np.array_equal(packed[9:], expected_last_eight)


def test_grid_api_matches_explicit_derivative_pipeline_and_preserves_2D3V() -> None:
    x = np.linspace(-0.4, 0.6, 7)
    y = np.linspace(-0.3, 0.8, 6)
    xx, yy = np.meshgrid(x, y)
    state = np.zeros((y.size, x.size, 17))
    state[..., 0] = 1.1 + 0.02 * xx - 0.015 * yy
    state[..., 1] = 0.07 * xx * (1.0 - yy)
    state[..., 2] = -0.04 * yy * (1.0 + xx)
    state[..., 3] = 0.95 + 0.025 * xx * yy
    for component in range(4, 17):
        state[..., component] = 0.001 * (component - 2) * np.sin(
            0.2 * component + xx
        ) * np.cos(yy)
    mu = 0.83 * np.sqrt(state[..., 3])
    force = np.zeros((y.size, x.size, 2))
    force[..., 0] = 0.004

    wrapped = bulk_residual_grid(state, x, y, mu=mu, body_force=force)
    tensors = planar_state_to_tensors(state)
    gradients = finite_difference_gradients(state, x=x, y=y)
    closures = closures_from_tensors(tensors, gradients, mu=mu)
    derivatives = closure_derivatives_on_grid(closures, x, y)
    acceleration = np.zeros((y.size, x.size, 3))
    acceleration[..., 0] = force[..., 0] / state[..., 0]
    direct = steady_r26_bulk_residual(
        tensors,
        gradients,
        closures,
        derivatives,
        mu=mu,
        acceleration=acceleration,
    ).as_planar17()
    assert wrapped.shape == state.shape
    assert np.isfinite(wrapped).all()
    assert np.array_equal(wrapped, direct)


def test_closure_grid_derivative_contractions_match_linear_manufactured_fields() -> None:
    rng = np.random.default_rng(2626)
    x = np.linspace(-0.5, 0.8, 6)
    y = np.linspace(-0.3, 0.9, 5)
    xx, yy = np.meshgrid(x, y)
    phi_x = rng.normal(size=(3, 3, 3, 3))
    phi_y = rng.normal(size=(3, 3, 3, 3))
    psi_x = rng.normal(size=(3, 3, 3))
    psi_y = rng.normal(size=(3, 3, 3))
    omega_x = rng.normal(size=3)
    omega_y = rng.normal(size=3)
    closures = R26Closures(
        phi=xx[..., None, None, None, None] * phi_x
        + yy[..., None, None, None, None] * phi_y,
        psi=xx[..., None, None, None] * psi_x + yy[..., None, None, None] * psi_y,
        Omega=xx[..., None] * omega_x + yy[..., None] * omega_y,
    )
    derivatives = closure_derivatives_on_grid(closures, x, y)
    expected_div_phi = phi_x[..., 0] + phi_y[..., 1]
    expected_div_psi = psi_x[..., 0] + psi_y[..., 1]
    assert np.allclose(derivatives.div_phi, expected_div_phi, rtol=0.0, atol=3.0e-14)
    assert np.allclose(derivatives.div_psi, expected_div_psi, rtol=0.0, atol=3.0e-14)
    assert np.allclose(derivatives.grad_Omega[..., 0, :], omega_x, rtol=0.0, atol=3.0e-14)
    assert np.allclose(derivatives.grad_Omega[..., 1, :], omega_y, rtol=0.0, atol=3.0e-14)
    assert np.array_equal(
        derivatives.grad_Omega[..., 2, :],
        np.zeros_like(derivatives.grad_Omega[..., 2, :]),
    )
