from __future__ import annotations

import numpy as np

from r26_state import StateTensors, planar_state_to_tensors, rotate_tensors
from r26_tensor_closures import (
    PHI_C1,
    PSI_Y1,
    R26Gradients,
    closures_from_tensors,
    finite_difference_gradients,
    gu_emerson_closures,
    rotate_gradients,
    stf2_project,
    stf3_project,
    stf4_project,
)


def _equilibrium(ny: int = 5, nx: int = 6) -> np.ndarray:
    u = np.zeros((ny, nx, 17), dtype=float)
    u[..., 0] = 1.0
    u[..., 3] = 1.0
    return u


def _assert_rank3_stf(a: np.ndarray, atol: float = 2.0e-11) -> None:
    assert np.allclose(a, np.swapaxes(a, -1, -2), rtol=0.0, atol=atol)
    assert np.allclose(a, np.swapaxes(a, -2, -3), rtol=0.0, atol=atol)
    assert np.max(np.abs(np.einsum("...iik->...k", a))) < atol


def _assert_rank4_stf(a: np.ndarray, atol: float = 2.0e-11) -> None:
    assert np.allclose(a, np.swapaxes(a, -1, -2), rtol=0.0, atol=atol)
    assert np.allclose(a, np.swapaxes(a, -2, -3), rtol=0.0, atol=atol)
    assert np.allclose(a, np.swapaxes(a, -3, -4), rtol=0.0, atol=atol)
    assert np.max(np.abs(np.einsum("...iikl->...kl", a))) < atol


def test_uniform_equilibrium_has_exactly_zero_high_order_closures() -> None:
    closure = gu_emerson_closures(_equilibrium(), dx=0.2, dy=0.25)
    assert np.array_equal(closure.phi, np.zeros_like(closure.phi))
    assert np.array_equal(closure.psi, np.zeros_like(closure.psi))
    assert np.array_equal(closure.Omega, np.zeros_like(closure.Omega))


def test_finite_difference_gradient_convention_on_linear_manufactured_state() -> None:
    x = np.linspace(-0.4, 0.6, 7)
    y = np.linspace(-0.3, 0.8, 6)
    xx, yy = np.meshgrid(x, y)
    u = _equilibrium(y.size, x.size)
    u[..., 0] = 1.2 + 0.10 * xx + 0.20 * yy
    u[..., 1] = 0.4 + 2.00 * xx + 3.00 * yy
    u[..., 2] = -0.2 + 4.00 * xx - 2.00 * yy
    u[..., 3] = 1.1 + 0.03 * xx - 0.02 * yy
    u[..., 4] = 0.01 + 0.05 * xx + 0.07 * yy
    u[..., 6] = 0.02 + 0.11 * xx - 0.09 * yy
    u[..., 12] = -0.03 + 0.08 * xx + 0.04 * yy
    u[..., 16] = 0.02 - 0.06 * xx + 0.03 * yy
    g = finite_difference_gradients(u, x=x, y=y)
    assert np.allclose(g.rho[..., 0], 0.10, rtol=0.0, atol=3.0e-14)
    assert np.allclose(g.rho[..., 1], 0.20, rtol=0.0, atol=3.0e-14)
    assert np.array_equal(g.rho[..., 2], np.zeros_like(g.rho[..., 2]))
    assert np.allclose(g.velocity[..., 0, 0], 2.0, rtol=0.0, atol=2.0e-14)
    assert np.allclose(g.velocity[..., 1, 0], 3.0, rtol=0.0, atol=2.0e-14)
    assert np.allclose(g.velocity[..., 0, 1], 4.0, rtol=0.0, atol=2.0e-14)
    assert np.allclose(g.velocity[..., 1, 1], -2.0, rtol=0.0, atol=2.0e-14)
    assert np.allclose(g.sigma[..., 0, 0, 0], 0.11, rtol=0.0, atol=3.0e-14)
    assert np.allclose(g.m[..., 1, 0, 0, 0], 0.04, rtol=0.0, atol=3.0e-14)
    assert np.allclose(g.Delta[..., 0], -0.06, rtol=0.0, atol=3.0e-14)


def test_nontrivial_grid_closures_are_full_3d_symmetric_and_trace_free() -> None:
    x = np.linspace(0.0, 1.0, 6)
    y = np.linspace(0.0, 1.0, 5)
    xx, yy = np.meshgrid(x, y)
    u = _equilibrium(y.size, x.size)
    u[..., 0] = 1.0 + 0.04 * xx - 0.02 * yy
    u[..., 1] = 0.15 * xx * (1.0 - yy)
    u[..., 2] = -0.06 * yy * (1.0 - xx)
    u[..., 3] = 1.0 + 0.03 * xx * yy
    for component in range(4, 17):
        if component == 16:
            scale = 0.004
        else:
            scale = 0.002 * (component - 3)
        u[..., component] = scale * np.sin((component - 2) * xx + 0.3) * np.cos(yy + 0.1)
    closure = gu_emerson_closures(u, x=x, y=y, mu=np.sqrt(u[..., 3]))
    _assert_rank4_stf(closure.phi)
    _assert_rank3_stf(closure.psi)
    assert np.isfinite(closure.Omega).all()
    assert np.linalg.norm(closure.phi) > 0.0
    assert np.linalg.norm(closure.psi) > 0.0
    assert np.linalg.norm(closure.Omega) > 0.0


def _random_full_point(seed: int = 2609) -> tuple[StateTensors, R26Gradients]:
    rng = np.random.default_rng(seed)
    sigma = 0.04 * stf2_project(rng.normal(size=(3, 3)))
    rr = 0.03 * stf2_project(rng.normal(size=(3, 3)))
    mm = 0.02 * stf3_project(rng.normal(size=(3, 3, 3)))
    tensors = StateTensors(
        rho=np.asarray(1.15),
        velocity=0.1 * rng.normal(size=3),
        theta=np.asarray(0.93),
        heat_flux=0.03 * rng.normal(size=3),
        sigma=sigma,
        R=rr,
        m=mm,
        Delta=np.asarray(0.015),
    )
    gradients = R26Gradients(
        rho=0.03 * rng.normal(size=3),
        velocity=0.05 * rng.normal(size=(3, 3)),
        theta=0.02 * rng.normal(size=3),
        heat_flux=0.025 * rng.normal(size=(3, 3)),
        sigma=0.03 * stf2_project(rng.normal(size=(3, 3, 3))),
        R=0.02 * stf2_project(rng.normal(size=(3, 3, 3))),
        m=0.015 * stf3_project(rng.normal(size=(3, 3, 3, 3))),
        Delta=0.01 * rng.normal(size=3),
    )
    return tensors, gradients


def test_closures_are_rotation_and_reflection_covariant() -> None:
    transforms = [
        np.diag([-1.0, 1.0, 1.0]),
        np.asarray(
            [
                [0.36, -0.48, 0.80],
                [0.80, 0.60, 0.00],
                [-0.48, 0.64, 0.60],
            ]
        ),
    ]
    for orthogonal in transforms:
        tensors, gradients = _random_full_point()
        original = closures_from_tensors(tensors, gradients, mu=0.87)
        transformed = closures_from_tensors(
            rotate_tensors(tensors, orthogonal), rotate_gradients(gradients, orthogonal), mu=0.87
        )
        expected_phi = np.einsum(
            "ai,bj,ck,dl,ijkl->abcd", orthogonal, orthogonal, orthogonal, orthogonal, original.phi
        )
        expected_psi = np.einsum("ai,bj,ck,ijk->abc", orthogonal, orthogonal, orthogonal, original.psi)
        expected_omega = np.einsum("ai,i->a", orthogonal, original.Omega)
        assert np.allclose(transformed.phi, expected_phi, rtol=2.0e-12, atol=3.0e-13)
        assert np.allclose(transformed.psi, expected_psi, rtol=3.0e-12, atol=5.0e-13)
        assert np.allclose(transformed.Omega, expected_omega, rtol=3.0e-12, atol=5.0e-13)


def test_v3_literal_equation25_contraction_is_frozen_by_a_unit_test() -> None:
    rng = np.random.default_rng(25)
    mm = stf3_project(rng.normal(size=(3, 3, 3)))
    alpha = 0.07
    tensors = StateTensors(
        rho=np.asarray(1.0),
        velocity=np.zeros(3),
        theta=np.asarray(1.0),
        heat_flux=np.zeros(3),
        sigma=np.zeros((3, 3)),
        R=np.zeros((3, 3)),
        m=mm,
        Delta=np.asarray(0.0),
    )
    gradients = R26Gradients(
        rho=np.zeros(3),
        velocity=alpha * np.eye(3),
        theta=np.zeros(3),
        heat_flux=np.zeros((3, 3)),
        sigma=np.zeros((3, 3, 3)),
        R=np.zeros((3, 3, 3)),
        m=np.zeros((3, 3, 3, 3)),
        Delta=np.zeros(3),
    )
    closure = closures_from_tensors(tensors, gradients, mu=1.0)
    expected = -(96.0 / 7.0) * alpha * mm / PSI_Y1
    assert np.allclose(closure.psi, expected, rtol=2.0e-13, atol=2.0e-13)
    assert np.max(np.abs(closure.phi)) < 1.0e-14
    assert np.max(np.abs(closure.Omega)) < 1.0e-14
    try:
        closures_from_tensors(tensors, gradients, equation25_mode="speculative")
    except ValueError as exc:
        assert "v3-literal" in str(exc)
    else:
        raise AssertionError("unsupported Eq. (25) mode was not rejected")


def test_equation25_m_grad_u_uses_derivative_then_velocity_indices() -> None:
    """Distinguish m_mij d_k u_m from the tempting transposed contraction."""

    rng = np.random.default_rng(2501)
    mm = stf3_project(rng.normal(size=(3, 3, 3)))
    gu = np.asarray(
        [
            [0.11, -0.37, 0.23],
            [0.41, -0.19, 0.07],
            [-0.13, 0.29, 0.05],
        ]
    )
    tensors = StateTensors(
        rho=np.asarray(1.0),
        velocity=np.zeros(3),
        theta=np.asarray(1.0),
        heat_flux=np.zeros(3),
        sigma=np.zeros((3, 3)),
        R=np.zeros((3, 3)),
        m=mm,
        Delta=np.asarray(0.0),
    )
    gradients = R26Gradients(
        rho=np.zeros(3),
        velocity=gu,
        theta=np.zeros(3),
        heat_flux=np.zeros((3, 3)),
        sigma=np.zeros((3, 3, 3)),
        R=np.zeros((3, 3, 3)),
        m=np.zeros((3, 3, 3, 3)),
        Delta=np.zeros(3),
    )
    closure = closures_from_tensors(tensors, gradients, mu=1.0)
    div_u = float(np.trace(gu))
    raw_primary = np.zeros((3, 3, 3))
    raw_transposed = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m_index in range(3):
                    # Primary: m_mij * d_k u_m = m[m,i,j] * gu[k,m].
                    raw_primary[i, j, k] += (54.0 / 7.0) * mm[m_index, i, j] * gu[k, m_index]
                    # Historical tempting error: m_mij * d_m u_k.
                    raw_transposed[i, j, k] += (54.0 / 7.0) * mm[m_index, i, j] * gu[m_index, k]
                raw_primary[i, j, k] += 8.0 * mm[i, j, k] * div_u - 6.0 * mm[i, j, k] * div_u
                raw_transposed[i, j, k] += 8.0 * mm[i, j, k] * div_u - 6.0 * mm[i, j, k] * div_u
    expected = -stf3_project(raw_primary) / PSI_Y1
    transposed = -stf3_project(raw_transposed) / PSI_Y1
    expected_omega = np.zeros(3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                expected_omega[i] += -8.0 * mm[i, j, k] * gu[k, j]
    assert np.max(np.abs(expected - transposed)) > 1.0e-3
    assert np.allclose(closure.psi, expected, rtol=2.0e-13, atol=2.0e-13)
    assert np.allclose(closure.Omega, expected_omega, rtol=2.0e-13, atol=2.0e-13)


def test_phi_velocity_gradient_indices_match_primary_component_loops() -> None:
    """Audit sigma_ij*d_l u_k and R_ij*d_l u_k with gu[d,i]."""

    rng = np.random.default_rng(2301)
    sigma = stf2_project(rng.normal(size=(3, 3)))
    rr = stf2_project(rng.normal(size=(3, 3)))
    gu = rng.normal(size=(3, 3))
    tensors = StateTensors(
        rho=np.asarray(1.0), velocity=np.zeros(3), theta=np.asarray(1.0),
        heat_flux=np.zeros(3), sigma=sigma, R=rr, m=np.zeros((3, 3, 3)), Delta=np.asarray(0.0),
    )
    zero = np.zeros(3)
    common = dict(
        rho=zero, theta=zero, heat_flux=np.zeros((3, 3)), sigma=np.zeros((3, 3, 3)),
        R=np.zeros((3, 3, 3)), m=np.zeros((3, 3, 3, 3)), Delta=zero,
    )
    active = closures_from_tensors(tensors, R26Gradients(velocity=gu, **common), mu=1.0)
    baseline = closures_from_tensors(tensors, R26Gradients(velocity=np.zeros((3, 3)), **common), mu=1.0)
    raw_sigma = np.zeros((3, 3, 3, 3))
    raw_R = np.zeros_like(raw_sigma)
    raw_sigma_transposed = np.zeros_like(raw_sigma)
    raw_R_transposed = np.zeros_like(raw_sigma)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for ell in range(3):
                    raw_sigma[i, j, k, ell] = sigma[i, j] * gu[ell, k]
                    raw_R[i, j, k, ell] = rr[i, j] * gu[ell, k]
                    raw_sigma_transposed[i, j, k, ell] = sigma[i, j] * gu[k, ell]
                    raw_R_transposed[i, j, k, ell] = rr[i, j] * gu[k, ell]
    expected = -12.0 / PHI_C1 * stf4_project(raw_sigma)
    expected -= 12.0 / (7.0 * PHI_C1) * stf4_project(raw_R)
    # Total STF4 symmetrization makes the k/l transpose algebraically neutral.
    assert np.allclose(stf4_project(raw_sigma), stf4_project(raw_sigma_transposed), atol=3.0e-13)
    assert np.allclose(stf4_project(raw_R), stf4_project(raw_R_transposed), atol=3.0e-13)
    assert np.allclose(active.phi - baseline.phi, expected, rtol=3.0e-13, atol=3.0e-13)


def test_q_velocity_gradient_terms_in_psi_and_omega_match_component_loops() -> None:
    """Audit q_i*d_k u_j and q_j*(d_j u_i+d_i u_j)."""

    q = np.asarray([0.31, -0.17, 0.23])
    gu = np.asarray([[0.13, -0.41, 0.29], [0.37, -0.11, 0.07], [-0.19, 0.43, 0.05]])
    tensors = StateTensors(
        rho=np.asarray(1.0), velocity=np.zeros(3), theta=np.asarray(1.0), heat_flux=q,
        sigma=np.zeros((3, 3)), R=np.zeros((3, 3)), m=np.zeros((3, 3, 3)), Delta=np.asarray(0.0),
    )
    zero = np.zeros(3)
    gradients = R26Gradients(
        rho=zero, velocity=gu, theta=zero, heat_flux=np.zeros((3, 3)),
        sigma=np.zeros((3, 3, 3)), R=np.zeros((3, 3, 3)), m=np.zeros((3, 3, 3, 3)), Delta=zero,
    )
    closure = closures_from_tensors(tensors, gradients, mu=1.0)
    raw_psi = np.zeros((3, 3, 3))
    raw_psi_transposed = np.zeros_like(raw_psi)
    expected_omega = np.zeros(3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                raw_psi[i, j, k] = q[i] * gu[k, j]
                raw_psi_transposed[i, j, k] = q[i] * gu[j, k]
            expected_omega[i] += -56.0 / 5.0 * q[j] * (gu[j, i] + gu[i, j])
    expected_psi = -108.0 / (5.0 * PSI_Y1) * stf3_project(raw_psi)
    # Total STF3 symmetrization makes the j/k transpose algebraically neutral.
    assert np.allclose(stf3_project(raw_psi), stf3_project(raw_psi_transposed), atol=3.0e-13)
    assert np.allclose(closure.psi, expected_psi, rtol=3.0e-13, atol=3.0e-13)
    assert np.allclose(closure.Omega, expected_omega, rtol=3.0e-13, atol=3.0e-13)


def test_sigma_velocity_scalar_in_psi_and_omega_uses_d_l_u_m() -> None:
    """Audit sigma_ml*d_l u_m; symmetry makes its transpose equivalent."""

    rng = np.random.default_rng(2511)
    sigma = stf2_project(rng.normal(size=(3, 3)))
    mm = stf3_project(rng.normal(size=(3, 3, 3)))
    q = rng.normal(size=3)
    gu = rng.normal(size=(3, 3))
    tensors = StateTensors(
        rho=np.asarray(1.0), velocity=np.zeros(3), theta=np.asarray(1.0), heat_flux=q,
        sigma=sigma, R=np.zeros((3, 3)), m=mm, Delta=np.asarray(0.0),
    )
    zero = np.zeros(3)
    common = dict(
        rho=zero, theta=zero, heat_flux=np.zeros((3, 3)), sigma=np.zeros((3, 3, 3)),
        R=np.zeros((3, 3, 3)), m=np.zeros((3, 3, 3, 3)), Delta=zero,
    )
    active = closures_from_tensors(tensors, R26Gradients(velocity=gu, **common), mu=1.0)
    baseline = closures_from_tensors(tensors, R26Gradients(velocity=np.zeros((3, 3)), **common), mu=1.0)
    scalar = 0.0
    scalar_transposed = 0.0
    t2 = np.zeros(3)
    t3 = np.zeros(3)
    for m_index in range(3):
        for ell in range(3):
            scalar += sigma[m_index, ell] * gu[ell, m_index]
            scalar_transposed += sigma[m_index, ell] * gu[m_index, ell]
        for j in range(3):
            t2[m_index] += -56.0 / 5.0 * q[j] * (gu[j, m_index] + gu[m_index, j])
            for k in range(3):
                t3[m_index] += -8.0 * mm[m_index, j, k] * gu[k, j]

    div_u = float(np.trace(gu))
    eq25 = np.zeros((3, 3, 3))
    q_grad_u = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                q_grad_u[i, j, k] = q[i] * gu[k, j]
                eq25[i, j, k] = 2.0 * mm[i, j, k] * div_u
                for m_index in range(3):
                    eq25[i, j, k] += 54.0 / 7.0 * mm[m_index, i, j] * gu[k, m_index]
    expected_psi_delta = -108.0 / (5.0 * PSI_Y1) * stf3_project(q_grad_u)
    expected_psi_delta += 6.0 / PSI_Y1 * mm * scalar - stf3_project(eq25) / PSI_Y1
    expected_omega_delta = t2 + t3 + (56.0 / 3.0) * q * scalar
    # sigma and m are symmetric in the exchanged indices, so these two
    # remaining transpose choices are algebraically neutral.
    t3_transposed = -8.0 * np.einsum("ijk,jk->i", mm, gu)
    assert np.isclose(scalar, scalar_transposed, rtol=0.0, atol=3.0e-13)
    assert np.allclose(t3, t3_transposed, rtol=0.0, atol=3.0e-13)
    assert np.allclose(active.psi - baseline.psi, expected_psi_delta, rtol=5.0e-13, atol=6.0e-13)
    assert np.allclose(active.Omega - baseline.Omega, expected_omega_delta, rtol=5.0e-13, atol=6.0e-13)


def test_planar_astr_mapping_produces_the_same_closure_as_direct_tensors() -> None:
    u = _equilibrium(5, 5)
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(0.0, 1.0, 5)
    xx, yy = np.meshgrid(x, y)
    u[..., 1] = 0.1 * xx
    u[..., 3] = 1.0 + 0.02 * yy
    u[..., 6] = 0.01 * xx * yy
    u[..., 12] = 0.005 * xx
    direct = closures_from_tensors(
        planar_state_to_tensors(u), finite_difference_gradients(u, x=x, y=y), mu=1.0
    )
    wrapped = gu_emerson_closures(u, x=x, y=y, mu=1.0)
    assert np.array_equal(direct.phi, wrapped.phi)
    assert np.array_equal(direct.psi, wrapped.psi)
    assert np.array_equal(direct.Omega, wrapped.Omega)
