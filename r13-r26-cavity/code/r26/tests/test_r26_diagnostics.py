from __future__ import annotations

import numpy as np

from r26_diagnostics import physical_diagnostics, rana_global_metrics, realizability_matrix


def _equilibrium(n: int = 5) -> np.ndarray:
    state = np.zeros((n, n, 17))
    state[..., 0] = 1.0
    state[..., 3] = 1.0
    return state


def test_equilibrium_realizability_is_identity_and_mass_safe() -> None:
    state = _equilibrium()
    matrix = realizability_matrix(state)
    assert np.array_equal(matrix, np.broadcast_to(np.eye(3), matrix.shape))
    result = physical_diagnostics(state)
    assert result.finite
    assert result.realizability_eigenvalue_min == 1.0
    assert result.realizability_negative_count == 0
    assert result.mass_error == 0.0


def test_rana_global_metrics_use_wall_nodes_and_reduced_stress_factor() -> None:
    n = 9
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    state = _equilibrium(n)
    lid = 0.2
    state[..., 1] = lid * y[:, None]
    state[-1, :, 7] = -0.01
    metrics = rana_global_metrics(state, x, y, lid_velocity=lid)
    assert np.isclose(metrics["D"], np.sqrt(2.0) * 0.01 / lid)
    assert np.isclose(metrics["G"], 0.5)


def test_nonpositive_eq35_denominator_is_rejected() -> None:
    state = _equilibrium()
    state[..., 16] = -6.1
    try:
        realizability_matrix(state)
    except FloatingPointError as exc:
        assert "denominator" in str(exc)
    else:
        raise AssertionError("non-positive Eq. (35) denominator was accepted")
