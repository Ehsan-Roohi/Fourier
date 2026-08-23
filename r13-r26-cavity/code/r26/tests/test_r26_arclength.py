#!/usr/bin/env python3
"""Unit tests for bounded pseudo-arclength continuation."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix

from r26_arclength import (
    ArcLengthCorrectorOptions,
    ArcLengthMetric,
    arclength_constraint,
    interpolate_bracketed_state,
    normalized_secant_tangent,
    solve_bordered_newton_direction,
)
from r26_cases import jfm_maxwell_cavity_case
from r26_discretization import R26NodeBVP
from r26_fv_backend import (
    compatible_fv_bulk_residual,
    wall_bounded_control_volume_weights,
)


def test_secant_tangent_is_unit_normalized_and_reference_oriented() -> None:
    metric = ArcLengthMetric(3, parameter_scale=0.2)
    previous = np.asarray((0.0, 0.0, 0.0))
    current = np.asarray((0.3, -0.6, 0.9))
    tangent = normalized_secant_tangent(previous, 0.1, current, 0.16, metric)
    np.testing.assert_allclose(
        metric.inner(tangent.state, tangent.parameter, tangent.state, tangent.parameter),
        1.0,
        rtol=0.0,
        atol=2.0e-15,
    )
    reversed_tangent = normalized_secant_tangent(
        current,
        0.16,
        previous,
        0.1,
        metric,
        reference=tangent,
    )
    assert metric.inner(
        tangent.state,
        tangent.parameter,
        reversed_tangent.state,
        reversed_tangent.parameter,
    ) > 0.0


def test_arclength_hyperplane_is_exact_at_predictor() -> None:
    metric = ArcLengthMetric(2, parameter_scale=0.04)
    tangent = normalized_secant_tangent(
        np.asarray((0.0, 0.0)),
        0.0,
        np.asarray((0.1, -0.2)),
        0.02,
        metric,
    )
    predicted_state = np.asarray((0.15, -0.3))
    value, row, parameter_entry = arclength_constraint(
        predicted_state,
        0.03,
        predicted_state,
        0.03,
        tangent,
        metric,
        scale=0.5,
    )
    assert value == 0.0
    np.testing.assert_allclose(row, metric.state_weight * tangent.state / 0.5)
    assert parameter_entry == metric.parameter_weight * tangent.parameter / 0.5


def test_bordered_system_remains_invertible_at_a_simple_fold() -> None:
    state_direction, parameter_direction, method = solve_bordered_newton_direction(
        csc_matrix([[0.0]]),
        np.asarray((1.0,)),
        np.asarray((1.0,)),
        0.0,
        np.asarray((0.1,)),
        0.2,
    )
    np.testing.assert_allclose(state_direction, (-0.2,), rtol=0.0, atol=1.0e-15)
    assert abs(parameter_direction + 0.1) <= 1.0e-15
    assert method == "bordered-splu"


def test_bracket_interpolation_preserves_positivity_and_exact_mass() -> None:
    case = jfm_maxwell_cavity_case(5, kn=0.20, grid_stretch_beta=0.0)
    problem = R26NodeBVP(
        case,
        bulk_operator=compatible_fv_bulk_residual,
        mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
    )
    lower = case.equilibrium_state()
    upper = lower.copy()
    pattern = np.linspace(-0.08, 0.08, 25).reshape(5, 5)
    upper[..., 0] *= np.exp(pattern)
    upper[..., 0] *= case.mean_density / problem.mean_density(upper)
    upper[..., 3] *= np.exp(-0.5 * pattern)
    upper[..., 1] = 0.01 * pattern
    interpolated = interpolate_bracketed_state(
        problem,
        lower,
        0.30,
        upper,
        0.40,
        0.35,
    )
    assert float(np.min(interpolated[..., 0])) > 0.0
    assert float(np.min(interpolated[..., 3])) > 0.0
    assert abs(problem.mass_constraint(interpolated)) <= 2.0e-15


def test_arclength_controls_are_bounded_and_fail_closed() -> None:
    controls = ArcLengthCorrectorOptions()
    assert controls.maximum_iterations == 80
    assert controls.maximum_jacobians == 7
    assert controls.maximum_objective_evaluations == 6000
    assert controls.pseudo_transient_chord_limit == 12
    assert controls.newton_chord_limit == 3
    assert controls.pseudo_time_minimum <= controls.pseudo_time_initial
    try:
        ArcLengthCorrectorOptions(pseudo_time_minimum=2.0)
    except ValueError as error:
        assert "pseudo-time limits" in str(error)
    else:
        raise AssertionError("invalid pseudo-time ordering was accepted")
    try:
        ArcLengthCorrectorOptions(maximum_iterations=0)
    except ValueError as error:
        assert "work limits" in str(error)
    else:
        raise AssertionError("zero nonlinear-iteration limit was accepted")
