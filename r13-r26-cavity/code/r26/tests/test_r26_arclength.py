#!/usr/bin/env python3
"""Unit tests for bounded pseudo-arclength continuation."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix

from r26_arclength import (
    ArcLengthCorrectorOptions,
    ArcLengthMetric,
    arclength_constraint,
    balanced_parameter_scale,
    interpolate_bracketed_state,
    normalized_secant_tangent,
    secant_metric_diagnostics,
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


def test_secant_balanced_metric_assigns_requested_squared_norm_fraction() -> None:
    previous = np.asarray((0.0, 0.0, 0.0, 0.0))
    current = np.asarray((0.002, -0.001, 0.003, -0.002))
    previous_parameter = 0.36935558170895255
    current_parameter = 0.37021571065841696
    scale = balanced_parameter_scale(
        previous,
        previous_parameter,
        current,
        current_parameter,
        parameter_fraction=0.5,
    )
    diagnostic = secant_metric_diagnostics(
        previous,
        previous_parameter,
        current,
        current_parameter,
        ArcLengthMetric(previous.size, parameter_scale=scale),
    )
    np.testing.assert_allclose(
        (diagnostic.state_fraction, diagnostic.parameter_fraction),
        (0.5, 0.5),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_historical_small_parameter_scale_exposes_fixed_parameter_degeneracy() -> None:
    # This ratio reproduces the N30 diagnostic: encoded-state RMS 1.275e-4
    # for a lid increment 8.60129e-4.  parameter_scale=0.04 assigns more
    # than 99.99% of the squared arclength norm to the lid parameter.
    previous = np.zeros(4)
    current = np.full(4, 1.275e-4)
    diagnostic = secant_metric_diagnostics(
        previous,
        0.36935558170895255,
        current,
        0.37021571065841696,
        ArcLengthMetric(previous.size, parameter_scale=0.04),
    )
    assert diagnostic.parameter_fraction > 0.9999
    assert diagnostic.state_fraction < 1.0e-4


def test_fixed_calibrated_metric_allows_parameter_tangent_to_vanish_at_fold() -> None:
    metric = ArcLengthMetric(4, parameter_scale=2.0)
    previous = np.zeros(4)
    current = np.asarray((0.02, -0.02, 0.02, -0.02))
    diagnostic = secant_metric_diagnostics(
        previous,
        0.3703661695101804,
        current,
        0.37038324507064474,
        metric,
    )
    assert diagnostic.parameter_fraction < 1.0e-6
    tangent = normalized_secant_tangent(
        previous,
        0.3703661695101804,
        current,
        0.37038324507064474,
        metric,
    )
    np.testing.assert_allclose(
        metric.inner(tangent.state, tangent.parameter, tangent.state, tangent.parameter),
        1.0,
        rtol=0.0,
        atol=2.0e-15,
    )


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
    assert controls.enforce_parameter_metric_fraction_bounds is False
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
    try:
        ArcLengthCorrectorOptions(
            minimum_parameter_metric_fraction=0.95,
            maximum_parameter_metric_fraction=0.9,
        )
    except ValueError as error:
        assert "metric-fraction bounds" in str(error)
    else:
        raise AssertionError("invalid metric-fraction bounds were accepted")
    try:
        ArcLengthCorrectorOptions(enforce_parameter_metric_fraction_bounds=1)
    except TypeError as error:
        assert "must be boolean" in str(error)
    else:
        raise AssertionError("non-boolean metric-fraction enforcement was accepted")
