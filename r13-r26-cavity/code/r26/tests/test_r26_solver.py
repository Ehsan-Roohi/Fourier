from __future__ import annotations

import numpy as np
from unittest.mock import patch

from r26_cases import (
    CavityCase,
    KnudsenConvention,
    ViscosityModel,
    equilibrium_mu_star,
    rana_first_case,
)
from r26_discretization import R26NodeBVP, bilinear_corner_residuals
from r26_solver import (
    analytic_mass_jacobian_row,
    EncodedR26MassContinuityObjective,
    EncodedR26Objective,
    EncodedR26RawMassContinuityObjective,
    LogStateTransform,
    interpolate_state_grid,
    jacobian_sparsity,
    pseudo_transient_diagonal,
    residual_family_row_scales,
    SolveOptions,
    secant_predict_state,
    solve_r26_bvp,
)
from r26_state import NVAR
from r26_tensor_closures import R26Closures


def _case(nodes: int = 5) -> CavityCase:
    return CavityCase(
        name="unit-equilibrium",
        nodes=nodes,
        kn=0.01,
        kn_convention=KnudsenConvention.RANA,
        lid_velocity=0.0,
        viscosity=ViscosityModel.power_law(1.0),
    )


def _mock_bulk(
    state: np.ndarray, *, x: np.ndarray, y: np.ndarray, mu: np.ndarray, case: CavityCase
) -> np.ndarray:
    del x, y, mu
    equilibrium = np.zeros_like(state)
    equilibrium[..., 0] = case.mean_density
    equilibrium[..., 3] = case.wall_temperature
    return state - equilibrium


def _zero_closures(
    state: np.ndarray, *, x: np.ndarray, y: np.ndarray, mu: np.ndarray
) -> R26Closures:
    del x, y, mu
    grid = state.shape[:2]
    return R26Closures(
        phi=np.zeros(grid + (3, 3, 3, 3)),
        psi=np.zeros(grid + (3, 3, 3)),
        Omega=np.zeros(grid + (3,)),
    )


def _problem(nodes: int = 5) -> R26NodeBVP:
    return R26NodeBVP(_case(nodes), bulk_operator=_mock_bulk, closure_operator=_zero_closures)


def test_equilibrium_has_zero_full_bulk_wall_mass_and_corner_residual() -> None:
    problem = _problem()
    evaluation = problem.evaluate(problem.case.equilibrium_state())
    assert np.array_equal(evaluation.residual, np.zeros(problem.shape))
    assert evaluation.diagnostics.total_linf == 0.0
    assert evaluation.diagnostics.held_out_continuity == 0.0


def test_actual_bulk_closure_wall_integration_has_exact_equilibrium_residual() -> None:
    case = _case()
    evaluation = R26NodeBVP(case).evaluate(case.equilibrium_state())
    assert np.array_equal(evaluation.residual, np.zeros((5, 5, NVAR)))
    assert evaluation.diagnostics.raw_total_linf == 0.0


def test_equation_accounting_and_corners_are_explicitly_excluded() -> None:
    problem = _problem(nodes=6)
    accounting = problem.equation_accounting
    assert accounting["interior_nodes"] == 16
    assert accounting["smooth_wall_nodes"] == 16
    assert accounting["excluded_corner_nodes"] == 4
    assert accounting["total"] == 6 * 6 * NVAR
    assert (
        accounting["interior_equations"]
        + accounting["smooth_wall_equations"]
        + accounting["corner_model_equations"]
        == accounting["total"]
    )


def test_bilinear_corner_model_is_a_zero_mixed_second_difference() -> None:
    case = _case()
    state = case.equilibrium_state()
    state[0, 0, 4] = 0.25
    corners = bilinear_corner_residuals(state)
    assert corners["bottom_left"][4] == 0.25
    assert all(np.all(value == 0.0) for name, value in corners.items() if name != "bottom_left")
    evaluation = _problem().evaluate(state)
    assert evaluation.diagnostics.corner_linf == 0.25


def test_mass_border_replaces_one_row_and_reports_held_out_continuity() -> None:
    problem = _problem()
    state = problem.case.equilibrium_state()
    state[..., 0] = 1.1
    evaluation = problem.evaluate(state)
    j, i, component = evaluation.mass_row
    assert component == 0 and 0 < j < 4 and 0 < i < 4
    assert np.isclose(evaluation.unscaled_residual[j, i, 0], 0.1)
    assert np.isclose(evaluation.diagnostics.mass_error, 0.1)
    assert np.isclose(evaluation.diagnostics.held_out_continuity, 0.1)


def test_log_transform_guarantees_positive_density_and_temperature() -> None:
    case = _case()
    transform = LogStateTransform(case.equilibrium_state().shape)
    encoded = transform.encode(case.equilibrium_state())
    reshaped = encoded.reshape(case.nodes, case.nodes, NVAR)
    reshaped[..., 0] = -20.0
    reshaped[..., 3] = 10.0
    decoded = transform.decode(reshaped.ravel())
    assert np.all(decoded[..., 0] > 0.0)
    assert np.all(decoded[..., 3] > 0.0)
    assert np.allclose(transform.decode(transform.encode(decoded)), decoded)


def test_full_objective_jvp_matches_exact_linear_mock_direction() -> None:
    problem = _problem()
    transform = LogStateTransform(problem.shape)
    encoded = transform.encode(problem.case.equilibrium_state())
    direction = np.zeros_like(encoded).reshape(problem.shape)
    # Delta is not one of the six linearly extrapolated wall quantities, so
    # this central direction exercises one exact mock-bulk row only.
    direction[2, 2, 16] = 1.0
    direction = direction.ravel()
    objective = EncodedR26Objective(problem, transform, penalty=1.0e8)
    jvp = objective.jvp(encoded, direction, relative_step=1.0e-6)
    assert np.allclose(jvp, direction, rtol=2.0e-10, atol=2.0e-10)
    assert objective.invalid_evaluations == 0


def test_full_objective_caches_the_unscaled_acceptance_gate() -> None:
    problem = _problem()
    transform = LogStateTransform(problem.shape)
    state = problem.case.equilibrium_state()
    state[..., 0] = 1.1
    state[2, 2, 16] = 0.2
    evaluation = problem.evaluate(state)
    objective = EncodedR26Objective(problem, transform, penalty=1.0e8)

    residual = objective(transform.encode(state))
    expected_raw_linf = max(
        evaluation.diagnostics.raw_total_linf,
        abs(evaluation.diagnostics.held_out_continuity),
        abs(evaluation.diagnostics.mass_error),
    )

    assert np.array_equal(residual, evaluation.flat)
    assert objective.last_raw_linf == expected_raw_linf


def test_raw_linf_guard_rejects_a_scaled_descent_that_worsens_raw_gate() -> None:
    import r26_solver as solver_module

    problem = _problem(nodes=5)
    state = problem.case.equilibrium_state()
    state[2, 2, 16] = 0.1
    initial_encoded = LogStateTransform(problem.shape).encode(state)
    delta_index = int(np.ravel_multi_index((2, 2, 16), problem.shape))

    class OpposedRawObjective:
        def __init__(self, problem: object, transform: object, penalty: float) -> None:
            del problem, transform, penalty
            self.invalid_evaluations = 0
            self.last_invalid_error = None
            self.last_raw_linf = float("inf")

        def __call__(self, vector: np.ndarray) -> np.ndarray:
            value = float(vector[delta_index])
            residual = np.zeros_like(vector)
            residual[delta_index] = value
            # The scaled objective descends toward value=0, while the raw
            # acceptance gate strictly worsens from its initial value of one.
            self.last_raw_linf = 1.1 - value
            return residual

    with patch.object(solver_module, "EncodedR26Objective", OpposedRawObjective):
        result = solve_r26_bvp(
            problem,
            state,
            options=SolveOptions(
                method="colored_newton",
                residual_tolerance=1.0e-12,
                held_out_continuity_tolerance=1.0e-12,
                max_iterations=1,
                pseudo_transient=True,
                pseudo_time_initial=1.0e8,
                pseudo_time_maximum=1.0e8,
                require_raw_linf_decrease=True,
                max_jacobian_evaluations=1,
            ),
        )

    assert np.array_equal(result.encoded_state, initial_encoded)
    assert result.pseudo_transient_steps == 0
    assert result.final_pseudo_time_step == 2.5e7


def test_raw_guard_options_are_fail_closed() -> None:
    invalid_options = (
        {"pseudo_time_minimum_accepted_alpha": -0.1},
        {"pseudo_time_minimum_accepted_alpha": 1.1},
        {"require_raw_linf_decrease": True},
    )
    for values in invalid_options:
        try:
            SolveOptions(**values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"SolveOptions unexpectedly accepted {values}")


def test_augmented_objective_restores_continuity_and_appends_mass() -> None:
    problem = _problem()
    transform = LogStateTransform(problem.shape)
    state = problem.case.equilibrium_state()
    state[..., 0] = 1.1
    encoded = transform.encode(state)
    square = EncodedR26Objective(problem, transform, penalty=1.0e8)(encoded)
    augmented = EncodedR26MassContinuityObjective(
        problem, transform, penalty=1.0e8
    )(encoded)
    mass_index = np.ravel_multi_index((problem.mass_j, problem.mass_i, 0), problem.shape)

    assert augmented.shape == (problem.unknown_count + 1,)
    assert np.array_equal(
        np.delete(augmented[:-1], mass_index),
        np.delete(square, mass_index),
    )
    assert np.isclose(augmented[mass_index], 0.1 / problem.case.scaling.bulk[0])
    assert np.isclose(augmented[-1], 0.1 / problem.case.scaling.mass)


def test_raw_augmented_objective_uses_unscaled_rows_and_raw_mass() -> None:
    case = rana_first_case(5).with_lid_velocity(0.0, suffix="raw-objective-unit")
    problem = R26NodeBVP(
        case,
        bulk_operator=_mock_bulk,
        closure_operator=_zero_closures,
    )
    transform = LogStateTransform(problem.shape)
    state = case.equilibrium_state()
    state[2, 1, 16] = 0.2
    state[..., 0] = 1.1
    encoded = transform.encode(state)
    evaluation = problem.evaluate(state)
    objective = EncodedR26RawMassContinuityObjective(
        problem, transform, penalty=1.0e8
    )(encoded)
    mass_index = int(np.ravel_multi_index(evaluation.mass_row, problem.shape))
    expected = evaluation.unscaled_residual.ravel().copy()
    expected[mass_index] = evaluation.diagnostics.held_out_continuity
    expected = np.concatenate(
        (expected, np.asarray((evaluation.diagnostics.mass_error,), dtype=float))
    )

    assert objective.shape == (problem.unknown_count + 1,)
    assert np.array_equal(objective, expected)
    delta_index = int(np.ravel_multi_index((2, 1, 16), problem.shape))
    assert np.isclose(objective[delta_index], 0.2)
    assert not np.isclose(
        objective[delta_index],
        EncodedR26MassContinuityObjective(problem, transform, penalty=1.0e8)(encoded)[
            delta_index
        ],
    )


def test_residual_has_no_hidden_history() -> None:
    problem = _problem()
    state = problem.case.equilibrium_state()
    state[2, 2, 5] = 0.03125
    first = problem.evaluate(state)
    second = problem.evaluate(state.copy())
    assert np.array_equal(first.residual, second.residual)
    assert first.diagnostics == second.diagnostics


def test_grid_restart_interpolation_preserves_positive_fields_and_mass() -> None:
    state = _case(5).equilibrium_state()
    y, x = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5), indexing="ij")
    state[..., 0] = np.exp(0.2 * x - 0.1 * y)
    state[..., 3] = np.exp(0.1 * x + 0.05 * y)
    state[..., 1] = x * (1.0 - y)
    refined = interpolate_state_grid(state, 8, target_mean_density=1.0)
    assert refined.shape == (8, 8, NVAR)
    assert np.min(refined[..., 0]) > 0.0 and np.min(refined[..., 3]) > 0.0
    from r26_discretization import trapezoidal_node_weights

    assert np.isclose(np.sum(trapezoidal_node_weights(8) * refined[..., 0]), 1.0, atol=2.0e-14)


def test_kn_conventions_and_viscosity_laws_are_not_silently_mixed() -> None:
    rana = equilibrium_mu_star(0.1, KnudsenConvention.RANA)
    gu = equilibrium_mu_star(0.1, KnudsenConvention.GU_MEAN_FREE_PATH)
    assert rana == 0.1
    assert np.isclose(gu, 0.1 * np.sqrt(2.0 / np.pi))
    maxwell = ViscosityModel.power_law(1.0)
    archive = ViscosityModel.power_law(0.5)
    assert np.isclose(maxwell.ratio(4.0), 4.0)
    assert np.isclose(archive.ratio(4.0), 2.0)
    sutherland = ViscosityModel.gu_sutherland(reference_temperature_K=300.0)
    assert np.isclose(sutherland.ratio(1.0), 1.0)
    rana = rana_first_case(5)
    assert rana.scaling.bulk[12] == 3.0 / (2.0 * rana.mu_equilibrium)


def test_colored_fd_sparsity_includes_radius_two_and_global_mass_row() -> None:
    problem = _problem(nodes=5)
    pattern = jacobian_sparsity(problem)
    assert pattern.shape == (problem.unknown_count, problem.unknown_count)
    mass_row = np.ravel_multi_index((problem.mass_j, problem.mass_i, 0), problem.shape)
    rho_columns = np.arange(0, problem.unknown_count, NVAR)
    assert np.all(pattern[mass_row, rho_columns].toarray() == 1)
    central_delta_row = np.ravel_multi_index((2, 2, 16), problem.shape)
    far_corner_q = np.ravel_multi_index((0, 0, 4), problem.shape)
    # On N=5, a radius-two closure stencil reaches the corner from the centre.
    assert pattern[central_delta_row, far_corner_q]


def test_analytic_mass_border_matches_directional_difference_and_removes_fd_row() -> None:
    problem = _problem(nodes=5)
    transform = LogStateTransform(problem.shape)
    state = problem.case.equilibrium_state()
    state[..., 0] = np.exp(np.linspace(-0.1, 0.1, 25).reshape(5, 5))
    state[..., 0] *= problem.case.mean_density / problem.mean_density(state)
    encoded = transform.encode(state)
    mass_row, rho_columns, values = analytic_mass_jacobian_row(
        problem, transform, encoded
    )
    direction = np.zeros(problem.unknown_count)
    direction[rho_columns] = np.linspace(-0.7, 0.9, rho_columns.size)
    step = 1.0e-6
    objective = EncodedR26Objective(problem, transform, penalty=1.0e8)
    finite_difference = (
        objective(encoded + step * direction)[mass_row]
        - objective(encoded - step * direction)[mass_row]
    ) / (2.0 * step)

    assert np.isclose(np.dot(values, direction[rho_columns]), finite_difference, rtol=2e-9, atol=2e-11)
    assert jacobian_sparsity(problem, include_mass_border=False)[mass_row].nnz == 0


def test_pseudo_transient_diagonal_is_bulk_only_scaled_and_log_aware() -> None:
    problem = _problem(nodes=5)
    transform = LogStateTransform(problem.shape)
    state = problem.case.equilibrium_state()
    state[2, 1, 0] = 1.2
    state[2, 1, 3] = 0.8
    diagonal = pseudo_transient_diagonal(
        problem, transform, transform.encode(state)
    ).reshape(problem.shape)

    assert np.all(diagonal[0] == 0.0)
    assert np.all(diagonal[-1] == 0.0)
    assert np.all(diagonal[:, 0] == 0.0)
    assert np.all(diagonal[:, -1] == 0.0)
    assert diagonal[problem.mass_j, problem.mass_i, 0] == 0.0
    assert np.isclose(diagonal[2, 1, 0], 1.2 / problem.case.scaling.bulk[0])
    assert np.isclose(diagonal[2, 1, 3], 0.8 / problem.case.scaling.bulk[3])
    interior = diagonal[1:-1, 1:-1].copy()
    interior[problem.mass_j - 1, problem.mass_i - 1, 0] = 1.0
    assert np.all(interior > 0.0)


def test_secant_predictor_preserves_positivity_mass_and_linear_moments() -> None:
    problem = _problem(nodes=5)
    previous = problem.case.equilibrium_state()
    current = previous.copy()
    current[..., 0] *= np.exp(0.02)
    current[..., 0] *= problem.case.mean_density / problem.mean_density(current)
    current[..., 3] *= np.exp(0.03)
    current[..., 16] = 0.04
    predicted = secant_predict_state(
        problem,
        previous,
        current,
        previous_parameter=0.0,
        current_parameter=0.1,
        target_parameter=0.2,
    )

    assert np.min(predicted[..., 0]) > 0.0
    assert np.min(predicted[..., 3]) > 0.0
    assert np.isclose(problem.mean_density(predicted), problem.case.mean_density, atol=2e-15)
    assert np.allclose(predicted[..., 16], 0.08)


def test_ser_ptc_uses_a_shift_then_polishes_the_same_mock_root() -> None:
    problem = _problem(nodes=5)
    state = problem.case.equilibrium_state()
    state[2, 2, 16] = 0.1
    result = solve_r26_bvp(
        problem,
        state,
        options=SolveOptions(
            method="colored_newton",
            residual_tolerance=1.0e-10,
            held_out_continuity_tolerance=1.0e-10,
            max_iterations=6,
            analytic_mass_jacobian=True,
            pseudo_transient=True,
            pseudo_time_initial=1.0e8,
            pseudo_time_maximum=1.0e8,
            newton_switch_tolerance=5.0e-2,
            max_jacobian_evaluations=2,
        ),
    )

    assert result.converged and result.scipy_success
    assert result.pseudo_transient_steps == 1
    assert result.jacobian_evaluations == 1
    assert result.diagnostics.total_linf <= 1.0e-10
    assert np.array_equal(result.state, problem.case.equilibrium_state())


def test_jacobian_row_equilibration_is_componentwise_and_family_local() -> None:
    problem = _problem(nodes=5)
    diagonal = np.ones(problem.unknown_count)
    shaped = diagonal.reshape(problem.shape)
    for component in range(NVAR):
        shaped[1:-1, 1:-1, component] = 10.0 + component
    for node in problem.boundary_nodes:
        shaped[node.j, node.i, :11] = 100.0 + np.arange(11)
        shaped[node.j, node.i, 11:] = 200.0 + np.arange(6)
    for j, i in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        shaped[j, i] = 300.0 + np.arange(NVAR)
    shaped[problem.mass_j, problem.mass_i, 0] = 999.0

    scales = residual_family_row_scales(problem, np.diag(diagonal)).reshape(problem.shape)
    for component in range(NVAR):
        values = scales[1:-1, 1:-1, component].copy()
        if component == 0:
            values[problem.mass_j - 1, problem.mass_i - 1] = 10.0
        assert np.all(values == 10.0 + component)
    assert scales[problem.mass_j, problem.mass_i, 0] == 999.0
    for node in problem.boundary_nodes:
        assert np.array_equal(scales[node.j, node.i, :11], 100.0 + np.arange(11))
        assert np.array_equal(scales[node.j, node.i, 11:], 200.0 + np.arange(6))
    for j, i in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert np.array_equal(scales[j, i], 300.0 + np.arange(NVAR))


def test_colored_newton_uses_absolute_fd_floor_for_small_moments() -> None:
    """The solver must not scale Jacobian perturbations by tiny moments.

    A relative-only step made valid near-zero R26 moment coordinates use a
    roundoff-sized perturbation.  Capture the production differentiation call
    so this regression cannot silently return.
    """

    import r26_solver as solver_module

    problem = _problem(nodes=5)
    state = problem.case.equilibrium_state()
    state[2, 2, 16] = 0.1
    original = solver_module.approx_derivative
    captured: dict[str, np.ndarray | None] = {"abs_step": None}

    def recording_derivative(*args: object, **kwargs: object) -> object:
        captured["abs_step"] = np.asarray(kwargs.get("abs_step"), dtype=float)
        return original(*args, **kwargs)

    with patch.object(solver_module, "approx_derivative", recording_derivative):
        result = solve_r26_bvp(
            problem,
            state,
            options=SolveOptions(
                method="colored_newton",
                residual_tolerance=1.0e-10,
                held_out_continuity_tolerance=1.0e-10,
                max_iterations=3,
            ),
        )

    assert result.converged
    assert captured["abs_step"] is not None
    assert captured["abs_step"].shape == (problem.unknown_count,)
    assert np.min(captured["abs_step"]) >= 2.0e-6


def test_colored_newton_stops_at_the_objective_evaluation_budget() -> None:
    problem = _problem(nodes=5)
    state = problem.case.equilibrium_state()
    state[2, 2, 16] = 0.1
    result = solve_r26_bvp(
        problem,
        state,
        options=SolveOptions(
            method="colored_newton",
            residual_tolerance=1.0e-12,
            held_out_continuity_tolerance=1.0e-12,
            max_iterations=100,
            max_objective_evaluations=1,
        ),
    )

    assert not result.converged
    assert not result.scipy_success
    assert result.function_evaluations == 1
    assert "objective-evaluation limit reached (1)" in result.message
