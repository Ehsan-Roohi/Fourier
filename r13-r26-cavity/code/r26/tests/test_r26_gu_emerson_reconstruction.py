from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np

from r26_cases import gu_asme2009_cavity_case
from r26_gu_emerson_algorithm import GU_EMERSON_STAGE_ORDER
from r26_gu_emerson_reconstruction import (
    CODE_SATURNE_V5_COMMIT,
    CODE_SATURNE_V5_STEADY_FIELD_RELAXATION,
    CODE_SATURNE_V5_STEADY_PRESSURE_RELAXATION,
    FIELD_SLOTS,
    RANA_SOURCE_HISTORY_RELAXATION,
    RANA_THERMOPHYSICAL_HISTORY_RELAXATION,
    GuEmersonReconstructionOptions,
    _SegregatedReconstructionOperators,
    _anderson_candidate,
    make_gu_emerson_reconstruction_problem,
    solve_gu_emerson_reconstruction,
)
from r26_gu_emerson_saturne_contract import saturne_carrier_evidence
from r26_gu_emerson_variables import gu_emerson_fields_from_state


def test_reconstruction_controls_are_complete_and_explicitly_nonpaper() -> None:
    options = GuEmersonReconstructionOptions(max_outer_iterations=1)
    disclosure = options.disclosure
    disclosure.require_production_authorization()
    assert disclosure.production_authorized
    assert set(FIELD_SLOTS) == {
        "velocity", "temperature", "g", "h", "omega", "gamma", "chi"
    }
    controls = disclosure.controls or {}
    assert "not specified by Gu--Emerson" in controls["linear_solver"].provenance
    assert "not specified by Gu--Emerson" in controls["under_relaxation_factors"].provenance
    assert "d92e0142776d90499e2beea4a8b3b37b590597f66b61f43bb49f58ade73a884b" in controls["source_term_linearisation"].provenance
    assert RANA_SOURCE_HISTORY_RELAXATION == {
        "g": 1.0e-2,
        "h": 1.0e-2,
        "omega": 5.0e-1,
        "gamma": 5.0e-1,
        "chi": 1.0e-1,
    }


def test_safeguarded_equation63_profile_is_bounded_and_disclosed() -> None:
    options = GuEmersonReconstructionOptions.asme2009_equation63_safeguarded_n8(
        max_outer_iterations=1
    )
    assert options.equation_backend == "equation63-transformed-fv"
    assert options.scalar_block_safeguard
    assert options.outer_anderson_acceleration
    assert options.outer_anderson_depth == 1
    assert options.outer_sweep_safeguard
    assert options.outer_nonmonotone_window == 10
    assert options.outer_minimum_step == 1.0 / 4096.0
    assert options.wall_relaxation == 0.25
    assert options.chi_relaxation == 1.0
    controls = options.disclosure.controls or {}
    assert "nonmonotone_window=10" in controls["outer_sweep_globalization"].value
    assert "depth-1 Anderson" in controls["outer_fixed_point_acceleration"].value


def test_equation63_safeguards_reject_incompatible_backends() -> None:
    try:
        GuEmersonReconstructionOptions(outer_sweep_safeguard=True)
    except ValueError as exc:
        assert "direct backend" in str(exc)
    else:
        raise AssertionError("safeguard must not enter the physical-defect backend")


def test_depth_one_anderson_affine_mix_minimizes_opposite_residuals() -> None:
    first = np.ones((1, 1, 17))
    second = np.ones((1, 1, 17))
    first[..., 1] = 0.5
    second[..., 1] = 1.5
    first_residual = np.zeros_like(first)
    second_residual = np.zeros_like(second)
    first_residual[..., 1] = 1.0
    second_residual[..., 1] = -1.0
    candidate, current_weight, used = _anderson_candidate(
        [first, second],
        [first_residual, second_residual],
        np.ones(17),
    )
    assert used
    assert current_weight == 0.5
    np.testing.assert_allclose(candidate, 0.5 * (first + second))


def test_zero_lid_equilibrium_survives_one_complete_published_order_sweep() -> None:
    case = replace(
        gu_asme2009_cavity_case(5, kn=0.1, lid_speed_m_per_s=10.0),
        lid_velocity=0.0,
    )
    problem = make_gu_emerson_reconstruction_problem(case)
    result = solve_gu_emerson_reconstruction(
        problem,
        case.equilibrium_state(),
        options=GuEmersonReconstructionOptions(max_outer_iterations=1),
    )
    assert result.converged
    assert result.outer_iterations == 1
    assert result.records[0].stage_order == GU_EMERSON_STAGE_ORDER
    assert result.records[0].raw_gate < 1.0e-12
    assert result.block_factorizations == 0


def test_saturne_profile_starts_nonzero_lid_equilibrium_without_false_simple_matrix() -> None:
    case = gu_asme2009_cavity_case(
        5, kn=0.1, lid_speed_m_per_s=10.0,
        wall_temperature_K=273.0, grid_stretch_beta=0.0,
    )
    problem = make_gu_emerson_reconstruction_problem(case)
    result = solve_gu_emerson_reconstruction(
        problem,
        case.equilibrium_state(),
        options=GuEmersonReconstructionOptions.code_saturne_v5_rana_diagnostic(
            max_outer_iterations=1
        ),
    )
    assert result.outer_iterations == 1
    assert result.records[0].stage_order == GU_EMERSON_STAGE_ORDER
    assert result.records[0].min_density > 0.0
    assert result.records[0].min_temperature > 0.0


def test_simple_uses_the_underrelaxed_velocity_block_diagonal() -> None:
    case = replace(
        gu_asme2009_cavity_case(5, kn=0.1, lid_speed_m_per_s=10.0),
        lid_velocity=0.0,
    )
    state = case.equilibrium_state()
    state[2, 2, 1] = 1.0e-3
    problem = make_gu_emerson_reconstruction_problem(case)
    options = GuEmersonReconstructionOptions(
        max_outer_iterations=1,
        matrix_refresh_interval=1,
        use_rana_source_history=False,
    )
    fields = gu_emerson_fields_from_state(
        state, x=case.x, y=case.y, mu=case.mu(state[..., 3])
    )
    operators = _SegregatedReconstructionOperators(problem, options)
    provisional = operators.solve_velocity(fields)
    matrix = operators._block_matrices["velocity"]
    raw_diagonal = matrix.diagonal().reshape(3, 3, 2)
    coefficients = operators._simple_inverse_momentum_diagonal
    assert coefficients is not None
    d_x, d_y = coefficients
    np.testing.assert_allclose(
        d_x[1:-1, 1:-1], options.velocity_relaxation / raw_diagonal[..., 0]
    )
    np.testing.assert_allclose(
        d_y[1:-1, 1:-1], options.velocity_relaxation / raw_diagonal[..., 1]
    )
    corrected = operators.simple_pressure_correction(provisional)
    assert np.isfinite(corrected.rho).all()
    assert np.min(corrected.rho) > 0.0


def test_code_saturne_diagnostic_declares_source_backed_carrier_defaults() -> None:
    options = GuEmersonReconstructionOptions.code_saturne_v5_rana_diagnostic(
        max_outer_iterations=1
    )
    assert CODE_SATURNE_V5_COMMIT == "e17068ce692ad2d90c694d375b7c098043b16969"
    assert options.velocity_relaxation == CODE_SATURNE_V5_STEADY_FIELD_RELAXATION
    assert options.pressure_relaxation == CODE_SATURNE_V5_STEADY_PRESSURE_RELAXATION
    assert options.density_property_relaxation == RANA_THERMOPHYSICAL_HISTORY_RELAXATION
    assert options.pressure_density_coupling == "code_saturne_v5_lagged_total_pressure_diagnostic"
    assert options.matrix_refresh_interval == 1
    assert options.wall_relaxation == 1.0
    controls = options.disclosure.controls or {}
    assert CODE_SATURNE_V5_COMMIT in controls["under_relaxation_factors"].provenance
    assert "p_total/theta" in controls["rhie_chow_face_coefficient"].value


def test_code_saturne_pressure_is_independent_and_density_update_is_lagged() -> None:
    case = replace(
        gu_asme2009_cavity_case(5, kn=0.1, lid_speed_m_per_s=10.0),
        lid_velocity=0.0,
    )
    state = case.equilibrium_state()
    state[2, 2, 1] = 1.0e-3
    problem = make_gu_emerson_reconstruction_problem(case)
    options = GuEmersonReconstructionOptions.code_saturne_v5_rana_diagnostic(
        max_outer_iterations=1,
        use_rana_source_history=False,
    )
    fields = gu_emerson_fields_from_state(
        state, x=case.x, y=case.y, mu=case.mu(state[..., 3])
    )
    operators = _SegregatedReconstructionOperators(problem, options)
    provisional = operators.solve_velocity(fields)
    rho_before_pressure = np.asarray(provisional.rho).copy()
    corrected = operators.simple_pressure_correction(provisional)
    np.testing.assert_array_equal(corrected.rho, rho_before_pressure)
    assert operators._saturne_total_pressure is not None
    pressure_after = operators._saturne_total_pressure.copy()
    lagged = operators._apply_lagged_density_property(corrected)
    alpha = RANA_THERMOPHYSICAL_HISTORY_RELAXATION
    expected = rho_before_pressure.copy()
    expected[1:-1, 1:-1] = (
        alpha * pressure_after[1:-1, 1:-1] / corrected.theta[1:-1, 1:-1]
        + (1.0 - alpha) * rho_before_pressure[1:-1, 1:-1]
    )
    np.testing.assert_allclose(lagged.rho, expected, rtol=0.0, atol=2.0e-15)
    np.testing.assert_array_equal(lagged.rho[0], corrected.rho[0])
    assert np.min(lagged.rho) > 0.0


def test_code_saturne_independent_pressure_enters_momentum_and_mass_fluxes() -> None:
    case = replace(
        gu_asme2009_cavity_case(5, kn=0.1, lid_speed_m_per_s=10.0),
        lid_velocity=0.0,
    )
    state = case.equilibrium_state()
    state[2, 2, 1] = 1.0e-3
    problem = make_gu_emerson_reconstruction_problem(case)
    options = GuEmersonReconstructionOptions.code_saturne_v5_rana_diagnostic(
        max_outer_iterations=1, use_rana_source_history=False
    )
    fields = gu_emerson_fields_from_state(
        state, x=case.x, y=case.y, mu=case.mu(state[..., 3])
    )
    operators = _SegregatedReconstructionOperators(problem, options)
    provisional = operators.solve_velocity(fields)
    assert operators._saturne_total_pressure is not None
    base_pressure = operators._saturne_total_pressure.copy()
    xx, yy = np.meshgrid(case.x, case.y)
    operators._saturne_total_pressure = base_pressure + 2.0e-3 * xx * yy
    physical = operators._physical(provisional)
    momentum = operators._saturne_pressure_momentum_correction(physical)
    assert np.max(np.abs(momentum[1:-1, 1:-1])) > 0.0
    continuity = operators._saturne_predicted_continuity(provisional)
    assert np.isfinite(continuity).all()
    operators._saturne_total_pressure = base_pressure + 7.0e-4
    constant = operators._saturne_pressure_momentum_correction(physical)
    np.testing.assert_allclose(constant, 0.0, rtol=0.0, atol=2.0e-14)


def test_saturne_contract_fails_closed_for_missing_case_and_high_speed_energy() -> None:
    evidence = saturne_carrier_evidence()
    assert evidence["published_cavity_lid_speed_m_s"] == 10.0
    assert evidence["historical_case_inputs_available"] is False
    assert evidence["historical_reproduction_authorized"] is False
    assert evidence["high_speed_authorized"] is False
    assert set(evidence["missing_historical_inputs"]) == {
        "setup.xml", "run.cfg", "mesh", "listing/convergence history"
    }
    for key in ("n24_authorized", "n28_authorized", "n29_authorized", "n30_authorized"):
        assert evidence[key] is False


def test_reconstruction_callback_observes_each_accepted_sweep_state() -> None:
    case = replace(
        gu_asme2009_cavity_case(5, kn=0.1, lid_speed_m_per_s=10.0),
        lid_velocity=0.0,
    )
    problem = make_gu_emerson_reconstruction_problem(case)
    observed: list[tuple[object, np.ndarray]] = []
    result = solve_gu_emerson_reconstruction(
        problem,
        case.equilibrium_state(),
        options=GuEmersonReconstructionOptions(max_outer_iterations=1),
        record_callback=lambda record, state: observed.append((record, state.copy())),
    )
    assert result.converged
    assert len(observed) == 1
    assert observed[0][0] == result.records[0]
    np.testing.assert_array_equal(observed[0][1], result.state)


def test_cross_gate_is_source_locked_bounded_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "analysis" / "run_r26_gu_emerson_reconstruction_cross_gate.py").read_text()
    validator = (root / "tools" / "validate_r26_gu_emerson_reconstruction_gate.py").read_text()
    slurm = (root / "hpc" / "r26_gu_emerson_reconstruction_cross_gate_n8_n16.slurm").read_text()
    submit = (root / "hpc" / "submit_r26_gu_emerson_reconstruction_cross_gate_n8_n16.sh").read_text()
    assert "R26_GE_RECON_REF" in slurm and "checkout --detach FETCH_HEAD" in slurm
    assert "run_tests.py" in slurm and "validate_r26_gu_emerson_reconstruction_gate.py" in slurm
    assert "R26_GE_RECON_REF" in submit and "raw.githubusercontent.com" in submit
    assert '"standalone_from_equilibrium_passed": False' in runner
    assert '"standalone_from_equilibrium_attempted": False' in runner
    for authorization in (
        "production_accepted", "n24_authorized", "n28_authorized", "n29_authorized", "n30_authorized"
    ):
        assert f'"{authorization}": False' in runner
        assert f'"{authorization}": False' in validator
    for forbidden in ("homotopy", "pseudo_arclength"):
        assert forbidden not in runner


def test_standalone_ladder_has_two_independent_paths_and_blocks_refinement() -> None:
    root = Path(__file__).resolve().parents[1]
    runner_path = root / "analysis" / "run_r26_gu_emerson_standalone_ladder.py"
    runner = runner_path.read_text()
    validator = (root / "tools" / "validate_r26_gu_emerson_standalone_ladder.py").read_text()
    slurm = (root / "hpc" / "r26_gu_emerson_standalone_ladder_n8_n16.slurm").read_text()
    submit = (root / "hpc" / "submit_r26_gu_emerson_standalone_ladder_n8_n16.sh").read_text()
    for stage in (
        "N8_FROM_EQUILIBRIUM",
        "N8_FROM_PERTURBED",
        "N8_ROOT_COMPARISON",
        "N16_FROM_N8_EQUILIBRIUM",
        "N16_FROM_N8_PERTURBED",
        "N16_ROOT_COMPARISON",
    ):
        assert stage in runner
    assert "if record[\"accepted\"] is not True" in runner
    assert "max_outer_iterations=480" in runner
    assert '"n24_authorized": passed' in runner
    for key in ("production_accepted", "n28_authorized", "n29_authorized", "n30_authorized"):
        assert f'"{key}": False' in runner
        assert f'"{key}": False' in validator
    for forbidden_call in (
        "solve_r26_bvp(",
        "solve_r26_thor_bvp(",
        "solve_lid_continuation(",
        "solve_r26_pseudo_arclength_step(",
    ):
        assert forbidden_call not in runner
    assert "run_tests.py" in slurm
    assert "validate_r26_gu_emerson_reconstruction_gate.py" in slurm
    assert "validate_r26_gu_emerson_standalone_ladder.py" in slurm
    assert "c9c3bc07d14a691d2d4ed70533b46f8daed53726" in slurm
    assert "R26_GE_STANDALONE_REF" in submit and "raw.githubusercontent.com" in submit


def test_standalone_perturbed_start_is_deterministic_positive_and_mass_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    runner_path = root / "analysis" / "run_r26_gu_emerson_standalone_ladder.py"
    spec = importlib.util.spec_from_file_location("r26_ge_standalone_ladder", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = gu_asme2009_cavity_case(
        8, kn=0.1, lid_speed_m_per_s=10.0,
        wall_temperature_K=273.0, grid_stretch_beta=0.0,
    )
    first = module.deterministic_perturbation(case)
    second = module.deterministic_perturbation(case)
    np.testing.assert_array_equal(first, second)
    assert np.min(first[..., 0]) > 0.0
    assert np.min(first[..., 3]) > 0.0
    weights = make_gu_emerson_reconstruction_problem(case).mass_weights
    assert abs(float(np.sum(weights * first[..., 0])) - case.mean_density) < 2.0e-15
    assert not np.array_equal(first, case.equilibrium_state())
    evidence = module.RANA_CODE_SATURNE_CONTROL_EVIDENCE
    assert evidence["source_sha256"] == "0ce53e0811b00154fc0b3c7cb370cfce92a9382d53741a04758499c8132a13ca"
    assert evidence["velocity_linear_relative_tolerance"] == 1.0e-6
    assert evidence["user_scalar_linear_relative_tolerance"] == 1.0e-5
    assert evidence["custom_under_relaxation_factors_declared_in_source"] is False
    assert evidence["nonlinear_source_history_declared_in_cs_user_modules"] is True
    assert evidence["cs_user_modules_sha256"] == "d92e0142776d90499e2beea4a8b3b37b590597f66b61f43bb49f58ade73a884b"


def test_source_linearized_n16_resume_is_bounded_source_locked_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "analysis" / "run_r26_gu_emerson_source_linearized_n16_resume.py").read_text()
    validator = (root / "tools" / "validate_r26_gu_emerson_source_linearized_n16_resume.py").read_text()
    slurm = (root / "hpc" / "r26_gu_emerson_source_linearized_n16_resume.slurm").read_text()
    submit = (root / "hpc" / "submit_r26_gu_emerson_source_linearized_n16_resume.sh").read_text()
    reconstruction = (root / "r26_gu_emerson_reconstruction.py").read_text()
    for factor in ("1.0e-2", "5.0e-1", "1.0e-1"):
        assert factor in reconstruction
    assert "matrix_refresh_interval=1" in runner
    assert "MAX_OUTER_ITERATIONS = 720" in runner
    assert '"reused_failed_n16_state": False' in runner
    assert "latest_checkpoint.npz" not in runner
    assert "state_sha256(state)" in runner
    assert "json.dumps(jsonable(record)" in runner
    assert "problem.evaluate(state)" in validator
    assert "run_tests.py" in slurm
    assert "checkout --detach FETCH_HEAD" in slurm
    assert "R26_GE_FAILED_STANDALONE_DIR" in submit
    assert "raw.githubusercontent.com" in submit
    for key in ("production_accepted", "n28_authorized", "n29_authorized", "n30_authorized"):
        assert f'"{key}": False' in runner
        assert f'"{key}": False' in validator


def test_momentum_simple_n16_resume_is_source_locked_bounded_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (
        root / "analysis" / "run_r26_gu_emerson_momentum_simple_n16_resume.py"
    ).read_text()
    validator = (
        root / "tools" / "validate_r26_gu_emerson_momentum_simple_n16_resume.py"
    ).read_text()
    slurm = (
        root / "hpc" / "r26_gu_emerson_momentum_simple_n16_resume.slurm"
    ).read_text()
    submit = (
        root / "hpc" / "submit_r26_gu_emerson_momentum_simple_n16_resume.sh"
    ).read_text()
    assert "563442d5ce7976b63d82c9592efd6ec3ef620830" in runner
    assert "MAX_OUTER_ITERATIONS = 720" in runner
    assert "use_rana_source_history=False" in runner
    assert '"reused_failed_n16_state": False' in runner
    assert "json.dumps(jsonable(record)" in runner
    assert "latest_checkpoint.npz" not in runner
    for path in (
        "N16_MOMENTUM_SIMPLE_FROM_N8_EQUILIBRIUM",
        "N16_MOMENTUM_SIMPLE_FROM_N8_PERTURBED",
    ):
        assert path in runner
        assert path in validator
    assert "problem.evaluate(state)" in validator
    assert "use_rana_source_history=False" in validator
    assert "run_tests.py" in slurm
    assert "checkout --detach FETCH_HEAD" in slurm
    assert "validate_r26_gu_emerson_momentum_simple_n16_resume.py" in slurm
    assert "R26_GE_FAILED_STANDALONE_DIR" in submit
    assert "R26_GE_FAILED_SOURCE16_DIR" in submit
    assert "raw.githubusercontent.com" in submit
    for key in ("production_accepted", "n28_authorized", "n29_authorized", "n30_authorized"):
        assert f'"{key}": False' in runner
        assert f'"{key}": False' in validator
