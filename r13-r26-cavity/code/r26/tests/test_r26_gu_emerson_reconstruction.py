from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np

from r26_cases import gu_asme2009_cavity_case
from r26_gu_emerson_algorithm import GU_EMERSON_STAGE_ORDER
from r26_gu_emerson_reconstruction import (
    FIELD_SLOTS,
    GuEmersonReconstructionOptions,
    make_gu_emerson_reconstruction_problem,
    solve_gu_emerson_reconstruction,
)


def test_reconstruction_controls_are_complete_and_explicitly_nonpaper() -> None:
    options = GuEmersonReconstructionOptions(max_outer_iterations=1)
    disclosure = options.disclosure
    disclosure.require_production_authorization()
    assert disclosure.production_authorized
    assert set(FIELD_SLOTS) == {
        "velocity", "temperature", "g", "h", "omega", "gamma", "chi"
    }
    assert all(
        "not specified by Gu--Emerson" in item.provenance
        for item in (disclosure.controls or {}).values()
    )


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
