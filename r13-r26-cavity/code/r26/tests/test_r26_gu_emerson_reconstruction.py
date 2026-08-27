from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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

