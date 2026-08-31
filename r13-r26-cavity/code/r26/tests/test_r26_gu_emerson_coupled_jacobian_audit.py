import numpy as np

from r26_cases import gu_asme2009_cavity_case
from r26_gu_emerson_coupled_jacobian_audit import analyze_scaled_coupled_matrix
from r26_gu_emerson_reconstruction import make_gu_emerson_reconstruction_problem
from r26_gu_emerson_variables import gu_emerson_fields_from_state
from r26_gu_emerson_coupled_jacobian_audit import audit_gu_emerson_coupled_jacobian


def test_scaled_coupled_audit_detects_and_localizes_one_null_direction() -> None:
    nodes = 3
    size = nodes * nodes * 17
    matrix = np.eye(size)
    matrix[:, 1] = matrix[:, 0]
    report = analyze_scaled_coupled_matrix(matrix, nodes=nodes)
    assert report.numerical_rank == size - 1
    assert report.rank_deficiency == 1
    assert not report.full_rank
    assert report.scaled_reciprocal_condition < 1.0e-14
    assert np.isclose(sum(value for _, value in report.weakest_unknown_slot_energy), 1.0)
    assert np.isclose(sum(value for _, value in report.weakest_unknown_region_energy), 1.0)
    wall_total = dict(report.weakest_equation_region_energy)["wall"]
    assert np.isclose(
        sum(value for _, value in report.weakest_wall_equation_energy), wall_total
    )
    assert len(report.weakest_wall_equation_energy) == 17
    assert all(0 <= index < nodes for index in report.dominant_unknown_location[:2])
    assert 0 <= report.dominant_unknown_location[2] < 17


def test_coupled_jacobian_audit_blocks_grids_above_n8() -> None:
    case = gu_asme2009_cavity_case(9, kn=0.1, lid_speed_m_per_s=10.0)
    state = case.equilibrium_state()
    fields = gu_emerson_fields_from_state(
        state, x=case.x, y=case.y, mu=case.mu(state[..., 3])
    )
    try:
        audit_gu_emerson_coupled_jacobian(
            make_gu_emerson_reconstruction_problem(case), fields
        )
    except ValueError as exc:
        assert "restricted to N8" in str(exc)
    else:
        raise AssertionError("coupled Jacobian audit must remain blocked above N8")
