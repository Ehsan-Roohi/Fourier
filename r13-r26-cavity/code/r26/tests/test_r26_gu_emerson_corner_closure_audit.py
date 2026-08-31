from dataclasses import replace

import numpy as np

from r26_cases import gu_asme2009_cavity_case
from r26_gu_emerson_corner_closure_audit import (
    corner_excluding_wall_gradients,
    make_corner_excluding_gu_emerson_problem,
)
from r26_gu_emerson_monolithic_oracle import EncodedGuEmersonMonolithicObjective
from r26_gu_emerson_variables import (
    gu_emerson_fields_as_planar17,
    gu_emerson_fields_from_state,
)
from r26_solver import LogStateTransform
from r26_tensor_closures import finite_difference_gradients


def _linear_state(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(x, y)
    state = np.zeros((y.size, x.size, 17), dtype=float)
    state[..., 0] = 1.2 + 0.03 * xx - 0.02 * yy
    state[..., 3] = 0.9 + 0.01 * xx + 0.04 * yy
    for slot in range(17):
        if slot not in (0, 3):
            state[..., slot] = slot * (0.002 * xx - 0.003 * yy)
    return state


def test_corner_excluding_gradients_preserve_linear_manufactured_derivatives() -> None:
    x = np.asarray([0.0, 0.08, 0.24, 0.51, 0.78, 0.93, 1.0])
    y = np.asarray([0.0, 0.12, 0.31, 0.58, 0.82, 1.0])
    state = _linear_state(x, y)
    baseline = finite_difference_gradients(state, x=x, y=y, edge_order=2)
    corrected = corner_excluding_wall_gradients(state, x=x, y=y)
    for name in baseline.__dataclass_fields__:
        np.testing.assert_allclose(
            getattr(corrected, name), getattr(baseline, name), rtol=0.0, atol=2.0e-13
        )


def test_corner_value_cannot_leak_into_adjacent_smooth_wall_tangent() -> None:
    x = np.linspace(0.0, 1.0, 8)
    y = np.linspace(0.0, 1.0, 8)
    baseline_state = _linear_state(x, y)
    changed = baseline_state.copy()
    changed[-1, 0, 10] += 0.4

    global_before = finite_difference_gradients(baseline_state, x=x, y=y)
    global_after = finite_difference_gradients(changed, x=x, y=y)
    corrected_before = corner_excluding_wall_gradients(baseline_state, x=x, y=y)
    corrected_after = corner_excluding_wall_gradients(changed, x=x, y=y)

    assert not np.array_equal(
        global_before.R[-2, 0, 1], global_after.R[-2, 0, 1]
    )
    np.testing.assert_array_equal(
        corrected_before.R[-2, 0, 1], corrected_after.R[-2, 0, 1]
    )


def test_corner_excluding_objective_is_exact_at_stationary_equilibrium() -> None:
    case = replace(
        gu_asme2009_cavity_case(5, kn=0.1, lid_speed_m_per_s=10.0),
        lid_velocity=0.0,
    )
    state = case.equilibrium_state()
    fields = gu_emerson_fields_from_state(
        state, x=case.x, y=case.y, mu=case.mu(state[..., 3])
    )
    packed = gu_emerson_fields_as_planar17(fields)
    problem = make_corner_excluding_gu_emerson_problem(case)
    transform = LogStateTransform(problem.shape)
    objective = EncodedGuEmersonMonolithicObjective(problem, transform, 1.0e8)
    assert np.max(np.abs(objective(transform.encode(packed)))) == 0.0
