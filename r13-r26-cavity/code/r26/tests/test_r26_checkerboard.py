from __future__ import annotations

import numpy as np

from r26_cases import CavityCase, KnudsenConvention, ViscosityModel
from r26_corner_bvp import R26PhysicalCornerBVP
from r26_staggered_backend import (
    make_oriented_second_order_operators,
    oriented_second_order_matrix,
)


def _case(nodes: int = 5, lid: float = 0.0) -> CavityCase:
    return CavityCase(
        name="checkerboard-test",
        nodes=nodes,
        kn=0.01,
        kn_convention=KnudsenConvention.RANA,
        lid_velocity=lid,
        viscosity=ViscosityModel.power_law(1.0),
    )


def _problem(nodes: int, ox: str, oy: str, lid: float = 0.0) -> R26PhysicalCornerBVP:
    bulk, closures = make_oriented_second_order_operators(ox, oy)
    return R26PhysicalCornerBVP(
        _case(nodes, lid), bulk_operator=bulk, closure_operator=closures
    )


def test_o2_biased_matrices_are_quadratic_exact_and_reflected_pairs() -> None:
    for nodes in (5, 6, 9):
        x = np.linspace(0.0, 1.0, nodes)
        forward = oriented_second_order_matrix(x, "forward")
        backward = oriented_second_order_matrix(x, "backward")
        reflection = np.eye(nodes)[::-1]
        assert np.allclose(forward @ (x * x), 2.0 * x, rtol=0.0, atol=2.0e-14)
        assert np.allclose(backward @ (x * x), 2.0 * x, rtol=0.0, atol=2.0e-14)
        assert np.allclose(backward, -reflection @ forward @ reflection, rtol=0.0, atol=2.0e-14)
        assert np.linalg.matrix_rank(forward) == nodes - 1
        assert np.linalg.matrix_rank(backward) == nodes - 1


def test_o2_equilibrium_residual_is_exact_on_n5_and_n6() -> None:
    for nodes in (5, 6):
        problem = _problem(nodes, "forward", "forward")
        evaluation = problem.evaluate(problem.case.equilibrium_state())
        assert np.max(np.abs(evaluation.residual)) <= 4.0e-15
        assert evaluation.diagnostics.held_out_continuity == 0.0


def test_explicit_corner_rows_are_two_normal_kinematic_and_isothermal() -> None:
    problem = _problem(5, "forward", "forward")
    state = problem.case.equilibrium_state()
    state[0, 0, 1] = 0.02
    state[0, 0, 2] = -0.03
    state[0, 0, 3] = 1.04
    residual = problem.evaluate(state).unscaled_residual
    assert np.allclose(
        residual[0, 0, 1:4], np.asarray((0.02, -0.03, 0.04)), rtol=0.0, atol=1.0e-15
    )


def test_y_backward_has_exact_wall_only_lid_branch_and_is_rejected() -> None:
    lid = 1.0e-4
    backward_y = _problem(5, "forward", "backward", lid=lid)
    state = backward_y.case.equilibrium_state()
    state[-1, 1:-1, 1] = lid
    # This exact residual-zero state has no interior response.  The test
    # deliberately records why y-backward must never be accepted physically.
    rejected = backward_y.evaluate(state)
    assert rejected.diagnostics.total_linf == 0.0
    interior_nonequilibrium = np.concatenate(
        (state[1:-1, 1:-1, 1:3].ravel(), state[1:-1, 1:-1, 4:].ravel())
    )
    assert np.max(np.abs(interior_nonequilibrium), initial=0.0) == 0.0

    forward_y = _problem(5, "forward", "forward", lid=lid)
    coupled_residual = forward_y.evaluate(state)
    assert coupled_residual.diagnostics.total_linf >= lid
