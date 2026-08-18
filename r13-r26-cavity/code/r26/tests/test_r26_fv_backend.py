from __future__ import annotations

import numpy as np

from r26_bulk_equations import bulk_residual_grid
from r26_cases import rana_first_case
from r26_discretization import R26NodeBVP
from r26_fv_backend import (
    DEFAULT_FV_FD_STEP_SCALE,
    compatible_face_fields,
    compatible_fv_bulk_residual,
    compatible_wall_fluxes,
    fv_absolute_difference_step,
    impermeable_wall_mass_divergence,
    interior_control_volume_widths,
    wall_bounded_control_volume_weights,
    wall_bounded_face_divergence,
)
from r26_state import NVAR, planar_state_to_tensors
from r26_tensor_closures import closures_from_tensors, finite_difference_gradients


def _alternating(nodes: int) -> np.ndarray:
    j, i = np.indices((nodes, nodes))
    return (-1.0) ** (i + j)


def test_fv_finite_difference_step_has_a_floor_for_tiny_moments() -> None:
    encoded = np.asarray((0.0, 1.0e-14, -1.0e-8, 2.0, -5.0))
    step = fv_absolute_difference_step(encoded)
    assert np.array_equal(
        step,
        DEFAULT_FV_FD_STEP_SCALE * (1.0 + np.abs(encoded)),
    )
    assert np.min(step) == DEFAULT_FV_FD_STEP_SCALE
    try:
        fv_absolute_difference_step(encoded, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a nonpositive finite-difference scale must be rejected")


def test_impermeable_wall_mass_divergence_has_exact_conservation_identity() -> None:
    # The deliberately nonuniform node positions make the first/last interior
    # volumes visibly different from a node-centred midpoint volume.  Their
    # widths are binary-exact here so the global identity is exactly zero,
    # rather than merely small within a tolerance.
    coordinate = np.asarray((0.0, 3.0 / 16.0, 5.0 / 16.0, 11.0 / 16.0, 1.0))
    mass_x = np.arange(20.0).reshape(5, 4) - 7.0
    mass_y = np.arange(20.0).reshape(4, 5)[::-1] - 11.0
    divergence = impermeable_wall_mass_divergence(
        mass_x, mass_y, coordinate, coordinate
    )
    widths = interior_control_volume_widths(coordinate)
    integrated = np.sum(
        divergence[1:-1, 1:-1] * widths[:, None] * widths[None, :]
    )
    assert integrated == 0.0

    # Wall-to-first-interior interpolants are not physical wall fluxes.  They
    # must not enter the interior continuity control volumes, whose actual
    # boundary normal flux is exactly zero for an impermeable cavity.
    changed_x = mass_x.copy()
    changed_y = mass_y.copy()
    changed_x[:, (0, -1)] += 1.0e6
    changed_y[(0, -1), :] -= 1.0e6
    assert np.array_equal(
        divergence,
        impermeable_wall_mass_divergence(
            changed_x, changed_y, coordinate, coordinate
        ),
    )


def test_wall_bounded_weights_match_the_balance_geometry() -> None:
    x = np.asarray((0.0, 0.125, 0.375, 0.75, 1.0))
    y = np.asarray((0.0, 0.25, 0.5, 0.625, 1.0))
    weights = wall_bounded_control_volume_weights(x, y)
    expected = np.zeros((5, 5))
    expected[1:-1, 1:-1] = (
        interior_control_volume_widths(y)[:, None]
        * interior_control_volume_widths(x)[None, :]
    )
    assert np.allclose(weights, expected, rtol=0.0, atol=2.0e-16)
    assert float(np.sum(weights)) == 1.0
    assert np.array_equal(weights[[0, -1]], np.zeros((2, 5)))
    assert np.array_equal(weights[:, [0, -1]], np.zeros((5, 2)))


def test_common_wall_bounded_tensor_fluxes_telescope_to_physical_walls() -> None:
    coordinate = np.asarray((0.0, 3.0 / 16.0, 5.0 / 16.0, 11.0 / 16.0, 1.0))
    trailing = (2, 3)
    face_x = np.arange(120.0).reshape((5, 4) + trailing) - 31.0
    face_y = (np.arange(120.0).reshape((4, 5) + trailing) - 47.0)[::-1]
    wall_x = np.arange(150.0).reshape((5, 5) + trailing) / 7.0
    wall_y = (np.arange(150.0).reshape((5, 5) + trailing) - 91.0) / 11.0
    divergence = wall_bounded_face_divergence(
        face_x, face_y, wall_x, wall_y, coordinate, coordinate
    )
    widths = interior_control_volume_widths(coordinate)
    integrated = np.sum(
        divergence[1:-1, 1:-1]
        * widths[:, None, None, None]
        * widths[None, :, None, None],
        axis=(0, 1),
    )
    expected = np.sum(
        widths[:, None, None]
        * (wall_x[1:-1, -1] - wall_x[1:-1, 0]),
        axis=0,
    ) + np.sum(
        widths[:, None, None]
        * (wall_y[-1, 1:-1] - wall_y[0, 1:-1]),
        axis=0,
    )
    assert np.allclose(integrated, expected, rtol=0.0, atol=2.0e-13)

    # Wall-to-interior midpoint fluxes are outside the physical control volume
    # and cannot change any balance family, including tensor-valued ones.
    changed_x = face_x.copy()
    changed_y = face_y.copy()
    changed_x[:, (0, -1)] += 1.0e8
    changed_y[(0, -1)] -= 1.0e8
    changed = wall_bounded_face_divergence(
        changed_x, changed_y, wall_x, wall_y, coordinate, coordinate
    )
    assert np.array_equal(changed, divergence)


def test_compatible_fv_equilibrium_is_exact_and_face_tensors_remain_stf() -> None:
    case = rana_first_case(5)
    state = case.equilibrium_state()
    mu = case.mu(state[..., 3])
    residual = compatible_fv_bulk_residual(state, case.x, case.y, mu)
    assert np.array_equal(residual, np.zeros((5, 5, NVAR)))
    gradients = finite_difference_gradients(state, x=case.x, y=case.y)
    closures = closures_from_tensors(planar_state_to_tensors(state), gradients, mu=mu)
    faces = compatible_face_fields(state, case.x, case.y, mu, closures)
    for rank2 in (faces.sigma_x, faces.sigma_y, faces.R_x, faces.R_y):
        assert np.max(np.abs(np.trace(rank2, axis1=-2, axis2=-1))) < 2.0e-13
    for rank3 in (faces.m_x, faces.m_y, faces.psi_x, faces.psi_y):
        assert np.max(np.abs(np.einsum("...iik->...k", rank3))) < 2.0e-12
    for rank4 in (faces.phi_x, faces.phi_y):
        assert np.max(np.abs(np.einsum("...iikl->...kl", rank4))) < 2.0e-11


def test_physical_wall_fluxes_are_the_raw_R26_coordinate_fluxes() -> None:
    case, state = _smooth_state(7)
    mu = case.mu(state[..., 3])
    tensors = planar_state_to_tensors(state)
    gradients = finite_difference_gradients(state, x=case.x, y=case.y)
    closures = closures_from_tensors(tensors, gradients, mu=mu)
    walls = compatible_wall_fluxes(state, closures)
    pressure = np.asarray(tensors.rho) * np.asarray(tensors.theta)
    sigma = np.asarray(tensors.sigma)
    q = np.asarray(tensors.heat_flux)
    mm = np.asarray(tensors.m)
    rr = np.asarray(tensors.R)

    assert np.array_equal(walls.mass_x, np.zeros_like(pressure))
    assert np.array_equal(walls.mass_y, np.zeros_like(pressure))
    expected_momentum_x = sigma[..., :, 0].copy()
    expected_momentum_y = sigma[..., :, 1].copy()
    expected_momentum_x[..., 0] += pressure
    expected_momentum_y[..., 1] += pressure
    assert np.array_equal(walls.momentum_x, expected_momentum_x)
    assert np.array_equal(walls.momentum_y, expected_momentum_y)
    assert np.array_equal(walls.theta_x, 2.0 / 3.0 * q[..., 0])
    assert np.array_equal(walls.theta_y, 2.0 / 3.0 * q[..., 1])
    assert np.array_equal(walls.stress_x, mm[..., :, :, 0])
    assert np.array_equal(walls.stress_y, mm[..., :, :, 1])
    assert np.array_equal(walls.heat_x, 0.5 * rr[..., :, 0])
    assert np.array_equal(walls.heat_y, 0.5 * rr[..., :, 1])
    assert np.array_equal(walls.m_x, np.asarray(closures.phi)[..., :, :, :, 0])
    assert np.array_equal(walls.m_y, np.asarray(closures.phi)[..., :, :, :, 1])
    assert np.array_equal(walls.R_x, np.asarray(closures.psi)[..., :, :, 0])
    assert np.array_equal(walls.R_y, np.asarray(closures.psi)[..., :, :, 1])
    assert np.array_equal(walls.Delta_x, np.asarray(closures.Omega)[..., 0])
    assert np.array_equal(walls.Delta_y, np.asarray(closures.Omega)[..., 1])


def test_rhie_chow_correction_vanishes_for_a_linear_pressure_field() -> None:
    case = rana_first_case(7)
    state = case.equilibrium_state()
    state[..., 0] = 1.0 + 0.01 * np.broadcast_to(case.x, (case.nodes, case.nodes))
    mu = case.mu(state[..., 3])
    central = bulk_residual_grid(state, case.x, case.y, mu)
    compatible = compatible_fv_bulk_residual(state, case.x, case.y, mu)
    assert np.allclose(
        compatible[1:-1, 1:-1], central[1:-1, 1:-1], rtol=0.0, atol=1.0e-12
    )


def test_compatible_transport_detects_all_three_collocated_checkerboards() -> None:
    case = rana_first_case(5)
    alternating = _alternating(case.nodes)
    amplitudes: dict[str, float] = {}
    for name in ("velocity", "pressure", "isobaric_temperature"):
        state = case.equilibrium_state()
        if name == "velocity":
            state[..., 1] = 1.0e-5 * alternating
        elif name == "pressure":
            state[..., 0] = 1.0 + 1.0e-5 * alternating
        else:
            state[..., 0] = 1.0 + 1.0e-5 * alternating
            state[..., 3] = 1.0 / state[..., 0]
        mu = case.mu(state[..., 3])
        central = bulk_residual_grid(state, case.x, case.y, mu)
        compatible = compatible_fv_bulk_residual(state, case.x, case.y, mu)
        assert np.max(np.abs(central[1:-1, 1:-1])) < 2.0e-12
        amplitudes[name] = float(np.max(np.abs(compatible[1:-1, 1:-1])))
    assert amplitudes["velocity"] > 1.0e-6
    assert amplitudes["pressure"] > 1.0e-4
    assert amplitudes["isobaric_temperature"] > 1.0e-6


def test_transport_replacement_does_not_change_body_force_or_nonmomentum_rows() -> None:
    case, state = _smooth_state(9)
    mu = case.mu(state[..., 3])
    force = np.zeros((case.nodes, case.nodes, 2))
    y, x = np.meshgrid(case.y, case.x, indexing="ij")
    force[..., 0] = 0.013 * (1.0 + x - 0.5 * y)
    force[..., 1] = -0.009 * (1.0 - 0.25 * x + y)
    central_zero = bulk_residual_grid(state, case.x, case.y, mu)
    central_force = bulk_residual_grid(state, case.x, case.y, mu, body_force=force)
    compatible_zero = compatible_fv_bulk_residual(state, case.x, case.y, mu)
    compatible_force = compatible_fv_bulk_residual(
        state, case.x, case.y, mu, body_force=force
    )
    central_delta = central_force - central_zero
    compatible_delta = compatible_force - compatible_zero
    assert np.allclose(compatible_delta, central_delta, rtol=0.0, atol=3.0e-15)
    assert np.allclose(compatible_delta[..., 1:3], -force, rtol=0.0, atol=3.0e-15)
    other = np.delete(compatible_delta, (1, 2), axis=-1)
    assert np.max(np.abs(other)) < 3.0e-15


def _smooth_state(nodes: int) -> tuple[object, np.ndarray]:
    case = rana_first_case(nodes)
    y, x = np.meshgrid(case.y, case.x, indexing="ij")
    state = case.equilibrium_state()
    state[..., 0] = 1.0 + 0.02 * np.sin(np.pi * x) * np.sin(np.pi * y)
    state[..., 1] = 0.03 * np.sin(np.pi * x) * np.sin(np.pi * y)
    state[..., 2] = -0.02 * np.cos(np.pi * x) * np.sin(np.pi * y)
    state[..., 3] = 1.0 + 0.015 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)
    for component in range(4, NVAR):
        state[..., component] = (
            0.001
            / (component + 1.0)
            * np.sin((1 + component % 2) * np.pi * x)
            * np.sin((1 + component % 3) * np.pi * y)
        )
    return case, state


def test_bulk_continuity_rows_telescope_to_zero_wall_flux() -> None:
    case, state = _smooth_state(9)
    # Deliberately perturb normal velocities on boundary nodes.  Such a state
    # is rejected by the independent wall rows, but the continuity operator
    # must still apply the prescribed impermeable boundary flux rather than a
    # wall-to-interior interpolation.
    state[:, 0, 1] = 0.17
    state[:, -1, 1] = -0.11
    state[0, :, 2] = 0.13
    state[-1, :, 2] = -0.19
    residual = compatible_fv_bulk_residual(
        state, case.x, case.y, case.mu(state[..., 3])
    )
    dx = interior_control_volume_widths(case.x)
    dy = interior_control_volume_widths(case.y)
    terms = residual[1:-1, 1:-1, 0] * dy[:, None] * dx[None, :]
    roundoff = 16.0 * np.finfo(float).eps * max(float(np.sum(np.abs(terms))), 1.0)
    assert abs(float(np.sum(terms))) <= roundoff

    # The square BVP may therefore replace one redundant continuity row with
    # the independent mean-mass border without discarding physics.  Restoring
    # the reported held row reconstructs the exact nonlinear identity.
    problem = R26NodeBVP(
        case,
        bulk_operator=compatible_fv_bulk_residual,
        mass_weights=wall_bounded_control_volume_weights(case.x, case.y),
    )
    evaluation = problem.evaluate(state)
    continuity = evaluation.unscaled_residual[1:-1, 1:-1, 0].copy()
    held_j = problem.mass_j - 1
    held_i = problem.mass_i - 1
    continuity[held_j, held_i] = evaluation.diagnostics.held_out_continuity
    bordered_terms = continuity * dy[:, None] * dx[None, :]
    assert abs(float(np.sum(bordered_terms))) <= roundoff
    other_sum = float(np.sum(bordered_terms) - bordered_terms[held_j, held_i])
    held_weight = float(dy[held_j] * dx[held_i])
    predicted_held = -other_sum / held_weight
    assert abs(predicted_held - evaluation.diagnostics.held_out_continuity) <= roundoff


def test_boundary_inclusive_compatible_vs_point_difference_shrinks_for_all_rows() -> None:
    component_errors: list[np.ndarray] = []
    for nodes in (17, 33):
        case, state = _smooth_state(nodes)
        mu = case.mu(state[..., 3])
        central = bulk_residual_grid(state, case.x, case.y, mu)
        compatible = compatible_fv_bulk_residual(state, case.x, case.y, mu)
        difference = compatible[1:-1, 1:-1] - central[1:-1, 1:-1]
        component_errors.append(np.sqrt(np.mean(difference * difference, axis=(0, 1))))
    # Every one of the 17 physical rows, not merely a global norm dominated
    # by continuity/energy, approaches the independent point-central form.
    assert np.all(component_errors[1] < component_errors[0] / 1.8)


def test_wall_bounded_continuity_converges_on_no_penetration_streamfunction() -> None:
    errors: list[float] = []
    for nodes in (33, 65):
        case = rana_first_case(nodes)
        y, x = np.meshgrid(case.y, case.x, indexing="ij")
        state = case.equilibrium_state()
        # psi=sin(pi*x)^2 sin(pi*y)^2 gives analytic div(u)=0 and zero
        # normal velocity on all four walls.
        state[..., 1] = (
            2.0
            * np.pi
            * np.sin(np.pi * x) ** 2
            * np.sin(np.pi * y)
            * np.cos(np.pi * y)
        )
        state[..., 2] = (
            -2.0
            * np.pi
            * np.sin(np.pi * x)
            * np.cos(np.pi * x)
            * np.sin(np.pi * y) ** 2
        )
        residual = compatible_fv_bulk_residual(
            state, case.x, case.y, case.mu(state[..., 3])
        )
        errors.append(float(np.max(np.abs(residual[1:-1, 1:-1, 0]))))
    # The boundary-adjacent volumes are first-order because their stored node
    # is not at the geometric centroid; the full-domain operator must still be
    # consistent, and the observed factor approaches two under h refinement.
    assert errors[1] < errors[0] / 1.8
