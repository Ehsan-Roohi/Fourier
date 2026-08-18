#!/usr/bin/env python3
"""Independent validation diagnostics for private wall-inclusive R26 states.

These checks intentionally do not reuse the scalar nonlinear merit function.
They evaluate telescoping global balances, wall traction and heat budgets,
Gu--Emerson effective pressure, and the leading small-Kn R13/NSF limits from
the decoded physical state.  A small solver residual is therefore necessary
but not sufficient for a state to pass this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from r26_cases import CavityCase
from r26_fv_backend import (
    compatible_wall_fluxes,
    interior_control_volume_widths,
)
from r26_postprocess import rana_global_metrics
from r26_state import STATE_INDEX, planar_state_to_tensors, validate_planar_state
from r26_tensor_closures import (
    closures_from_tensors,
    finite_difference_gradients,
    stf2_project,
    stf3_project,
)
from r26_wall_conditions import (
    WallParameters,
    effective_pressure,
    extract_face_quantities,
    project_closures,
    square_wall_frame,
    wall_residual,
)


def _rms(value: np.ndarray) -> float:
    a = np.asarray(value, dtype=float)
    return float(np.sqrt(np.mean(a * a)))


def _cv_integral(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    value = np.asarray(field, dtype=float)
    dx = interior_control_volume_widths(x)
    dy = interior_control_volume_widths(y)
    weights = dy[:, None] * dx[None, :]
    trailing = (1,) * (value.ndim - 2)
    return np.sum(value[1:-1, 1:-1] * weights.reshape(weights.shape + trailing), axis=(0, 1))


def _quotient_gradient(
    field: np.ndarray,
    rho: np.ndarray,
    gradient: np.ndarray,
    grad_rho: np.ndarray,
) -> np.ndarray:
    rank = field.ndim - rho.ndim
    rho_field = rho[(Ellipsis,) + (None,) * rank]
    rho_gradient = rho[(Ellipsis,) + (None,) * (rank + 1)]
    return gradient / rho_gradient - np.expand_dims(field, axis=2) * grad_rho[
        (Ellipsis,) + (None,) * rank
    ] / rho_gradient**2


def leading_r13_nsf_diagnostics(
    state: np.ndarray,
    case: CavityCase,
    *,
    core_layers: int = 2,
) -> dict[str, Any]:
    """Compare a state to its leading Chapman--Enskog R13/NSF closures.

    The R13 quantities here are the leading algebraic terms obtained from the
    R26 ``m/R/Delta`` collision balances, not a full external R13 solution.
    This is an asymptotic regression diagnostic and must not be presented as
    validation against R13 or DSMC.
    """

    u = validate_planar_state(state)
    tensors = planar_state_to_tensors(u)
    gradients = finite_difference_gradients(u, x=case.x, y=case.y, edge_order=2)
    rho = np.asarray(tensors.rho)
    mu = np.asarray(case.mu(tensors.theta))
    sigma = np.asarray(tensors.sigma)
    q = np.asarray(tensors.heat_flux)
    mm = np.asarray(tensors.m)
    rr = np.asarray(tensors.R)
    delta = np.asarray(tensors.Delta)

    sigma_nsf = -2.0 * mu[..., None, None] * stf2_project(gradients.velocity)
    q_nsf = -15.0 / 4.0 * mu[..., None] * gradients.theta
    grad_sigma_ratio = _quotient_gradient(sigma, rho, gradients.sigma, gradients.rho)
    grad_q_ratio = _quotient_gradient(q, rho, gradients.heat_flux, gradients.rho)
    m_r13 = -2.0 * mu[..., None, None, None] * stf3_project(
        np.moveaxis(grad_sigma_ratio, 2, -1)
    )
    R_r13 = -24.0 / 5.0 * mu[..., None, None] * stf2_project(
        np.swapaxes(grad_q_ratio, 2, 3)
    )
    div_q_over_rho = np.einsum("...ii->...", grad_q_ratio)
    Delta_r13 = -12.0 * mu * div_q_over_rho

    n = case.nodes
    layers = min(max(int(core_layers), 1), max(1, (n - 2) // 2))
    mask = np.zeros((n, n), dtype=bool)
    mask[layers : n - layers, layers : n - layers] = True
    if not np.any(mask):
        mask[1:-1, 1:-1] = True

    def comparison(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
        a = np.asarray(actual)[mask]
        p = np.asarray(predicted)[mask]
        defect = a - p
        scale = max(_rms(p), _rms(a), 1.0e-30)
        return {
            "actual_rms": _rms(a),
            "leading_rms": _rms(p),
            "defect_rms": _rms(defect),
            "relative_defect_rms": _rms(defect) / scale,
        }

    return {
        "semantics": (
            "leading small-Kn algebraic R13/NSF regression only; not an "
            "external R13 solution and not validation"
        ),
        "core_layers_excluded": layers,
        "sigma_vs_NSF": comparison(sigma, sigma_nsf),
        "q_vs_NSF": comparison(q, q_nsf),
        "m_vs_leading_R13": comparison(mm, m_r13),
        "R_vs_leading_R13": comparison(rr, R_r13),
        "Delta_vs_leading_R13": comparison(delta, Delta_r13),
    }


def global_balance_diagnostics(state: np.ndarray, case: CavityCase) -> dict[str, Any]:
    """Evaluate independent wall and domain-integrated balances."""

    u = validate_planar_state(state)
    tensors = planar_state_to_tensors(u)
    gradients = finite_difference_gradients(u, x=case.x, y=case.y, edge_order=2)
    mu = np.asarray(case.mu(tensors.theta))
    closures = closures_from_tensors(
        tensors,
        gradients,
        mu=mu,
        coefficient_mode=case.r26_closure_mode,
    )
    walls = compatible_wall_fluxes(u, closures)
    dx = interior_control_volume_widths(case.x)
    dy = interior_control_volume_widths(case.y)

    momentum_net = (
        np.sum(dy[:, None] * (walls.momentum_x[1:-1, -1] - walls.momentum_x[1:-1, 0]), axis=0)
        + np.sum(dx[:, None] * (walls.momentum_y[-1, 1:-1] - walls.momentum_y[0, 1:-1]), axis=0)
    )
    theta_flux_net = float(
        np.sum(dy * (walls.theta_x[1:-1, -1] - walls.theta_x[1:-1, 0]))
        + np.sum(dx * (walls.theta_y[-1, 1:-1] - walls.theta_y[0, 1:-1]))
    )
    q_flux_net = 1.5 * theta_flux_net

    pressure = np.asarray(tensors.rho) * np.asarray(tensors.theta)
    div_u = np.einsum("...ii->...", gradients.velocity)
    sigma_grad_u = np.einsum("...ij,...ij->...", tensors.sigma, gradients.velocity)
    internal_work = float(_cv_integral(pressure * div_u + sigma_grad_u, case.x, case.y))

    top_shear_state = u[-1, 1:-1, STATE_INDEX["sigma_xy"]]
    top_shear_flux = walls.momentum_y[-1, 1:-1, 0]
    top_traction_mismatch = float(np.max(np.abs(top_shear_state - top_shear_flux), initial=0.0))
    smooth_top_integral = float(np.sum(dx * top_shear_state))
    reduction = np.sqrt(2.0) / abs(case.lid_velocity)
    rana_trapezoid = rana_global_metrics(
        u,
        lid_velocity=case.lid_velocity,
        x=case.x,
        y=case.y,
    )

    palpha: list[float] = []
    wall_normal_velocity: list[float] = []
    wall_residual_linf: list[float] = []
    boundary_entries = (
        ("left", ((j, 0) for j in range(1, case.nodes - 1))),
        ("right", ((j, case.nodes - 1) for j in range(1, case.nodes - 1))),
        ("bottom", ((0, i) for i in range(1, case.nodes - 1))),
        ("top", ((case.nodes - 1, i) for i in range(1, case.nodes - 1))),
    )
    velocity = np.asarray(tensors.velocity)
    for side, entries in boundary_entries:
        frame = square_wall_frame(side)
        parameters = WallParameters(
            wall_temperature=case.wall_temperature,
            accommodation=case.accommodation,
            gas_constant=case.gas_constant,
            wall_velocity=case.wall_velocity(side),
        )
        for j, i in entries:
            local_tensors = planar_state_to_tensors(u[j, i])
            local_closures = type(closures)(
                phi=np.asarray(closures.phi[j, i]),
                psi=np.asarray(closures.psi[j, i]),
                Omega=np.asarray(closures.Omega[j, i]),
                equation25_mode=closures.equation25_mode,
                provenance=closures.provenance,
                coefficient_mode=closures.coefficient_mode,
            )
            free, unknowns = extract_face_quantities(
                local_tensors, frame, gas_constant=case.gas_constant
            )
            projected = project_closures(local_closures, frame)
            palpha.append(effective_pressure(free, unknowns, projected, parameters))
            wall_normal_velocity.append(
                float(np.dot(velocity[j, i] - case.wall_velocity(side), frame.normal))
            )
            residual = wall_residual(
                u[j, i],
                local_closures,
                frame.normal,
                frame.tangent,
                case.wall_velocity(side),
                case.wall_temperature,
                alpha=case.accommodation,
                gas_constant=case.gas_constant,
            )
            wall_residual_linf.append(float(np.max(np.abs(residual))))

    return {
        "mass_boundary_flux": 0.0,
        "momentum_boundary_flux": momentum_net,
        "momentum_boundary_flux_linf": float(np.max(np.abs(momentum_net))),
        "theta_boundary_flux": theta_flux_net,
        "heat_boundary_flux": q_flux_net,
        "volume_pressure_plus_viscous_work": internal_work,
        "internal_energy_balance_error": q_flux_net + internal_work,
        "top_traction_state_vs_flux_linf": top_traction_mismatch,
        "top_shear_integral_smooth_common_cv": smooth_top_integral,
        "D_smooth_common_cv": abs(reduction * smooth_top_integral),
        "D_corner_inclusive_trapezoid": rana_trapezoid["D"],
        "G_corner_inclusive_trapezoid": rana_trapezoid["G"],
        "wall_effective_pressure_min": float(np.min(palpha)),
        "wall_effective_pressure_max": float(np.max(palpha)),
        "wall_normal_velocity_linf": float(np.max(np.abs(wall_normal_velocity))),
        "independent_wall_residual_linf": float(np.max(wall_residual_linf)),
        "thermodynamic_pressure_min": float(np.min(pressure)),
        "thermodynamic_pressure_max": float(np.max(pressure)),
        "corner_policy": (
            "global smooth-wall flux quadrature excludes all four corner nodes; "
            "legacy Rana D/G trapezoid is reported separately"
        ),
    }


__all__ = ["global_balance_diagnostics", "leading_r13_nsf_diagnostics"]
