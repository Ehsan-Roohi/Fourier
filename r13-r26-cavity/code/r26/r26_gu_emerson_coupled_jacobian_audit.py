#!/usr/bin/env python3
"""N8-only rank audit of the complete transformed Gu--Emerson objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from r26_discretization import R26NodeBVP
from r26_fv_backend import fv_absolute_difference_step
from r26_gu_emerson_monolithic_oracle import EncodedGuEmersonMonolithicObjective
from r26_gu_emerson_variables import GuEmersonFields, gu_emerson_fields_as_planar17
from r26_solver import LogStateTransform, jacobian_sparsity
from r26_wall_conditions import WALL_EQUATION_ORDER


TRANSFORMED_SLOT_NAMES = (
    "rho",
    "vx",
    "vy",
    "theta",
    "h_x",
    "h_y",
    "g_xx",
    "g_xy",
    "g_yy",
    "gamma_xx",
    "gamma_xy",
    "gamma_yy",
    "omega_xxx",
    "omega_xxy",
    "omega_xyy",
    "omega_yyy",
    "chi",
)
WALL_ROW_NAMES = WALL_EQUATION_ORDER + (
    "extrap_pressure",
    "extrap_sigma_nt",
    "extrap_q_n",
    "extrap_m_nnn",
    "extrap_m_ntt",
    "extrap_R_nt",
)


@dataclass(frozen=True)
class GuEmersonCoupledJacobianReport:
    unknown_count: int
    numerical_rank: int
    rank_deficiency: int
    residual_evaluations: int
    invalid_evaluations: int
    jacobian_nonzeros: int
    largest_scaled_singular_value: float
    smallest_scaled_singular_value: float
    scaled_reciprocal_condition: float
    numerical_rank_tolerance: float
    full_rank: bool
    weakest_unknown_slot_energy: tuple[tuple[str, float], ...]
    weakest_unknown_region_energy: tuple[tuple[str, float], ...]
    weakest_unknown_side_energy: tuple[tuple[str, float], ...]
    weakest_equation_region_energy: tuple[tuple[str, float], ...]
    weakest_equation_side_energy: tuple[tuple[str, float], ...]
    weakest_bulk_equation_energy: tuple[tuple[str, float], ...]
    weakest_wall_equation_energy: tuple[tuple[str, float], ...]
    dominant_unknown_slot: str
    dominant_unknown_region: str
    dominant_unknown_side: str
    dominant_equation_region: str
    dominant_equation_side: str
    dominant_wall_equation: str
    dominant_unknown_location: tuple[int, int, int]
    dominant_equation_location: tuple[int, int, int]


def _normalized_energy(values: np.ndarray) -> np.ndarray:
    energy = np.square(np.asarray(values, dtype=float))
    total = float(np.sum(energy))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("singular vector has invalid zero energy")
    return energy / total


def _slot_energy(vector: np.ndarray, nodes: int) -> tuple[tuple[str, float], ...]:
    energy = _normalized_energy(vector.reshape(nodes, nodes, 17))
    values = np.sum(energy, axis=(0, 1))
    return tuple((name, float(values[index])) for index, name in enumerate(TRANSFORMED_SLOT_NAMES))


def _region_energy(vector: np.ndarray, nodes: int) -> tuple[tuple[str, float], ...]:
    energy = _normalized_energy(vector.reshape(nodes, nodes, 17))
    interior = float(np.sum(energy[1:-1, 1:-1]))
    corners = float(
        np.sum(energy[0, 0])
        + np.sum(energy[0, -1])
        + np.sum(energy[-1, 0])
        + np.sum(energy[-1, -1])
    )
    wall = float(1.0 - interior - corners)
    return (("bulk", interior), ("wall", wall), ("corner", corners))


def _side_energy(vector: np.ndarray, nodes: int) -> tuple[tuple[str, float], ...]:
    energy = _normalized_energy(vector.reshape(nodes, nodes, 17))
    return (
        ("left", float(np.sum(energy[1:-1, 0]))),
        ("right", float(np.sum(energy[1:-1, -1]))),
        ("bottom", float(np.sum(energy[0, 1:-1]))),
        ("top", float(np.sum(energy[-1, 1:-1]))),
    )


def _equation_energy(
    vector: np.ndarray, nodes: int, *, region: str
) -> tuple[tuple[str, float], ...]:
    energy = _normalized_energy(vector.reshape(nodes, nodes, 17))
    if region == "bulk":
        values = np.sum(energy[1:-1, 1:-1], axis=(0, 1))
        names = TRANSFORMED_SLOT_NAMES
    elif region == "wall":
        values = (
            np.sum(energy[1:-1, 0], axis=0)
            + np.sum(energy[1:-1, -1], axis=0)
            + np.sum(energy[0, 1:-1], axis=0)
            + np.sum(energy[-1, 1:-1], axis=0)
        )
        names = WALL_ROW_NAMES
    else:
        raise ValueError("equation energy region must be bulk or wall")
    return tuple((name, float(values[index])) for index, name in enumerate(names))


def analyze_scaled_coupled_matrix(
    matrix: np.ndarray, *, nodes: int
) -> GuEmersonCoupledJacobianReport:
    """Rank and localize a square planar-17 matrix after max-norm scaling."""

    value = np.asarray(matrix, dtype=float)
    expected = nodes * nodes * 17
    if value.shape != (expected, expected) or not np.isfinite(value).all():
        raise ValueError("coupled Jacobian must be a finite square planar-17 matrix")
    row = np.max(np.abs(value), axis=1)
    scaled = value.copy()
    if np.all(row > 0.0):
        scaled /= row[:, None]
    column = np.max(np.abs(scaled), axis=0)
    if np.all(column > 0.0):
        scaled /= column[None, :]
    left, singular, right_transpose = np.linalg.svd(scaled, full_matrices=False)
    largest = float(singular[0])
    smallest = float(singular[-1])
    tolerance = float(max(scaled.shape) * np.finfo(float).eps * largest)
    rank = int(np.count_nonzero(singular > tolerance))
    unknown_slots = _slot_energy(right_transpose[-1], nodes)
    unknown_regions = _region_energy(right_transpose[-1], nodes)
    unknown_sides = _side_energy(right_transpose[-1], nodes)
    equation_regions = _region_energy(left[:, -1], nodes)
    equation_sides = _side_energy(left[:, -1], nodes)
    bulk_equations = _equation_energy(left[:, -1], nodes, region="bulk")
    wall_equations = _equation_energy(left[:, -1], nodes, region="wall")
    unknown_location = tuple(
        int(index)
        for index in np.unravel_index(
            np.argmax(np.abs(right_transpose[-1])), (nodes, nodes, 17)
        )
    )
    equation_location = tuple(
        int(index)
        for index in np.unravel_index(
            np.argmax(np.abs(left[:, -1])), (nodes, nodes, 17)
        )
    )
    return GuEmersonCoupledJacobianReport(
        unknown_count=expected,
        numerical_rank=rank,
        rank_deficiency=expected - rank,
        residual_evaluations=0,
        invalid_evaluations=0,
        jacobian_nonzeros=int(np.count_nonzero(value)),
        largest_scaled_singular_value=largest,
        smallest_scaled_singular_value=smallest,
        scaled_reciprocal_condition=(smallest / largest if largest > 0.0 else 0.0),
        numerical_rank_tolerance=tolerance,
        full_rank=rank == expected,
        weakest_unknown_slot_energy=unknown_slots,
        weakest_unknown_region_energy=unknown_regions,
        weakest_unknown_side_energy=unknown_sides,
        weakest_equation_region_energy=equation_regions,
        weakest_equation_side_energy=equation_sides,
        weakest_bulk_equation_energy=bulk_equations,
        weakest_wall_equation_energy=wall_equations,
        dominant_unknown_slot=max(unknown_slots, key=lambda item: item[1])[0],
        dominant_unknown_region=max(unknown_regions, key=lambda item: item[1])[0],
        dominant_unknown_side=max(unknown_sides, key=lambda item: item[1])[0],
        dominant_equation_region=max(equation_regions, key=lambda item: item[1])[0],
        dominant_equation_side=max(equation_sides, key=lambda item: item[1])[0],
        dominant_wall_equation=max(wall_equations, key=lambda item: item[1])[0],
        dominant_unknown_location=unknown_location,
        dominant_equation_location=equation_location,
    )


def audit_gu_emerson_coupled_jacobian(
    problem: R26NodeBVP,
    fields: GuEmersonFields,
    *,
    invalid_penalty: float = 1.0e8,
) -> GuEmersonCoupledJacobianReport:
    """Build, scale and densely rank the complete N8 transformed Jacobian."""

    if problem.case.nodes > 8:
        raise ValueError("coupled Gu--Emerson Jacobian audit is restricted to N8")
    packed = gu_emerson_fields_as_planar17(fields)
    transform = LogStateTransform(problem.shape)
    encoded = transform.encode(packed)
    objective = EncodedGuEmersonMonolithicObjective(
        problem, transform, invalid_penalty
    )
    evaluations = 0

    def sampled(vector: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1
        return objective(vector)

    jacobian = approx_derivative(
        sampled,
        encoded,
        method="2-point",
        abs_step=fv_absolute_difference_step(encoded),
        sparsity=jacobian_sparsity(problem),
    ).tocsc()
    report = analyze_scaled_coupled_matrix(
        jacobian.toarray(), nodes=problem.case.nodes
    )
    return GuEmersonCoupledJacobianReport(
        **{
            **report.__dict__,
            "residual_evaluations": evaluations,
            "invalid_evaluations": objective.invalid_evaluations,
            "jacobian_nonzeros": int(jacobian.nnz),
        }
    )


__all__ = [
    "GuEmersonCoupledJacobianReport",
    "TRANSFORMED_SLOT_NAMES",
    "WALL_ROW_NAMES",
    "analyze_scaled_coupled_matrix",
    "audit_gu_emerson_coupled_jacobian",
]
