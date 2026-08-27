#!/usr/bin/env python3
"""Independent numerical-rank and cross-discretization checks for THOR R26.

The THOR candidate and the historical SER--PTC candidate solve the same
Maxwell-molecule R26 case with different spatial/nonlinear algorithms.  This
module deliberately compares their physical states without requiring either
state to satisfy the other discretization exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize._numdiff import approx_derivative

from r26_fv_backend import fv_absolute_difference_step
from r26_postprocess import rana_global_metrics
from r26_solver import EncodedR26Objective, LogStateTransform, jacobian_sparsity
from r26_state import STATE_ORDER, validate_planar_state


@dataclass(frozen=True)
class NumericalRankReport:
    unknown_count: int
    numerical_rank: int
    residual_evaluations: int
    invalid_evaluations: int
    jacobian_nonzeros: int
    largest_scaled_singular_value: float
    smallest_scaled_singular_value: float
    scaled_reciprocal_condition: float
    numerical_rank_tolerance: float
    minimum_required_reciprocal_condition: float
    full_rank: bool
    passed: bool


def state_sha256(state: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(state, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(b"|<f8|")
    digest.update(value.tobytes())
    return digest.hexdigest()


def raw_acceptance_gate(problem: object, state: np.ndarray) -> float:
    diagnostics = problem.evaluate(validate_planar_state(state)).diagnostics
    return float(
        max(
            diagnostics.raw_total_linf,
            abs(diagnostics.held_out_continuity),
            abs(diagnostics.mass_error),
        )
    )


def scaled_singular_spectrum(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Return singular values after deterministic max-norm row/column scaling."""

    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("numerical-rank matrix must be square")
    if not np.isfinite(value).all():
        raise FloatingPointError("numerical-rank matrix contains NaN or infinity")
    row = np.max(np.abs(value), axis=1)
    if np.any(row == 0.0):
        singular = np.linalg.svd(value, compute_uv=False)
        return singular, 0.0
    scaled = value / row[:, None]
    column = np.max(np.abs(scaled), axis=0)
    if np.any(column == 0.0):
        singular = np.linalg.svd(scaled, compute_uv=False)
        return singular, 0.0
    scaled = scaled / column[None, :]
    singular = np.linalg.svd(scaled, compute_uv=False)
    reciprocal = float(singular[-1] / singular[0]) if singular[0] > 0.0 else 0.0
    return singular, reciprocal


def numerical_jacobian_rank(
    problem: object,
    state: np.ndarray,
    *,
    minimum_reciprocal_condition: float = 1.0e-8,
    invalid_penalty: float = 1.0e8,
) -> NumericalRankReport:
    """Build and densely rank the final colored numerical Jacobian.

    Only the N8/N16 audit calls this routine.  Dense SVD is intentional here:
    it is an independent numerical-rank test rather than another sparsity or
    factorization-success proxy.
    """

    if not np.isfinite(minimum_reciprocal_condition) or not (
        0.0 < minimum_reciprocal_condition < 1.0
    ):
        raise ValueError("minimum reciprocal condition must lie in (0,1)")
    physical = validate_planar_state(state)
    transform = LogStateTransform(problem.shape)
    encoded = transform.encode(physical)
    objective = EncodedR26Objective(problem, transform, invalid_penalty)
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
    row = np.asarray(abs(jacobian).max(axis=1).toarray()).ravel()
    if np.any(row == 0.0):
        dense = jacobian.toarray()
    else:
        scaled = jacobian.multiply((1.0 / row)[:, None]).tocsc()
        column = np.asarray(abs(scaled).max(axis=0).toarray()).ravel()
        if np.any(column == 0.0):
            dense = scaled.toarray()
        else:
            dense = scaled.multiply((1.0 / column)[None, :]).toarray()
    singular = np.linalg.svd(dense, compute_uv=False)
    largest = float(singular[0])
    smallest = float(singular[-1])
    tolerance = float(max(dense.shape) * np.finfo(float).eps * largest)
    rank = int(np.count_nonzero(singular > tolerance))
    reciprocal = smallest / largest if largest > 0.0 else 0.0
    full_rank = rank == problem.unknown_count
    passed = bool(
        full_rank
        and objective.invalid_evaluations == 0
        and reciprocal >= minimum_reciprocal_condition
    )
    return NumericalRankReport(
        unknown_count=problem.unknown_count,
        numerical_rank=rank,
        residual_evaluations=evaluations,
        invalid_evaluations=objective.invalid_evaluations,
        jacobian_nonzeros=int(jacobian.nnz),
        largest_scaled_singular_value=largest,
        smallest_scaled_singular_value=smallest,
        scaled_reciprocal_condition=float(reciprocal),
        numerical_rank_tolerance=tolerance,
        minimum_required_reciprocal_condition=minimum_reciprocal_condition,
        full_rank=full_rank,
        passed=passed,
    )


def _interpolate_components(
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    target_n: int,
) -> dict[str, np.ndarray]:
    value = validate_planar_state(state)
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    if xv.shape != (value.shape[1],) or yv.shape != (value.shape[0],):
        raise ValueError("coordinates do not match state")
    if target_n < 8:
        raise ValueError("target_n must be at least eight")
    centers = (np.arange(target_n, dtype=float) + 0.5) / target_n
    X, Y = np.meshgrid(centers, centers)
    points = np.column_stack((Y.ravel(), X.ravel()))
    result: dict[str, np.ndarray] = {}
    for index, name in enumerate(STATE_ORDER):
        interpolation = RegularGridInterpolator(
            (yv, xv), value[..., index], method="linear", bounds_error=True
        )
        result[name] = interpolation(points).reshape(target_n, target_n)
    return result


def compare_cross_solver_profiles(
    reference_state: np.ndarray,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    candidate_state: np.ndarray,
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    *,
    lid_velocity: float,
    target_n: int = 128,
) -> dict[str, object]:
    """Compare all 17 R26 components, two lines, and Rana D/G metrics."""

    reference = _interpolate_components(
        reference_state, reference_x, reference_y, target_n
    )
    candidate = _interpolate_components(
        candidate_state, candidate_x, candidate_y, target_n
    )
    ix = int(np.argmin(np.abs((np.arange(target_n) + 0.5) / target_n - 0.5)))
    iy = int(np.argmin(np.abs((np.arange(target_n) + 0.5) / target_n - 0.9)))
    fields: list[dict[str, object]] = []
    for name in STATE_ORDER:
        first = reference[name]
        second = candidate[name]
        equilibrium = 1.0 if name in {"rho", "theta"} else 0.0
        scale = max(
            float(np.sqrt(np.mean((first - equilibrium) ** 2))),
            float(np.sqrt(np.mean((second - equilibrium) ** 2))),
            np.finfo(float).tiny,
        )
        difference = second - first
        vertical = difference[:, ix]
        horizontal = difference[iy, :]
        vertical_scale = max(
            float(np.sqrt(np.mean((first[:, ix] - equilibrium) ** 2))),
            float(np.sqrt(np.mean((second[:, ix] - equilibrium) ** 2))),
            np.finfo(float).tiny,
        )
        horizontal_scale = max(
            float(np.sqrt(np.mean((first[iy, :] - equilibrium) ** 2))),
            float(np.sqrt(np.mean((second[iy, :] - equilibrium) ** 2))),
            np.finfo(float).tiny,
        )
        fields.append(
            {
                "field": name,
                "normalized_rms_difference": float(
                    np.sqrt(np.mean(difference * difference)) / scale
                ),
                "maximum_absolute_difference": float(np.max(np.abs(difference))),
                "vertical_centerline_normalized_rms_difference": float(
                    np.sqrt(np.mean(vertical * vertical)) / vertical_scale
                ),
                "horizontal_y0p9_normalized_rms_difference": float(
                    np.sqrt(np.mean(horizontal * horizontal)) / horizontal_scale
                ),
            }
        )
    reference_metrics = rana_global_metrics(
        reference_state,
        lid_velocity=lid_velocity,
        x=reference_x,
        y=reference_y,
    )
    candidate_metrics = rana_global_metrics(
        candidate_state,
        lid_velocity=lid_velocity,
        x=candidate_x,
        y=candidate_y,
    )

    def relative(key: str) -> float:
        first = float(reference_metrics[key])
        second = float(candidate_metrics[key])
        return abs(second - first) / max(abs(first), abs(second), np.finfo(float).tiny)

    return {
        "target_common_grid": target_n,
        "fields": fields,
        "maximum_normalized_rms_difference": max(
            float(row["normalized_rms_difference"]) for row in fields
        ),
        "maximum_line_normalized_rms_difference": max(
            max(
                float(row["vertical_centerline_normalized_rms_difference"]),
                float(row["horizontal_y0p9_normalized_rms_difference"]),
            )
            for row in fields
        ),
        "D_relative_difference": relative("D"),
        "G_relative_difference": relative("G"),
        "reference_D": float(reference_metrics["D"]),
        "candidate_D": float(candidate_metrics["D"]),
        "reference_G": float(reference_metrics["G"]),
        "candidate_G": float(candidate_metrics["G"]),
    }


__all__ = [
    "NumericalRankReport",
    "compare_cross_solver_profiles",
    "numerical_jacobian_rank",
    "raw_acceptance_gate",
    "scaled_singular_spectrum",
    "state_sha256",
]
