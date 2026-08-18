#!/usr/bin/env python3
"""Node-collocated private R26 square-cavity boundary-value residual.

The unknown has shape ``(N,N,17)`` and includes the wall nodes.  Every
interior node carries all 17 R26 balance equations.  A smooth non-corner wall
node carries the 11 Gu--Emerson wall relations and six one-sided linear
extrapolation constraints.  Corner nodes are *not* claimed to be paper-exact:
they use a stated bilinear adjacent-face extension and are excluded from wall
validation metrics.

This module orchestrates the independently audited bulk, closure, and wall
operators.  It intentionally contains no fallback moment equations.  If the
bulk implementation is absent or has the wrong API, construction/evaluation
fails loudly instead of silently solving an R13 or partial-R26 system.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import inspect
from typing import Callable, Final, Protocol

import numpy as np

from r26_cases import CavityCase
from r26_state import NVAR, planar_state_to_tensors, validate_planar_state
from r26_tensor_closures import R26Closures, gu_emerson_closures
from r26_wall_conditions import free_extrapolation_values, square_wall_frame, wall_residual


class BulkOperator(Protocol):
    def __call__(
        self,
        state: np.ndarray,
        *,
        x: np.ndarray,
        y: np.ndarray,
        mu: np.ndarray,
        case: CavityCase,
    ) -> np.ndarray: ...


class ClosureOperator(Protocol):
    def __call__(
        self,
        state: np.ndarray,
        *,
        x: np.ndarray,
        y: np.ndarray,
        mu: np.ndarray,
        case: CavityCase,
    ) -> R26Closures: ...


@dataclass(frozen=True)
class BoundaryNode:
    side: str
    j: int
    i: int
    near_j: int
    near_i: int
    next_j: int
    next_i: int


@dataclass(frozen=True)
class ResidualDiagnostics:
    bulk_linf: float
    wall_linf: float
    extrapolation_linf: float
    corner_linf: float
    mass_error: float
    held_out_continuity: float
    total_linf: float
    total_l2_rms: float
    raw_bulk_linf: float
    raw_wall_linf: float
    raw_extrapolation_linf: float
    raw_corner_linf: float
    raw_total_linf: float
    min_density: float
    min_temperature: float
    interior_nodes: int
    smooth_wall_nodes: int
    excluded_corner_nodes: int = 4


@dataclass(frozen=True)
class ResidualEvaluation:
    residual: np.ndarray
    unscaled_residual: np.ndarray
    diagnostics: ResidualDiagnostics
    mass_row: tuple[int, int, int]

    @property
    def flat(self) -> np.ndarray:
        return self.residual.ravel()


def _default_bulk_operator() -> BulkOperator:
    """Resolve the private audited bulk operator without a numerical fallback."""

    module = import_module("r26_bulk_equations")
    for name in ("bulk_residual_grid", "r26_bulk_residual", "bulk_residual"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    raise ImportError(
        "r26_bulk_equations must export bulk_residual_grid(state, x=, y=, mu=, case=); "
        "no partial/R13 fallback is permitted"
    )


def _default_closure_operator(
    state: np.ndarray, *, x: np.ndarray, y: np.ndarray, mu: np.ndarray, case: CavityCase
) -> R26Closures:
    return gu_emerson_closures(
        state,
        x=x,
        y=y,
        mu=mu,
        edge_order=2,
        coefficient_mode=case.r26_closure_mode,
    )


def _point_closure(closures: R26Closures, j: int, i: int) -> R26Closures:
    return R26Closures(
        phi=np.asarray(closures.phi[j, i]),
        psi=np.asarray(closures.psi[j, i]),
        Omega=np.asarray(closures.Omega[j, i]),
        equation25_mode=closures.equation25_mode,
        provenance=closures.provenance,
        coefficient_mode=closures.coefficient_mode,
    )


def trapezoidal_node_weights(nodes: int) -> np.ndarray:
    """Tensor-product trapezoidal weights, normalized to unit sum."""

    if nodes < 2:
        raise ValueError("at least two nodes are required")
    one = np.ones(nodes)
    one[[0, -1]] = 0.5
    weights = np.outer(one, one)
    return weights / np.sum(weights)


def smooth_boundary_nodes(nodes: int) -> tuple[BoundaryNode, ...]:
    """Enumerate each non-corner wall node exactly once."""

    if nodes < 5:
        raise ValueError("two interior layers are needed for wall extrapolation")
    entries: list[BoundaryNode] = []
    for j in range(1, nodes - 1):
        entries.append(BoundaryNode("left", j, 0, j, 1, j, 2))
        entries.append(BoundaryNode("right", j, nodes - 1, j, nodes - 2, j, nodes - 3))
    for i in range(1, nodes - 1):
        entries.append(BoundaryNode("bottom", 0, i, 1, i, 2, i))
        entries.append(BoundaryNode("top", nodes - 1, i, nodes - 2, i, nodes - 3, i))
    return tuple(entries)


def linear_wall_extrapolation(
    wall_coordinate: float,
    near_coordinate: float,
    next_coordinate: float,
    near_value: np.ndarray,
    next_value: np.ndarray,
) -> np.ndarray:
    """Extrapolate a value linearly from two interior nodes to a wall.

    The earlier ``2*near-next`` formula is only valid on a uniform grid.  This
    coordinate form is exact for every affine profile and therefore preserves
    the intended six free-moment completion on a stretched mesh.
    """

    wall = float(wall_coordinate)
    near = float(near_coordinate)
    nxt = float(next_coordinate)
    denominator = nxt - near
    if not all(np.isfinite(value) for value in (wall, near, nxt)) or denominator == 0.0:
        raise ValueError("wall extrapolation coordinates must be finite and distinct")
    a = np.asarray(near_value, dtype=float)
    b = np.asarray(next_value, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("wall extrapolation values must be finite with equal shape")
    return a + (wall - near) / denominator * (b - a)


CORNER_INDICES: Final[tuple[str, ...]] = (
    "bottom_left",
    "bottom_right",
    "top_left",
    "top_right",
)


def bilinear_corner_residuals(state: np.ndarray) -> dict[str, np.ndarray]:
    """Return the explicit non-paper corner extension residuals.

    For example, the bottom-left relation is
    ``U[0,0] - U[0,1] - U[1,0] + U[1,1] = 0``.  It is the zero mixed-second-
    difference bilinear extension from the two adjacent smooth faces.
    """

    u = validate_planar_state(state)
    if u.ndim != 3 or min(u.shape[:2]) < 3:
        raise ValueError("corner residual expects a two-dimensional node grid")
    return {
        "bottom_left": u[0, 0] - u[0, 1] - u[1, 0] + u[1, 1],
        "bottom_right": u[0, -1] - u[0, -2] - u[1, -1] + u[1, -2],
        "top_left": u[-1, 0] - u[-1, 1] - u[-2, 0] + u[-2, 1],
        "top_right": u[-1, -1] - u[-1, -2] - u[-2, -1] + u[-2, -2],
    }


class R26NodeBVP:
    """Stateless square residual with an explicit mass-border replacement."""

    def __init__(
        self,
        case: CavityCase,
        *,
        bulk_operator: BulkOperator | None = None,
        closure_operator: ClosureOperator | None = None,
        wall_operator: Callable[..., np.ndarray] = wall_residual,
        mass_weights: np.ndarray | None = None,
    ) -> None:
        self.case = case
        self.bulk_operator = _default_bulk_operator() if bulk_operator is None else bulk_operator
        self.closure_operator = _default_closure_operator if closure_operator is None else closure_operator
        self.wall_operator = wall_operator
        self.boundary_nodes = smooth_boundary_nodes(case.nodes)
        if mass_weights is None:
            self.mass_weights = trapezoidal_node_weights(case.nodes)
        else:
            weights = np.asarray(mass_weights, dtype=float)
            if weights.shape != (case.nodes, case.nodes):
                raise ValueError("mass_weights must match the node grid")
            if not np.isfinite(weights).all() or np.any(weights < 0.0):
                raise ValueError("mass_weights must be finite and nonnegative")
            total_weight = float(np.sum(weights))
            if total_weight <= 0.0:
                raise ValueError("mass_weights must have positive total weight")
            self.mass_weights = weights.copy() / total_weight
        # A central interior equation is least exposed to one-sided stencils.
        self.mass_j = max(1, min(case.nodes - 2, case.nodes // 2))
        self.mass_i = max(1, min(case.nodes - 2, case.nodes // 2))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.case.nodes, self.case.nodes, NVAR)

    @property
    def unknown_count(self) -> int:
        return int(np.prod(self.shape))

    @property
    def equation_accounting(self) -> dict[str, int]:
        n = self.case.nodes
        interior = (n - 2) ** 2
        smooth = 4 * (n - 2)
        return {
            "interior_nodes": interior,
            "interior_equations": 17 * interior,
            "smooth_wall_nodes": smooth,
            "smooth_wall_equations": 17 * smooth,
            "excluded_corner_nodes": 4,
            "corner_model_equations": 68,
            "total": self.unknown_count,
        }

    def mean_density(self, state: np.ndarray) -> float:
        u = validate_planar_state(state)
        return float(np.sum(self.mass_weights * u[..., 0]))

    def mass_constraint(self, state: np.ndarray) -> float:
        return self.mean_density(state) - self.case.mean_density

    def _bulk(self, state: np.ndarray, mu: np.ndarray) -> np.ndarray:
        parameters = inspect.signature(self.bulk_operator).parameters
        accepts_extra = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        kwargs: dict[str, object] = {"x": self.case.x, "y": self.case.y, "mu": mu}
        if "case" in parameters or accepts_extra:
            kwargs["case"] = self.case
        value = self.bulk_operator(state, **kwargs)
        residual = np.asarray(value, dtype=float)
        if residual.shape != self.shape or not np.isfinite(residual).all():
            raise ValueError(f"bulk residual must be finite with shape {self.shape}, got {residual.shape}")
        return residual

    def _closures(self, state: np.ndarray, mu: np.ndarray) -> R26Closures:
        parameters = inspect.signature(self.closure_operator).parameters
        accepts_extra = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        kwargs: dict[str, object] = {"x": self.case.x, "y": self.case.y, "mu": mu}
        if "case" in parameters or accepts_extra:
            kwargs["case"] = self.case
        closures = self.closure_operator(state, **kwargs)
        expected = self.shape[:2]
        if np.asarray(closures.phi).shape != expected + (3, 3, 3, 3):
            raise ValueError("closure phi grid shape does not match state")
        if np.asarray(closures.psi).shape != expected + (3, 3, 3):
            raise ValueError("closure psi grid shape does not match state")
        if np.asarray(closures.Omega).shape != expected + (3,):
            raise ValueError("closure Omega grid shape does not match state")
        return closures

    def evaluate(self, state: np.ndarray) -> ResidualEvaluation:
        """Assemble one complete square residual without mutating solver state."""

        u = validate_planar_state(state)
        if u.shape != self.shape:
            raise ValueError(f"state must have shape {self.shape}, got {u.shape}")
        mu = np.asarray(self.case.mu(u[..., 3]), dtype=float)
        bulk = self._bulk(u, mu)
        closures = self._closures(u, mu)

        raw = np.zeros_like(u)
        scaled = np.zeros_like(u)
        bulk_scale = self.case.scaling.bulk
        raw[1:-1, 1:-1] = bulk[1:-1, 1:-1]
        scaled[1:-1, 1:-1] = raw[1:-1, 1:-1] / bulk_scale

        wall_max = 0.0
        extrap_max = 0.0
        raw_wall_max = 0.0
        raw_extrap_max = 0.0
        for node in self.boundary_nodes:
            frame = square_wall_frame(node.side)
            local = _point_closure(closures, node.j, node.i)
            wall = np.asarray(
                self.wall_operator(
                    u[node.j, node.i],
                    local,
                    frame.normal,
                    frame.tangent,
                    self.case.wall_velocity(node.side),
                    self.case.wall_temperature,
                    alpha=self.case.accommodation,
                    gas_constant=self.case.gas_constant,
                ),
                dtype=float,
            )
            if wall.shape != (11,) or not np.isfinite(wall).all():
                raise ValueError("smooth-wall operator must return 11 finite residuals")
            wall_free = free_extrapolation_values(
                u[node.j, node.i], frame.normal, frame.tangent, gas_constant=self.case.gas_constant
            )
            near_free = free_extrapolation_values(
                u[node.near_j, node.near_i],
                frame.normal,
                frame.tangent,
                gas_constant=self.case.gas_constant,
            )
            next_free = free_extrapolation_values(
                u[node.next_j, node.next_i],
                frame.normal,
                frame.tangent,
                gas_constant=self.case.gas_constant,
            )
            if node.side in {"left", "right"}:
                wall_coordinate = self.case.x[node.i]
                near_coordinate = self.case.x[node.near_i]
                next_coordinate = self.case.x[node.next_i]
            else:
                wall_coordinate = self.case.y[node.j]
                near_coordinate = self.case.y[node.near_j]
                next_coordinate = self.case.y[node.next_j]
            extrapolated_free = linear_wall_extrapolation(
                wall_coordinate,
                near_coordinate,
                next_coordinate,
                near_free,
                next_free,
            )
            extrapolation = wall_free - extrapolated_free
            raw[node.j, node.i, :11] = wall
            raw[node.j, node.i, 11:] = extrapolation
            scaled[node.j, node.i, :11] = wall / self.case.scaling.wall
            scaled[node.j, node.i, 11:] = extrapolation / self.case.scaling.extrapolation
            wall_max = max(wall_max, float(np.max(np.abs(scaled[node.j, node.i, :11]))))
            extrap_max = max(extrap_max, float(np.max(np.abs(scaled[node.j, node.i, 11:]))))
            raw_wall_max = max(raw_wall_max, float(np.max(np.abs(wall))))
            raw_extrap_max = max(raw_extrap_max, float(np.max(np.abs(extrapolation))))

        corner_values = bilinear_corner_residuals(u)
        corner_locations = {
            "bottom_left": (0, 0),
            "bottom_right": (0, -1),
            "top_left": (-1, 0),
            "top_right": (-1, -1),
        }
        corner_max = 0.0
        raw_corner_max = 0.0
        for name, value in corner_values.items():
            j, i = corner_locations[name]
            raw[j, i] = value
            scaled[j, i] = value / self.case.scaling.corner
            corner_max = max(corner_max, float(np.max(np.abs(scaled[j, i]))))
            raw_corner_max = max(raw_corner_max, float(np.max(np.abs(value))))

        held_out = float(raw[self.mass_j, self.mass_i, 0])
        mass = self.mass_constraint(u)
        raw[self.mass_j, self.mass_i, 0] = mass
        scaled[self.mass_j, self.mass_i, 0] = mass / self.case.scaling.mass

        interior_scaled = scaled[1:-1, 1:-1].copy()
        # Do not let the replacement hide the remaining bulk equation norms.
        interior_scaled[self.mass_j - 1, self.mass_i - 1, 0] = 0.0
        interior_raw = raw[1:-1, 1:-1].copy()
        interior_raw[self.mass_j - 1, self.mass_i - 1, 0] = 0.0
        total = scaled.ravel()
        diagnostics = ResidualDiagnostics(
            bulk_linf=float(np.max(np.abs(interior_scaled), initial=0.0)),
            wall_linf=wall_max,
            extrapolation_linf=extrap_max,
            corner_linf=corner_max,
            mass_error=mass,
            held_out_continuity=held_out,
            total_linf=float(np.max(np.abs(total), initial=0.0)),
            total_l2_rms=float(np.sqrt(np.mean(total * total))),
            raw_bulk_linf=float(np.max(np.abs(interior_raw), initial=0.0)),
            raw_wall_linf=raw_wall_max,
            raw_extrapolation_linf=raw_extrap_max,
            raw_corner_linf=raw_corner_max,
            raw_total_linf=float(np.max(np.abs(raw), initial=0.0)),
            min_density=float(np.min(u[..., 0])),
            min_temperature=float(np.min(u[..., 3])),
            interior_nodes=(self.case.nodes - 2) ** 2,
            smooth_wall_nodes=4 * (self.case.nodes - 2),
        )
        return ResidualEvaluation(
            residual=scaled,
            unscaled_residual=raw,
            diagnostics=diagnostics,
            mass_row=(self.mass_j, self.mass_i, 0),
        )

    def residual(self, state: np.ndarray) -> np.ndarray:
        return self.evaluate(state).flat


__all__ = [
    "BoundaryNode",
    "BulkOperator",
    "ClosureOperator",
    "CORNER_INDICES",
    "R26NodeBVP",
    "ResidualDiagnostics",
    "ResidualEvaluation",
    "bilinear_corner_residuals",
    "linear_wall_extrapolation",
    "smooth_boundary_nodes",
    "trapezoidal_node_weights",
]
