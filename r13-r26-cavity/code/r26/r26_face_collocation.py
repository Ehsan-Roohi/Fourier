#!/usr/bin/env python3
"""Corner-free face-augmented collocation prototype for the private R26 cavity.

This discretization addresses two structural defects of the first nodal BVP:

1. smooth-wall relations were attached to shared corner degrees of freedom,
   although the primary R26 source defines only one-normal smooth faces; and
2. centred collocated derivatives admitted alternating interior modes.

The gas unknowns here live at ``N x N`` cell centres.  Each of the four walls
owns ``N`` independent face states at tangential cell-centre locations, so no
corner degree of freedom exists and no corner convention is needed.  The 17
equations are

* all 17 R26 bulk balances at every gas centre; and
* 11 Gu--Emerson wall equations plus the six documented gas-side
  extrapolation relations at every wall face.

Normal derivatives use the unique polynomial through the two face values and
the row/column of cell centres.  Tangential face derivatives use the unique
polynomial through that face's own points.  This is a monolithic spectral
collocation BVP, not Gu--Emerson's FV/SIMPLE algorithm, and it adds no penalty,
constraint, filtering, or regularization.  It is intended as an independent
small-grid solvability audit; equispaced high-order polynomial collocation is
not recommended for production refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from r26_bulk_equations import R26ClosureDerivatives, steady_r26_bulk_residual
from r26_cases import CavityCase
from r26_spectral_backend import differentiate_axis, polynomial_first_derivative_matrix
from r26_state import NVAR, StateTensors, planar_state_to_tensors, validate_planar_state
from r26_tensor_closures import R26Closures, R26Gradients, closures_from_tensors
from r26_wall_conditions import free_extrapolation_values, square_wall_frame, wall_residual


@dataclass(frozen=True)
class FaceGridState:
    """Cell states and four independent smooth-face state arrays."""

    cells: np.ndarray
    left: np.ndarray
    right: np.ndarray
    bottom: np.ndarray
    top: np.ndarray

    def validated(self) -> "FaceGridState":
        cells = validate_planar_state(self.cells)
        if cells.ndim != 3 or cells.shape[0] != cells.shape[1]:
            raise ValueError("cells must have shape (N,N,17)")
        n = cells.shape[0]
        faces = {}
        for name in ("left", "right", "bottom", "top"):
            value = validate_planar_state(getattr(self, name))
            if value.shape != (n, NVAR):
                raise ValueError(f"{name} face must have shape ({n},17)")
            faces[name] = value
        return FaceGridState(cells, **faces)

    @property
    def nodes(self) -> int:
        return int(np.asarray(self.cells).shape[0])

    @property
    def unknown_count(self) -> int:
        n = self.nodes
        return NVAR * (n * n + 4 * n)

    def blocks(self) -> tuple[np.ndarray, ...]:
        return (self.cells, self.left, self.right, self.bottom, self.top)

    def flat(self) -> np.ndarray:
        return np.concatenate(tuple(np.asarray(value).ravel() for value in self.blocks()))


@dataclass(frozen=True)
class FaceGridResidual:
    residual: FaceGridState
    held_out_continuity: float
    mass_error: float
    bulk_linf: float
    wall_linf: float
    extrapolation_linf: float
    total_linf: float

    @property
    def flat(self) -> np.ndarray:
        return self.residual.flat()


def _make_tensor_blocks(state: FaceGridState) -> dict[str, StateTensors]:
    return {
        name: planar_state_to_tensors(getattr(state, name))
        for name in ("cells", "left", "right", "bottom", "top")
    }


def _field_block_gradients(
    values: dict[str, np.ndarray],
    *,
    x_centres: np.ndarray,
    y_centres: np.ndarray,
    tensor_rank: int,
) -> dict[str, np.ndarray]:
    """Return physical x/y/z gradients at cells and all four faces."""

    x_all = np.concatenate(((0.0,), x_centres, (1.0,)))
    y_all = np.concatenate(((0.0,), y_centres, (1.0,)))
    dx_all = polynomial_first_derivative_matrix(x_all)
    dy_all = polynomial_first_derivative_matrix(y_all)
    dx_tangent = polynomial_first_derivative_matrix(x_centres)
    dy_tangent = polynomial_first_derivative_matrix(y_centres)

    cells = values["cells"]
    x_lines = np.concatenate(
        (values["left"][:, None], cells, values["right"][:, None]), axis=1
    )
    y_lines = np.concatenate(
        (values["bottom"][None, :], cells, values["top"][None, :]), axis=0
    )
    dx_lines = differentiate_axis(x_lines, dx_all, axis=1)
    dy_lines = differentiate_axis(y_lines, dy_all, axis=0)

    derivatives = {
        "cells": (dx_lines[:, 1:-1], dy_lines[1:-1]),
        "left": (dx_lines[:, 0], differentiate_axis(values["left"], dy_tangent, axis=0)),
        "right": (dx_lines[:, -1], differentiate_axis(values["right"], dy_tangent, axis=0)),
        "bottom": (differentiate_axis(values["bottom"], dx_tangent, axis=0), dy_lines[0]),
        "top": (differentiate_axis(values["top"], dx_tangent, axis=0), dy_lines[-1]),
    }
    result = {}
    for name, (ddx, ddy) in derivatives.items():
        result[name] = np.stack(
            (ddx, ddy, np.zeros_like(ddx)), axis=-(tensor_rank + 1)
        )
    return result


def face_grid_gradients(
    state: FaceGridState, *, x_centres: np.ndarray, y_centres: np.ndarray
) -> dict[str, R26Gradients]:
    """Build full tensor gradients at gas centres and independent wall faces."""

    state = state.validated()
    tensors = _make_tensor_blocks(state)
    ranks = {
        "rho": 0,
        "velocity": 1,
        "theta": 0,
        "heat_flux": 1,
        "sigma": 2,
        "R": 2,
        "m": 3,
        "Delta": 0,
    }
    by_field = {}
    for field, rank in ranks.items():
        by_field[field] = _field_block_gradients(
            {name: np.asarray(getattr(value, field)) for name, value in tensors.items()},
            x_centres=x_centres,
            y_centres=y_centres,
            tensor_rank=rank,
        )
    return {
        name: R26Gradients(**{field: by_field[field][name] for field in ranks})
        for name in tensors
    }


def _closure_derivatives_at_cells(
    closures: dict[str, R26Closures], *, x_centres: np.ndarray, y_centres: np.ndarray
) -> R26ClosureDerivatives:
    field_gradients = {}
    for field, rank in (("phi", 4), ("psi", 3), ("Omega", 1)):
        field_gradients[field] = _field_block_gradients(
            {name: np.asarray(getattr(value, field)) for name, value in closures.items()},
            x_centres=x_centres,
            y_centres=y_centres,
            tensor_rank=rank,
        )["cells"]
    return R26ClosureDerivatives(
        div_phi=np.einsum("...lijkl->...ijk", field_gradients["phi"], optimize=True),
        div_psi=np.einsum("...kijk->...ij", field_gradients["psi"], optimize=True),
        grad_Omega=field_gradients["Omega"],
    )


def _point_closure(closures: R26Closures, index: int) -> R26Closures:
    return R26Closures(
        phi=np.asarray(closures.phi[index]),
        psi=np.asarray(closures.psi[index]),
        Omega=np.asarray(closures.Omega[index]),
        equation25_mode=closures.equation25_mode,
        provenance=closures.provenance,
        coefficient_mode=closures.coefficient_mode,
    )


class R26FaceCollocationBVP:
    """Square, corner-free cell/face R26 residual."""

    def __init__(self, case: CavityCase) -> None:
        self.case = case
        self.n = case.nodes
        h = case.length / self.n
        self.x_centres = (np.arange(self.n) + 0.5) * h
        self.y_centres = (np.arange(self.n) + 0.5) * h
        self.mass_j = self.n // 2
        self.mass_i = self.n // 2

    @property
    def unknown_count(self) -> int:
        return NVAR * (self.n * self.n + 4 * self.n)

    def equilibrium_state(self) -> FaceGridState:
        cell = np.zeros((self.n, self.n, NVAR))
        face = np.zeros((self.n, NVAR))
        cell[..., 0] = face[..., 0] = self.case.mean_density
        cell[..., 3] = face[..., 3] = self.case.wall_temperature
        return FaceGridState(cell, face.copy(), face.copy(), face.copy(), face.copy())

    def _split_vector(self, vector: np.ndarray) -> FaceGridState:
        value = np.asarray(vector, dtype=float)
        if value.shape != (self.unknown_count,) or not np.isfinite(value).all():
            raise ValueError("face-grid vector has wrong shape or non-finite values")
        sizes = (self.n * self.n * NVAR,) + (self.n * NVAR,) * 4
        cuts = np.cumsum((0,) + sizes)
        blocks = [value[cuts[k] : cuts[k + 1]] for k in range(5)]
        # Encoded rho/theta entries are logarithms and need not be positive;
        # validation happens after ``decode`` exponentiates them.
        return FaceGridState(
            blocks[0].reshape(self.n, self.n, NVAR),
            *(block.reshape(self.n, NVAR) for block in blocks[1:]),
        )

    def encode(self, state: FaceGridState) -> np.ndarray:
        encoded = []
        for block in state.validated().blocks():
            value = np.asarray(block).copy()
            value[..., 0] = np.log(value[..., 0])
            value[..., 3] = np.log(value[..., 3])
            encoded.append(value.ravel())
        return np.concatenate(encoded)

    def decode(self, vector: np.ndarray) -> FaceGridState:
        encoded = self._split_vector(vector)
        decoded = []
        for block in encoded.blocks():
            value = np.asarray(block).copy()
            value[..., 0] = np.exp(value[..., 0])
            value[..., 3] = np.exp(value[..., 3])
            decoded.append(value)
        return FaceGridState(*decoded).validated()

    def evaluate(self, state: FaceGridState) -> FaceGridResidual:
        state = state.validated()
        tensors = _make_tensor_blocks(state)
        gradients = face_grid_gradients(
            state, x_centres=self.x_centres, y_centres=self.y_centres
        )
        mu = {
            name: np.asarray(self.case.mu(value.theta), dtype=float)
            for name, value in tensors.items()
        }
        closures = {
            name: closures_from_tensors(
                tensors[name],
                gradients[name],
                mu=mu[name],
                coefficient_mode=self.case.r26_closure_mode,
            )
            for name in tensors
        }
        derivatives = _closure_derivatives_at_cells(
            closures, x_centres=self.x_centres, y_centres=self.y_centres
        )
        bulk = steady_r26_bulk_residual(
            tensors["cells"],
            gradients["cells"],
            closures["cells"],
            derivatives,
            mu=mu["cells"],
        ).as_planar17()
        bulk_scaled = bulk / self.case.scaling.bulk
        held_out = float(bulk[self.mass_j, self.mass_i, 0])
        mass_error = float(np.mean(state.cells[..., 0]) - self.case.mean_density)
        bulk_scaled[self.mass_j, self.mass_i, 0] = mass_error / self.case.scaling.mass

        face_residuals = {}
        wall_max = 0.0
        extrapolation_max = 0.0
        for side in ("left", "right", "bottom", "top"):
            frame = square_wall_frame(side)
            current = getattr(state, side)
            output = np.zeros_like(current)
            for k in range(self.n):
                wall = wall_residual(
                    current[k],
                    _point_closure(closures[side], k),
                    frame.normal,
                    frame.tangent,
                    self.case.wall_velocity(side),
                    self.case.wall_temperature,
                    alpha=self.case.accommodation,
                    gas_constant=self.case.gas_constant,
                )
                if side == "left":
                    near, nxt = state.cells[k, 0], state.cells[k, 1]
                elif side == "right":
                    near, nxt = state.cells[k, -1], state.cells[k, -2]
                elif side == "bottom":
                    near, nxt = state.cells[0, k], state.cells[1, k]
                else:
                    near, nxt = state.cells[-1, k], state.cells[-2, k]
                free_wall = free_extrapolation_values(
                    current[k], frame.normal, frame.tangent, gas_constant=self.case.gas_constant
                )
                free_near = free_extrapolation_values(
                    near, frame.normal, frame.tangent, gas_constant=self.case.gas_constant
                )
                free_next = free_extrapolation_values(
                    nxt, frame.normal, frame.tangent, gas_constant=self.case.gas_constant
                )
                extrapolation = free_wall - (1.5 * free_near - 0.5 * free_next)
                output[k, :11] = wall / self.case.scaling.wall
                output[k, 11:] = extrapolation / self.case.scaling.extrapolation
                wall_max = max(wall_max, float(np.max(np.abs(output[k, :11]))))
                extrapolation_max = max(
                    extrapolation_max, float(np.max(np.abs(output[k, 11:])))
                )
            face_residuals[side] = output

        residual = FaceGridState(bulk_scaled, **face_residuals)
        flat = residual.flat()
        bulk_copy = bulk_scaled.copy()
        bulk_copy[self.mass_j, self.mass_i, 0] = 0.0
        return FaceGridResidual(
            residual=residual,
            held_out_continuity=held_out,
            mass_error=mass_error,
            bulk_linf=float(np.max(np.abs(bulk_copy), initial=0.0)),
            wall_linf=wall_max,
            extrapolation_linf=extrapolation_max,
            total_linf=float(np.max(np.abs(flat), initial=0.0)),
        )

    def objective(self, encoded_vector: np.ndarray) -> np.ndarray:
        return self.evaluate(self.decode(encoded_vector)).flat


__all__ = [
    "FaceGridResidual",
    "FaceGridState",
    "R26FaceCollocationBVP",
    "face_grid_gradients",
]
