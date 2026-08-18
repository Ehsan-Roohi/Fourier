#!/usr/bin/env python3
"""Explicit physical-intersection corner model for the private R26 cavity.

Gu--Emerson's wall conditions are smooth-face conditions and do not define a
two-normal sharp-corner law.  The baseline node BVP therefore used a bilinear
extension for all 17 corner components.  Linear audits showed that extension
leaves three corner-heavy null modes: two kinematic and one thermodynamic.

This module provides a *declared alternative model*, not a hidden numerical
regularizer.  At each corner it replaces the bilinear rows for ``u_x``,
``u_y``, and ``theta`` by the physical intersection conditions

* no penetration through both intersecting stationary wall normals; and
* the common isothermal wall temperature.

At a top corner, the lid velocity is discontinuous from the stationary side
wall.  ``top_corner_lid_fraction`` exposes rather than hides that point-value
convention: 0 chooses the side-wall trace (and satisfies both normal
conditions); 1 chooses the moving-lid trace.  Results must be reported with a
0/1 sensitivity.  The other 14 components retain the documented bilinear
extension because no primary source supplies a coupled corner law.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from r26_discretization import R26NodeBVP, ResidualEvaluation


class R26PhysicalCornerBVP(R26NodeBVP):
    """R26 node BVP with explicit velocity/temperature corner conditions."""

    def __init__(
        self,
        *args: object,
        top_corner_lid_fraction: float = 0.0,
        mass_row: tuple[int, int] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        fraction = float(top_corner_lid_fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("top_corner_lid_fraction must lie in [0,1]")
        self.top_corner_lid_fraction = fraction
        if mass_row is not None:
            j, i = (int(mass_row[0]), int(mass_row[1]))
            if not (1 <= j < self.case.nodes - 1 and 1 <= i < self.case.nodes - 1):
                raise ValueError("mass_row must select an interior node")
            self.mass_j, self.mass_i = j, i

    @property
    def corner_model(self) -> str:
        return (
            "two-normal no-penetration + isothermal temperature; remaining "
            "14 rows bilinear; top-corner lid fraction="
            f"{self.top_corner_lid_fraction:g}"
        )

    def evaluate(self, state: np.ndarray) -> ResidualEvaluation:
        base = super().evaluate(state)
        u = np.asarray(state, dtype=float)
        raw = base.unscaled_residual.copy()
        scaled = base.residual.copy()
        corners = ((0, 0, False), (0, -1, False), (-1, 0, True), (-1, -1, True))
        for j, i, is_top in corners:
            target_x = (
                self.top_corner_lid_fraction * self.case.lid_velocity if is_top else 0.0
            )
            values = np.asarray(
                (
                    u[j, i, 1] - target_x,
                    u[j, i, 2],
                    u[j, i, 3] - self.case.wall_temperature,
                )
            )
            raw[j, i, 1:4] = values
            scaled[j, i, 1:4] = values / self.case.scaling.corner[1:4]

        raw_corner_max = max(
            float(np.max(np.abs(raw[j, i])))
            for j, i, _ in corners
        )
        corner_max = max(
            float(np.max(np.abs(scaled[j, i])))
            for j, i, _ in corners
        )
        flat = scaled.ravel()
        diagnostics = replace(
            base.diagnostics,
            corner_linf=corner_max,
            raw_corner_linf=raw_corner_max,
            total_linf=float(np.max(np.abs(flat), initial=0.0)),
            total_l2_rms=float(np.sqrt(np.mean(flat * flat))),
            raw_total_linf=float(np.max(np.abs(raw), initial=0.0)),
        )
        return ResidualEvaluation(
            residual=scaled,
            unscaled_residual=raw,
            diagnostics=diagnostics,
            mass_row=base.mass_row,
        )


__all__ = ["R26PhysicalCornerBVP"]
