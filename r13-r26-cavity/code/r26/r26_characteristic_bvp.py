#!/usr/bin/env python3
"""Characteristic-consistent smooth-face completion for the R26 node BVP.

The Gu--Emerson wall system supplies 11 relations and explicitly treats six
face moments as free gas-side data: ``p, sigma_nt, q_n, m_nnn, m_ntt, R_nt``.
The initial prototype completed the 17 wall rows by linearly extrapolating
those six quantities.  That is simple, but it discards their six outgoing
bulk balances and was found to participate in null/near-null modes.

This alternative keeps the 11 exact wall relations and completes each smooth
face with the six corresponding one-sided bulk balance rows.  It is a standard
boundary-collocation choice: incoming characteristics are supplied by the
wall law while outgoing/free moments retain their governing equations.  It
adds no algebraic constraint, filtering, penalty, or regularization.

For an axis-aligned planar wall the selected balance rows are:

* mass;
* the ``sigma_nt`` stress balance;
* the normal heat-flux balance;
* the ``m_nnn`` and ``m_ntt`` balances; and
* the ``R_nt`` balance.

The sharp-corner model remains separately explicit and is not modified here.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from r26_discretization import R26NodeBVP, ResidualEvaluation


_SIDE_BULK_ROWS = {
    # state/bulk planar row order:
    # rho,vx,vy,theta,qx,qy,sxx,sxy,syy,Rxx,Rxy,Ryy,
    # mxxx,mxxy,mxyy,myyy,Delta
    "left": np.asarray((0, 7, 4, 12, 14, 10), dtype=int),
    "right": np.asarray((0, 7, 4, 12, 14, 10), dtype=int),
    "bottom": np.asarray((0, 7, 5, 15, 13, 10), dtype=int),
    "top": np.asarray((0, 7, 5, 15, 13, 10), dtype=int),
}


class R26CharacteristicBoundaryBVP(R26NodeBVP):
    """Use one-sided outgoing bulk rows instead of six extrapolation rows."""

    boundary_completion = (
        "11 Gu--Emerson WBC + one-sided balances for "
        "mass,sigma_nt,q_n,m_nnn,m_ntt,R_nt"
    )

    def evaluate(self, state: np.ndarray) -> ResidualEvaluation:
        base = super().evaluate(state)
        u = np.asarray(state, dtype=float)
        mu = np.asarray(self.case.mu(u[..., 3]), dtype=float)
        bulk = self._bulk(u, mu)
        raw = base.unscaled_residual.copy()
        scaled = base.residual.copy()

        completion_max = 0.0
        raw_completion_max = 0.0
        for node in self.boundary_nodes:
            indices = _SIDE_BULK_ROWS[node.side]
            values = bulk[node.j, node.i, indices]
            raw[node.j, node.i, 11:] = values
            scaled[node.j, node.i, 11:] = values / self.case.scaling.bulk[indices]
            completion_max = max(
                completion_max,
                float(np.max(np.abs(scaled[node.j, node.i, 11:])))
            )
            raw_completion_max = max(
                raw_completion_max,
                float(np.max(np.abs(values)))
            )

        flat = scaled.ravel()
        diagnostics = replace(
            base.diagnostics,
            # Retain the field name for API compatibility; in this subclass
            # it means the declared outgoing-balance completion norm.
            extrapolation_linf=completion_max,
            raw_extrapolation_linf=raw_completion_max,
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


__all__ = ["R26CharacteristicBoundaryBVP"]
