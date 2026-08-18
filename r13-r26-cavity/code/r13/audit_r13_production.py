#!/usr/bin/env python3
"""Generate a deterministic JSON audit of the archived R13 production code."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_DIR = ROOT / "work/article_final_v2/data/kn020_models/r13_N60"
LEGACY_COEFFICIENTS = SOURCE_DIR / "rana_original_coefficients.py"
LEGACY_SOLVER = SOURCE_DIR / "rana_original_reference_solver.py"

from r13_maxwell_production import (  # noqa: E402
    STATE_ORDER_PRINTED_EQ11,
    STATE_ORDER_SOLVER,
    appendix_a_reduced_production_matrix,
    kn_rana_from_gu,
    legacy_sqrt_temperature_prefactor,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_legacy():
    spec = importlib.util.spec_from_file_location("legacy_r13_coefficients", LEGACY_COEFFICIENTS)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load archived coefficient module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    legacy = load_legacy()
    state = np.asarray(
        [
            1.37, 0.07, -0.04, 1.11, 0.013, -0.021, 0.031, -0.017,
            -0.024, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003,
            -0.004, 0.005,
        ],
        dtype=float,
    )
    kn = kn_rana_from_gu(0.20)
    archived_matrix = legacy.production(
        state, kn, rb=1.0, ra=1.0, ma=1.0
    ) / legacy_sqrt_temperature_prefactor(state, kn)
    printed_matrix = appendix_a_reduced_production_matrix(state)
    delta = archived_matrix - printed_matrix
    mismatch_indices = np.argwhere(np.abs(delta) > 1.0e-13)
    mismatches = [
        {
            "row": int(row),
            "column": int(column),
            "row_name": STATE_ORDER_SOLVER[int(row)],
            "column_name": STATE_ORDER_SOLVER[int(column)],
            "archived_unscaled": float(archived_matrix[row, column]),
            "appendix_a": float(printed_matrix[row, column]),
            "difference": float(delta[row, column]),
        }
        for row, column in mismatch_indices
    ]
    report = {
        "reference": {
            "title": "A robust numerical method for the R13 equations of rarefied gas dynamics: Application to lid driven cavity",
            "doi": "10.1016/j.jcp.2012.11.023",
            "author_pdf": "https://www.engr.uvic.ca/~struchtr/2013_JCP_Lidcavity.pdf",
            "evidence": [
                "Section 2.1 states Maxwell molecules.",
                "Equation (11) defines the 17-state quasilinear system.",
                "Appendix A, journal page 184, prints the reduced production matrix.",
                "Equation (29) defines Kn_Rana = mu0/(rho0 sqrt(theta0) L).",
            ],
        },
        "archived_sources": {
            "coefficient_path": str(LEGACY_COEFFICIENTS.relative_to(ROOT)),
            "coefficient_sha256": sha256(LEGACY_COEFFICIENTS),
            "solver_path": str(LEGACY_SOLVER.relative_to(ROOT)),
            "solver_sha256": sha256(LEGACY_SOLVER),
        },
        "state_order": {
            "printed_below_eq_11": list(STATE_ORDER_PRINTED_EQ11),
            "executable_flux_wall_order": list(STATE_ORDER_SOLVER),
            "archived_solver_labels": [
                "rho", "vx", "vy", "theta", "qx", "qy", "sigma_xx",
                "sigma_xy", "sigma_yy", "R_xx", "R_xy", "R_yy",
                "m_xxx", "m_xxy", "m_xyy", "m_yyy", "Delta",
            ],
            "internal_evidence": [
                "The printed Eq. (11) state list puts m_xyy in slot 13 and m_xxy in slot 14.",
                "The Appendix-A x-flux matrix and the archived solver have A[7,13]=1, identifying slot 13 as m_xxy.",
                "The Appendix-A x-wall matrix applies the m_ssn=m_xyy condition at slot 14, while slot 13 is an identity row.",
            ],
            "integration_decision": "Preserve the executable A/X order; do not permute only the production operator.",
        },
        "prefactor_audit": {
            "archived": "rho*sqrt(theta)/Kn_Rana",
            "archived_implied_viscosity_power": "mu/mu0 = sqrt(theta/theta0)",
            "maxwell_collision_consistent": "rho/Kn_Rana",
            "maxwell_viscosity_power": "mu/mu0 = theta/theta0",
            "appendix_literal": "1/Kn_Rana",
            "appendix_literal_caveat": "The literal printed Eq. (11)+Appendix-A factor and the factor derived from Eqs. (3)-(4) with mu proportional to theta are not identical away from rho=1.",
        },
        "coefficient_audit": {
            "evaluation_state": state.tolist(),
            "mismatch_count": len(mismatches),
            "mismatch_rows": sorted({item["row_name"] for item in mismatches}),
            "rows_0_to_9_match": bool(np.allclose(archived_matrix[:10], printed_matrix[:10], rtol=2e-14, atol=2e-14)),
            "row_10_R_xy_matches": bool(np.allclose(archived_matrix[10], printed_matrix[10], rtol=2e-14, atol=2e-14)),
            "row_11_R_yy_matches": bool(np.allclose(archived_matrix[11], printed_matrix[11], rtol=2e-14, atol=2e-14)),
            "row_16_matches": bool(np.allclose(archived_matrix[16], printed_matrix[16], rtol=2e-14, atol=2e-14)),
            "mismatches": mismatches,
        },
        "kn_contract": {
            "formula": "Kn_Rana = sqrt(2/pi) Kn_Gu",
            "KnGu_0.05": {"Kn_Rana": kn_rana_from_gu(0.05)},
            "KnGu_0.20": {"Kn_Rana": kn_rana_from_gu(0.20)},
        },
        "claims_not_made": [
            "No new cavity state has been solved by this audit.",
            "No agreement with DSMC is inferred from coefficient matching.",
            "No branch is called an author-code reproduction without an external benchmark.",
        ],
        "candidate_status": "coefficient-tested and integration-order-tested; no nonlinear cavity solve or external benchmark completed",
    }
    output = HERE / "audit_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
