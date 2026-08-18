#!/usr/bin/env python3
"""Fail-closed integration adapter for the audited R13 production candidate.

The adapter does not run a cavity calculation.  It verifies the exact source
hashes and the mixed-third-moment ordering used by the archived flux and wall
operators before installing the candidate production function into the
already-loaded solver module.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from r13_maxwell_production import STATE_ORDER_SOLVER, production_maxwell


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_DIR = ROOT / "work/article_final_v2/data/kn020_models/r13_N60"
COEFFICIENT_PATH = SOURCE_DIR / "rana_original_coefficients.py"
SOLVER_PATH = SOURCE_DIR / "rana_original_reference_solver.py"

EXPECTED_COEFFICIENT_SHA256 = "08caba3895db19c72cc69fe8c9be4b41fb5676b47e140ce123798df548a0b6fd"
EXPECTED_SOLVER_SHA256 = "9b10862a3582ae59e91303292865ef142eabb022c103a4b18abbe0956f7f2e24"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_archived_solver() -> ModuleType:
    """Load the audited archived solver without changing article files."""

    actual = {
        "coefficient": sha256(COEFFICIENT_PATH),
        "solver": sha256(SOLVER_PATH),
    }
    expected = {
        "coefficient": EXPECTED_COEFFICIENT_SHA256,
        "solver": EXPECTED_SOLVER_SHA256,
    }
    if actual != expected:
        raise RuntimeError(f"archived R13 source hash mismatch: {actual!r}")
    sys.path.insert(0, str(SOURCE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "audited_archived_r13_solver", SOLVER_PATH
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load archived R13 solver")
        module = importlib.util.module_from_spec(spec)
        # dataclasses inspects sys.modules during class construction.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(SOURCE_DIR))
        except ValueError:  # pragma: no cover - defensive only.
            pass


def verify_solver_order(solver: ModuleType) -> dict[str, object]:
    """Prove the order used jointly by STATE_ORDER, A(U), and X(U)."""

    if tuple(solver.STATE_ORDER) != STATE_ORDER_SOLVER:
        raise RuntimeError(
            "solver state order is not the audited conventional order: "
            f"{tuple(solver.STATE_ORDER)!r}"
        )
    u = np.asarray(
        [
            1.1, 0.02, -0.03, 1.05, 0.01, -0.012, 0.02, -0.008,
            -0.013, 0.001, -0.0015, 0.002, 0.0004, -0.0005,
            0.0006, -0.0007, 0.0008,
        ],
        dtype=float,
    )
    ax = solver.flux_x(u)
    # sigma_xy balance contains d(m_xxy)/dx, hence A[7,13] = 1.
    if ax[7, 13] != 1.0 or ax[7, 14] != 0.0:
        raise RuntimeError("flux_x does not identify slot 13 as m_xxy")
    wall = solver.wall_matrix(
        u,
        axis="x",
        normal=1,
        accommodation=1.0,
        pressure_mode="paper-tangential",
    )
    identity = np.eye(17)
    # x-wall m_ssn is m_xyy: slot 14 is a boundary row and slot 13 remains an
    # identity row.  This is independent of the isolated Eq. (11) list typo.
    if not np.array_equal(wall[13], identity[13]):
        raise RuntimeError("x-wall slot 13 is not the expected m_xxy identity row")
    if np.array_equal(wall[14], identity[14]):
        raise RuntimeError("x-wall slot 14 is not the expected m_xyy boundary row")
    return {
        "state_order": list(STATE_ORDER_SOLVER),
        "flux_proof": "flux_x[7,13]=1 and flux_x[7,14]=0",
        "wall_proof": "x-wall row 13 is identity; row 14 is m_ssn boundary row",
    }


def install_candidate(solver: ModuleType) -> dict[str, object]:
    """Install the candidate only after all integration invariants pass."""

    order_evidence = verify_solver_order(solver)
    previous = solver.production
    solver.production = production_maxwell
    return {
        **order_evidence,
        "previous_production_module": previous.__module__,
        "installed_production_module": production_maxwell.__module__,
        "candidate_status": "not-run-ready; coefficient/integration tests only",
    }

