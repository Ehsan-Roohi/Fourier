#!/usr/bin/env python3
"""Run one bounded THOR-style R26 validation candidate.

The driver solves a fixed physical case; it does not perform lid continuation
or promote a refined grid automatically.  A successful record means the
unscaled equations, balances, positivity and structural-rank checks passed.
It deliberately records ``production_accepted=false`` until an independent
final numerical-Jacobian rank check and cross-solver/profile comparison have
also been completed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.sparse.csgraph import structural_rank

from r26_cases import gu_asme2009_cavity_case, jfm_maxwell_cavity_case
from r26_postprocess import rana_global_metrics
from r26_solver import interpolate_state_grid, jacobian_sparsity
from r26_thor_solver import (
    THOR_SOLVER_PROVENANCE,
    ThorSolveOptions,
    make_thor_problem,
    solve_r26_thor_bvp,
)
from r26_validation import global_balance_diagnostics, leading_r13_nsf_diagnostics


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code" if (ROOT / "code" / "r26_cases.py").is_file() else ROOT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(state, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(b"|<f8|")
    digest.update(value.tobytes())
    return digest.hexdigest()


def jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Enum):
        return value.value
    return value


def source_manifest() -> dict[str, str]:
    files = tuple(sorted(CODE.glob("r26_*.py"))) + (Path(__file__).resolve(),)
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in files
    }


def load_initial_state(path: Path, case: object, problem: object) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        if "accepted" in archive and not bool(np.asarray(archive["accepted"]).item()):
            raise ValueError("initial state is explicitly marked rejected")
        state = np.asarray(archive["state"], dtype=float)
        old_x = np.asarray(archive["x"], dtype=float)
        old_y = np.asarray(archive["y"], dtype=float)
    if state.shape != problem.shape or not np.array_equal(old_x, case.x) or not np.array_equal(old_y, case.y):
        state = interpolate_state_grid(
            state,
            case.nodes,
            old_x=old_x,
            old_y=old_y,
            new_x=case.x,
            new_y=case.y,
            mass_weights=problem.mass_weights,
        )
        action = "interpolated_and_mass_corrected"
    else:
        action = "same_grid"
    return state, {
        "kind": "explicit_restart",
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "action": action,
        "state_sha256": state_sha256(state),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-family", choices=("jfm-maxwell", "gu-asme2009"), default="jfm-maxwell")
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--kn-gu", type=float, default=0.2)
    parser.add_argument("--lid-speed-m-s", type=float, default=100.0)
    parser.add_argument("--wall-temperature-k", type=float)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--raw-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--inner-max-iterations", type=int, default=100)
    parser.add_argument("--ilu-drop-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--ilu-fill-factor", type=float, default=48.0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be absent or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.case_family == "jfm-maxwell":
        wall_temperature = 300.0 if args.wall_temperature_k is None else args.wall_temperature_k
        case = jfm_maxwell_cavity_case(
            args.nodes,
            kn=args.kn_gu,
            lid_speed_m_per_s=args.lid_speed_m_s,
            wall_temperature_K=wall_temperature,
            grid_stretch_beta=args.beta,
        )
    else:
        wall_temperature = 273.0 if args.wall_temperature_k is None else args.wall_temperature_k
        case = gu_asme2009_cavity_case(
            args.nodes,
            kn=args.kn_gu,
            lid_speed_m_per_s=args.lid_speed_m_s,
            wall_temperature_K=wall_temperature,
            grid_stretch_beta=args.beta,
        )
    problem = make_thor_problem(case)
    if args.initial_state is None:
        state = case.equilibrium_state()
        initial = {"kind": "analytic_equilibrium", "state_sha256": state_sha256(state)}
    else:
        state, initial = load_initial_state(args.initial_state, case, problem)

    options = ThorSolveOptions(
        residual_tolerance=args.solver_tolerance,
        raw_tolerance=args.raw_tolerance,
        held_out_continuity_tolerance=args.raw_tolerance,
        mass_tolerance=min(1.0e-10, args.raw_tolerance),
        max_iterations=args.max_iterations,
        inner_max_iterations=args.inner_max_iterations,
        ilu_drop_tolerance=args.ilu_drop_tolerance,
        ilu_fill_factor=args.ilu_fill_factor,
    )
    started = time.time()
    result = solve_r26_thor_bvp(problem, state, options=options)
    elapsed = time.time() - started
    solution = result.solution
    balances = global_balance_diagnostics(solution.state, case)
    pattern = jacobian_sparsity(problem)
    rank = int(structural_rank(pattern))
    physical_gate = bool(
        solution.converged
        and result.raw_gate_passed
        and balances["wall_effective_pressure_min"] > 0.0
        and balances["momentum_boundary_flux_linf"] <= 10.0 * args.raw_tolerance
        and abs(balances["internal_energy_balance_error"]) <= 10.0 * args.raw_tolerance
        and rank == problem.unknown_count
        and result.ilu_available
    )
    record: dict[str, object] = {
        "status": "R26_THOR_VALIDATION_CANDIDATE_PASSED" if physical_gate else "R26_THOR_VALIDATION_CANDIDATE_FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": asdict(case),
        "initial_state": initial,
        "solver": {
            "method": solution.solver_method,
            "message": solution.message,
            "scipy_success": solution.scipy_success,
            "iterations": solution.iterations,
            "function_evaluations": solution.function_evaluations,
            "invalid_evaluations": solution.invalid_evaluations,
            "pressure_factorizations": result.pressure_factorizations,
            "preconditioner_applications": result.preconditioner_applications,
            "frozen_jacobian_residual_evaluations": result.frozen_jacobian_residual_evaluations,
            "frozen_jacobian_nonzeros": result.frozen_jacobian_nonzeros,
            "ilu_available": result.ilu_available,
        },
        "diagnostics": asdict(solution.diagnostics),
        "raw_acceptance_gate": result.raw_acceptance_gate,
        "global_balances": balances,
        "rana_metrics": rana_global_metrics(
            solution.state,
            lid_velocity=case.lid_velocity,
            x=case.x,
            y=case.y,
        ),
        "leading_r13_nsf": leading_r13_nsf_diagnostics(solution.state, case),
        "structural_jacobian_rank": rank,
        "unknown_count": problem.unknown_count,
        "physical_candidate_gate_passed": physical_gate,
        "production_accepted": False,
        "production_blocker": "independent final numerical-Jacobian rank and cross-solver/profile checks are not yet complete",
        "elapsed_seconds": elapsed,
        "state_sha256": state_sha256(solution.state),
        "provenance": THOR_SOLVER_PROVENANCE,
        "source_manifest": source_manifest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    np.savez_compressed(
        args.output_dir / "thor_state.npz",
        state=solution.state,
        x=case.x,
        y=case.y,
        lid_velocity=case.lid_velocity,
        lid_speed_m_s=args.lid_speed_m_s,
        kn_input=case.kn,
        kn_convention=case.kn_convention.value,
        mu_equilibrium=case.mu_equilibrium,
        beta=case.grid_stretch_beta,
        accepted=physical_gate,
        production_accepted=False,
    )
    (args.output_dir / "thor_validation.json").write_text(
        json.dumps(jsonable(record), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(jsonable(record), sort_keys=True))
    raise SystemExit(0 if physical_gate else 1)


if __name__ == "__main__":
    main()
