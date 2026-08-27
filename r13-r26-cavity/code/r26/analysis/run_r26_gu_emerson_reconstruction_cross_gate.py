#!/usr/bin/env python3
"""Cross-solver fixed-point gate for the documented Gu--Emerson reconstruction.

The independently solved THOR-style roots are reference states only.  This
driver converts each root to the printed Gu--Emerson variables, executes one
complete field-by-field sweep, and requires the full unscaled R26 gate to
remain closed.  It does not claim convergence from equilibrium and never
authorizes a production or refined-grid run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path

import numpy as np

from r26_cases import gu_asme2009_cavity_case
from r26_gu_emerson_algorithm import GU_EMERSON_STAGE_ORDER
from r26_gu_emerson_reconstruction import (
    GuEmersonReconstructionOptions,
    make_gu_emerson_reconstruction_problem,
    solve_gu_emerson_reconstruction,
)
from r26_thor_audit import state_sha256


def jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Enum):
        return value.value
    return value


def load_reference(root: Path, nodes: int) -> tuple[np.ndarray, dict[str, object]]:
    directory = root / f"REFERENCE_N{nodes}"
    record = json.loads((directory / "thor_validation.json").read_text())
    if record.get("status") != "R26_THOR_VALIDATION_CANDIDATE_PASSED":
        raise RuntimeError(f"N{nodes} independent reference record did not pass")
    if record.get("physical_candidate_gate_passed") is not True:
        raise RuntimeError(f"N{nodes} reference physical gate is not true")
    if record.get("production_accepted") is not False:
        raise RuntimeError(f"N{nodes} reference must remain non-production")
    case = record.get("case", {})
    required = {
        "nodes": nodes,
        "kn": 0.1,
        "kn_convention": "gu_lambda_over_L",
        "r26_closure_mode": "asme2009-cavity",
    }
    for name, expected in required.items():
        if case.get(name) != expected:
            raise RuntimeError(
                f"N{nodes} reference case mismatch for {name}: {case.get(name)!r}"
            )
    with np.load(directory / "thor_state.npz", allow_pickle=False) as archive:
        if not bool(np.asarray(archive["accepted"]).item()):
            raise RuntimeError(f"N{nodes} reference state is explicitly rejected")
        if bool(np.asarray(archive["production_accepted"]).item()):
            raise RuntimeError(f"N{nodes} reference state is incorrectly production-marked")
        state = np.asarray(archive["state"], dtype=float)
        lid_speed = float(np.asarray(archive["lid_speed_m_s"]).item())
        kn = float(np.asarray(archive["kn_input"]).item())
    if lid_speed != 10.0 or kn != 0.1:
        raise RuntimeError(f"N{nodes} reference metadata is not Kn=0.1, U=10 m/s")
    if state_sha256(state) != record.get("state_sha256"):
        raise RuntimeError(f"N{nodes} reference state hash mismatch")
    return state, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not __import__("re").fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("source commit must be an immutable lowercase 40-character SHA")

    options = GuEmersonReconstructionOptions(max_outer_iterations=1)
    options.disclosure.require_production_authorization()
    grids: list[dict[str, object]] = []
    all_passed = True
    for nodes in (8, 16):
        reference_state, reference_record = load_reference(args.reference_root, nodes)
        case = gu_asme2009_cavity_case(
            nodes,
            kn=0.1,
            lid_speed_m_per_s=10.0,
            wall_temperature_K=273.0,
            grid_stretch_beta=0.0,
        )
        problem = make_gu_emerson_reconstruction_problem(case)
        before = problem.evaluate(reference_state).diagnostics
        result = solve_gu_emerson_reconstruction(problem, reference_state, options=options)
        after = problem.evaluate(result.state).diagnostics
        delta_linf = float(np.max(np.abs(result.state - reference_state)))
        stage_order = result.records[-1].stage_order if result.records else ()
        passed = bool(
            result.converged
            and before.raw_total_linf <= options.raw_tolerance
            and after.raw_total_linf <= options.raw_tolerance
            and abs(after.held_out_continuity) <= options.held_continuity_tolerance
            and abs(after.mass_error) <= options.mass_tolerance
            and after.min_density > 0.0
            and after.min_temperature > 0.0
            and stage_order == GU_EMERSON_STAGE_ORDER
            and delta_linf <= 1.0e-8
        )
        all_passed = all_passed and passed
        target = args.output / f"N{nodes}"
        target.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            target / "gu_emerson_reconstruction_state.npz",
            state=result.state,
            x=case.x,
            y=case.y,
            lid_speed_m_s=10.0,
            lid_velocity=case.lid_velocity,
            kn_input=case.kn,
            accepted=passed,
            production_accepted=False,
            standalone_from_equilibrium=False,
        )
        grid_record = {
            "nodes": nodes,
            "passed": passed,
            "case": asdict(case),
            "reference_solver": "thor-simple-krylov; independent raw-root reference only",
            "reference_state_sha256": reference_record["state_sha256"],
            "reference_raw_gate": reference_record["raw_acceptance_gate"],
            "reference_diagnostics_recomputed": asdict(before),
            "reconstruction_state_sha256": state_sha256(result.state),
            "reconstruction_diagnostics": asdict(after),
            "state_change_linf": delta_linf,
            "executed_stage_order": list(stage_order),
            "solver": {
                "outer_iterations": result.outer_iterations,
                "residual_evaluations": result.residual_evaluations,
                "block_factorizations": result.block_factorizations,
                "lsmr_fallbacks": result.lsmr_fallbacks,
                "wall_solves": result.wall_solves,
                "wall_function_evaluations": result.wall_function_evaluations,
                "message": result.message,
            },
        }
        (target / "gu_emerson_reconstruction.json").write_text(
            json.dumps(jsonable(grid_record), indent=2, sort_keys=True) + "\n"
        )
        grids.append(grid_record)

    controls = {
        name: asdict(source)
        for name, source in (options.disclosure.controls or {}).items()
    }
    final = {
        "status": (
            "R26_GU_EMERSON_RECONSTRUCTION_CROSS_GATE_PASSED"
            if all_passed
            else "R26_GU_EMERSON_RECONSTRUCTION_CROSS_GATE_FAILED"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "case_reference": "Gu--John--Tang--Emerson ASME HT2009-88293",
        "case_fixed": {
            "kn_gu": 0.1,
            "lid_speed_m_s": 10.0,
            "wall_temperature_K": 273.0,
            "grids": [8, 16],
        },
        "published_stage_order": list(GU_EMERSON_STAGE_ORDER),
        "reconstruction_controls": controls,
        "controls_fully_declared": options.disclosure.production_authorized,
        "cross_solver_fixed_point_gate_passed": all_passed,
        "standalone_from_equilibrium_passed": False,
        "standalone_from_equilibrium_attempted": False,
        "production_accepted": False,
        "n24_authorized": False,
        "n28_authorized": False,
        "n29_authorized": False,
        "n30_authorized": False,
        "next_required_stage": (
            "bounded standalone N8 solve from equilibrium and a declared perturbed start; "
            "N16 remains blocked until N8 passes"
        ),
        "grids": grids,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "GU_EMERSON_RECONSTRUCTION_CROSS_GATE.json").write_text(
        json.dumps(jsonable(final), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(jsonable(final), sort_keys=True))
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
