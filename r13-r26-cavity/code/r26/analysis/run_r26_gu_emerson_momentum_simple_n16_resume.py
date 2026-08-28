#!/usr/bin/env python3
"""Resume N16 with the momentum-equation diagonal required by SIMPLE.

Gu--Emerson JFM 636 (2009), Sec. 5.2, requires the velocity equation to be
solved first and its SIMPLE pressure-correction equation to update pressure
and velocity second.  This gate therefore uses the component-wise diagonal of
that same under-relaxed velocity block (alpha_u/a_P).  It deliberately disables
the transferred Rana nonlinear-source history that failed in Job 63725331.
Only the two independently accepted N8 roots are reusable; every failed N16
iterate remains excluded.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import traceback

import numpy as np

from r26_cases import gu_asme2009_cavity_case
from r26_gu_emerson_reconstruction import (
    GuEmersonReconstructionOptions,
    make_gu_emerson_reconstruction_problem,
)
from r26_solver import interpolate_state_grid

from run_r26_gu_emerson_source_linearized_n16_resume import (
    FAILED_STANDALONE_COMMIT,
    N8_PATHS,
    load_accepted_n8,
    resolve_failed_directory,
)
from run_r26_gu_emerson_standalone_ladder import (
    CASE,
    comparison,
    jsonable,
    run_path,
    write_json,
)
FAILED_SOURCE_LINEARIZED_COMMIT = "563442d5ce7976b63d82c9592efd6ec3ef620830"
MAX_OUTER_ITERATIONS = 720
N16_PATHS = (
    "N16_MOMENTUM_SIMPLE_FROM_N8_EQUILIBRIUM",
    "N16_MOMENTUM_SIMPLE_FROM_N8_PERTURBED",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve_failed_source_linearized(path: Path) -> Path:
    root = path.resolve()
    data = root / "SOURCE_LINEARIZED" if (root / "SOURCE_LINEARIZED").is_dir() else root
    candidates = (
        data / "GU_EMERSON_SOURCE_LINEARIZED_N16_RESUME_FAILED.json",
        root / "GU_EMERSON_SOURCE_LINEARIZED_N16_RESUME_FAILED.json",
    )
    records = [candidate for candidate in candidates if candidate.is_file()]
    require(bool(records), "source-linearized failure record is missing")
    record = json.loads(records[0].read_text())
    require(
        record.get("status") == "R26_GU_EMERSON_SOURCE_LINEARIZED_N16_RESUME_FAILED",
        "source-linearized predecessor is not explicitly FAILED",
    )
    require(
        record.get("source_commit") == FAILED_SOURCE_LINEARIZED_COMMIT,
        "source-linearized predecessor commit mismatch",
    )
    require(record.get("reused_failed_n16_state") is False, "predecessor reused failed N16")
    require(record.get("n24_authorized") is False, "failed predecessor authorized N24")
    return records[0]


def final_record(
    *,
    status: str,
    source_commit: str,
    standalone_failure_record: Path,
    source_linearized_failure_record: Path,
    completed_stages: list[str],
    failure_stage: str | None,
    comparisons: dict[str, object],
    reused_n8: list[dict[str, object]],
    exception: BaseException | None = None,
) -> dict[str, object]:
    passed = status == "R26_GU_EMERSON_MOMENTUM_SIMPLE_N16_RESUME_PASSED"
    record: dict[str, object] = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "failed_standalone_commit": FAILED_STANDALONE_COMMIT,
        "failed_standalone_record": str(standalone_failure_record),
        "failed_source_linearized_commit": FAILED_SOURCE_LINEARIZED_COMMIT,
        "failed_source_linearized_record": str(source_linearized_failure_record),
        "case_fixed": CASE,
        "simple_momentum_coefficient": "component-wise alpha_u/a_P from the velocity block solved in stage (i)",
        "simple_velocity_correction": "full pressure-correction velocity update",
        "simple_pressure_update": "fixed pressure under-relaxation applied through p=rho*theta",
        "rana_source_history_used": False,
        "matrix_refresh_interval": 1,
        "max_outer_iterations_per_path": MAX_OUTER_ITERATIONS,
        "reused_n8_states": reused_n8,
        "reused_failed_n16_state": False,
        "completed_stages": completed_stages,
        "failure_stage": failure_stage,
        "root_comparisons": comparisons,
        "standalone_from_equilibrium_passed": passed,
        "independent_start_consistency_passed": passed,
        "production_accepted": False,
        "n24_authorized": passed,
        "n28_authorized": False,
        "n29_authorized": False,
        "n30_authorized": False,
        "next_required_stage": (
            "bounded N24 momentum-diagonal SIMPLE reconstruction from both accepted N16 roots"
            if passed
            else "inspect the recorded momentum-diagonal SIMPLE N16 failure; N24 and finer grids remain blocked"
        ),
    }
    if exception is not None:
        record["exception_type"] = type(exception).__name__
        record["exception_message"] = str(exception)
        record["traceback"] = traceback.format_exc()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-standalone-dir", type=Path, required=True)
    parser.add_argument("--failed-source-linearized-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        parser.error("source commit must be an immutable lowercase 40-character SHA")
    args.output.mkdir(parents=True, exist_ok=False)

    completed: list[str] = []
    comparisons: dict[str, object] = {}
    reused_n8: list[dict[str, object]] = []
    stage = "SOURCE_AUTHENTICATION"
    standalone_record = args.failed_standalone_dir / "GU_EMERSON_STANDALONE_LADDER_FAILED.json"
    source_linearized_record = (
        args.failed_source_linearized_dir
        / "SOURCE_LINEARIZED"
        / "GU_EMERSON_SOURCE_LINEARIZED_N16_RESUME_FAILED.json"
    )
    try:
        data, standalone_record = resolve_failed_directory(args.failed_standalone_dir)
        source_linearized_record = resolve_failed_source_linearized(
            args.failed_source_linearized_dir
        )
        completed.append(stage)

        states: list[np.ndarray] = []
        for name in N8_PATHS:
            stage = f"REVALIDATE_{name}"
            state, audit = load_accepted_n8(data, name)
            states.append(state)
            reused_n8.append(audit)
            completed.append(stage)
        comparisons["N8"] = comparison(states[0], states[1])
        write_json(args.output / "N8_REUSED_ROOT_COMPARISON.json", comparisons["N8"])
        require(comparisons["N8"]["passed"] is True, "reused N8 roots disagree")
        completed.append("N8_REUSED_ROOT_COMPARISON")

        options = GuEmersonReconstructionOptions(
            max_outer_iterations=MAX_OUTER_ITERATIONS,
            matrix_refresh_interval=1,
            use_rana_source_history=False,
        )
        write_json(
            args.output / "MOMENTUM_SIMPLE_N16_PLAN.json",
            {
                "source_commit": args.source_commit,
                "case_fixed": CASE,
                "options": asdict(options),
                "simple_momentum_coefficient": "component-wise alpha_u/a_P from the just-solved velocity block",
                "simple_velocity_correction": "full correction; pressure relaxation is not applied twice",
                "rana_source_history_used": False,
                "failed_n16_checkpoints_reused": False,
                "ordered_stages": [
                    "REVALIDATE_N8_FROM_EQUILIBRIUM",
                    "REVALIDATE_N8_FROM_PERTURBED",
                    "N8_REUSED_ROOT_COMPARISON",
                    *N16_PATHS,
                    "N16_MOMENTUM_SIMPLE_ROOT_COMPARISON",
                ],
                "stop_on_first_failed_stage": True,
            },
        )
        case8 = gu_asme2009_cavity_case(
            8, kn=0.1, lid_speed_m_per_s=10.0,
            wall_temperature_K=273.0, grid_stretch_beta=0.0,
        )
        case16 = gu_asme2009_cavity_case(
            16, kn=0.1, lid_speed_m_per_s=10.0,
            wall_temperature_K=273.0, grid_stretch_beta=0.0,
        )
        weights16 = make_gu_emerson_reconstruction_problem(case16).mass_weights
        seeds = [
            interpolate_state_grid(
                state, 16,
                target_mean_density=case16.mean_density,
                mass_weights=weights16,
                old_x=case8.x, old_y=case8.y,
                new_x=case16.x, new_y=case16.y,
            )
            for state in states
        ]
        results: list[np.ndarray] = []
        for path_name, seed in zip(N16_PATHS, seeds, strict=True):
            stage = path_name
            state, record = run_path(
                args.output,
                nodes=16,
                path_name=path_name,
                initial_state=seed,
                initial_kind="independently_revalidated_accepted_N8_interpolated",
                options=options,
            )
            require(record.get("accepted") is True, f"{path_name} did not pass the raw gate")
            results.append(state)
            completed.append(stage)
        stage = "N16_MOMENTUM_SIMPLE_ROOT_COMPARISON"
        comparisons["N16"] = comparison(results[0], results[1])
        write_json(args.output / "N16_MOMENTUM_SIMPLE_ROOT_COMPARISON.json", comparisons["N16"])
        require(comparisons["N16"]["passed"] is True, "momentum-SIMPLE N16 roots disagree")
        completed.append(stage)
    except Exception as exc:
        record = final_record(
            status="R26_GU_EMERSON_MOMENTUM_SIMPLE_N16_RESUME_FAILED",
            source_commit=args.source_commit,
            standalone_failure_record=standalone_record,
            source_linearized_failure_record=source_linearized_record,
            completed_stages=completed,
            failure_stage=stage,
            comparisons=comparisons,
            reused_n8=reused_n8,
            exception=exc,
        )
        write_json(args.output / "GU_EMERSON_MOMENTUM_SIMPLE_N16_RESUME_FAILED.json", record)
        print(json.dumps(jsonable(record), sort_keys=True), flush=True)
        raise

    record = final_record(
        status="R26_GU_EMERSON_MOMENTUM_SIMPLE_N16_RESUME_PASSED",
        source_commit=args.source_commit,
        standalone_failure_record=standalone_record,
        source_linearized_failure_record=source_linearized_record,
        completed_stages=completed,
        failure_stage=None,
        comparisons=comparisons,
        reused_n8=reused_n8,
    )
    write_json(args.output / "GU_EMERSON_MOMENTUM_SIMPLE_N16_RESUME_PASSED.json", record)
    print(json.dumps(jsonable(record), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
