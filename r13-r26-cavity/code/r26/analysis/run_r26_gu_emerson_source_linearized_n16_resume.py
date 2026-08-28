#!/usr/bin/env python3
"""Resume the Gu--Emerson standalone gate with Rana source linearisation.

Only the two independently accepted N8 roots from the failed standalone job
are reusable.  The failed N16 iterate and every checkpoint produced from it
are ignored.  Each N8 state is independently re-evaluated with the immutable
physical gate, interpolated to N16, and advanced by the printed Gu--Emerson
field order with the nonlinear-source history factors present in the supplied
Rana Code_Saturne R26 source.
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
    RANA_SOURCE_HISTORY_RELAXATION,
    make_gu_emerson_reconstruction_problem,
)
from r26_solver import interpolate_state_grid
from r26_thor_audit import state_sha256

from run_r26_gu_emerson_standalone_ladder import (
    CASE,
    ROOT_ABSOLUTE_TOLERANCE,
    comparison,
    jsonable,
    run_path,
    write_json,
)


FAILED_STANDALONE_COMMIT = "a1661f698dd1080394b54abbb38e5fe6202d0bcd"
N8_PATHS = (
    "N8_FROM_EQUILIBRIUM",
    "N8_FROM_PERTURBED",
)
MAX_OUTER_ITERATIONS = 720
RANA_SOURCE_MODULE_SHA256 = (
    "d92e0142776d90499e2beea4a8b3b37b590597f66b61f43bb49f58ade73a884b"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve_failed_directory(path: Path) -> tuple[Path, Path]:
    root = path.resolve()
    data = root / "STANDALONE" if (root / "STANDALONE").is_dir() else root
    require(data.is_dir(), "failed standalone data directory does not exist")
    candidates = (
        root / "GU_EMERSON_STANDALONE_LADDER_FAILED.json",
        data / "GU_EMERSON_STANDALONE_LADDER_FAILED.json",
    )
    records = [candidate for candidate in candidates if candidate.is_file()]
    require(bool(records), "failed standalone final record is missing")
    record = json.loads(records[0].read_text())
    require(
        record.get("status") == "R26_GU_EMERSON_STANDALONE_LADDER_FAILED",
        "source standalone run is not explicitly FAILED",
    )
    require(
        record.get("source_commit") == FAILED_STANDALONE_COMMIT,
        "source standalone commit mismatch",
    )
    return data, records[0]


def load_accepted_n8(data: Path, name: str) -> tuple[np.ndarray, dict[str, object]]:
    target = data / name
    record = json.loads((target / "standalone_path.json").read_text())
    require(
        record.get("status") == "R26_GU_EMERSON_STANDALONE_PATH_PASSED",
        f"{name} path status is not PASSED",
    )
    require(record.get("accepted") is True, f"{name} is not accepted")
    require(record.get("production_accepted") is False, f"{name} is production-marked")
    with np.load(target / "standalone_state.npz", allow_pickle=False) as archive:
        require(bool(np.asarray(archive["accepted"]).item()), f"{name} archive is rejected")
        require(
            not bool(np.asarray(archive["production_accepted"]).item()),
            f"{name} archive is production-marked",
        )
        require(int(np.asarray(archive["nodes"]).item()) == 8, f"{name} node mismatch")
        require(float(np.asarray(archive["lid_speed_m_s"]).item()) == 10.0, f"{name} lid mismatch")
        require(float(np.asarray(archive["kn_input"]).item()) == 0.1, f"{name} Kn mismatch")
        state = np.asarray(archive["state"], dtype=float)
    require(state_sha256(state) == record.get("state_sha256"), f"{name} hash mismatch")
    case = gu_asme2009_cavity_case(
        8,
        kn=CASE["kn_gu"],
        lid_speed_m_per_s=CASE["lid_speed_m_s"],
        wall_temperature_K=CASE["wall_temperature_K"],
        grid_stretch_beta=CASE["grid_stretch_beta"],
    )
    diagnostics = make_gu_emerson_reconstruction_problem(case).evaluate(state).diagnostics
    raw = float(
        max(
            diagnostics.raw_total_linf,
            abs(diagnostics.held_out_continuity),
            abs(diagnostics.mass_error),
        )
    )
    require(raw <= 1.0e-8, f"{name} independent raw gate failed: {raw:.3e}")
    require(diagnostics.total_linf <= 1.0e-8, f"{name} independent scaled gate failed")
    require(abs(diagnostics.held_out_continuity) <= 1.0e-8, f"{name} continuity failed")
    require(abs(diagnostics.mass_error) <= 1.0e-10, f"{name} mass gate failed")
    require(diagnostics.min_density > 0.0, f"{name} density is non-positive")
    require(diagnostics.min_temperature > 0.0, f"{name} temperature is non-positive")
    return state, {
        "path_name": name,
        "state_sha256": state_sha256(state),
        "independent_raw_gate": raw,
        "diagnostics": asdict(diagnostics),
    }


def final_record(
    *,
    status: str,
    source_commit: str,
    source_failure_record: Path,
    completed_stages: list[str],
    failure_stage: str | None,
    comparisons: dict[str, object],
    reused_n8: list[dict[str, object]],
    exception: BaseException | None = None,
) -> dict[str, object]:
    passed = status == "R26_GU_EMERSON_SOURCE_LINEARIZED_N16_RESUME_PASSED"
    record: dict[str, object] = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "failed_standalone_commit": FAILED_STANDALONE_COMMIT,
        "failed_standalone_record": str(source_failure_record),
        "case_fixed": CASE,
        "source_module_sha256": RANA_SOURCE_MODULE_SHA256,
        "source_history_relaxation": RANA_SOURCE_HISTORY_RELAXATION,
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
            "bounded N24 source-linearized Gu--Emerson reconstruction from both accepted N16 roots"
            if passed
            else "inspect the recorded N16 source-linearized failure; N24 and finer grids remain blocked"
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
    source_failure_record = args.failed_standalone_dir / "GU_EMERSON_STANDALONE_LADDER_FAILED.json"
    try:
        data, source_failure_record = resolve_failed_directory(args.failed_standalone_dir)
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
        )
        write_json(
            args.output / "SOURCE_LINEARIZED_N16_PLAN.json",
            {
                "source_commit": args.source_commit,
                "case_fixed": CASE,
                "source_module_sha256": RANA_SOURCE_MODULE_SHA256,
                "source_history_relaxation": RANA_SOURCE_HISTORY_RELAXATION,
                "options": asdict(options),
                "ordered_stages": [
                    "REVALIDATE_N8_FROM_EQUILIBRIUM",
                    "REVALIDATE_N8_FROM_PERTURBED",
                    "N8_REUSED_ROOT_COMPARISON",
                    "N16_SOURCE_LINEARIZED_FROM_N8_EQUILIBRIUM",
                    "N16_SOURCE_LINEARIZED_FROM_N8_PERTURBED",
                    "N16_SOURCE_LINEARIZED_ROOT_COMPARISON",
                ],
                "failed_n16_checkpoints_reused": False,
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
                state,
                16,
                target_mean_density=case16.mean_density,
                mass_weights=weights16,
                old_x=case8.x,
                old_y=case8.y,
                new_x=case16.x,
                new_y=case16.y,
            )
            for state in states
        ]
        results: list[np.ndarray] = []
        for suffix, seed in zip(("EQUILIBRIUM", "PERTURBED"), seeds, strict=True):
            stage = f"N16_SOURCE_LINEARIZED_FROM_N8_{suffix}"
            state, record = run_path(
                args.output,
                nodes=16,
                path_name=stage,
                initial_state=seed,
                initial_kind=f"independently_revalidated_accepted_N8_{suffix.lower()}_interpolated",
                options=options,
            )
            require(record.get("accepted") is True, f"{stage} did not pass the raw gate")
            results.append(state)
            completed.append(stage)
        stage = "N16_SOURCE_LINEARIZED_ROOT_COMPARISON"
        comparisons["N16"] = comparison(results[0], results[1])
        write_json(args.output / "N16_SOURCE_LINEARIZED_ROOT_COMPARISON.json", comparisons["N16"])
        require(comparisons["N16"]["passed"] is True, "source-linearized N16 roots disagree")
        completed.append(stage)
    except Exception as exc:
        record = final_record(
            status="R26_GU_EMERSON_SOURCE_LINEARIZED_N16_RESUME_FAILED",
            source_commit=args.source_commit,
            source_failure_record=source_failure_record,
            completed_stages=completed,
            failure_stage=stage,
            comparisons=comparisons,
            reused_n8=reused_n8,
            exception=exc,
        )
        write_json(args.output / "GU_EMERSON_SOURCE_LINEARIZED_N16_RESUME_FAILED.json", record)
        print(json.dumps(jsonable(record), sort_keys=True), flush=True)
        raise

    record = final_record(
        status="R26_GU_EMERSON_SOURCE_LINEARIZED_N16_RESUME_PASSED",
        source_commit=args.source_commit,
        source_failure_record=source_failure_record,
        completed_stages=completed,
        failure_stage=None,
        comparisons=comparisons,
        reused_n8=reused_n8,
    )
    write_json(args.output / "GU_EMERSON_SOURCE_LINEARIZED_N16_RESUME_PASSED.json", record)
    print(json.dumps(jsonable(record), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
