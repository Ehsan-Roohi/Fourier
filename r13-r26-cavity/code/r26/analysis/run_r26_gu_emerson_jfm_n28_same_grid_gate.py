#!/usr/bin/env python3
"""Run the immutable JFM Maxwell N28-to-N28 transformed-variable gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r26_cases import jfm_maxwell_cavity_case
from r26_gu_emerson_jfm_same_grid_gate import (
    JFM_N28_REFERENCE_STATE_SHA256,
    jsonable,
    run_jfm_n28_same_grid_gate,
)
from r26_thor_audit import state_sha256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_summary(summary: dict[str, object]) -> dict[str, object]:
    require(summary.get("termination") == "target_accepted", "target was not accepted")
    case = summary.get("case", {})
    require(isinstance(case, dict), "summary case metadata is missing")
    required = {
        "family": "jfm-maxwell",
        "molecular_model": "maxwell_molecules",
        "kn_convention": "gu_lambda_over_L",
        "nodes": 28,
        "viscosity_kind": "power_law",
        "viscosity_exponent": 1.0,
        "closure_mode": "jfm2009",
        "wall_accommodation": 1.0,
        "wall_temperature_K": 300.0,
        "lid_speed_m_per_s": 100.0,
        "beta": 0.0,
    }
    for name, expected in required.items():
        require(case.get(name) == expected, f"case metadata mismatch for {name}")
    require(
        math.isclose(float(case.get("kn_input")), 0.2, rel_tol=0.0, abs_tol=2.0e-15),
        "Kn mismatch",
    )
    require("Pure Maxwell-molecule" in str(case.get("provenance")), "provenance mismatch")
    attempts = summary.get("attempts", [])
    require(isinstance(attempts, list) and bool(attempts), "attempt history is missing")
    final = attempts[-1]
    require(isinstance(final, dict), "final attempt metadata is missing")
    accepted = final.get("accepted")
    require(
        isinstance(accepted, (bool, int)) and int(accepted) == 1,
        "final attempt rejected",
    )
    require(
        str(final.get("state_sha256")) == JFM_N28_REFERENCE_STATE_SHA256,
        "summary state SHA-256 mismatch",
    )
    require(float(final.get("raw_acceptance_gate")) <= 1.0e-8, "summary raw gate failed")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("source commit must be an immutable lowercase 40-character SHA")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    args.output_dir.mkdir(parents=True)

    record_path = args.output_dir / "JFM_N28_SAME_GRID_GATE.json"
    try:
        summary_path = args.reference_dir / "run_summary.json"
        state_path = args.reference_dir / "last_accepted_state.npz"
        require(summary_path.is_file(), "frozen run_summary.json is missing")
        require(state_path.is_file(), "frozen last_accepted_state.npz is missing")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(isinstance(summary, dict), "run_summary.json is not an object")
        validate_summary(summary)

        with np.load(state_path, allow_pickle=False) as archive:
            state = np.asarray(archive["state"], dtype=float)
            x = np.asarray(archive["x"], dtype=float)
            y = np.asarray(archive["y"], dtype=float)
            require(
                str(np.asarray(archive["kn_convention"]).item())
                == "gu_lambda_over_L",
                "state Kn convention mismatch",
            )
            require(
                math.isclose(
                    float(np.asarray(archive["kn_input"]).item()),
                    0.2,
                    rel_tol=0.0,
                    abs_tol=2.0e-15,
                ),
                "state Kn mismatch",
            )
            require(
                math.isclose(
                    float(np.asarray(archive["beta"]).item()),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=2.0e-15,
                ),
                "state beta mismatch",
            )
        require(
            state_sha256(state) == JFM_N28_REFERENCE_STATE_SHA256,
            "decoded frozen state SHA-256 mismatch",
        )

        case = jfm_maxwell_cavity_case(
            28,
            kn=0.2,
            lid_speed_m_per_s=100.0,
            wall_temperature_K=300.0,
            grid_stretch_beta=0.0,
        )
        result = run_jfm_n28_same_grid_gate(
            state,
            x,
            y,
            case=case,
            source_commit=args.source_commit,
        )
        record = dict(result.record)
        record.update(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "reference_directory": str(args.reference_dir.resolve()),
                "reference_state_file_sha256": sha256_file(state_path),
                "reference_summary_file_sha256": sha256_file(summary_path),
                "source_manifest": {
                    str(path.relative_to(ROOT)): sha256_file(path)
                    for path in (
                        ROOT / "r26_gu_emerson_variables.py",
                        ROOT / "r26_gu_emerson_transformed_fv.py",
                        ROOT / "r26_gu_emerson_jfm_same_grid_gate.py",
                        Path(__file__).resolve(),
                    )
                },
            }
        )
        passed = record.get("same_grid_gate_passed") is True
        np.savez_compressed(
            args.output_dir / "gu_emerson_jfm_n28_same_grid_candidate.npz",
            state=result.candidate_state,
            transformed_state=result.transformed_state,
            x=case.x,
            y=case.y,
            kn_input=case.kn,
            kn_convention=case.kn_convention.value,
            beta=case.grid_stretch_beta,
            lid_velocity=case.lid_velocity,
            reference_state_sha256=JFM_N28_REFERENCE_STATE_SHA256,
            source_commit=args.source_commit,
            accepted=passed,
            production_accepted=False,
            n32_authorized=passed,
        )
    except Exception as exc:
        passed = False
        record = {
            "status": "R26_GU_EMERSON_JFM_N28_SAME_GRID_GATE_FAILED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "reference_directory": str(args.reference_dir.resolve()),
            "failure": f"{type(exc).__name__}: {exc}",
            "same_grid_gate_passed": False,
            "candidate_accepted": False,
            "production_accepted": False,
            "n32_authorized": False,
            "n40_authorized": False,
            "n44_authorized": False,
        }

    record_path.write_text(
        json.dumps(jsonable(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(record), sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
