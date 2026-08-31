#!/usr/bin/env python3
"""Independently replay the JFM Maxwell N28 same-grid acceptance gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"R26_JFM_N28_SAME_GRID_VALIDATION_FAILED: {message}")


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"record missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"record is not an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args()
    require(
        re.fullmatch(r"[0-9a-f]{40}", args.expected_source_commit) is not None,
        "expected source commit is not a full lowercase SHA",
    )
    output = args.run_dir / "JFM_N28_SAME_GRID_GATE_INDEPENDENT_VALIDATION.json"
    require(not output.exists(), "independent validation record already exists")

    record = read_json(args.run_dir / "JFM_N28_SAME_GRID_GATE.json")
    require(
        record.get("status") == "R26_GU_EMERSON_JFM_N28_SAME_GRID_GATE_PASSED",
        "primary gate status did not pass",
    )
    require(record.get("source_commit") == args.expected_source_commit, "source commit mismatch")
    require(record.get("same_grid_gate_passed") is True, "same-grid gate is not true")
    require(record.get("candidate_accepted") is True, "candidate is not accepted")
    require(record.get("production_accepted") is False, "premature production claim")
    require(record.get("n32_authorized") is True, "N32 was not explicitly authorized")
    require(record.get("n40_authorized") is False, "N40 was prematurely authorized")
    require(record.get("n44_authorized") is False, "N44 was prematurely authorized")
    require(record.get("higher_grid_run_attempted") is False, "higher grid was run inside N28 gate")

    state_path = args.reference_dir / "last_accepted_state.npz"
    candidate_path = args.run_dir / "gu_emerson_jfm_n28_same_grid_candidate.npz"
    require(state_path.is_file(), "frozen reference state is missing")
    require(candidate_path.is_file(), "candidate archive is missing")
    with np.load(state_path, allow_pickle=False) as archive:
        reference_state = np.asarray(archive["state"], dtype=float)
        reference_x = np.asarray(archive["x"], dtype=float)
        reference_y = np.asarray(archive["y"], dtype=float)
    require(
        state_sha256(reference_state) == JFM_N28_REFERENCE_STATE_SHA256,
        "frozen reference state hash mismatch",
    )
    with np.load(candidate_path, allow_pickle=False) as archive:
        candidate_state = np.asarray(archive["state"], dtype=float)
        transformed_state = np.asarray(archive["transformed_state"], dtype=float)
        require(bool(np.asarray(archive["accepted"]).item()), "archive candidate is rejected")
        require(
            not bool(np.asarray(archive["production_accepted"]).item()),
            "archive makes a production claim",
        )
        require(bool(np.asarray(archive["n32_authorized"]).item()), "archive blocks N32")
        require(
            str(np.asarray(archive["source_commit"]).item())
            == args.expected_source_commit,
            "archive source commit mismatch",
        )

    case = jfm_maxwell_cavity_case(
        28,
        kn=0.2,
        lid_speed_m_per_s=100.0,
        wall_temperature_K=300.0,
        grid_stretch_beta=0.0,
    )
    replay = run_jfm_n28_same_grid_gate(
        reference_state,
        reference_x,
        reference_y,
        case=case,
        source_commit=args.expected_source_commit,
    )
    require(replay.record.get("same_grid_gate_passed") is True, "independent replay failed")
    require(np.array_equal(candidate_state, replay.candidate_state), "candidate state is not reproducible")
    require(
        np.array_equal(transformed_state, replay.transformed_state),
        "transformed state is not reproducible",
    )
    require(
        state_sha256(candidate_state)
        == record.get("candidate", {}).get("state_sha256"),
        "candidate state hash differs from primary record",
    )
    for section, key in (
        ("reference", "independent_raw_gate"),
        ("candidate", "independent_raw_gate"),
        ("candidate", "transformed_interior_linf"),
        ("candidate", "transformed_vs_compatible_physical_linf"),
        ("candidate", "physical_roundtrip_linf"),
        ("candidate", "transformed_storage_roundtrip_linf"),
    ):
        primary = float(record[section][key])
        repeated = float(replay.record[section][key])
        require(
            math.isclose(primary, repeated, rel_tol=0.0, abs_tol=5.0e-15),
            f"replayed metric differs: {section}.{key}",
        )

    validation = {
        "status": "R26_GU_EMERSON_JFM_N28_SAME_GRID_INDEPENDENT_VALIDATION_PASSED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.expected_source_commit,
        "reference_state_sha256": JFM_N28_REFERENCE_STATE_SHA256,
        "candidate_state_sha256": state_sha256(candidate_state),
        "primary_gate_replayed": True,
        "candidate_arrays_exactly_reproduced": True,
        "production_accepted": False,
        "n32_authorized": True,
        "n40_authorized": False,
        "n44_authorized": False,
    }
    output.write_text(
        json.dumps(jsonable(validation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(validation), sort_keys=True))


if __name__ == "__main__":
    main()
