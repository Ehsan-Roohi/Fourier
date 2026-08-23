#!/usr/bin/env python3
"""Fail-closed validation of the N8/N16 balanced-arclength metric gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"R26_BALANCED_METRIC_VALIDATION_FAILED: {message}")


def state_sha256(state: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(state, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(b"|<f8|")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def close(value: object, expected: float, tolerance: float = 2.0e-14) -> bool:
    return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate_dir", type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument(
        "--expected-historical-source-commit",
        default="8cbd874eea68dd475faa3f5e3fb318b49cc0c665",
    )
    parser.add_argument("--expected-parameter-fraction", type=float, default=0.5)
    parser.add_argument("--raw-tolerance", type=float, default=1.0e-8)
    args = parser.parse_args()

    require(args.gate_dir.is_dir(), "gate directory missing")
    require(len(args.expected_source_commit) == 40, "source commit SHA has wrong length")
    require(
        len(args.expected_historical_source_commit) == 40,
        "historical source commit SHA has wrong length",
    )
    require(
        0.1 <= args.expected_parameter_fraction <= 0.9,
        "expected parameter fraction lies outside the admissible interval",
    )
    require(args.raw_tolerance > 0.0, "raw tolerance must be positive")

    root_path = args.gate_dir / "BALANCED_METRIC_GATE_PASSED.json"
    nested_path = args.gate_dir / "GATE" / "BALANCED_METRIC_GATE_PASSED.json"
    for path in (root_path, nested_path):
        require(path.is_file(), f"required gate record missing: {path.name}")
    root = json.loads(root_path.read_text(encoding="utf-8"))
    nested = json.loads(nested_path.read_text(encoding="utf-8"))

    require(
        root.get("status") == "R26_BALANCED_ARCLENGTH_METRIC_GATE_PASSED",
        "root gate status is not PASS",
    )
    require(
        root.get("source_commit") == args.expected_source_commit,
        "root gate source commit mismatch",
    )
    require(root.get("n30_authorized") is False, "root gate changed N30 authorization")
    require(
        nested.get("status") == "R26_BALANCED_ARCLENGTH_METRIC_GATE_PASS",
        "nested metric replay status is not PASS",
    )
    require(
        nested.get("historical_gate_source_commit")
        == args.expected_historical_source_commit,
        "historical gate source commit mismatch",
    )
    require(
        nested.get("n30_authorized") is False,
        "nested gate changed N30 authorization",
    )
    require(
        close(
            nested.get("parameter_metric_fraction"),
            args.expected_parameter_fraction,
        ),
        "declared parameter metric fraction mismatch",
    )

    records = nested.get("records", [])
    require(len(records) == 2, "metric gate must contain exactly N8 and N16")
    summaries: list[dict[str, object]] = []
    for record, nodes in zip(records, (8, 16), strict=True):
        replay_path = args.gate_dir / "GATE" / f"N{nodes}" / "balanced_metric_replay.json"
        state_path = args.gate_dir / "GATE" / f"N{nodes}" / "balanced_metric_state.npz"
        for path in (replay_path, state_path):
            require(path.is_file(), f"N{nodes} artifact missing: {path.name}")
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        require(replay == record, f"N{nodes} replay disagrees with the nested gate record")
        require(
            replay.get("status") == f"R26_N{nodes}_BALANCED_METRIC_REPLAY_PASS",
            f"N{nodes} replay status is not PASS",
        )
        require(int(replay.get("nodes", -1)) == nodes, f"N{nodes} node count mismatch")
        root_record = Path(str(root.get(f"n{nodes}_record", "")))
        require(
            root_record.parts[-3:]
            == ("GATE", f"N{nodes}", "balanced_metric_replay.json"),
            f"N{nodes} root record path mismatch",
        )

        metric = replay.get("metric", {})
        parameter_fraction = float(metric.get("parameter_fraction", -1.0))
        state_fraction = float(metric.get("state_fraction", -1.0))
        require(
            close(parameter_fraction, args.expected_parameter_fraction),
            f"N{nodes} parameter fraction mismatch",
        )
        require(
            close(state_fraction, 1.0 - args.expected_parameter_fraction),
            f"N{nodes} state fraction mismatch",
        )
        require(
            close(parameter_fraction + state_fraction, 1.0),
            f"N{nodes} metric fractions do not sum to one",
        )
        require(float(metric.get("parameter_scale", 0.0)) > 0.0, f"N{nodes} scale invalid")
        raw_gate = float(replay.get("raw_acceptance_gate", math.inf))
        independent_raw = float(replay.get("independent_raw_gate", math.inf))
        require(raw_gate <= args.raw_tolerance, f"N{nodes} raw gate failed")
        require(
            independent_raw <= args.raw_tolerance,
            f"N{nodes} independent raw gate failed",
        )
        solver = replay.get("solver", {})
        jacobians = int(solver.get("jacobian_evaluations", -1))
        require(1 <= jacobians <= 7, f"N{nodes} Jacobian count invalid")
        require(
            int(solver.get("pseudo_transient_steps", -1)) >= 1,
            f"N{nodes} did not exercise SER-PTC",
        )
        require(
            solver.get("message") == "pseudo-arclength raw physical gate reached",
            f"N{nodes} solver did not reach the raw physical gate",
        )

        with np.load(state_path, allow_pickle=False) as archive:
            state = np.asarray(archive["state"], dtype=float)
            lid = float(np.asarray(archive["lid_velocity"]).item())
            accepted = bool(np.asarray(archive["accepted"]).item())
            kn_input = float(np.asarray(archive["kn_input"]).item())
            beta = float(np.asarray(archive["beta"]).item())
        require(state.shape == (nodes, nodes, 17), f"N{nodes} state shape mismatch")
        require(np.isfinite(state).all(), f"N{nodes} state is nonfinite")
        require(accepted, f"N{nodes} state is explicitly rejected")
        require(close(lid, float(replay["corrected_parameter"])), f"N{nodes} lid mismatch")
        require(close(kn_input, 0.20), f"N{nodes} Kn mismatch")
        require(close(beta, 0.0), f"N{nodes} beta mismatch")
        require(
            state_sha256(state) == replay.get("state_sha256"),
            f"N{nodes} state checksum mismatch",
        )
        summaries.append(
            {
                "nodes": nodes,
                "raw_gate": raw_gate,
                "parameter_fraction": parameter_fraction,
                "jacobians": jacobians,
            }
        )

    print(
        json.dumps(
            {
                "status": "R26_BALANCED_METRIC_GATE_VALIDATION_PASS",
                "source_commit": args.expected_source_commit,
                "n30_authorized": False,
                "summaries": summaries,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
