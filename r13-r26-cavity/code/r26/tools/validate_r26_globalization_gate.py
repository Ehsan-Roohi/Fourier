#!/usr/bin/env python3
"""Validate the immutable N8/N16 SER-PTC gate before an N30 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


REQUIRED_ARTIFACTS = {
    "N8/last_accepted_state.npz",
    "N8/run_summary.json",
    "N16/last_accepted_state.npz",
    "N16/run_summary.json",
    "N16_GATE_PASSED.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"R26_GLOBALIZATION_GATE_VALIDATION_FAILED: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_relative_path(recorded_path: str) -> str:
    normalized = recorded_path.replace("\\", "/")
    for grid in ("N8", "N16"):
        marker = f"/{grid}/"
        if marker in normalized:
            return f"{grid}/{normalized.split(marker, 1)[1]}"
    if normalized.endswith("/N16_GATE_PASSED.json"):
        return "N16_GATE_PASSED.json"
    raise ValueError(f"unrecognized manifest path: {recorded_path}")


def validate_manifest(gate_dir: Path) -> dict[str, str]:
    manifest_path = gate_dir / "N8_N16_GATE.sha256"
    require(manifest_path.is_file(), "N8_N16_GATE.sha256 missing")
    observed: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, f"malformed checksum line {line_number}")
        expected, recorded_path = fields
        require(
            re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            f"invalid SHA-256 on line {line_number}",
        )
        try:
            relative = manifest_relative_path(recorded_path.strip())
        except ValueError as error:
            raise SystemExit(
                f"R26_GLOBALIZATION_GATE_VALIDATION_FAILED: {error}"
            ) from error
        require(relative not in observed, f"duplicate checksum entry for {relative}")
        artifact = gate_dir / relative
        require(artifact.is_file(), f"manifest artifact missing: {relative}")
        actual = sha256(artifact)
        require(actual == expected, f"checksum mismatch for {relative}")
        observed[relative] = actual
    require(set(observed) == REQUIRED_ARTIFACTS, "checksum manifest has wrong artifact set")
    return observed


def validate_summary(
    gate_dir: Path,
    nodes: int,
    raw_tolerance: float,
) -> dict[str, object]:
    summary_path = gate_dir / f"N{nodes}" / "run_summary.json"
    require(summary_path.is_file(), f"N{nodes} run_summary.json missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    case = summary.get("case", {})
    solver = summary.get("nonlinear_solver", {})
    attempts = summary.get("attempts", [])
    require(summary.get("termination") == "target_accepted", f"N{nodes} target not accepted")
    require(int(case.get("nodes", -1)) == nodes, f"N{nodes} grid declaration mismatch")
    require(case.get("family") == "jfm-maxwell", f"N{nodes} case family mismatch")
    require(case.get("molecular_model") == "maxwell_molecules", f"N{nodes} molecular-model lock missing")
    require(case.get("kn_convention") == "gu_lambda_over_L", f"N{nodes} Kn convention mismatch")
    require(math.isclose(float(case.get("kn_input")), 0.20, rel_tol=0.0, abs_tol=2e-15), f"N{nodes} Kn mismatch")
    require(math.isclose(float(case.get("beta")), 0.0, rel_tol=0.0, abs_tol=2e-15), f"N{nodes} beta mismatch")
    require(summary.get("input_provenance", {}).get("kind") == "analytic_equilibrium", f"N{nodes} did not start from equilibrium")
    require(bool(attempts), f"N{nodes} has no continuation attempts")
    require(all(bool(row.get("accepted")) for row in attempts), f"N{nodes} contains a rejected attempt")
    require(float(attempts[-1].get("raw_acceptance_gate")) <= raw_tolerance, f"N{nodes} raw gate failed")
    require(bool(solver.get("analytic_mass_jacobian")), f"N{nodes} analytic mass Jacobian disabled")
    require(bool(solver.get("secant_predictor")), f"N{nodes} secant predictor disabled")
    require(bool(solver.get("ser_pseudo_transient")), f"N{nodes} SER-PTC disabled")
    jacobian_cap = int(solver.get("max_jacobians_per_attempt", -1))
    require(1 <= jacobian_cap <= 5, f"N{nodes} Jacobian cap invalid")
    require(
        max(int(row.get("solver", {}).get("jacobian_evaluations", -1)) for row in attempts)
        <= jacobian_cap,
        f"N{nodes} exceeded its Jacobian cap",
    )
    target = float(case.get("lid_target"))
    accepted = float(case.get("lid_last_accepted"))
    require(math.isclose(accepted, target, rel_tol=0.0, abs_tol=2e-14), f"N{nodes} did not reach the target lid")
    return {
        "nodes": nodes,
        "attempts": len(attempts),
        "rejections": sum(not bool(row.get("accepted")) for row in attempts),
        "final_raw_gate": float(attempts[-1]["raw_acceptance_gate"]),
        "maximum_jacobians_per_attempt": max(
            int(row["solver"]["jacobian_evaluations"]) for row in attempts
        ),
        "state_file_sha256": sha256(gate_dir / f"N{nodes}" / "last_accepted_state.npz"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate_dir", type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--raw-tolerance", type=float, default=1.0e-8)
    args = parser.parse_args()

    require(args.gate_dir.is_dir(), "gate directory missing")
    require(
        re.fullmatch(r"[0-9a-f]{40}", args.expected_source_commit) is not None,
        "expected source commit must be a lowercase 40-character SHA",
    )
    require(args.raw_tolerance > 0.0, "raw tolerance must be positive")

    gate_record_path = args.gate_dir / "N16_GATE_PASSED.json"
    require(gate_record_path.is_file(), "N16_GATE_PASSED.json missing")
    gate_record = json.loads(gate_record_path.read_text(encoding="utf-8"))
    require(gate_record.get("status") == "R26_N16_GATE_PASSED", "gate status is not PASS")
    require(
        gate_record.get("source_commit") == args.expected_source_commit,
        "gate source commit mismatch",
    )
    require(gate_record.get("n30_authorized") is False, "historical gate semantics changed")

    artifacts = validate_manifest(args.gate_dir)
    summaries = [
        validate_summary(args.gate_dir, nodes, args.raw_tolerance)
        for nodes in (8, 16)
    ]
    print(
        json.dumps(
            {
                "status": "R26_GLOBALIZATION_GATE_VALIDATION_PASS",
                "source_commit": args.expected_source_commit,
                "artifacts": artifacts,
                "summaries": summaries,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
