#!/usr/bin/env python3
"""Standalone fail-closed validator for the N8/N16 reconstruction cross-gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np

from r26_cases import gu_asme2009_cavity_case
from r26_gu_emerson_algorithm import GU_EMERSON_STAGE_ORDER
from r26_gu_emerson_reconstruction import (
    GuEmersonReconstructionOptions,
    make_gu_emerson_reconstruction_problem,
)
from r26_thor_audit import state_sha256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args()
    require(
        re.fullmatch(r"[0-9a-f]{40}", args.expected_source_commit) is not None,
        "invalid expected source commit",
    )
    gate_path = args.run_dir / "GU_EMERSON_RECONSTRUCTION_CROSS_GATE.json"
    gate = json.loads(gate_path.read_text())
    require(
        gate.get("status") == "R26_GU_EMERSON_RECONSTRUCTION_CROSS_GATE_PASSED",
        "cross-gate status is not PASSED",
    )
    require(gate.get("source_commit") == args.expected_source_commit, "source commit mismatch")
    require(gate.get("cross_solver_fixed_point_gate_passed") is True, "cross-gate boolean is not true")
    require(gate.get("controls_fully_declared") is True, "controls are not fully declared")
    require(gate.get("standalone_from_equilibrium_passed") is False, "standalone status must remain false")
    require(gate.get("standalone_from_equilibrium_attempted") is False, "standalone attempt must remain false")
    for key in ("production_accepted", "n24_authorized", "n28_authorized", "n29_authorized", "n30_authorized"):
        require(gate.get(key) is False, f"{key} must remain false")

    options = GuEmersonReconstructionOptions(max_outer_iterations=1)
    summaries: list[dict[str, object]] = []
    for nodes in (8, 16):
        record = json.loads(
            (args.run_dir / f"N{nodes}" / "gu_emerson_reconstruction.json").read_text()
        )
        require(record.get("passed") is True, f"N{nodes} record did not pass")
        require(
            tuple(record.get("executed_stage_order", ())) == GU_EMERSON_STAGE_ORDER,
            f"N{nodes} stage order mismatch",
        )
        path = args.run_dir / f"N{nodes}" / "gu_emerson_reconstruction_state.npz"
        with np.load(path, allow_pickle=False) as archive:
            require(bool(np.asarray(archive["accepted"]).item()), f"N{nodes} state rejected")
            require(not bool(np.asarray(archive["production_accepted"]).item()), f"N{nodes} production flag true")
            require(not bool(np.asarray(archive["standalone_from_equilibrium"]).item()), f"N{nodes} standalone flag true")
            state = np.asarray(archive["state"], dtype=float)
            require(float(np.asarray(archive["lid_speed_m_s"]).item()) == 10.0, f"N{nodes} lid mismatch")
            require(float(np.asarray(archive["kn_input"]).item()) == 0.1, f"N{nodes} Kn mismatch")
        require(state_sha256(state) == record.get("reconstruction_state_sha256"), f"N{nodes} state hash mismatch")
        case = gu_asme2009_cavity_case(nodes, kn=0.1, lid_speed_m_per_s=10.0)
        diagnostics = make_gu_emerson_reconstruction_problem(case).evaluate(state).diagnostics
        raw = max(
            diagnostics.raw_total_linf,
            abs(diagnostics.held_out_continuity),
            abs(diagnostics.mass_error),
        )
        require(raw <= options.raw_tolerance, f"N{nodes} independent raw gate failed: {raw:.3e}")
        require(diagnostics.total_linf <= options.scaled_tolerance, f"N{nodes} scaled gate failed")
        require(abs(diagnostics.held_out_continuity) <= options.held_continuity_tolerance, f"N{nodes} held continuity failed")
        require(abs(diagnostics.mass_error) <= options.mass_tolerance, f"N{nodes} mass failed")
        require(diagnostics.min_density > 0.0 and diagnostics.min_temperature > 0.0, f"N{nodes} positivity failed")
        summaries.append({"nodes": nodes, "raw_gate": raw, "diagnostics": asdict(diagnostics)})

    validation = {
        "status": "R26_GU_EMERSON_RECONSTRUCTION_CROSS_GATE_VALIDATION_PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.expected_source_commit,
        "production_accepted": False,
        "n24_authorized": False,
        "n28_authorized": False,
        "n29_authorized": False,
        "n30_authorized": False,
        "summaries": summaries,
    }
    (args.run_dir / "GU_EMERSON_RECONSTRUCTION_VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
