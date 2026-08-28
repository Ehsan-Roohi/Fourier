#!/usr/bin/env python3
"""Independent fail-closed validator for the N8/N16 standalone ladder."""

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


PATHS = (
    (8, "N8_FROM_EQUILIBRIUM"),
    (8, "N8_FROM_PERTURBED"),
    (16, "N16_FROM_N8_EQUILIBRIUM"),
    (16, "N16_FROM_N8_PERTURBED"),
)


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
    final_path = args.run_dir / "GU_EMERSON_STANDALONE_LADDER_PASSED.json"
    final = json.loads(final_path.read_text())
    require(final.get("status") == "R26_GU_EMERSON_STANDALONE_LADDER_PASSED", "ladder status is not PASSED")
    require(final.get("source_commit") == args.expected_source_commit, "source commit mismatch")
    require(final.get("standalone_from_equilibrium_attempted") is True, "equilibrium attempt boolean is not true")
    require(final.get("standalone_from_equilibrium_passed") is True, "equilibrium pass boolean is not true")
    require(final.get("independent_start_consistency_passed") is True, "independent-start gate is not true")
    require(final.get("production_accepted") is False, "production flag must remain false")
    require(final.get("n24_authorized") is True, "N24 must be authorized after a passed ladder")
    for key in ("n28_authorized", "n29_authorized", "n30_authorized"):
        require(final.get(key) is False, f"{key} must remain false")
    require(final.get("failure_stage") is None, "passed ladder has a failure stage")
    require(len(final.get("completed_stages", ())) == 6, "all six ordered stages were not completed")
    for nodes in (8, 16):
        comparison = json.loads((args.run_dir / f"N{nodes}_ROOT_COMPARISON.json").read_text())
        require(comparison.get("passed") is True, f"N{nodes} root comparison failed")
        require(
            float(comparison.get("maximum_absolute_state_difference"))
            <= float(comparison.get("required_maximum_absolute_state_difference")),
            f"N{nodes} root difference exceeds its declared bound",
        )

    options = GuEmersonReconstructionOptions(max_outer_iterations=480)
    summaries: list[dict[str, object]] = []
    for nodes, name in PATHS:
        record = json.loads((args.run_dir / name / "standalone_path.json").read_text())
        require(record.get("status") == "R26_GU_EMERSON_STANDALONE_PATH_PASSED", f"{name} status failed")
        require(record.get("accepted") is True, f"{name} accepted boolean is not true")
        require(record.get("production_accepted") is False, f"{name} production flag is true")
        archive_path = args.run_dir / name / "standalone_state.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            require(bool(np.asarray(archive["accepted"]).item()), f"{name} state is rejected")
            require(not bool(np.asarray(archive["production_accepted"]).item()), f"{name} state is production-marked")
            require(int(np.asarray(archive["nodes"]).item()) == nodes, f"{name} node count mismatch")
            require(float(np.asarray(archive["lid_speed_m_s"]).item()) == 10.0, f"{name} lid speed mismatch")
            require(float(np.asarray(archive["kn_input"]).item()) == 0.1, f"{name} Kn mismatch")
            state = np.asarray(archive["state"], dtype=float)
        require(state_sha256(state) == record.get("state_sha256"), f"{name} state hash mismatch")
        case = gu_asme2009_cavity_case(
            nodes, kn=0.1, lid_speed_m_per_s=10.0,
            wall_temperature_K=273.0, grid_stretch_beta=0.0,
        )
        diagnostics = make_gu_emerson_reconstruction_problem(case).evaluate(state).diagnostics
        raw = float(max(diagnostics.raw_total_linf, abs(diagnostics.held_out_continuity), abs(diagnostics.mass_error)))
        require(raw <= options.raw_tolerance, f"{name} independent raw gate failed: {raw:.3e}")
        require(diagnostics.total_linf <= options.scaled_tolerance, f"{name} scaled gate failed")
        require(abs(diagnostics.held_out_continuity) <= options.held_continuity_tolerance, f"{name} held continuity failed")
        require(abs(diagnostics.mass_error) <= options.mass_tolerance, f"{name} mass gate failed")
        require(diagnostics.min_density > 0.0 and diagnostics.min_temperature > 0.0, f"{name} positivity failed")
        history = record.get("history", ())
        require(bool(history), f"{name} history is empty")
        require(tuple(history[-1].get("stage_order", ())) == GU_EMERSON_STAGE_ORDER, f"{name} stage order mismatch")
        summaries.append({"nodes": nodes, "path_name": name, "raw_gate": raw, "diagnostics": asdict(diagnostics)})

    validation = {
        "status": "R26_GU_EMERSON_STANDALONE_LADDER_VALIDATION_PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.expected_source_commit,
        "summaries": summaries,
        "standalone_from_equilibrium_passed": True,
        "independent_start_consistency_passed": True,
        "production_accepted": False,
        "n24_authorized": True,
        "n28_authorized": False,
        "n29_authorized": False,
        "n30_authorized": False,
    }
    (args.run_dir / "GU_EMERSON_STANDALONE_LADDER_VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
