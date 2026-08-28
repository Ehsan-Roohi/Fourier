#!/usr/bin/env python3
"""Independent fail-closed validator for the source-linearized N16 resume."""

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
    RANA_SOURCE_HISTORY_RELAXATION,
    make_gu_emerson_reconstruction_problem,
)
from r26_thor_audit import state_sha256


PATHS = (
    "N16_SOURCE_LINEARIZED_FROM_N8_EQUILIBRIUM",
    "N16_SOURCE_LINEARIZED_FROM_N8_PERTURBED",
)
EXPECTED_SOURCE_MODULE_SHA256 = (
    "d92e0142776d90499e2beea4a8b3b37b590597f66b61f43bb49f58ade73a884b"
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
    final = json.loads(
        (args.run_dir / "GU_EMERSON_SOURCE_LINEARIZED_N16_RESUME_PASSED.json").read_text()
    )
    require(
        final.get("status") == "R26_GU_EMERSON_SOURCE_LINEARIZED_N16_RESUME_PASSED",
        "resume status is not PASSED",
    )
    require(final.get("source_commit") == args.expected_source_commit, "source commit mismatch")
    require(final.get("source_module_sha256") == EXPECTED_SOURCE_MODULE_SHA256, "source module hash mismatch")
    require(final.get("source_history_relaxation") == RANA_SOURCE_HISTORY_RELAXATION, "source factors mismatch")
    require(final.get("matrix_refresh_interval") == 1, "matrix was not reassembled every sweep")
    require(final.get("reused_failed_n16_state") is False, "failed N16 state was reused")
    require(len(final.get("reused_n8_states", ())) == 2, "two accepted N8 states were not reused")
    require(final.get("standalone_from_equilibrium_passed") is True, "standalone gate is false")
    require(final.get("independent_start_consistency_passed") is True, "independent gate is false")
    require(final.get("production_accepted") is False, "production flag must remain false")
    require(final.get("n24_authorized") is True, "N24 is not authorized")
    for key in ("n28_authorized", "n29_authorized", "n30_authorized"):
        require(final.get(key) is False, f"{key} must remain false")
    comparison = json.loads(
        (args.run_dir / "N16_SOURCE_LINEARIZED_ROOT_COMPARISON.json").read_text()
    )
    require(comparison.get("passed") is True, "N16 root comparison failed")

    options = GuEmersonReconstructionOptions(
        max_outer_iterations=720,
        matrix_refresh_interval=1,
    )
    case = gu_asme2009_cavity_case(
        16, kn=0.1, lid_speed_m_per_s=10.0,
        wall_temperature_K=273.0, grid_stretch_beta=0.0,
    )
    problem = make_gu_emerson_reconstruction_problem(case)
    summaries: list[dict[str, object]] = []
    hashes: list[str] = []
    for name in PATHS:
        record = json.loads((args.run_dir / name / "standalone_path.json").read_text())
        require(record.get("status") == "R26_GU_EMERSON_STANDALONE_PATH_PASSED", f"{name} status failed")
        require(record.get("accepted") is True, f"{name} accepted flag is false")
        require(record.get("production_accepted") is False, f"{name} is production-marked")
        history = record.get("history", ())
        require(bool(history), f"{name} history is empty")
        require(tuple(history[-1].get("stage_order", ())) == GU_EMERSON_STAGE_ORDER, f"{name} stage order mismatch")
        with np.load(args.run_dir / name / "standalone_state.npz", allow_pickle=False) as archive:
            require(bool(np.asarray(archive["accepted"]).item()), f"{name} archive is rejected")
            require(not bool(np.asarray(archive["production_accepted"]).item()), f"{name} archive is production-marked")
            require(int(np.asarray(archive["nodes"]).item()) == 16, f"{name} nodes mismatch")
            state = np.asarray(archive["state"], dtype=float)
        digest = state_sha256(state)
        require(digest == record.get("state_sha256"), f"{name} state hash mismatch")
        hashes.append(digest)
        diagnostics = problem.evaluate(state).diagnostics
        raw = float(
            max(
                diagnostics.raw_total_linf,
                abs(diagnostics.held_out_continuity),
                abs(diagnostics.mass_error),
            )
        )
        require(raw <= options.raw_tolerance, f"{name} raw gate failed: {raw:.3e}")
        require(diagnostics.total_linf <= options.scaled_tolerance, f"{name} scaled gate failed")
        require(abs(diagnostics.held_out_continuity) <= options.held_continuity_tolerance, f"{name} continuity failed")
        require(abs(diagnostics.mass_error) <= options.mass_tolerance, f"{name} mass gate failed")
        require(diagnostics.min_density > 0.0 and diagnostics.min_temperature > 0.0, f"{name} positivity failed")
        summaries.append({"path_name": name, "state_sha256": digest, "raw_gate": raw, "diagnostics": asdict(diagnostics)})
    require(hashes[0] != "" and hashes[1] != "", "N16 state hashes are empty")
    validation = {
        "status": "R26_GU_EMERSON_SOURCE_LINEARIZED_N16_VALIDATION_PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.expected_source_commit,
        "source_module_sha256": EXPECTED_SOURCE_MODULE_SHA256,
        "summaries": summaries,
        "production_accepted": False,
        "n24_authorized": True,
        "n28_authorized": False,
        "n29_authorized": False,
        "n30_authorized": False,
    }
    (args.run_dir / "GU_EMERSON_SOURCE_LINEARIZED_N16_VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
