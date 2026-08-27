#!/usr/bin/env python3
"""Compare accepted THOR-style R26 candidates on one common grid.

The report is a grid-sensitivity diagnostic, not a formal asymptotic
convergence claim.  Density and temperature errors are normalized by their
departures from equilibrium; every other field uses its fine-grid RMS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from r26_postprocess import COMMON_FIELDS, interpolate_state, rana_global_metrics


def load(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        if "accepted" not in archive or not bool(np.asarray(archive["accepted"]).item()):
            raise ValueError(f"candidate is not accepted: {path}")
        state = np.asarray(archive["state"], dtype=float)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        lid = float(np.asarray(archive["lid_velocity"]).item())
        kn = float(np.asarray(archive["kn_input"]).item())
    return {"path": path, "state": state, "x": x, "y": y, "lid": lid, "kn": kn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--target-n", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidates) < 2:
        parser.error("at least two candidates are required")
    candidates = [load(path) for path in args.candidates]
    for previous, current in zip(candidates, candidates[1:]):
        if current["kn"] != previous["kn"] or current["lid"] != previous["lid"]:
            parser.error("all candidates must have identical Kn and lid velocity")

    common = [
        interpolate_state(
            item["state"],
            x=item["x"],
            y=item["y"],
            target_n=args.target_n,
        )
        for item in candidates
    ]
    pairs: list[dict[str, object]] = []
    for coarse, fine, coarse_fields, fine_fields in zip(
        candidates, candidates[1:], common, common[1:]
    ):
        fields: list[dict[str, float | str]] = []
        for name in COMMON_FIELDS:
            first = np.asarray(coarse_fields[name], dtype=float)
            second = np.asarray(fine_fields[name], dtype=float)
            signal = second - 1.0 if name in {"rho", "theta"} else second
            difference = first - second
            rms = float(np.sqrt(np.mean(difference * difference)))
            signal_rms = float(np.sqrt(np.mean(signal * signal)))
            fields.append(
                {
                    "field": name,
                    "rms_difference": rms,
                    "fine_signal_rms": signal_rms,
                    "normalized_rms_difference": rms / max(signal_rms, np.finfo(float).tiny),
                    "max_abs_difference": float(np.max(np.abs(difference))),
                }
            )
        coarse_metrics = rana_global_metrics(
            coarse["state"],
            lid_velocity=coarse["lid"],
            x=coarse["x"],
            y=coarse["y"],
        )
        fine_metrics = rana_global_metrics(
            fine["state"],
            lid_velocity=fine["lid"],
            x=fine["x"],
            y=fine["y"],
        )
        pairs.append(
            {
                "coarse_nodes": int(np.asarray(coarse["state"]).shape[0]),
                "fine_nodes": int(np.asarray(fine["state"]).shape[0]),
                "fields": fields,
                "maximum_normalized_rms_difference": max(
                    float(row["normalized_rms_difference"]) for row in fields
                ),
                "D_relative_change": abs(float(coarse_metrics["D"]) - float(fine_metrics["D"]))
                / max(abs(float(fine_metrics["D"])), np.finfo(float).tiny),
                "G_relative_change": abs(float(coarse_metrics["G"]) - float(fine_metrics["G"]))
                / max(abs(float(fine_metrics["G"])), np.finfo(float).tiny),
            }
        )
    report = {
        "status": "R26_THOR_GRID_SENSITIVITY_REPORTED",
        "target_common_grid": args.target_n,
        "pairs": pairs,
        "formal_asymptotic_grid_convergence_claim": False,
        "production_accepted": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
