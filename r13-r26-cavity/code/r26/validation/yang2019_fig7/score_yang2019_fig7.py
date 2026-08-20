#!/usr/bin/env python3
"""Score computed R26 centreline profiles against Yang et al. (2019), Fig. 7."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


INDEX = {"ux": 1, "uy": 2, "qx": 4, "qy": 5}
PANELS = (("a", "ux"), ("b", "uy"), ("c", "qx"), ("d", "qy"))


def load_reference(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[float, float]]] = {q: [] for _, q in PANELS}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[row["quantity"]].append(
                (float(row["coordinate_over_L0"]), float(row["published_R26_value"]))
            )
    result = {}
    for quantity, pairs in grouped.items():
        if len(pairs) < 15:
            raise SystemExit(f"reference profile {quantity} is incomplete")
        pairs.sort()
        result[quantity] = (np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs]))
    return result


def load_profile(path: Path, quantity: str, coordinate: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive["state"], dtype=float)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
    if state.shape != (len(y), len(x), 17) or not np.isfinite(state).all():
        raise SystemExit(f"invalid state archive: {path}")
    centred = np.clip(coordinate + 0.5, 0.0, 1.0)
    if quantity in ("ux", "qx"):
        points = np.column_stack((centred, np.full_like(centred, 0.5)))
    else:
        points = np.column_stack((np.full_like(centred, 0.5), centred))
    raw = RegularGridInterpolator((y, x), state[..., INDEX[quantity]], bounds_error=True)(points)
    # Yang et al. use sqrt(2 R T0) rather than sqrt(R T0).  A velocity moment
    # of rank k therefore changes by (1/sqrt(2))**k.
    return raw / (np.sqrt(2.0) if quantity.startswith("u") else 2.0 * np.sqrt(2.0))


def metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    difference = prediction - reference
    span = max(float(np.ptp(reference)), np.finfo(float).eps)
    return {
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(reference), np.finfo(float).eps)),
        "range_normalized_rmse": float(np.sqrt(np.mean(difference**2)) / span),
        "max_abs_error": float(np.max(np.abs(difference))),
        "correlation": float(np.corrcoef(reference, prediction)[0, 1]),
    }


def parse_state(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("state must be LABEL=/path/to/last_accepted_state.npz")
    label, path = spec.split("=", 1)
    return label, Path(path)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", action="append", type=parse_state, required=True)
    parser.add_argument("--reference", type=Path, default=here / "reference/yang2019_fig7_r26_vector.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--velocity-relative-l2-limit", type=float, default=0.15)
    parser.add_argument("--heat-flux-relative-l2-limit", type=float, default=0.30)
    parser.add_argument("--velocity-grid-change-limit", type=float, default=0.10)
    parser.add_argument("--heat-flux-grid-change-limit", type=float, default=0.15)
    args = parser.parse_args()
    if len(args.state) < 2:
        parser.error("at least two accepted grids are required")
    for _, path in args.state:
        if not path.is_file():
            parser.error(f"state file missing: {path}")

    reference = load_reference(args.reference)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for label, path in args.state:
        predictions[label] = {}
        scores[label] = {}
        for quantity, (coordinate, target) in reference.items():
            prediction = load_profile(path, quantity, coordinate)
            predictions[label][quantity] = prediction
            scores[label][quantity] = metrics(target, prediction)

    fine_label = args.state[-1][0]
    previous_label = args.state[-2][0]
    grid_change = {}
    for quantity, (_, target) in reference.items():
        fine = predictions[fine_label][quantity]
        previous = predictions[previous_label][quantity]
        grid_change[quantity] = float(np.linalg.norm(fine - previous) / max(np.linalg.norm(fine), np.finfo(float).eps))

    gates = {}
    for quantity in INDEX:
        field_limit = args.velocity_relative_l2_limit if quantity.startswith("u") else args.heat_flux_relative_l2_limit
        grid_limit = args.velocity_grid_change_limit if quantity.startswith("u") else args.heat_flux_grid_change_limit
        gates[quantity] = {
            "published_profile_relative_l2": scores[fine_label][quantity]["relative_l2"],
            "published_profile_limit": field_limit,
            "published_profile_pass": scores[fine_label][quantity]["relative_l2"] <= field_limit,
            "last_two_grids_relative_l2": grid_change[quantity],
            "grid_change_limit": grid_limit,
            "grid_change_pass": grid_change[quantity] <= grid_limit,
        }
    overall = all(item["published_profile_pass"] and item["grid_change_pass"] for item in gates.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "YANG2019_FIG7_VALIDATION_PASS" if overall else "YANG2019_FIG7_VALIDATION_FAIL",
        "source": {
            "citation": "Yang, Tang & Yang, Applied Sciences 9 (2019) 2733",
            "doi": "10.3390/app9132733",
            "figure": "Figure 7",
            "reference_csv_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        },
        "case_lock": {
            "Kn_Gu": 0.1,
            "wall_temperature_K": 273.0,
            "lid_speed_m_per_s": 10.0,
            "walls": "fully diffuse",
            "closure": "final Gu--Emerson JFM-2009 R26 Maxwell-molecule equations",
        },
        "normalization": {
            "coordinates": "x/L0-0.5 or y/L0-0.5",
            "velocity": "u/sqrt(2 R T0)",
            "heat_flux": "q/[rho0 (sqrt(2 R T0))^3]",
        },
        "states": {label: str(path.resolve()) for label, path in args.state},
        "scores": scores,
        "grid_change": {f"{previous_label}_to_{fine_label}": grid_change},
        "gates": gates,
        "gate_policy": "limits declared in published code before the validation run; all four profiles and both profile/grid gates must pass",
    }
    (args.output_dir / "yang2019_fig7_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(args.state)))
    for axis, (panel, quantity) in zip(axes.flat, PANELS):
        coordinate, target = reference[quantity]
        axis.plot(coordinate, target, "ko", ms=4.5, label="Yang et al. R26 (digitised)")
        for color, (label, _) in zip(colors, args.state):
            axis.plot(coordinate, predictions[label][quantity], color=color, lw=2.0, label=f"present R26 {label}")
        axis.set_xlabel(("$y/L_0$" if quantity in ("ux", "qx") else "$x/L_0$") + " (centred)")
        axis.set_ylabel({"ux": r"$u_x/\sqrt{2RT_0}$", "uy": r"$u_y/\sqrt{2RT_0}$", "qx": r"$q_x/[\rho_0(\sqrt{2RT_0})^3]$", "qy": r"$q_y/[\rho_0(\sqrt{2RT_0})^3]$"}[quantity])
        axis.text(0.02, 0.96, f"({panel})", transform=axis.transAxes, ha="left", va="top", weight="bold", fontsize=13)
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.savefig(args.output_dir / "yang2019_fig7_validation.pdf", dpi=300)
    fig.savefig(args.output_dir / "yang2019_fig7_validation.png", dpi=220)
    plt.close(fig)

    print(json.dumps({"status": report["status"], "report": str(args.output_dir / "yang2019_fig7_validation.json")}, sort_keys=True))
    raise SystemExit(0 if overall else 2)


if __name__ == "__main__":
    main()
