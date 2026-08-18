#!/usr/bin/env python3
"""Compare accepted R26 grid states on the finest supplied node grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


GROUPS = {
    "rho_perturbation": (0,),
    "velocity": (1, 2),
    "temperature_perturbation": (3,),
    "heat_flux_q": (4, 5),
    "stress_sigma": (6, 7, 8),
    "R_tensor": (9, 10, 11),
    "m_tensor": (12, 13, 14, 15),
    "Delta": (16,),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"R26_GRID_COMPARISON_FAILED: {message}")


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(path.is_file(), f"missing state {path}")
    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive["state"], dtype=float)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
    require(state.shape == (len(y), len(x), 17), f"invalid state shape in {path}")
    require(np.isfinite(state).all(), f"non-finite state in {path}")
    return state, x, y


def interpolate(
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    yy, xx = np.meshgrid(target_y, target_x, indexing="ij")
    points = np.column_stack((yy.ravel(), xx.ravel()))
    result = np.empty((len(target_y), len(target_x), state.shape[-1]))
    for component in range(state.shape[-1]):
        result[..., component] = RegularGridInterpolator(
            (y, x), state[..., component], bounds_error=True
        )(points).reshape(len(target_y), len(target_x))
    return result


def relative_l2(difference: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference.ravel())), 1.0e-300)
    return float(np.linalg.norm(difference.ravel()) / denominator)


def grouped_change(coarse: np.ndarray, fine: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, components in GROUPS.items():
        coarse_group = coarse[..., components].copy()
        fine_group = fine[..., components].copy()
        if name == "rho_perturbation":
            coarse_group -= 1.0
            fine_group -= 1.0
        elif name == "temperature_perturbation":
            coarse_group -= 1.0
            fine_group -= 1.0
        result[name] = relative_l2(coarse_group - fine_group, fine_group)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(len(args.state) == len(args.label) >= 2, "state and label counts must match")

    loaded = [load(path.resolve()) for path in args.state]
    target_index = int(np.argmax([item[0].shape[0] for item in loaded]))
    target_x, target_y = loaded[target_index][1], loaded[target_index][2]
    common = [interpolate(state, x, y, target_x, target_y) for state, x, y in loaded]

    comparisons: dict[str, dict[str, float]] = {}
    for index in range(len(common) - 1):
        key = f"{args.label[index]}_to_{args.label[index + 1]}"
        comparisons[key] = grouped_change(common[index], common[index + 1])
    comparisons[f"{args.label[0]}_to_{args.label[-1]}"] = grouped_change(
        common[0], common[-1]
    )
    report = {
        "status": "R26_GRID_COMPARISON_COMPLETE",
        "comparison_grid_nodes": int(len(target_x)),
        "labels": args.label,
        "states": [str(path.resolve()) for path in args.state],
        "relative_l2": comparisons,
        "interpretation": (
            "sensitivity study on accepted algebraic roots; not a claim of asymptotic "
            "grid convergence unless the reported sequence is demonstrably monotone"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
