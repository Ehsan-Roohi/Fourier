#!/usr/bin/env python3
"""Aggregate the blocked Maxwell-VSS KnGu=0.20 DSMC campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter


FIELDS = ["nrho", "u", "v", "w", "T", "qx", "qy", "Pxx", "Pxy",
          "Pyy", "Pzz", "B1xx", "B1xy", "B1yy", "B1zz"]
SEEDS = (104729, 130363, 155921, 196613, 242081, 32452843, 49979687, 67867967)
DESIGNS = {
    "primary_N160_P256": ("primary_N160_P256", 160, 256, 8),
    "grid_coarse_N120_P256": ("grid_coarse_N120_P256", 120, 256, 4),
    "grid_fine_N200_P256": ("grid_fine_N200_P256", 200, 256, 4),
    "ppc_low_N160_P128": ("ppc_low_N160_P128", 160, 128, 4),
    "ppc_high_N160_P512": ("ppc_high_N160_P512", 160, 512, 4),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"MAXWELL_ENSEMBLE_ANALYSIS_FAILED: {message}")


def dump_values(path: Path, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, skiprows=9)
    require(data.shape == (nx * nx, 18), f"unexpected shape for {path}")
    data = data[np.lexsort((data[:, 1], data[:, 2]))]
    x = np.unique(data[:, 1]); y = np.unique(data[:, 2])
    require(len(x) == nx and len(y) == nx, f"coordinate count mismatch for {path}")
    return data[:, 3:].reshape(nx, nx, 15), x, y


def nondimensional(state: np.ndarray, meta: dict[str, object]) -> np.ndarray:
    kb = 1.380649e-23
    mass = float(meta["argon_mass_kg"])
    tw = float(meta["wall_temperature_K"])
    n0 = float(meta["number_density_m-3"])
    rho0 = n0 * mass
    c0 = math.sqrt(kb * tw / mass)
    out = state.copy()
    out[..., 0] /= n0
    out[..., 1:4] /= c0
    out[..., 4] /= tw
    out[..., 5:7] /= rho0 * c0**3
    out[..., 7:11] /= rho0 * c0**2
    out[..., 11:15] /= c0**4
    return out


def rel(diff: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(reference.ravel()), 1e-300))


def observable(state: np.ndarray, name: str) -> np.ndarray:
    if name == "rho_prime": return state[..., 0] - 1.0
    if name == "temperature_prime": return state[..., 4] - 1.0
    if name == "velocity": return state[..., 1:3]
    if name == "heat_flux": return state[..., 5:7]
    if name == "stress": return state[..., 7:11]
    if name == "sonine_B1": return state[..., 11:15]
    raise KeyError(name)


def uncertainty_component(state: np.ndarray, name: str) -> np.ndarray:
    if name == "rho_prime": return state[..., 0]
    if name == "temperature_prime": return state[..., 4]
    return observable(state, name)


def interp(state: np.ndarray, sx: np.ndarray, sy: np.ndarray,
           tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
    yy, xx = np.meshgrid(ty, tx, indexing="ij")
    points = np.column_stack((yy.ravel(), xx.ravel()))
    out = np.empty((len(ty), len(tx), state.shape[-1]))
    for k in range(state.shape[-1]):
        out[..., k] = RegularGridInterpolator((sy, sx), state[..., k],
                                               bounds_error=True)(points).reshape(len(ty), len(tx))
    return out


def af_mask(state: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = uniform_filter(state[..., 4], 15, mode="nearest")
    qx = uniform_filter(state[..., 5], 15, mode="nearest")
    qy = uniform_filter(state[..., 6], 15, mode="nearest")
    dtdy, dtdx = np.gradient(t, y, x, edge_order=2)
    xx, yy = np.meshgrid(x, y)
    eligible = ~(((xx < .05) | (xx > .95)) & (yy > .95))
    qm = np.hypot(qx, qy); gm = np.hypot(dtdx, dtdy)
    active = eligible & (qm > .05 * qm[eligible].max()) & (gm > .05 * gm[eligible].max())
    return active, active & ((qx * dtdx + qy * dtdy) > 0.0)


def load_run(path: Path, nx: int) -> dict[str, object]:
    meta = json.loads((path / "case_metadata.json").read_text(encoding="utf-8"))
    require(json.loads((path / "validation_report.json").read_text(encoding="utf-8"))["status"] ==
            "MAXWELL_ENSEMBLE_VALIDATION_PASS", f"validator did not pass: {path}")
    final, x, y = dump_values(next(path.glob("grid.final.*")), nx)
    blocks = [dump_values(p, nx)[0] for p in sorted(path.glob("grid.block.*"))]
    require(len(blocks) == 10, f"expected ten blocks: {path}")
    return {"path": path, "meta": meta, "x": x, "y": y,
            "final": nondimensional(final, meta),
            "blocks": np.stack([nondimensional(b, meta) for b in blocks])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    runs: dict[str, list[dict[str, object]]] = {}
    for key, (dirname, nx, ppc, count) in DESIGNS.items():
        group = []
        for seed in SEEDS[:count]:
            path = args.results / dirname / f"seed_{seed}"
            require(path.is_dir(), f"missing run {path}")
            group.append(load_run(path, nx))
        runs[key] = group

    primary = runs["primary_N160_P256"]
    stack = np.stack([r["final"] for r in primary])
    mean = stack.mean(axis=0); sample_std = stack.std(axis=0, ddof=1)
    x = primary[0]["x"]; y = primary[0]["y"]
    observables = ("rho_prime", "temperature_prime", "velocity", "heat_flux", "stress", "sonine_B1")
    metrics: list[dict[str, object]] = []
    thresholds = {"rho_prime": .10, "temperature_prime": .10, "velocity": .03,
                  "heat_flux": .08, "stress": .10, "sonine_B1": .20}
    gate_failures: list[str] = []

    for name in observables:
        ref = observable(mean, name)
        se = uncertainty_component(sample_std / math.sqrt(len(primary)), name)
        se_rel = rel(se, ref)
        seed_errors = [rel(observable(s, name) - ref, ref) for s in stack]
        half_drifts = []
        for run in primary:
            blocks = run["blocks"]
            first = blocks[:5].mean(axis=0); second = blocks[5:].mean(axis=0)
            half_drifts.append(rel(observable(second, name) - observable(first, name), ref))
        metrics.append({"comparison": "primary_sampling", "observable": name,
                        "relative_standard_error": se_rel,
                        "seed_error_mean": float(np.mean(seed_errors)),
                        "seed_error_max": float(np.max(seed_errors)),
                        "split_half_drift_mean": float(np.mean(half_drifts)),
                        "split_half_drift_max": float(np.max(half_drifts)),
                        "predeclared_se_limit": thresholds[name]})
        if se_rel > thresholds[name]:
            gate_failures.append(f"{name} ensemble SE {se_rel:.4g} > {thresholds[name]:.4g}")

    sensitivity_limits = {"rho_prime": .20, "temperature_prime": .20, "velocity": .08,
                          "heat_flux": .20, "stress": .20, "sonine_B1": .35}
    for design, group in runs.items():
        if design == "primary_N160_P256": continue
        group_mean_native = np.stack([r["final"] for r in group]).mean(axis=0)
        group_mean = interp(group_mean_native, group[0]["x"], group[0]["y"], x, y)
        for name in observables:
            ref = observable(mean, name)
            difference = rel(observable(group_mean, name) - ref, ref)
            metrics.append({"comparison": design, "observable": name,
                            "relative_difference_from_primary": difference,
                            "predeclared_difference_limit": sensitivity_limits[name]})
            if difference > sensitivity_limits[name]:
                gate_failures.append(f"{design} {name} difference {difference:.4g} > {sensitivity_limits[name]:.4g}")

    active_mean, af_mean = af_mask(mean, x, y)
    af_rows = []
    for run in primary:
        active, mask = af_mask(run["final"], x, y)
        union = (mask | af_mean) & active_mean
        inter = mask & af_mean & active_mean
        af_rows.append({"seed": int(run["meta"]["seed"]),
                        "active_fraction": float(active.mean()),
                        "anti_fourier_fraction_on_mean_active": float((mask & active_mean).sum()/max(active_mean.sum(), 1)),
                        "jaccard_to_ensemble_mean": float(inter.sum()/max(union.sum(), 1))})

    with (args.output / "convergence_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = sorted({key for row in metrics for key in row})
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(metrics)
    with (args.output / "anti_fourier_seed_stability.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(af_rows[0])); writer.writeheader(); writer.writerows(af_rows)
    np.savez_compressed(args.output / "primary_ensemble_mean_and_se.npz", x=x, y=y,
                        fields=np.asarray(FIELDS), mean=mean,
                        standard_error=sample_std/math.sqrt(len(primary)))
    report = {
        "status": "PUBLICATION_GATE_PASS" if not gate_failures else "PUBLICATION_GATE_FAIL",
        "case": "Maxwell-VSS lid-driven cavity KnGu=0.20",
        "primary_independent_realisations": 8,
        "sensitivity_realisations": 16,
        "production_samples_per_cell_per_run": 20000,
        "production_steps_per_run": 2000000,
        "independent_blocks_per_run": 10,
        "anti_fourier_filter_cells": 15,
        "gate_failures": gate_failures,
        "metrics": metrics,
        "anti_fourier_seed_stability": af_rows,
    }
    (args.output / "ensemble_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gate_failure_count": len(gate_failures),
                      "output": str(args.output)}, indent=2))
    if args.fail_on_gate and gate_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
