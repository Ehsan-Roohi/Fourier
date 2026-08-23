#!/usr/bin/env python3
"""Private, fail-closed post-processing for planar Python R26 cavity states.

The input is a *wall-inclusive* ``(ny, nx, 17)`` state in the ordering declared
by :mod:`r26_state`.  This module contains no solver and does not infer a
corner convention: the supplied wall and corner nodes are treated as data.

Two comparisons are supported.

``Rana``
    Evaluate the audited Eq. (30) drag/flow-rate pair and compare the vertical
    centreline velocity with the digitised R13 curve from Rana et al.  That
    curve is an R13 reference only; agreement is not an R26 validation test.

``JFM``
    Interpolate to the manuscript's 160 x 160 cell-centred grid, apply the
    disclosed seven-point box smoothing, evaluate the anti-Fourier and
    fourth-order-channel diagnostics, and compare all common fields with the
    available diagnostic DSMC CSV.  The CSV is a legacy single-seed auxiliary
    dataset, not the authoritative eight-seed VHS production ensemble.  The
    manuscript scalar values are therefore reported separately and retain
    authority over the auxiliary field comparison.

All generated titles and reports say ``R26`` explicitly.  A converged algebraic
state is never labelled publication-grade by this module.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-private-r26")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter

from r26_state import NVAR, STATE_INDEX, validate_planar_state


TARGET_N = 160
SMOOTH_POINTS = 7
MASS_ARGON_KG = 6.6335e-26
BOLTZMANN = 1.380649e-23
DEFAULT_WALL_TEMPERATURE_K = 300.0
DEFAULT_LENGTH_M = 1.0e-6
DEFAULT_VELOCITY_SCALE_M_S = math.sqrt(
    BOLTZMANN * DEFAULT_WALL_TEMPERATURE_K / MASS_ARGON_KG
)

COMMON_FIELDS: tuple[str, ...] = (
    "rho",
    "vx",
    "vy",
    "theta",
    "qx",
    "qy",
    "sigma_xx",
    "sigma_xy",
    "sigma_yy",
    "Rxx",
    "Rxy",
    "Ryy",
    "Delta",
)

STATE_TO_FIELD: Mapping[str, str] = {
    "rho": "rho",
    "vx": "vx",
    "vy": "vy",
    "theta": "theta",
    "qx": "qx",
    "qy": "qy",
    "sigma_xx": "sigma_xx",
    "sigma_xy": "sigma_xy",
    "sigma_yy": "sigma_yy",
    "Rxx": "R_xx",
    "Rxy": "R_xy",
    "Ryy": "R_yy",
    "Delta": "Delta",
}

MANUSCRIPT_SCALARS: Mapping[str, Mapping[str, float | None]] = {
    "Kn005_U100": {
        "kn": 0.05,
        "lid_velocity_m_s": 100.0,
        "f_AF_active": 0.343,
        "mean_IAF_AF": 0.342,
        "PDelta_over_PR": 0.063,
        "mean_chiDelta": 0.097,
    },
    "Kn005_U200": {
        "kn": 0.05,
        "lid_velocity_m_s": 200.0,
        "f_AF_active": 0.049,
        "mean_IAF_AF": 0.358,
        "PDelta_over_PR": None,
        "mean_chiDelta": 0.056,
    },
    "Kn010_U100": {
        "kn": 0.10,
        "lid_velocity_m_s": 100.0,
        "f_AF_active": 0.685,
        "mean_IAF_AF": 0.300,
        "PDelta_over_PR": None,
        "mean_chiDelta": 0.085,
    },
}

METRIC_LABELS: Mapping[str, str] = {
    "f_AF_active": r"$f_{AF\mid active}$",
    "mean_IAF_AF": r"$\langle I_{AF}\rangle_{AF}$",
    "PDelta_over_PR": r"RMS$(P_\Delta)$/RMS$(P_R)$",
    "mean_chiDelta": r"$\langle\chi_\Delta\rangle_{AF}$",
}

DSMC_EXPECTED_COLUMNS: tuple[str, ...] = (
    "ix", "iy", "x", "y", "n", "ux", "uy", "uz", "theta", "T",
    "qxN", "qyN", "qx_phys", "qy_phys", "sigma_xxN", "sigma_xyN",
    "sigma_yyN", "sigma_zzN", "sigma_xx_phys", "sigma_xy_phys",
    "sigma_yy_phys", "sigma_zz_phys", "R_xxN", "R_xyN", "R_yyN",
    "R_zzN", "DeltaN", "A_xxN", "A_xyN", "A_yyN", "particle_count",
    "collision_count", "nsamples", "time", "sample_time", "FNUM", "n0",
    "Kn", "Uwall", "Twall",
)

DSMC_PROVENANCE_CAVEAT = (
    "The field-level DSMC CSV is a legacy single-seed auxiliary 200x200 "
    "dataset with geometric hard-sphere n0, an undocumented collision-kernel "
    "provenance, and known finite-particle fourth-moment bias. It is not the "
    "authoritative eight-seed 160x160 VHS production ensemble described by "
    "the manuscript. Field and line comparisons are diagnostic only; the "
    "manuscript scalar values are reported independently."
)

RANA_REFERENCE_CAVEAT = (
    "The digitised Rana curve and tabulated D/G values are published R13 "
    "results. The present R26 result is a predictive extension, so this is a "
    "cross-model comparison rather than an R26 validation target."
)


@dataclass(frozen=True)
class CaseMetadata:
    """Minimal dimensional and provenance metadata for one post-processing run."""

    case_name: str
    family: str
    kn: float
    lid_velocity: float
    wall_temperature_K: float = DEFAULT_WALL_TEMPERATURE_K
    velocity_scale_m_s: float = DEFAULT_VELOCITY_SCALE_M_S
    length_m: float = DEFAULT_LENGTH_M
    model: str = "private Python R26"
    converged: bool = False
    publication_grade: bool = False
    provenance: str = "private R26 state"

    def __post_init__(self) -> None:
        if self.family not in {"rana", "jfm"}:
            raise ValueError("family must be 'rana' or 'jfm'")
        if not self.case_name:
            raise ValueError("case_name cannot be empty")
        for name in ("kn", "wall_temperature_K", "velocity_scale_m_s", "length_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.lid_velocity) or self.lid_velocity == 0.0:
            raise ValueError("lid_velocity must be finite and nonzero")
        if self.publication_grade:
            raise ValueError(
                "post-processing cannot promote a private R26 state to publication-grade"
            )

    @property
    def lid_velocity_m_s(self) -> float:
        return float(self.lid_velocity * self.velocity_scale_m_s)


def _trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    function = getattr(np, "trapezoid", np.trapz)
    return float(function(values, coordinates))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            return None
        return result
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows supplied for {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_jsonable(list(rows)))


def validate_wall_inclusive_state(state: np.ndarray) -> np.ndarray:
    """Validate a square wall-inclusive state without inventing wall values."""

    array = validate_planar_state(state)
    if array.ndim != 3 or array.shape[-1] != NVAR:
        raise ValueError(f"expected a wall-inclusive (ny,nx,{NVAR}) state")
    if array.shape[0] < 5 or array.shape[1] < 5:
        raise ValueError("post-processing needs at least 5x5 wall-inclusive nodes")
    if array.shape[0] != array.shape[1]:
        raise ValueError("the present cavity post-processing requires a square N x N grid")
    return array


def uniform_wall_grid(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = validate_wall_inclusive_state(state)
    return (
        np.linspace(0.0, 1.0, array.shape[1]),
        np.linspace(0.0, 1.0, array.shape[0]),
    )


def interpolate_state(
    state: np.ndarray,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    target_n: int = TARGET_N,
) -> dict[str, np.ndarray]:
    """Interpolate a wall-inclusive state to a cell-centred square grid."""

    array = validate_wall_inclusive_state(state)
    if target_n < 8:
        raise ValueError("target_n must be at least 8")
    default_x, default_y = uniform_wall_grid(array)
    xv = default_x if x is None else np.asarray(x, dtype=float)
    yv = default_y if y is None else np.asarray(y, dtype=float)
    if xv.shape != (array.shape[1],) or yv.shape != (array.shape[0],):
        raise ValueError("x/y coordinates do not match state shape")
    if (
        not np.isfinite(xv).all()
        or not np.isfinite(yv).all()
        or np.any(np.diff(xv) <= 0.0)
        or np.any(np.diff(yv) <= 0.0)
    ):
        raise ValueError("x and y must be finite and strictly increasing")
    centers = (np.arange(target_n, dtype=float) + 0.5) / target_n
    X, Y = np.meshgrid(centers, centers)
    points = np.column_stack((Y.ravel(), X.ravel()))
    result: dict[str, np.ndarray] = {"centers": centers, "X": X, "Y": Y}
    for output_name, state_name in STATE_TO_FIELD.items():
        interpolator = RegularGridInterpolator(
            (yv, xv),
            array[..., STATE_INDEX[state_name]],
            method="linear",
            bounds_error=True,
        )
        result[output_name] = interpolator(points).reshape(target_n, target_n)
    if not all(np.isfinite(result[name]).all() for name in COMMON_FIELDS):
        raise FloatingPointError("non-finite value after R26 interpolation")
    return result


def smooth_common_fields(
    fields: Mapping[str, np.ndarray],
    *,
    points: int = SMOOTH_POINTS,
    mode: str = "nearest",
) -> dict[str, np.ndarray]:
    if points < 1 or points % 2 == 0:
        raise ValueError("smoothing points must be a positive odd number")
    shape = np.asarray(fields["X"]).shape
    result: dict[str, np.ndarray] = {}
    for name in COMMON_FIELDS:
        value = np.asarray(fields[name], dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid field {name}")
        result[name] = uniform_filter(value, size=points, mode=mode)
    return result


def convert_velocity_normalization(
    fields: Mapping[str, np.ndarray],
    *,
    from_velocity_scale_m_s: float,
    to_velocity_scale_m_s: float,
) -> dict[str, np.ndarray]:
    """Express nondimensional moments on another thermal-velocity basis.

    Density and temperature ratios are unchanged.  A rank-k velocity moment
    is multiplied by ``(from_scale/to_scale)**k``.  This prevents the small
    ``R=208`` versus exact argon ``k_B/m`` difference from being silently
    folded into a model/DSMC field error.
    """

    source = float(from_velocity_scale_m_s)
    target = float(to_velocity_scale_m_s)
    if not np.isfinite(source) or not np.isfinite(target) or source <= 0.0 or target <= 0.0:
        raise ValueError("velocity scales must be finite and positive")
    ratio = source / target
    result = {key: np.asarray(fields[key]).copy() for key in ("centers", "X", "Y")}
    powers = {
        "rho": 0,
        "theta": 0,
        "vx": 1,
        "vy": 1,
        "qx": 3,
        "qy": 3,
        "sigma_xx": 2,
        "sigma_xy": 2,
        "sigma_yy": 2,
        "Rxx": 4,
        "Rxy": 4,
        "Ryy": 4,
        "Delta": 4,
    }
    for name, power in powers.items():
        result[name] = np.asarray(fields[name], dtype=float) * ratio**power
    return result


def corner_eligible_mask(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    *,
    corners: str = "top",
) -> np.ndarray:
    """Mask square corner neighborhoods without changing the underlying state."""

    x = np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("X and Y shape mismatch")
    if not 0.0 <= epsilon < 0.5:
        raise ValueError("epsilon must lie in [0,0.5)")
    if epsilon == 0.0:
        return np.ones_like(x, dtype=bool)
    side = (x < epsilon) | (x > 1.0 - epsilon)
    if corners == "top":
        excluded = side & (y > 1.0 - epsilon)
    elif corners == "all":
        excluded = side & ((y < epsilon) | (y > 1.0 - epsilon))
    else:
        raise ValueError("corners must be 'top' or 'all'")
    return ~excluded


def analyze_anti_fourier(
    fields: Mapping[str, np.ndarray],
    *,
    smooth_points: int = SMOOTH_POINTS,
    smoothing_mode: str = "nearest",
    activity_fraction: float = 0.05,
    eligible_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate the disclosed JFM anti-Fourier/fourth-order diagnostics exactly."""

    centers = np.asarray(fields["centers"], dtype=float)
    if centers.ndim != 1 or centers.size < 8 or np.any(np.diff(centers) <= 0.0):
        raise ValueError("invalid target centers")
    smoothed = smooth_common_fields(fields, points=smooth_points, mode=smoothing_mode)
    theta = smoothed["theta"]
    qx, qy = smoothed["qx"], smoothed["qy"]
    dtheta_dy, dtheta_dx = np.gradient(theta, centers, centers, edge_order=2)
    qmag = np.hypot(qx, qy)
    grad_t_mag = np.hypot(dtheta_dx, dtheta_dy)
    denominator = qmag * grad_t_mag
    valid = denominator > 1.0e-14
    if eligible_mask is None:
        eligible = np.ones_like(valid, dtype=bool)
    else:
        eligible = np.asarray(eligible_mask, dtype=bool)
        if eligible.shape != valid.shape or not np.any(eligible):
            raise ValueError("eligible_mask is empty or has the wrong shape")
    activity_fraction = float(activity_fraction)
    if not np.isfinite(activity_fraction) or not 0.0 < activity_fraction < 1.0:
        raise ValueError("activity_fraction must lie strictly between zero and one")

    iaf = np.full_like(theta, np.nan)
    iaf[valid] = (
        qx[valid] * dtheta_dx[valid] + qy[valid] * dtheta_dy[valid]
    ) / denominator[valid]
    eligible_q = np.where(eligible, qmag, -np.inf)
    eligible_grad = np.where(eligible, grad_t_mag, -np.inf)
    q_index = np.unravel_index(int(np.argmax(eligible_q)), qmag.shape)
    grad_index = np.unravel_index(int(np.argmax(eligible_grad)), grad_t_mag.shape)
    q_max = float(qmag[q_index])
    grad_max = float(grad_t_mag[grad_index])
    if q_max <= 0.0 or grad_max <= 0.0:
        raise RuntimeError("anti-Fourier diagnostic requires nonzero q and grad(T)")
    q_threshold = activity_fraction * q_max
    grad_threshold = activity_fraction * grad_max
    active = (
        eligible
        & valid
        & (qmag >= q_threshold)
        & (grad_t_mag >= grad_threshold)
    )
    af = active & (iaf > 0.0)
    if not np.any(active) or not np.any(af):
        raise RuntimeError("active or anti-Fourier subset is empty")

    Rxx, Rxy, Ryy = smoothed["Rxx"], smoothed["Rxy"], smoothed["Ryy"]
    Delta = smoothed["Delta"]
    dRxx_dy, dRxx_dx = np.gradient(Rxx, centers, centers, edge_order=2)
    dRxy_dy, dRxy_dx = np.gradient(Rxy, centers, centers, edge_order=2)
    dRyy_dy, _ = np.gradient(Ryy, centers, centers, edge_order=2)
    dDelta_dy, dDelta_dx = np.gradient(Delta, centers, centers, edge_order=2)
    div_R_x = dRxx_dx + dRxy_dy
    div_R_y = dRxy_dx + dRyy_dy
    PR = np.full_like(theta, np.nan)
    PDelta = np.full_like(theta, np.nan)
    q_valid = qmag > 1.0e-14
    PR[q_valid] = (
        qx[q_valid] * div_R_x[q_valid] + qy[q_valid] * div_R_y[q_valid]
    ) / qmag[q_valid]
    PDelta[q_valid] = (
        qx[q_valid] * dDelta_dx[q_valid]
        + qy[q_valid] * dDelta_dy[q_valid]
    ) / (3.0 * qmag[q_valid])
    closure_mask = af & np.isfinite(PR) & np.isfinite(PDelta)
    if not np.any(closure_mask):
        raise RuntimeError("fourth-order closure subset is empty")
    rms_PR = float(np.sqrt(np.mean(PR[closure_mask] ** 2)))
    rms_PDelta = float(np.sqrt(np.mean(PDelta[closure_mask] ** 2)))
    if rms_PR <= np.finfo(float).tiny:
        raise RuntimeError("RMS(P_R) is numerically zero")
    chi = np.full_like(theta, np.nan)
    chi[closure_mask] = np.abs(PDelta[closure_mask]) / (
        np.abs(PR[closure_mask]) + np.abs(PDelta[closure_mask]) + 1.0e-30
    )
    domain_count = int(np.count_nonzero(eligible))
    active_count = int(np.count_nonzero(active))
    af_count = int(np.count_nonzero(af))
    metrics = {
        "f_active_domain": float(active_count / domain_count),
        "f_AF_domain": float(af_count / domain_count),
        "f_AF_active": float(np.count_nonzero(af) / np.count_nonzero(active)),
        "mean_IAF_AF": float(np.mean(iaf[af])),
        "PDelta_over_PR": float(rms_PDelta / rms_PR),
        "mean_chiDelta": float(np.mean(chi[closure_mask])),
    }
    return {
        "metrics": metrics,
        "smoothed": smoothed,
        "iaf": iaf,
        "active": active,
        "af": af,
        "PR": PR,
        "PDelta": PDelta,
        "chiDelta": chi,
        "qmag": qmag,
        "grad_t_mag": grad_t_mag,
        "eligible": eligible,
        "eligible_count": domain_count,
        "active_count": active_count,
        "af_count": af_count,
        "closure_count": int(np.count_nonzero(closure_mask)),
        "q_threshold": q_threshold,
        "grad_t_threshold": grad_threshold,
        "q_max": q_max,
        "grad_t_max": grad_max,
        "activity_fraction": activity_fraction,
        "rms_PR": rms_PR,
        "rms_PDelta": rms_PDelta,
        "smoothing": f"{smooth_points}x{smooth_points} centered uniform, {smoothing_mode} edge",
    }


def rana_global_metrics(
    state: np.ndarray,
    *,
    lid_velocity: float,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> dict[str, float | str]:
    """Audited Rana Eq. (30) D/G definition on wall-inclusive nodes."""

    array = validate_wall_inclusive_state(state)
    default_x, default_y = uniform_wall_grid(array)
    xv = default_x if x is None else np.asarray(x, dtype=float)
    yv = default_y if y is None else np.asarray(y, dtype=float)
    speed = abs(float(lid_velocity))
    if speed <= 0.0:
        raise ValueError("lid velocity must be nonzero")
    top_shear = array[-1, :, STATE_INDEX["sigma_xy"]]
    sigma_integral = _trapezoid(top_shear, xv)
    center_velocity = np.asarray(
        [np.interp(0.5, xv, row[:, STATE_INDEX["vx"]]) for row in array]
    )
    reduction = math.sqrt(2.0) / speed
    return {
        "D": abs(reduction * sigma_integral),
        "D_signed": reduction * sigma_integral,
        "D_sigma_over_p0_signed": sigma_integral,
        "D_reduced_stress_factor": reduction,
        "G": _trapezoid(np.abs(center_velocity), yv) / speed,
        "provenance": "Rana Eq. (30), wall-inclusive node-grid trapezoid",
    }


def load_rana_digitized_reference(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    required = {
        "y_over_L",
        "rana_fig3_R13_vx_over_Ulid",
        "digitized_half_line_uncertainty",
    }
    if data.dtype.names is None or not required.issubset(data.dtype.names):
        raise ValueError("unexpected Rana digitized CSV schema")
    result = {name: np.asarray(data[name], dtype=float) for name in required}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("non-finite Rana digitized reference")
    if np.any(np.diff(result["y_over_L"]) <= 0.0):
        raise ValueError("Rana digitized y coordinate is not increasing")
    return result


def _zero_crossing(x: np.ndarray, value: np.ndarray) -> float | None:
    indices = np.where(value[:-1] * value[1:] <= 0.0)[0]
    if indices.size == 0:
        return None
    # The cavity centreline's physically relevant crossing is the last one.
    i = int(indices[-1])
    if value[i + 1] == value[i]:
        return float(0.5 * (x[i] + x[i + 1]))
    return float(x[i] - value[i] * (x[i + 1] - x[i]) / (value[i + 1] - value[i]))


def compare_rana_centerline(
    state: np.ndarray,
    reference: Mapping[str, np.ndarray],
    *,
    lid_velocity: float,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare R26 vx/U on x/L=0.5 with the digitised Rana R13 curve."""

    array = validate_wall_inclusive_state(state)
    default_x, default_y = uniform_wall_grid(array)
    xv = default_x if x is None else np.asarray(x, dtype=float)
    yv = default_y if y is None else np.asarray(y, dtype=float)
    y_ref = np.asarray(reference["y_over_L"], dtype=float)
    paper = np.asarray(reference["rana_fig3_R13_vx_over_Ulid"], dtype=float)
    uncertainty = np.asarray(reference["digitized_half_line_uncertainty"], dtype=float)
    native = np.asarray(
        [np.interp(0.5, xv, row[:, STATE_INDEX["vx"]]) for row in array]
    ) / float(lid_velocity)
    prediction = np.interp(y_ref, yv, native)
    difference = prediction - paper
    weights = 1.0 / np.maximum(uncertainty, 1.0e-12) ** 2
    weights /= np.sum(weights)
    result = {
        "y": y_ref,
        "R13_reference": paper,
        "digitized_uncertainty": uncertainty,
        "R26_prediction": prediction,
        "difference_R26_minus_R13": difference,
        "native_y": yv,
        "native_R26_prediction": native,
        "metrics": {
            "weighted_bias_R26_minus_R13": float(np.sum(weights * difference)),
            "weighted_mae": float(np.sum(weights * np.abs(difference))),
            "weighted_rmse": float(np.sqrt(np.sum(weights * difference**2))),
            "linf": float(np.max(np.abs(difference))),
            "fraction_within_digitized_line_width": float(
                np.mean(np.abs(difference) <= uncertainty)
            ),
            "R13_zero_crossing_y_over_L": _zero_crossing(y_ref, paper),
            "R26_zero_crossing_y_over_L": _zero_crossing(y_ref, prediction),
            "R13_profile_area_on_digitized_span": _trapezoid(np.abs(paper), y_ref),
            "R26_profile_area_on_digitized_span": _trapezoid(np.abs(prediction), y_ref),
        },
        "caveat": RANA_REFERENCE_CAVEAT,
    }
    first = result["metrics"]["R13_zero_crossing_y_over_L"]
    second = result["metrics"]["R26_zero_crossing_y_over_L"]
    result["metrics"]["zero_crossing_delta_R26_minus_R13"] = (
        None if first is None or second is None else float(second - first)
    )
    return result


def load_diagnostic_dsmc_csv(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and audit the legacy auxiliary DSMC field CSV."""

    with path.open("r", encoding="utf-8") as stream:
        header = tuple(stream.readline().strip().split(","))
    if header != DSMC_EXPECTED_COLUMNS:
        raise ValueError(f"unexpected DSMC header in {path.name}")
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("empty or malformed DSMC CSV")
    for name in data.dtype.names or ():
        if not np.isfinite(data[name]).all():
            raise FloatingPointError(f"non-finite DSMC column {name}")
    nx = int(np.max(data["ix"])) + 1
    ny = int(np.max(data["iy"])) + 1
    if data.size != nx * ny:
        raise ValueError("DSMC row count does not match ix/iy extent")
    if not (
        np.array_equal(data["ix"].astype(int), np.tile(np.arange(nx), ny))
        and np.array_equal(data["iy"].astype(int), np.repeat(np.arange(ny), nx))
    ):
        raise ValueError("DSMC rows are not iy-major/ix-minor")
    if np.min(data["n"]) <= 0.0 or np.min(data["T"]) <= 0.0:
        raise FloatingPointError("DSMC density/temperature is non-positive")
    for name in ("n0", "Kn", "Uwall", "Twall", "nsamples"):
        if np.ptp(data[name]) != 0.0:
            raise ValueError(f"DSMC column {name} is not constant")
    x_values = np.unique(data["x"])
    y_values = np.unique(data["y"])
    dx = float(np.median(np.diff(x_values)))
    dy = float(np.median(np.diff(y_values)))
    audit = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": int(data.size),
        "nx": nx,
        "ny": ny,
        "kn": float(data["Kn"][0]),
        "lid_velocity_m_s": float(data["Uwall"][0]),
        "wall_temperature_K": float(data["Twall"][0]),
        "inferred_length_x_m": float(x_values[-1] + 0.5 * dx),
        "inferred_length_y_m": float(y_values[-1] + 0.5 * dy),
        "n0": float(data["n0"][0]),
        "nsamples": int(data["nsamples"][0]),
        "mean_particle_count": float(np.mean(data["particle_count"])),
        "caveat": DSMC_PROVENANCE_CAVEAT,
    }
    return data, audit


def dsmc_to_target_fields(
    data: np.ndarray,
    *,
    target_n: int = TARGET_N,
    length_m: float = DEFAULT_LENGTH_M,
    molecular_mass_kg: float = MASS_ARGON_KG,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Normalize the diagnostic DSMC moments and interpolate to target cells."""

    nx = int(np.max(data["ix"])) + 1
    ny = int(np.max(data["iy"])) + 1
    x = np.unique(data["x"]) / length_m
    y = np.unique(data["y"]) / length_m
    centers = (np.arange(target_n, dtype=float) + 0.5) / target_n
    X, Y = np.meshgrid(centers, centers)
    points = np.column_stack((Y.ravel(), X.ravel()))
    outside = (X < x[0]) | (X > x[-1]) | (Y < y[0]) | (Y > y[-1])
    n0 = float(data["n0"][0])
    twall = float(data["Twall"][0])
    velocity_scale = math.sqrt(BOLTZMANN * twall / molecular_mass_kg)

    def interpolate(column: str, scale: float) -> np.ndarray:
        values = np.asarray(data[column], dtype=float).reshape(ny, nx) / scale
        function = RegularGridInterpolator(
            (y, x), values, bounds_error=False, fill_value=None
        )
        result = function(points).reshape(target_n, target_n)
        if not np.isfinite(result).all():
            raise FloatingPointError(f"non-finite DSMC interpolation of {column}")
        return result

    fields = {
        "centers": centers,
        "X": X,
        "Y": Y,
        "rho": interpolate("n", n0),
        "vx": interpolate("ux", velocity_scale),
        "vy": interpolate("uy", velocity_scale),
        "theta": interpolate("T", twall),
        "qx": interpolate("qxN", n0 * velocity_scale**3),
        "qy": interpolate("qyN", n0 * velocity_scale**3),
        "sigma_xx": interpolate("sigma_xxN", n0 * velocity_scale**2),
        "sigma_xy": interpolate("sigma_xyN", n0 * velocity_scale**2),
        "sigma_yy": interpolate("sigma_yyN", n0 * velocity_scale**2),
        "Rxx": interpolate("R_xxN", n0 * velocity_scale**4),
        "Rxy": interpolate("R_xyN", n0 * velocity_scale**4),
        "Ryy": interpolate("R_yyN", n0 * velocity_scale**4),
        "Delta": interpolate("DeltaN", n0 * velocity_scale**4),
    }
    metadata = {
        "source_grid": [ny, nx],
        "target_grid": [target_n, target_n],
        "target_points_outside_source_cell_center_extent": int(np.count_nonzero(outside)),
        "linear_boundary_extrapolation_used": bool(np.any(outside)),
        "velocity_scale_m_s": velocity_scale,
        "normalization": {
            "rho": "n/n0",
            "velocity": "u/sqrt(k_B*Twall/m_Ar)",
            "temperature": "T/Twall",
            "heat_flux": "qN/(n0*v0^3)",
            "stress": "sigmaN/(n0*v0^2)",
            "R_and_Delta": "number-based fourth moment/(n0*v0^4)",
        },
        "caveat": DSMC_PROVENANCE_CAVEAT,
    }
    return fields, metadata


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    a = np.asarray(first).ravel() - float(np.mean(first))
    b = np.asarray(second).ravel() - float(np.mean(second))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator <= np.finfo(float).tiny else float(np.dot(a, b) / denominator)


def compare_common_fields(
    reference: Mapping[str, np.ndarray],
    prediction: Mapping[str, np.ndarray],
    *,
    eligible_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Return error summaries for every common DSMC/R26 field."""

    shape = np.asarray(reference["rho"]).shape
    mask = np.ones(shape, dtype=bool) if eligible_mask is None else np.asarray(eligible_mask, bool)
    if mask.shape != shape or not np.any(mask):
        raise ValueError("field-comparison mask is empty or has the wrong shape")
    rows: list[dict[str, Any]] = []
    for name in COMMON_FIELDS:
        ref = np.asarray(reference[name], dtype=float)[mask]
        pred = np.asarray(prediction[name], dtype=float)[mask]
        difference = pred - ref
        ref_signal = ref - 1.0 if name in {"rho", "theta"} else ref
        pred_signal = pred - 1.0 if name in {"rho", "theta"} else pred
        rms_ref = float(np.sqrt(np.mean(ref_signal**2)))
        rmse = float(np.sqrt(np.mean(difference**2)))
        rows.append(
            {
                "field": name,
                "comparison": "private_Python_R26_minus_diagnostic_legacy_seed1_DSMC",
                "DSMC_mean": float(np.mean(ref)),
                "R26_mean": float(np.mean(pred)),
                "bias_R26_minus_DSMC": float(np.mean(difference)),
                "DSMC_signal_RMS": rms_ref,
                "R26_signal_RMS": float(np.sqrt(np.mean(pred_signal**2))),
                "RMSE": rmse,
                "NRMSE": rmse / max(rms_ref, np.finfo(float).tiny),
                "pearson_pattern_correlation": _correlation(ref, pred),
                "max_abs_error": float(np.max(np.abs(difference))),
                "included_cell_count": int(np.count_nonzero(mask)),
                "provenance_caveat": DSMC_PROVENANCE_CAVEAT,
            }
        )
    return rows


def line_profile_rows(
    dsmc: Mapping[str, np.ndarray],
    r26: Mapping[str, np.ndarray],
    *,
    orientation: str,
    location: float,
) -> list[dict[str, Any]]:
    """Return wide all-field DSMC/R26 profiles on a nearest target-grid line."""

    centers = np.asarray(r26["centers"], dtype=float)
    if not np.array_equal(centers, np.asarray(dsmc["centers"], dtype=float)):
        raise ValueError("DSMC and R26 target grids differ")
    index = int(np.argmin(np.abs(centers - location)))
    rows: list[dict[str, Any]] = []
    for k, coordinate in enumerate(centers):
        row: dict[str, Any] = {
            "orientation": orientation,
            "requested_fixed_coordinate": location,
            "actual_fixed_coordinate": float(centers[index]),
            "varying_coordinate": float(coordinate),
            "DSMC_provenance": "legacy single-seed auxiliary; diagnostic only",
        }
        for name in COMMON_FIELDS:
            if orientation == "vertical":
                reference_value = dsmc[name][k, index]
                prediction_value = r26[name][k, index]
            elif orientation == "horizontal":
                reference_value = dsmc[name][index, k]
                prediction_value = r26[name][index, k]
            else:
                raise ValueError("orientation must be vertical or horizontal")
            row[f"DSMC_{name}"] = float(reference_value)
            row[f"R26_{name}"] = float(prediction_value)
            row[f"R26_minus_DSMC_{name}"] = float(prediction_value - reference_value)
        rows.append(row)
    return rows


def _common_limit(first: np.ndarray, second: np.ndarray, *, departure: bool = False) -> tuple[float, float]:
    a = np.asarray(first, dtype=float) - (1.0 if departure else 0.0)
    b = np.asarray(second, dtype=float) - (1.0 if departure else 0.0)
    values = np.concatenate((a.ravel(), b.ravel()))
    low, high = np.percentile(values, [1.0, 99.0])
    if low == high:
        low, high = float(np.min(values)), float(np.max(values))
    if low == high:
        low -= 1.0
        high += 1.0
    return float(low), float(high)


def _save_field_pages(
    output: Path,
    dsmc: Mapping[str, np.ndarray],
    r26: Mapping[str, np.ndarray],
) -> None:
    X, Y = np.asarray(r26["X"]), np.asarray(r26["Y"])
    groups = {
        "primary": ("rho", "theta", "vx", "vy", "qx", "qy"),
        "stress_and_fourth_order": (
            "sigma_xx", "sigma_xy", "sigma_yy", "Rxx", "Rxy", "Ryy", "Delta"
        ),
    }
    for group, names in groups.items():
        fig, axes = plt.subplots(len(names), 3, figsize=(11.8, 2.55 * len(names)), constrained_layout=True)
        for row, name in enumerate(names):
            departure = name in {"rho", "theta"}
            reference = np.asarray(dsmc[name]) - (1.0 if departure else 0.0)
            prediction = np.asarray(r26[name]) - (1.0 if departure else 0.0)
            difference = prediction - reference
            low, high = _common_limit(dsmc[name], r26[name], departure=departure)
            levels = np.linspace(low, high, 33)
            diff_limit = max(float(np.percentile(np.abs(difference), 99.0)), 1.0e-15)
            for column, (values, title, local_levels) in enumerate(
                (
                    (reference, f"DSMC diagnostic: {name}", levels),
                    (prediction, f"Python R26: {name}", levels),
                    (difference, f"R26 - DSMC: {name}", np.linspace(-diff_limit, diff_limit, 33)),
                )
            ):
                image = axes[row, column].contourf(
                    X, Y, values, levels=local_levels, cmap="coolwarm", extend="both"
                )
                fig.colorbar(image, ax=axes[row, column], shrink=0.86)
                axes[row, column].set_title(title, fontsize=9)
                axes[row, column].set_aspect("equal")
                axes[row, column].set_xlabel("x/L")
                axes[row, column].set_ylabel("y/L")
        fig.suptitle(
            "Diagnostic legacy single-seed DSMC vs private Python R26\n"
            "7x7 smoothing; DSMC field comparison is not manuscript-production validation",
            fontsize=12,
        )
        fig.savefig(output / f"jfm_r26_vs_dsmc_{group}_fields.png", dpi=190)
        fig.savefig(output / f"jfm_r26_vs_dsmc_{group}_fields.pdf")
        plt.close(fig)


def _save_line_figures(
    output: Path,
    dsmc: Mapping[str, np.ndarray],
    r26: Mapping[str, np.ndarray],
) -> None:
    centers = np.asarray(r26["centers"])
    for orientation, location in (("vertical", 0.5), ("horizontal", 0.9)):
        index = int(np.argmin(np.abs(centers - location)))
        fig, axes = plt.subplots(4, 4, figsize=(13.5, 12.0), constrained_layout=True)
        for ax, name in zip(axes.ravel(), COMMON_FIELDS):
            if orientation == "vertical":
                dsmc_line, r26_line = dsmc[name][:, index], r26[name][:, index]
            else:
                dsmc_line, r26_line = dsmc[name][index, :], r26[name][index, :]
            ax.plot(centers, dsmc_line, label="DSMC diagnostic", lw=1.3)
            ax.plot(centers, r26_line, label="Python R26", lw=1.3)
            ax.set_title(name)
            ax.grid(alpha=0.25)
        for ax in axes.ravel()[len(COMMON_FIELDS):]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            f"All common fields: {orientation} line at nearest coordinate {centers[index]:.5f}\n"
            "Auxiliary DSMC field comparison is diagnostic only"
        )
        fig.savefig(output / f"jfm_r26_vs_dsmc_{orientation}_profiles.png", dpi=190)
        fig.savefig(output / f"jfm_r26_vs_dsmc_{orientation}_profiles.pdf")
        plt.close(fig)


def _save_anti_fourier_figure(
    output: Path,
    fields: Mapping[str, np.ndarray],
    analysis: Mapping[str, Any],
) -> None:
    X, Y = fields["X"], fields["Y"]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.5), constrained_layout=True)
    iaf = np.ma.masked_where(~analysis["active"], analysis["iaf"])
    image = axes[0, 0].contourf(
        X, Y, iaf, levels=np.linspace(-1.0, 1.0, 41), cmap="coolwarm", extend="both"
    )
    axes[0, 0].contour(X, Y, analysis["af"].astype(float), levels=[0.5], colors="k")
    fig.colorbar(image, ax=axes[0, 0], label=r"$I_{AF}$")
    panels = (("PR", r"$P_R$"), ("PDelta", r"$P_\Delta$"))
    for ax, (key, label) in zip((axes[0, 1], axes[1, 0]), panels):
        raw = np.asarray(analysis[key])
        values = np.ma.masked_where(~analysis["af"], raw)
        limit = max(float(np.nanpercentile(np.abs(raw[analysis["af"]]), 99.0)), 1.0e-15)
        image = ax.contourf(
            X, Y, values, levels=np.linspace(-limit, limit, 41), cmap="coolwarm", extend="both"
        )
        fig.colorbar(image, ax=ax, label=label)
    chi = np.ma.masked_where(~analysis["af"], analysis["chiDelta"])
    image = axes[1, 1].contourf(X, Y, chi, levels=np.linspace(0.0, 1.0, 41), cmap="viridis")
    fig.colorbar(image, ax=axes[1, 1], label=r"$\chi_\Delta$")
    for ax in axes.ravel():
        ax.set_xlabel("x/L")
        ax.set_ylabel("y/L")
        ax.set_aspect("equal")
    metrics = analysis["metrics"]
    fig.suptitle(
        "Private Python R26 anti-Fourier and fourth-order channels\n"
        f"f_AF|active={metrics['f_AF_active']:.3f}, mean I_AF={metrics['mean_IAF_AF']:.3f}, "
        f"PDelta/PR={metrics['PDelta_over_PR']:.3f}, mean chiDelta={metrics['mean_chiDelta']:.3f}"
    )
    fig.savefig(output / "jfm_r26_anti_fourier_channels.png", dpi=220)
    fig.savefig(output / "jfm_r26_anti_fourier_channels.pdf")
    plt.close(fig)


def _save_anti_fourier_comparison(
    output: Path,
    fields: Mapping[str, np.ndarray],
    dsmc_analysis: Mapping[str, Any],
    r26_analysis: Mapping[str, Any],
) -> None:
    """Render auxiliary DSMC and R26 diagnostic channels on common scales."""

    X, Y = fields["X"], fields["Y"]
    fig, axes = plt.subplots(4, 2, figsize=(9.8, 15.0), constrained_layout=True)
    analyses = (dsmc_analysis, r26_analysis)
    column_titles = (
        "Legacy single-seed DSMC\nauxiliary diagnostic only",
        "Private Python R26",
    )
    for column, (analysis, title) in enumerate(zip(analyses, column_titles)):
        masked = np.ma.masked_where(~analysis["active"], analysis["iaf"])
        image = axes[0, column].contourf(
            X, Y, masked, levels=np.linspace(-1.0, 1.0, 41), cmap="coolwarm", extend="both"
        )
        axes[0, column].contour(
            X, Y, analysis["af"].astype(float), levels=[0.5], colors="k", linewidths=0.8
        )
        fig.colorbar(image, ax=axes[0, column], label=r"active $I_{AF}$")
        axes[0, column].set_title(title)

    for row, (key, label) in enumerate(
        (("PR", r"$P_R$"), ("PDelta", r"$P_\Delta$")), start=1
    ):
        values = np.concatenate(
            [np.abs(np.asarray(analysis[key])[analysis["af"]]) for analysis in analyses]
        )
        limit = max(float(np.percentile(values, 99.0)), 1.0e-15)
        for column, analysis in enumerate(analyses):
            masked = np.ma.masked_where(~analysis["af"], analysis[key])
            image = axes[row, column].contourf(
                X,
                Y,
                masked,
                levels=np.linspace(-limit, limit, 41),
                cmap="coolwarm",
                extend="both",
            )
            fig.colorbar(image, ax=axes[row, column], label=label)
    for column, analysis in enumerate(analyses):
        masked = np.ma.masked_where(~analysis["af"], analysis["chiDelta"])
        image = axes[3, column].contourf(
            X, Y, masked, levels=np.linspace(0.0, 1.0, 41), cmap="viridis"
        )
        fig.colorbar(image, ax=axes[3, column], label=r"$\chi_\Delta$")
    for ax in axes.ravel():
        ax.set_xlabel("x/L")
        ax.set_ylabel("y/L")
        ax.set_aspect("equal")
    fig.suptitle(
        "Anti-Fourier and fourth-order channels: diagnostic DSMC vs private R26\n"
        "Common scales; auxiliary DSMC is not the manuscript production ensemble"
    )
    fig.savefig(output / "jfm_r26_vs_diagnostic_dsmc_anti_fourier_channels.png", dpi=200)
    fig.savefig(output / "jfm_r26_vs_diagnostic_dsmc_anti_fourier_channels.pdf")
    plt.close(fig)


def _save_scalar_comparison(
    output: Path,
    r26_metrics: Mapping[str, float],
    manuscript: Mapping[str, float | None],
) -> None:
    names = list(METRIC_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), constrained_layout=True)
    for ax, name in zip(axes.ravel(), names):
        published = manuscript.get(name)
        labels, values = ["Python R26"], [float(r26_metrics[name])]
        if published is not None:
            labels.insert(0, "Manuscript DSMC")
            values.insert(0, float(published))
        bars = ax.bar(labels, values, color=("tab:blue", "tab:orange")[: len(values)])
        ax.set_title(METRIC_LABELS[name])
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    fig.suptitle("JFM scalar diagnostics: authoritative manuscript DSMC vs private Python R26")
    fig.savefig(output / "jfm_r26_vs_manuscript_scalars.png", dpi=220)
    fig.savefig(output / "jfm_r26_vs_manuscript_scalars.pdf")
    plt.close(fig)


def run_rana_postprocess(
    state: np.ndarray,
    metadata: CaseMetadata,
    *,
    output: Path,
    digitized_csv: Path,
) -> dict[str, Any]:
    """Generate the complete private Rana/R26 comparison bundle."""

    if metadata.family != "rana":
        raise ValueError("Rana post-processing requires family='rana'")
    if not math.isclose(metadata.kn, 0.010, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("the digitised Rana Fig. 3 reference is the Kn=0.010 case")
    array = validate_wall_inclusive_state(state)
    output.mkdir(parents=True, exist_ok=True)
    reference = load_rana_digitized_reference(digitized_csv)
    globals_ = rana_global_metrics(array, lid_velocity=metadata.lid_velocity)
    profile = compare_rana_centerline(
        array, reference, lid_velocity=metadata.lid_velocity
    )
    rows = []
    for i, y_value in enumerate(profile["y"]):
        rows.append(
            {
                "y_over_L": y_value,
                "published_R13_vx_over_Ulid": profile["R13_reference"][i],
                "digitized_half_line_uncertainty": profile["digitized_uncertainty"][i],
                "private_R26_vx_over_Ulid": profile["R26_prediction"][i],
                "R26_minus_R13": profile["difference_R26_minus_R13"][i],
                "semantics": "published curve is R13; R26 is a cross-model prediction",
            }
        )
    write_csv(output / "rana_fig3_R13_vs_private_R26_centerline.csv", rows)

    fig, ax = plt.subplots(figsize=(6.7, 5.0), constrained_layout=True)
    ax.fill_betweenx(
        profile["y"],
        profile["R13_reference"] - profile["digitized_uncertainty"],
        profile["R13_reference"] + profile["digitized_uncertainty"],
        color="0.75",
        alpha=0.5,
        label="digitised R13 line-width uncertainty",
    )
    ax.plot(profile["R13_reference"], profile["y"], "k-", label="Rana published R13")
    ax.plot(profile["R26_prediction"], profile["y"], "C3--", label="private Python R26")
    ax.set_xlabel(r"$u_x/U_{lid}$")
    ax.set_ylabel("y/L")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Rana first cavity: vertical centreline\nR13 reference vs new R26 prediction")
    fig.savefig(output / "rana_fig3_R13_vs_private_R26_centerline.png", dpi=220)
    fig.savefig(output / "rana_fig3_R13_vs_private_R26_centerline.pdf")
    plt.close(fig)

    summary = {
        "model": "private Python R26",
        "metadata": asdict(metadata),
        "state_shape": list(array.shape),
        "global_metrics": globals_,
        "published_R13_context": {"D": 0.1585, "G": 0.1893},
        "global_relative_difference_R26_vs_published_R13": {
            "D": float(globals_["D"] / 0.1585 - 1.0),
            "G": float(globals_["G"] / 0.1893 - 1.0),
        },
        "centerline_metrics": profile["metrics"],
        "reference_csv": {
            "path": str(digitized_csv.resolve()),
            "sha256": sha256_file(digitized_csv),
        },
        "scientific_status": {
            "solver_converged_claim_from_metadata": metadata.converged,
            "comparison_is_validation": False,
            "publication_grade": False,
            "caveat": RANA_REFERENCE_CAVEAT,
        },
    }
    write_json(output / "rana_r26_comparison_summary.json", summary)
    report = f"""# Private Python R26 — Rana first-cavity comparison

This bundle post-processes a wall-inclusive R26 state. It does not solve or
alter the state.

## Result

- R26 D: `{globals_['D']:.9g}`
- R26 G: `{globals_['G']:.9g}`
- Published R13 context: D=`0.1585`, G=`0.1893`
- Centreline weighted RMSE (R26 minus digitised R13): `{profile['metrics']['weighted_rmse']:.6g}`
- Algebraic convergence asserted by input metadata: `{metadata.converged}`
- Publication-grade: `false`

## Interpretation

{RANA_REFERENCE_CAVEAT}
"""
    (output / "RANA_R26_PRIVATE_COMPARISON.md").write_text(report, encoding="utf-8")
    return summary


def _select_manuscript_regime(metadata: CaseMetadata) -> tuple[str | None, Mapping[str, float | None]]:
    for name, values in MANUSCRIPT_SCALARS.items():
        if (
            math.isclose(metadata.kn, float(values["kn"]), rel_tol=0.0, abs_tol=1.0e-12)
            and math.isclose(
                metadata.lid_velocity_m_s,
                float(values["lid_velocity_m_s"]),
                rel_tol=0.0,
                abs_tol=0.75,
            )
        ):
            return name, values
    return None, {}


def run_jfm_postprocess(
    state: np.ndarray,
    metadata: CaseMetadata,
    *,
    output: Path,
    dsmc_csv: Path,
    target_n: int = TARGET_N,
) -> dict[str, Any]:
    """Generate R26/JFM scalar, field, line, mask-sensitivity, and plot outputs."""

    if metadata.family != "jfm":
        raise ValueError("JFM post-processing requires family='jfm'")
    array = validate_wall_inclusive_state(state)
    output.mkdir(parents=True, exist_ok=True)
    r26_fields = interpolate_state(array, target_n=target_n)
    primary = analyze_anti_fourier(r26_fields)

    dsmc_data, dsmc_audit = load_diagnostic_dsmc_csv(dsmc_csv)
    if not math.isclose(metadata.kn, dsmc_audit["kn"], rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            f"R26 Kn={metadata.kn} does not match DSMC CSV Kn={dsmc_audit['kn']}"
        )
    if abs(metadata.lid_velocity_m_s - dsmc_audit["lid_velocity_m_s"]) > 0.25:
        raise ValueError(
            "R26 dimensional lid speed does not match the diagnostic DSMC CSV"
        )
    if abs(metadata.wall_temperature_K - dsmc_audit["wall_temperature_K"]) > 0.1:
        raise ValueError("R26 wall temperature does not match the diagnostic DSMC CSV")
    for key in ("inferred_length_x_m", "inferred_length_y_m"):
        if not math.isclose(
            metadata.length_m, dsmc_audit[key], rel_tol=5.0e-6, abs_tol=1.0e-15
        ):
            raise ValueError("R26 cavity length does not match the diagnostic DSMC grid")
    dsmc_fields, dsmc_interp = dsmc_to_target_fields(
        dsmc_data, target_n=target_n, length_m=metadata.length_m
    )
    dsmc_analysis = analyze_anti_fourier(dsmc_fields)
    r26_fields_on_dsmc_basis = convert_velocity_normalization(
        r26_fields,
        from_velocity_scale_m_s=metadata.velocity_scale_m_s,
        to_velocity_scale_m_s=dsmc_interp["velocity_scale_m_s"],
    )
    r26_comparison_analysis = analyze_anti_fourier(r26_fields_on_dsmc_basis)
    r26_smoothed = {
        **{key: r26_fields_on_dsmc_basis[key] for key in ("centers", "X", "Y")},
        **smooth_common_fields(r26_fields_on_dsmc_basis),
    }
    dsmc_smoothed = {
        **{key: dsmc_fields[key] for key in ("centers", "X", "Y")},
        **dsmc_analysis["smoothed"],
    }

    sensitivity_rows: list[dict[str, Any]] = []
    sensitivity: dict[str, Any] = {}
    for corners in ("top", "all"):
        for epsilon in (0.0, 0.025, 0.05, 0.10):
            mask = corner_eligible_mask(r26_fields["X"], r26_fields["Y"], epsilon, corners=corners)
            result = analyze_anti_fourier(r26_fields, eligible_mask=mask)
            key = f"{corners}_epsilon_{epsilon:.3f}"
            sensitivity[key] = result["metrics"]
            sensitivity_rows.append(
                {
                    "corner_set": corners,
                    "epsilon": epsilon,
                    "eligible_count": result["eligible_count"],
                    "active_count": result["active_count"],
                    "AF_count": result["af_count"],
                    **result["metrics"],
                }
            )
    for mode in ("nearest", "reflect", "mirror"):
        result = analyze_anti_fourier(r26_fields, smoothing_mode=mode)
        sensitivity[f"smoothing_mode_{mode}"] = result["metrics"]
        sensitivity_rows.append(
            {
                "corner_set": "none",
                "epsilon": 0.0,
                "smoothing_mode": mode,
                "eligible_count": result["eligible_count"],
                "active_count": result["active_count"],
                "AF_count": result["af_count"],
                **result["metrics"],
            }
        )
    write_csv(output / "jfm_r26_corner_and_smoothing_sensitivity.csv", sensitivity_rows)

    field_rows = compare_common_fields(dsmc_smoothed, r26_smoothed)
    write_csv(output / "jfm_r26_vs_diagnostic_dsmc_field_errors.csv", field_rows)
    field_mask_rows: list[dict[str, Any]] = []
    for epsilon in (0.0, 0.05, 0.10):
        mask = corner_eligible_mask(r26_fields["X"], r26_fields["Y"], epsilon, corners="top")
        for row in compare_common_fields(dsmc_smoothed, r26_smoothed, eligible_mask=mask):
            field_mask_rows.append({"top_corner_epsilon": epsilon, **row})
    write_csv(output / "jfm_r26_vs_diagnostic_dsmc_field_corner_sensitivity.csv", field_mask_rows)

    vertical_rows = line_profile_rows(dsmc_smoothed, r26_smoothed, orientation="vertical", location=0.5)
    horizontal_rows = line_profile_rows(dsmc_smoothed, r26_smoothed, orientation="horizontal", location=0.9)
    write_csv(output / "jfm_r26_vs_dsmc_vertical_x0p5_all_fields.csv", vertical_rows)
    write_csv(output / "jfm_r26_vs_dsmc_horizontal_y0p9_all_fields.csv", horizontal_rows)

    regime_name, manuscript = _select_manuscript_regime(metadata)
    manuscript_rows: list[dict[str, Any]] = []
    for name in METRIC_LABELS:
        published = manuscript.get(name)
        value = primary["metrics"][name]
        manuscript_rows.append(
            {
                "metric": name,
                "manuscript_DSMC": published,
                "private_Python_R26": value,
                "R26_relative_difference": (
                    None if published is None else float(value / float(published) - 1.0)
                ),
                "comparability": (
                    "same disclosed scalar definition"
                    if name != "PDelta_over_PR"
                    else "provisional: manuscript does not specify the norm behind its scale ratio"
                ),
            }
        )
    write_csv(output / "jfm_r26_vs_manuscript_scalar_metrics.csv", manuscript_rows)

    _save_field_pages(output, dsmc_smoothed, r26_smoothed)
    _save_line_figures(output, dsmc_smoothed, r26_smoothed)
    _save_anti_fourier_figure(output, r26_fields, primary)
    _save_anti_fourier_comparison(
        output, r26_fields_on_dsmc_basis, dsmc_analysis, r26_comparison_analysis
    )
    if manuscript:
        _save_scalar_comparison(output, primary["metrics"], manuscript)

    summary = {
        "model": "private Python R26",
        "metadata": asdict(metadata),
        "state_shape": list(array.shape),
        "target_grid": [target_n, target_n],
        "smoothing": primary["smoothing"],
        "R26_metrics": primary["metrics"],
        "diagnostic_DSMC_metrics_from_auxiliary_CSV": dsmc_analysis["metrics"],
        "manuscript_regime": regime_name,
        "authoritative_manuscript_scalars": dict(manuscript),
        "corner_and_smoothing_sensitivity": sensitivity,
        "DSMC_source_audit": dsmc_audit,
        "DSMC_interpolation": dsmc_interp,
        "R26_to_DSMC_velocity_basis_conversion": {
            "from_velocity_scale_m_s": metadata.velocity_scale_m_s,
            "to_velocity_scale_m_s": dsmc_interp["velocity_scale_m_s"],
            "ratio": metadata.velocity_scale_m_s / dsmc_interp["velocity_scale_m_s"],
            "moment_scaling": "rank-k moments multiplied by ratio**k",
        },
        "field_error_summary": field_rows,
        "scientific_status": {
            "solver_converged_claim_from_metadata": metadata.converged,
            "field_comparison_is_validation": False,
            "publication_grade": False,
            "DSMC_caveat": DSMC_PROVENANCE_CAVEAT,
        },
    }
    write_json(output / "jfm_r26_comparison_summary.json", summary)
    metrics = primary["metrics"]
    report = f"""# Private Python R26 — JFM cavity comparison

This bundle post-processes a wall-inclusive R26 state. It does not solve,
restart, or modify the state.

## R26 diagnostics

- `f_AF|active = {metrics['f_AF_active']:.9g}`
- `mean I_AF|AF = {metrics['mean_IAF_AF']:.9g}`
- `RMS(PDelta)/RMS(PR)|AF = {metrics['PDelta_over_PR']:.9g}`
- `mean chiDelta|AF = {metrics['mean_chiDelta']:.9g}`
- Algebraic convergence asserted by input metadata: `{metadata.converged}`
- Publication-grade: `false`

## Comparison semantics

{DSMC_PROVENANCE_CAVEAT}

The corner tables remove top or all corner squares from diagnostic masks only;
they never alter the supplied R26 state.  Smoothing-mode alternatives are
reported as implementation sensitivity, while 7x7-nearest is the primary
disclosed realization.
"""
    (output / "JFM_R26_PRIVATE_COMPARISON.md").write_text(report, encoding="utf-8")
    return summary


def load_metadata(path: Path) -> CaseMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("metadata JSON must contain one object")
    return CaseMetadata(**payload)


def _default_workspace() -> Path:
    # .../private_work/r26_python/code/r26_postprocess.py -> workspace root
    return Path(__file__).resolve().parents[3]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="family", required=True)
    for family in ("rana", "jfm"):
        child = subparsers.add_parser(family)
        child.add_argument("--state", type=Path, required=True)
        child.add_argument("--metadata", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    rana = subparsers.choices["rana"]
    rana.add_argument(
        "--digitized-csv",
        type=Path,
        default=_default_workspace()
        / "Rana_Fig3_Private_Comparison"
        / "Rana_Fig3_Kn0010_R13_vs_Python_N40.csv",
    )
    jfm = subparsers.choices["jfm"]
    jfm.add_argument(
        "--dsmc-csv",
        type=Path,
        default=_default_workspace()
        / "library_materialized"
        / "jfm_compare"
        / "cavity_Kn005_U100_seed1_VALID_final.csv",
    )
    jfm.add_argument("--target-n", type=int, default=TARGET_N)
    args = parser.parse_args(argv)
    state = np.load(args.state, allow_pickle=False)
    metadata = load_metadata(args.metadata)
    if metadata.family != args.family:
        raise ValueError("metadata family does not match CLI subcommand")
    if args.family == "rana":
        run_rana_postprocess(
            state,
            metadata,
            output=args.output.resolve(),
            digitized_csv=args.digitized_csv.resolve(),
        )
    else:
        run_jfm_postprocess(
            state,
            metadata,
            output=args.output.resolve(),
            dsmc_csv=args.dsmc_csv.resolve(),
            target_n=args.target_n,
        )
    return 0


__all__ = [
    "CaseMetadata",
    "COMMON_FIELDS",
    "DSMC_PROVENANCE_CAVEAT",
    "MANUSCRIPT_SCALARS",
    "RANA_REFERENCE_CAVEAT",
    "analyze_anti_fourier",
    "compare_common_fields",
    "compare_rana_centerline",
    "convert_velocity_normalization",
    "corner_eligible_mask",
    "dsmc_to_target_fields",
    "interpolate_state",
    "line_profile_rows",
    "load_diagnostic_dsmc_csv",
    "load_rana_digitized_reference",
    "rana_global_metrics",
    "run_jfm_postprocess",
    "run_rana_postprocess",
    "smooth_common_fields",
    "validate_wall_inclusive_state",
]


if __name__ == "__main__":
    raise SystemExit(main())
