from __future__ import annotations

import math

import numpy as np

from r26_postprocess import (
    COMMON_FIELDS,
    DSMC_PROVENANCE_CAVEAT,
    RANA_REFERENCE_CAVEAT,
    analyze_anti_fourier,
    compare_common_fields,
    compare_rana_centerline,
    convert_velocity_normalization,
    corner_eligible_mask,
    interpolate_state,
    line_profile_rows,
    rana_global_metrics,
)


def _equilibrium(n: int = 11) -> np.ndarray:
    state = np.zeros((n, n, 17))
    state[..., 0] = 1.0
    state[..., 3] = 1.0
    return state


def _synthetic_fields(n: int = 32) -> dict[str, np.ndarray]:
    centers = (np.arange(n, dtype=float) + 0.5) / n
    X, Y = np.meshgrid(centers, centers)
    fields: dict[str, np.ndarray] = {"centers": centers, "X": X, "Y": Y}
    for name in COMMON_FIELDS:
        fields[name] = np.zeros_like(X)
    fields["rho"] = np.ones_like(X)
    fields["theta"] = 1.0 + 0.1 * X
    fields["qx"] = np.ones_like(X)
    fields["Rxx"] = X.copy()
    fields["Delta"] = 0.6 * X
    return fields


def test_affine_wall_state_interpolation_is_exact() -> None:
    n = 9
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, y)
    state = _equilibrium(n)
    state[..., 0] = 1.0 + 0.1 * X + 0.2 * Y
    state[..., 1] = -0.3 * X + 0.4 * Y
    state[..., 3] = 1.0 + 0.05 * Y
    result = interpolate_state(state, target_n=24)
    assert result["rho"].shape == (24, 24)
    assert np.allclose(result["rho"], 1.0 + 0.1 * result["X"] + 0.2 * result["Y"])
    assert np.allclose(result["vx"], -0.3 * result["X"] + 0.4 * result["Y"])


def test_anti_fourier_metric_definitions_on_manufactured_linear_fields() -> None:
    result = analyze_anti_fourier(_synthetic_fields())
    metrics = result["metrics"]
    assert np.isclose(metrics["f_active_domain"], 1.0)
    assert np.isclose(metrics["f_AF_domain"], 1.0)
    assert np.isclose(metrics["f_AF_active"], 1.0)
    assert np.isclose(metrics["mean_IAF_AF"], 1.0)
    # Delta=0.6*x and PDelta=(1/3)dDelta/dx=0.2 while PR=dRxx/dx=1.
    assert np.isclose(metrics["PDelta_over_PR"], 0.2, rtol=1.0e-11, atol=1.0e-12)
    assert np.isclose(metrics["mean_chiDelta"], 1.0 / 6.0, rtol=1.0e-11, atol=1.0e-12)
    assert result["active_count"] == 32 * 32
    assert np.isclose(result["activity_fraction"], 0.05)


def test_activity_fraction_is_explicit_and_validated() -> None:
    fields = _synthetic_fields()
    result = analyze_anti_fourier(fields, activity_fraction=0.1)
    assert np.isclose(result["activity_fraction"], 0.1)
    for invalid in (0.0, 1.0, -0.1, np.nan):
        try:
            analyze_anti_fourier(fields, activity_fraction=invalid)
        except ValueError as error:
            assert "activity_fraction" in str(error)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError(f"invalid activity fraction accepted: {invalid}")


def test_corner_masks_only_change_eligibility() -> None:
    fields = _synthetic_fields(n=40)
    top = corner_eligible_mask(fields["X"], fields["Y"], 0.1, corners="top")
    all_corners = corner_eligible_mask(fields["X"], fields["Y"], 0.1, corners="all")
    assert np.count_nonzero(~top) == 32
    assert np.count_nonzero(~all_corners) == 64
    assert np.all(corner_eligible_mask(fields["X"], fields["Y"], 0.0))


def test_velocity_basis_conversion_uses_tensor_rank() -> None:
    fields = _synthetic_fields(n=8)
    fields["vx"][:] = 1.0
    fields["sigma_xy"][:] = 1.0
    fields["qx"][:] = 1.0
    fields["Delta"][:] = 1.0
    converted = convert_velocity_normalization(
        fields, from_velocity_scale_m_s=2.0, to_velocity_scale_m_s=1.0
    )
    assert np.all(converted["rho"] == fields["rho"])
    assert np.all(converted["vx"] == 2.0)
    assert np.all(converted["sigma_xy"] == 4.0)
    assert np.all(converted["qx"] == 8.0)
    assert np.all(converted["Delta"] == 16.0)


def test_rana_metrics_and_cross_model_centerline_semantics() -> None:
    n = 11
    y = np.linspace(0.0, 1.0, n)
    state = _equilibrium(n)
    lid = 0.2
    state[..., 1] = lid * y[:, None]
    state[-1, :, 7] = -0.01
    metrics = rana_global_metrics(state, lid_velocity=lid)
    assert np.isclose(metrics["D"], math.sqrt(2.0) * 0.01 / lid)
    assert np.isclose(metrics["G"], 0.5)

    sample_y = np.linspace(0.1, 0.9, 9)
    reference = {
        "y_over_L": sample_y,
        "rana_fig3_R13_vx_over_Ulid": sample_y - 0.01,
        "digitized_half_line_uncertainty": np.full(sample_y.shape, 0.02),
    }
    comparison = compare_rana_centerline(state, reference, lid_velocity=lid)
    assert np.allclose(comparison["R26_prediction"], sample_y)
    assert np.isclose(comparison["metrics"]["weighted_bias_R26_minus_R13"], 0.01)
    assert comparison["metrics"]["fraction_within_digitized_line_width"] == 1.0
    assert "R13" in comparison["caveat"] and "validation" in comparison["caveat"]
    assert RANA_REFERENCE_CAVEAT == comparison["caveat"]


def test_all_common_fields_and_line_profiles_are_compared() -> None:
    reference = _synthetic_fields(n=16)
    prediction = {key: value.copy() for key, value in reference.items()}
    prediction["qx"] += 0.25
    rows = compare_common_fields(reference, prediction)
    assert [row["field"] for row in rows] == list(COMMON_FIELDS)
    qx = next(row for row in rows if row["field"] == "qx")
    assert np.isclose(qx["bias_R26_minus_DSMC"], 0.25)
    assert np.isclose(qx["RMSE"], 0.25)
    assert all("diagnostic only" in row["provenance_caveat"] for row in rows)
    assert "single-seed" in DSMC_PROVENANCE_CAVEAT

    profile = line_profile_rows(reference, prediction, orientation="vertical", location=0.5)
    assert len(profile) == 16
    for name in COMMON_FIELDS:
        assert f"DSMC_{name}" in profile[0]
        assert f"R26_{name}" in profile[0]
        assert f"R26_minus_DSMC_{name}" in profile[0]
