#!/usr/bin/env python3
"""Unit tests for independent THOR numerical-rank/profile checks."""

from __future__ import annotations

import numpy as np

from r26_cases import jfm_maxwell_cavity_case
from r26_thor_audit import compare_cross_solver_profiles, scaled_singular_spectrum


def test_scaled_singular_spectrum_detects_full_rank_and_singularity() -> None:
    singular, reciprocal = scaled_singular_spectrum(np.diag((1.0e-9, 2.0, 5.0e7)))
    assert singular[-1] > 0.0
    assert np.isclose(reciprocal, 1.0)
    singular, reciprocal = scaled_singular_spectrum(
        np.asarray(((1.0, 2.0), (2.0, 4.0)))
    )
    assert singular[-1] < 1.0e-12
    assert reciprocal < 1.0e-12


def test_identical_state_cross_solver_profiles_are_exact() -> None:
    case = jfm_maxwell_cavity_case(8, kn=0.2, lid_speed_m_per_s=100.0)
    state = case.equilibrium_state()
    report = compare_cross_solver_profiles(
        state,
        case.x,
        case.y,
        state.copy(),
        case.x,
        case.y,
        lid_velocity=case.lid_velocity,
        target_n=16,
    )
    assert report["maximum_normalized_rms_difference"] == 0.0
    assert report["maximum_line_normalized_rms_difference"] == 0.0
    assert report["D_relative_difference"] == 0.0
    assert report["G_relative_difference"] == 0.0
