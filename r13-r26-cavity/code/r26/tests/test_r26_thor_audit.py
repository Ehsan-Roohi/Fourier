#!/usr/bin/env python3
"""Unit tests for independent THOR numerical-rank/profile checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import numpy as np

from r26_cases import jfm_maxwell_cavity_case
from r26_thor_audit import compare_cross_solver_profiles, scaled_singular_spectrum
from r26_thor_reconciliation import (
    EXPECTED_ROOT_FILE_SHA256,
    ladder_comparison_passed,
    load_immutable_root,
    n16_n24_profile_envelope,
    same_grid_cross_solver_passed,
)


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


def test_root_ladder_uses_the_observed_n16_n24_envelope() -> None:
    record = {
        "pairs": [
            {
                "coarse_nodes": 8,
                "fine_nodes": 16,
                "maximum_normalized_rms_difference": 0.52,
            },
            {
                "coarse_nodes": 16,
                "fine_nodes": 24,
                "maximum_normalized_rms_difference": 0.200778,
            },
        ]
    }
    assert n16_n24_profile_envelope(record) == 0.200778
    passing = {
        "maximum_normalized_rms_difference": 0.08,
        "D_relative_difference": 0.01,
        "G_relative_difference": 0.001,
    }
    assert ladder_comparison_passed(
        passing,
        maximum_profile_nrms=0.200778,
        maximum_dg_relative_difference=0.02,
    )
    failing = dict(passing, maximum_normalized_rms_difference=0.200779)
    assert not ladder_comparison_passed(
        failing,
        maximum_profile_nrms=0.200778,
        maximum_dg_relative_difference=0.02,
    )


def test_n28_cross_solver_gate_requires_every_declared_metric() -> None:
    report = {
        "maximum_normalized_rms_difference": 0.049,
        "maximum_line_normalized_rms_difference": 0.149,
        "D_relative_difference": 0.019,
        "G_relative_difference": 0.019,
    }
    assert same_grid_cross_solver_passed(
        report,
        maximum_profile_nrms=0.05,
        maximum_line_nrms=0.15,
        maximum_dg_relative_difference=0.02,
    )
    for key in tuple(report):
        incomplete = dict(report)
        incomplete.pop(key)
        assert not same_grid_cross_solver_passed(
            incomplete,
            maximum_profile_nrms=0.05,
            maximum_line_nrms=0.15,
            maximum_dg_relative_difference=0.02,
        )
    assert set(EXPECTED_ROOT_FILE_SHA256) == {24, 25, 27, 28}
    assert all(len(value) == 64 for value in EXPECTED_ROOT_FILE_SHA256.values())


def test_legacy_root_acceptance_uses_external_record_but_rejects_explicit_false() -> None:
    case = jfm_maxwell_cavity_case(8, kn=0.2, lid_speed_m_per_s=100.0)
    state = np.zeros((8, 8, 17), dtype=float)
    state[..., 0] = 1.0
    state[..., 3] = 1.0
    common = {
        "state": state,
        "x": case.x,
        "y": case.y,
        "lid_velocity": case.lid_velocity,
        "kn_input": 0.2,
        "beta": 0.0,
    }
    with tempfile.TemporaryDirectory() as directory:
        legacy = Path(directory) / "legacy.npz"
        np.savez_compressed(legacy, **common)
        digest = hashlib.sha256(legacy.read_bytes()).hexdigest()
        root = load_immutable_root(
            legacy,
            nodes=8,
            expected_file_sha256=digest,
            require_accepted_flag=False,
        )
        assert root.nodes == 8

        rejected = Path(directory) / "rejected.npz"
        np.savez_compressed(rejected, **common, accepted=False)
        rejected_digest = hashlib.sha256(rejected.read_bytes()).hexdigest()
        try:
            load_immutable_root(
                rejected,
                nodes=8,
                expected_file_sha256=rejected_digest,
                require_accepted_flag=False,
            )
        except ValueError as error:
            assert "explicitly rejected" in str(error)
        else:
            raise AssertionError("an explicit rejected state was accepted")
