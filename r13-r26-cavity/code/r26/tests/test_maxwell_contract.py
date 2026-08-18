#!/usr/bin/env python3
"""Fail-closed contract tests for the source-locked Maxwell R26 cases."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


R26_ROOT = Path(__file__).resolve().parents[1]
CODE = R26_ROOT
DRIVER = R26_ROOT / "analysis" / "run_jfm_observability_continuation.py"
sys.path.insert(0, str(CODE))

from r26_cases import (  # noqa: E402
    KnudsenConvention,
    SQRT_2_OVER_PI,
    ViscosityKind,
    jfm_maxwell_cavity_case,
)
from r26_tensor_closures import closure_coefficients  # noqa: E402
from r26_wall_conditions import WallParameters  # noqa: E402


BASELINE_CORE_HASHES = {
    "r26_bulk_equations.py": "9abe3943ce541e6c5243a61893c1428daea30cf8fae42ab3e90c140eb7ba6a06",
    "r26_tensor_closures.py": "13037256b49de8ce0737136c56ab31fa5b1641545a79a65e77c761c25bcbbbea",
    "r26_wall_conditions.py": "b3a7bf0bc4be58f3e0c42928c87f4b01802ae88d50055b58410e485e7bbcdd49",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MaxwellContract(unittest.TestCase):
    def test_collision_relevant_sources_are_hash_matched(self) -> None:
        for name, expected in BASELINE_CORE_HASHES.items():
            self.assertEqual(sha256(CODE / name), expected, name)

    def test_final_jfm2009_coefficients_are_unchanged(self) -> None:
        c = closure_coefficients("jfm2009")
        self.assertEqual((c.C1, c.C2, c.Y1, c.Y2, c.Y3), (2.097, 0.291, 1.698, 1.203, 0.854))

    def test_kn005_and_kn020_are_pure_maxwell(self) -> None:
        theta = np.asarray((0.73, 1.0, 1.31))
        for kn, nodes, beta in ((0.05, 40, 1.25), (0.20, 20, 0.0)):
            with self.subTest(kn=kn):
                case = jfm_maxwell_cavity_case(nodes, kn=kn, grid_stretch_beta=beta)
                self.assertIs(case.kn_convention, KnudsenConvention.GU_MEAN_FREE_PATH)
                self.assertEqual(case.viscosity.kind, ViscosityKind.POWER_LAW)
                self.assertEqual(case.viscosity.exponent, 1.0)
                np.testing.assert_allclose(case.mu(theta) / case.mu_equilibrium, theta, rtol=0.0, atol=2e-15)
                self.assertTrue(math.isclose(case.mu_equilibrium, kn * SQRT_2_OVER_PI, rel_tol=0.0, abs_tol=2e-15))
                self.assertEqual(case.r26_closure_mode, "jfm2009")
                self.assertEqual(case.accommodation, 1.0)
                self.assertEqual(case.wall_temperature, 1.0)
                self.assertTrue(math.isclose(case.lid_velocity, 100.0 / math.sqrt(208.0 * 300.0), rel_tol=0.0, abs_tol=2e-15))
                self.assertIn("Pure Maxwell-molecule", case.provenance)
                self.assertNotIn("VHS", case.provenance)

    def test_diffuse_wall_factor_is_one(self) -> None:
        case = jfm_maxwell_cavity_case(5, kn=0.05)
        wall = WallParameters(
            wall_temperature=case.wall_temperature,
            accommodation=case.accommodation,
            gas_constant=case.gas_constant,
            wall_velocity=case.wall_velocity("top"),
        )
        self.assertEqual(wall.A, 1.0)

    def test_driver_rejects_vhs_override_in_maxwell_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(DRIVER),
                    "--nodes", "5",
                    "--case-family", "jfm-maxwell",
                    "--kn-gu", "0.05",
                    "--vhs-omega", "0.81",
                    "--output-dir", str(Path(directory) / "out"),
                ],
                env={**dict(__import__("os").environ), "PYTHONPATH": str(CODE)},
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("do not pass --vhs-omega", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
