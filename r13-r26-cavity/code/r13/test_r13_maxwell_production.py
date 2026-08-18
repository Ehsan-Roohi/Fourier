#!/usr/bin/env python3
"""Exact-rational and floating-point tests for the audited R13 operator."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = (
    ROOT
    / "work/article_final_v2/data/kn020_models/r13_N60"
    / "rana_original_coefficients.py"
)
sys.path.insert(0, str(HERE))

from r13_maxwell_production import (  # noqa: E402
    STATE_ORDER_PRINTED_EQ11,
    STATE_ORDER_SOLVER,
    appendix_a_reduced_production_matrix,
    kn_gu_from_rana,
    kn_rana_from_gu,
    legacy_sqrt_temperature_prefactor,
    maxwell_collision_prefactor,
    production_maxwell,
)
from r13_maxwell_adapter import (  # noqa: E402
    install_candidate,
    load_archived_solver,
    verify_solver_order,
)


def load_legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_r13_coefficients", LEGACY)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load archived coefficient module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPaperMatrixExact(unittest.TestCase):
    def setUp(self) -> None:
        # Non-trivial rational state: every nonlinear term tested below is
        # non-zero and all denominators remain exact.
        self.uq = [
            F(7, 5), F(1, 9), F(-1, 8), F(11, 10), F(2, 7), F(-3, 8),
            F(5, 13), F(-2, 11), F(7, 17), F(1, 19), F(-2, 23),
            F(3, 29), F(1, 31), F(-2, 37), F(3, 41), F(-4, 43), F(5, 47),
        ]
        self.m = appendix_a_reduced_production_matrix(self.uq, exact=True)

    def test_printed_eq11_and_executable_orders_are_explicit(self) -> None:
        self.assertEqual(
            STATE_ORDER_PRINTED_EQ11[12:16],
            ("m_xxx", "m_xyy", "m_xxy", "m_yyy"),
        )
        self.assertEqual(
            STATE_ORDER_SOLVER[12:16],
            ("m_xxx", "m_xxy", "m_xyy", "m_yyy"),
        )

    def test_conserved_rows_are_exactly_zero(self) -> None:
        for row in range(4):
            for value in self.m[row]:
                self.assertEqual(value, F(0))

    def test_exact_rational_mxxx_row(self) -> None:
        rho, theta = self.uq[0], self.uq[3]
        qx, qy = self.uq[4], self.uq[5]
        sxx, sxy = self.uq[6], self.uq[7]
        expected = {
            4: -F(6, 25) * sxx / (theta * rho),
            5: F(4, 25) * sxy / (theta * rho),
            6: -F(4, 25) * qx / (theta * rho),
            7: F(8, 75) * qy / (theta * rho),
            12: F(1, 2),
        }
        for col in range(17):
            self.assertEqual(self.m[12, col], expected.get(col, F(0)))

    def test_exact_rational_rxy_row(self) -> None:
        """Appendix A contains two q-couplings absent from the archive."""
        rho, theta = self.uq[0], self.uq[3]
        qx, qy = self.uq[4], self.uq[5]
        sxx, sxy, syy = self.uq[6], self.uq[7], self.uq[8]
        expected = {
            4: -F(4, 15) * qy / (theta * rho),
            5: -F(4, 15) * qx / (theta * rho),
            6: -F(25, 84) * sxy / rho,
            7: -F(25, 84) * (sxx + syy) / rho,
            8: -F(25, 84) * sxy / rho,
            10: F(5, 24),
        }
        for col in range(17):
            self.assertEqual(self.m[10, col], expected.get(col, F(0)))

    def test_exact_rational_mxxy_row(self) -> None:
        rho, theta = self.uq[0], self.uq[3]
        qx, qy = self.uq[4], self.uq[5]
        sxx, sxy, syy = self.uq[6], self.uq[7], self.uq[8]
        expected = {
            4: -F(16, 75) * sxy / (theta * rho),
            5: -F(2, 75) * (5 * sxx - 2 * syy) / (theta * rho),
            6: -F(4, 45) * qy / (theta * rho),
            7: -F(32, 225) * qx / (theta * rho),
            8: F(8, 225) * qy / (theta * rho),
            13: F(1, 2),
        }
        for col in range(17):
            self.assertEqual(self.m[13, col], expected.get(col, F(0)))

    def test_exact_rational_mxyy_row(self) -> None:
        rho, theta = self.uq[0], self.uq[3]
        qx, qy = self.uq[4], self.uq[5]
        sxx, sxy, syy = self.uq[6], self.uq[7], self.uq[8]
        expected = {
            4: F(2, 75) * (2 * sxx - 5 * syy) / (theta * rho),
            5: -F(16, 75) * sxy / (theta * rho),
            6: F(8, 225) * qx / (theta * rho),
            7: -F(32, 225) * qy / (theta * rho),
            8: -F(4, 45) * qx / (theta * rho),
            14: F(1, 2),
        }
        for col in range(17):
            self.assertEqual(self.m[14, col], expected.get(col, F(0)))

    def test_exact_rational_myyy_row(self) -> None:
        rho, theta = self.uq[0], self.uq[3]
        qx, qy = self.uq[4], self.uq[5]
        sxy, syy = self.uq[7], self.uq[8]
        expected = {
            4: F(4, 25) * sxy / (theta * rho),
            5: -F(6, 25) * syy / (theta * rho),
            7: F(8, 75) * qx / (theta * rho),
            8: -F(4, 25) * qy / (theta * rho),
            15: F(1, 2),
        }
        for col in range(17):
            self.assertEqual(self.m[15, col], expected.get(col, F(0)))


class TestNumericAndLegacyAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.u = np.array(
            [
                1.37, 0.07, -0.04, 1.11, 0.013, -0.021, 0.031, -0.017,
                -0.024, 0.002, -0.003, 0.004, 0.001, -0.002, 0.003,
                -0.004, 0.005,
            ],
            dtype=float,
        )
        self.kn = 0.159576912160573

    def test_kn_contract_for_005_and_020(self) -> None:
        self.assertAlmostEqual(kn_rana_from_gu(0.05), 0.03989422804014327, 15)
        self.assertAlmostEqual(kn_rana_from_gu(0.20), 0.15957691216057307, 15)
        for kn_gu in (0.05, 0.20):
            self.assertAlmostEqual(kn_gu_from_rana(kn_rana_from_gu(kn_gu)), kn_gu, 15)

    def test_maxwell_prefactor_is_temperature_independent(self) -> None:
        u2 = self.u.copy()
        u2[3] = 2.4
        self.assertEqual(
            maxwell_collision_prefactor(self.u, self.kn),
            maxwell_collision_prefactor(u2, self.kn),
        )
        self.assertNotEqual(
            legacy_sqrt_temperature_prefactor(self.u, self.kn),
            legacy_sqrt_temperature_prefactor(u2, self.kn),
        )

    def test_numeric_maxwell_scaling(self) -> None:
        matrix = appendix_a_reduced_production_matrix(self.u)
        actual = production_maxwell(self.u, self.kn)
        np.testing.assert_allclose(actual, (self.u[0] / self.kn) * matrix, rtol=0, atol=0)

    def test_archived_matrix_disagrees_in_rxy_and_third_moment_rows(self) -> None:
        legacy = load_legacy_module()
        prefactor = legacy_sqrt_temperature_prefactor(self.u, self.kn)
        archived_matrix = legacy.production(
            self.u, self.kn, rb=1.0, ra=1.0, ma=1.0
        ) / prefactor
        appendix = appendix_a_reduced_production_matrix(self.u)
        np.testing.assert_allclose(archived_matrix[:10], appendix[:10], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(archived_matrix[11], appendix[11], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(archived_matrix[16], appendix[16], rtol=2e-14, atol=2e-14)
        self.assertGreater(np.count_nonzero(np.abs(archived_matrix[10] - appendix[10]) > 1e-14), 0)
        self.assertGreater(np.count_nonzero(np.abs(archived_matrix[12:16] - appendix[12:16]) > 1e-14), 0)

    def test_archived_prefactor_is_rho_sqrt_theta_over_kn(self) -> None:
        legacy = load_legacy_module()
        archived = legacy.production(self.u, self.kn, rb=1.0, ra=1.0, ma=1.0)
        # The qx relaxation diagonal is exactly 2/3 in the unscaled matrix.
        inferred = archived[4, 4] / (2.0 / 3.0)
        self.assertAlmostEqual(
            inferred,
            self.u[0] * np.sqrt(self.u[3]) / self.kn,
            14,
        )

    def test_adapter_proves_flux_and_wall_order_before_install(self) -> None:
        solver = load_archived_solver()
        evidence = verify_solver_order(solver)
        self.assertEqual(tuple(evidence["state_order"]), STATE_ORDER_SOLVER)
        old = solver.production
        installed = install_candidate(solver)
        self.assertIsNot(solver.production, old)
        self.assertIs(solver.production, production_maxwell)
        self.assertIn("not-run-ready", installed["candidate_status"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
