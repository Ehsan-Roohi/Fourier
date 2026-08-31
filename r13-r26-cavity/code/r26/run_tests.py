#!/usr/bin/env python3
"""Dependency-free runner for the private R26 unit tests."""

from __future__ import annotations

import inspect
import unittest

from tests import (
    test_r26_bulk_equations,
    test_r26_arclength,
    test_r26_checkerboard,
    test_r26_diagnostics,
    test_r26_fv_backend,
    test_r26_gu_emerson_algorithm,
    test_r26_gu_emerson_coupled_jacobian_audit,
    test_r26_gu_emerson_monolithic_oracle,
    test_r26_gu_emerson_reconstruction,
    test_r26_gu_emerson_transformed_fv,
    test_r26_postprocess,
    test_r26_raw_continuation,
    test_r26_solver,
    test_r26_state,
    test_r26_stretched_grid,
    test_r26_tensor_closures,
    test_r26_thor_audit,
    test_r26_thor_solver,
    test_r26_wall_conditions,
    test_maxwell_contract,
)


def suite() -> unittest.TestSuite:
    result = unittest.TestSuite()
    for module in (
        test_r26_state,
        test_r26_arclength,
        test_r26_stretched_grid,
        test_r26_tensor_closures,
        test_r26_thor_solver,
        test_r26_thor_audit,
        test_r26_bulk_equations,
        test_r26_checkerboard,
        test_r26_wall_conditions,
        test_r26_solver,
        test_r26_raw_continuation,
        test_r26_fv_backend,
        test_r26_gu_emerson_algorithm,
        test_r26_gu_emerson_coupled_jacobian_audit,
        test_r26_gu_emerson_transformed_fv,
        test_r26_gu_emerson_monolithic_oracle,
        test_r26_gu_emerson_reconstruction,
        test_r26_diagnostics,
        test_r26_postprocess,
    ):
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_") and len(inspect.signature(function).parameters) == 0:
                result.addTest(unittest.FunctionTestCase(function, description=f"{module.__name__}.{name}"))
    result.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_maxwell_contract))
    return result


if __name__ == "__main__":
    outcome = unittest.TextTestRunner(verbosity=2).run(suite())
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
