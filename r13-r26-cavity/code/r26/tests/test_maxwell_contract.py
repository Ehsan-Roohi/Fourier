#!/usr/bin/env python3
"""Fail-closed contract tests for the source-locked Maxwell R26 cases."""

from __future__ import annotations

import hashlib
import importlib.util
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
VALIDATOR = R26_ROOT / "tools" / "validate_r26_maxwell_run.py"
GATE_VALIDATOR = R26_ROOT / "tools" / "validate_r26_globalization_gate.py"
N8_N16_GATE_SLURM = R26_ROOT / "hpc" / "r26_kn020_ser_ptc_gate_n8_n16.slurm"
N30_PRODUCTION_SLURM = R26_ROOT / "hpc" / "r26_kn020_ser_ptc_fresh_n30.slurm"
N30_SUBMIT = R26_ROOT / "hpc" / "submit_r26_kn020_ser_ptc_fresh_n30.sh"
N30_ARCLENGTH_SLURM = (
    R26_ROOT / "hpc" / "r26_kn020_n30_pseudo_arclength_rescue.slurm"
)
N30_ARCLENGTH_SUBMIT = (
    R26_ROOT / "hpc" / "submit_r26_kn020_n30_pseudo_arclength_rescue.sh"
)
N30_ARCLENGTH_VALIDATOR = (
    R26_ROOT / "tools" / "validate_r26_arclength_rescue.py"
)
N30_ARCLENGTH_RESUME_SLURM = (
    R26_ROOT / "hpc" / "r26_kn020_n30_arclength_chord_resume.slurm"
)
N30_ARCLENGTH_RESUME_SUBMIT = (
    R26_ROOT / "hpc" / "submit_r26_kn020_n30_arclength_chord_resume.sh"
)
BALANCED_METRIC_GATE_SLURM = (
    R26_ROOT / "hpc" / "r26_kn020_balanced_metric_gate_n8_n16.slurm"
)
BALANCED_METRIC_GATE_SUBMIT = (
    R26_ROOT / "hpc" / "submit_r26_kn020_balanced_metric_gate_n8_n16.sh"
)
BALANCED_METRIC_GATE_VALIDATOR = (
    R26_ROOT / "tools" / "validate_r26_balanced_metric_gate.py"
)
N30_BALANCED_ARCLENGTH_SLURM = (
    R26_ROOT / "hpc" / "r26_kn020_n30_balanced_arclength_rescue.slurm"
)
N30_BALANCED_ARCLENGTH_SUBMIT = (
    R26_ROOT / "hpc" / "submit_r26_kn020_n30_balanced_arclength_rescue.sh"
)
sys.path.insert(0, str(CODE))

from analysis.run_jfm_observability_continuation import jsonable as continuation_jsonable  # noqa: E402
from analysis.run_r26_pseudo_arclength_rescue import jsonable as arclength_jsonable  # noqa: E402
from r26_cases import (  # noqa: E402
    KnudsenConvention,
    SQRT_2_OVER_PI,
    ViscosityKind,
    jfm_maxwell_cavity_case,
)
from analysis.run_r26_balanced_metric_gate import jsonable as metric_gate_jsonable  # noqa: E402
from r26_postprocess import _jsonable as postprocess_jsonable  # noqa: E402
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
    def test_balanced_metric_gate_preserves_boolean_json_types(self) -> None:
        # ``bool`` subclasses ``int`` in Python, so this is an ordering-sensitive
        # regression test for the fail-closed n30_authorized contract.
        serializers = (
            metric_gate_jsonable,
            arclength_jsonable,
            continuation_jsonable,
            postprocess_jsonable,
        )
        for serializer in serializers:
            with self.subTest(serializer=serializer.__module__):
                self.assertIs(serializer(False), False)
                self.assertIs(serializer(True), True)
                self.assertIs(serializer(np.bool_(False)), False)
                self.assertEqual(serializer(0), 0)
                self.assertIs(type(serializer(0)), int)

    def test_n30_balanced_arclength_is_gate_locked_and_fail_closed(self) -> None:
        script = N30_BALANCED_ARCLENGTH_SLURM.read_text()
        submit = N30_BALANCED_ARCLENGTH_SUBMIT.read_text()
        self.assertIn(
            'EXPECTED_BALANCED_GATE_REF="93fd4b55b8932bcfedb36f1c66e90b443e7744e2"',
            script,
        )
        self.assertIn(
            'EXPECTED_PRODUCTION_REF="312ee29799e5fdb4340d1146af5c408d72563d49"',
            script,
        )
        self.assertIn(
            'EXPECTED_FAILED_ARC_REF="380a5cb05f0620813c255a086ca899b4051ea2ae"',
            script,
        )
        self.assertIn("R26_BALANCED_GATE_DIR", script)
        self.assertIn("validate_r26_balanced_metric_gate.py", script)
        self.assertIn("--parameter-metric-fraction 0.5", script)
        self.assertNotIn("--parameter-scale", script)
        self.assertIn("--failed-arclength-dir", script)
        self.assertIn("arc_attempt_003_lid_0.370215710658.npz", script)
        self.assertIn("--maximum-attempts 24", script)
        self.assertIn("--maximum-jacobians 7", script)
        self.assertIn("--maximum-objective-evaluations 6000", script)
        self.assertIn("--expected-parameter-metric-fraction 0.5", script)
        self.assertIn("validate_r26_maxwell_run.py", script)
        self.assertIn("N30_BALANCED_ARCLENGTH_RESCUE_PASSED.json", script)
        self.assertIn("N30_BALANCED_ARCLENGTH_RESCUE_FAILED.json", script)
        self.assertIn('"n30_target_accepted": False', script)
        self.assertNotIn("--nodes 32", script)
        self.assertIn("r26_kn020_n30_balanced_arclength_rescue.slurm", submit)
        self.assertTrue(N30_BALANCED_ARCLENGTH_SLURM.is_file())
        self.assertTrue(N30_BALANCED_ARCLENGTH_SUBMIT.is_file())

    def test_balanced_arclength_metric_gate_is_n8_n16_only(self) -> None:
        script = BALANCED_METRIC_GATE_SLURM.read_text()
        submit = BALANCED_METRIC_GATE_SUBMIT.read_text()
        self.assertIn(
            'EXPECTED_GATE_REF="8cbd874eea68dd475faa3f5e3fb318b49cc0c665"',
            script,
        )
        self.assertIn("validate_r26_globalization_gate.py", script)
        self.assertIn("run_r26_balanced_metric_gate.py", script)
        self.assertIn("--parameter-metric-fraction 0.5", script)
        self.assertIn("BALANCED_METRIC_GATE_PASSED.json", script)
        self.assertIn('"n30_authorized": False', script)
        self.assertNotIn("--nodes 30", script)
        self.assertNotIn("run_r26_pseudo_arclength_rescue.py", script)
        self.assertNotIn("N30_PRODUCTION_PASSED", script)
        self.assertIn("r26_kn020_balanced_metric_gate_n8_n16.slurm", submit)
        self.assertTrue(BALANCED_METRIC_GATE_SLURM.is_file())
        self.assertTrue(BALANCED_METRIC_GATE_SUBMIT.is_file())

    def test_n8_n16_globalization_gate_remains_source_locked(self) -> None:
        script = N8_N16_GATE_SLURM.read_text()
        self.assertIn("run_gate 8", script)
        self.assertIn("run_gate 16", script)
        self.assertIn("--analytic-mass-jacobian", script)
        self.assertIn("--secant-predictor", script)
        self.assertIn("--ser-ptc", script)
        self.assertIn("--max-jacobians 5", script)
        self.assertIn("--smoke-lid 0.001", script)
        self.assertIn("--initial-step 0.04", script)
        self.assertIn("--minimum-step 0.0025", script)
        self.assertNotIn("--initial-state", script)
        self.assertNotIn("--reconcile-initial", script)
        self.assertNotIn("R26_N28_DIR", script)
        self.assertNotIn("run_gate 30", script)

    def test_n30_production_is_fresh_bounded_and_gate_locked(self) -> None:
        script = N30_PRODUCTION_SLURM.read_text()
        self.assertIn('EXPECTED_GATE_REF="8cbd874eea68dd475faa3f5e3fb318b49cc0c665"', script)
        self.assertIn("R26_N16_GATE_DIR", script)
        self.assertIn("validate_r26_globalization_gate.py", script)
        self.assertIn("--expected-source-commit", script)
        self.assertIn("--nodes 30", script)
        self.assertIn("--analytic-mass-jacobian", script)
        self.assertIn("--secant-predictor", script)
        self.assertIn("--ser-ptc", script)
        self.assertIn("--max-jacobians 5", script)
        self.assertIn("--max-objective-evaluations 4000", script)
        self.assertIn("--smoke-lid 0.001", script)
        self.assertIn("--initial-step 0.04", script)
        self.assertIn("--minimum-step 0.0025", script)
        self.assertIn("--require-modern-globalization", script)
        self.assertIn("N30_PRODUCTION_PASSED.json", script)
        self.assertIn("N30_PRODUCTION_FAILED.json", script)
        self.assertIn("_RESULTS.zip", script)
        self.assertNotIn("--initial-state", script)
        self.assertNotIn("--reconcile-initial", script)
        self.assertNotIn("R26_N28_DIR", script)
        self.assertNotIn("last_accepted_state.npz\" --reconcile", script)
        self.assertTrue(N30_SUBMIT.is_file())
        self.assertEqual(
            set((R26_ROOT / "hpc").glob("*n30*.slurm")),
            {
                N30_PRODUCTION_SLURM,
                N30_ARCLENGTH_SLURM,
                N30_ARCLENGTH_RESUME_SLURM,
                N30_BALANCED_ARCLENGTH_SLURM,
            },
        )

    def test_balanced_metric_validator_is_standalone_and_fail_closed(self) -> None:
        environment = dict(__import__("os").environ)
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, str(BALANCED_METRIC_GATE_VALIDATOR), "--help"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--expected-source-commit", result.stdout)
        self.assertIn("--expected-parameter-fraction", result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(BALANCED_METRIC_GATE_VALIDATOR),
                    directory,
                    "--expected-source-commit",
                    "93fd4b55b8932bcfedb36f1c66e90b443e7744e2",
                ],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required gate record missing", result.stderr)

    def test_n30_arclength_rescue_is_bounded_source_locked_and_fail_closed(self) -> None:
        script = N30_ARCLENGTH_SLURM.read_text()
        self.assertIn(
            'EXPECTED_FAILED_REF="312ee29799e5fdb4340d1146af5c408d72563d49"',
            script,
        )
        self.assertIn("N30_PRODUCTION_FAILED.json", script)
        self.assertIn("run_r26_pseudo_arclength_rescue.py", script)
        self.assertIn("--maximum-attempts 24", script)
        self.assertIn("--maximum-jacobians 7", script)
        self.assertIn("--maximum-objective-evaluations 6000", script)
        self.assertIn("landing_seed.npz", script)
        self.assertIn("--reconcile-initial", script)
        self.assertIn("--max-jacobians 8", script)
        self.assertIn("validate_r26_arclength_rescue.py", script)
        self.assertIn("N30_ARCLENGTH_RESCUE_PASSED.json", script)
        self.assertIn("N30_ARCLENGTH_RESCUE_FAILED.json", script)
        self.assertIn("_RESULTS.zip", script)
        self.assertNotIn("R26_N16_GATE_DIR", script)
        self.assertNotIn("R26_N28_DIR", script)
        self.assertTrue(N30_ARCLENGTH_SUBMIT.is_file())

    def test_n30_arclength_chord_resume_reuses_only_the_accepted_root(self) -> None:
        script = N30_ARCLENGTH_RESUME_SLURM.read_text()
        self.assertIn(
            'EXPECTED_PRODUCTION_REF="312ee29799e5fdb4340d1146af5c408d72563d49"',
            script,
        )
        self.assertIn(
            'EXPECTED_FAILED_ARC_REF="380a5cb05f0620813c255a086ca899b4051ea2ae"',
            script,
        )
        self.assertIn("R26_ARC_FAILED_DIR", script)
        self.assertIn("arc_attempt_003_lid_0.370215710658.npz", script)
        self.assertIn("--failed-arclength-dir", script)
        self.assertIn("--maximum-iterations 80", script)
        self.assertIn("--maximum-jacobians 7", script)
        self.assertIn("--maximum-objective-evaluations 6000", script)
        self.assertIn("--pseudo-transient-chord-limit 12", script)
        self.assertIn("--newton-chord-limit 3", script)
        self.assertIn("--expected-failed-arclength-source-commit", script)
        self.assertIn("N30_ARCLENGTH_CHORD_RESUME_PASSED.json", script)
        self.assertIn("N30_ARCLENGTH_CHORD_RESUME_FAILED.json", script)
        self.assertIn("_RESULTS.zip", script)
        self.assertNotIn("--initial-step-factor", script)
        self.assertNotIn("--minimum-step-factor", script)
        self.assertNotIn("--nodes 32", script)
        self.assertTrue(N30_ARCLENGTH_RESUME_SUBMIT.is_file())

    def test_arclength_validator_is_standalone_and_fail_closed(self) -> None:
        environment = dict(__import__("os").environ)
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, str(N30_ARCLENGTH_VALIDATOR), "--help"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--expected-target-lid", result.stdout)
        self.assertIn("--expected-failed-arclength-source-commit", result.stdout)

        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(N30_ARCLENGTH_VALIDATOR),
                    directory,
                    directory,
                    "--expected-target-lid",
                    "0.40032038451271784",
                ],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required artifact missing", result.stderr)

    def test_gate_validator_is_standalone_and_fail_closed(self) -> None:
        environment = dict(__import__("os").environ)
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, str(GATE_VALIDATOR), "--help"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--expected-source-commit", result.stdout)
        self.assertIn("--raw-tolerance", result.stdout)

        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(GATE_VALIDATOR),
                    directory,
                    "--expected-source-commit",
                    "8cbd874eea68dd475faa3f5e3fb318b49cc0c665",
                ],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("N16_GATE_PASSED.json missing", result.stderr)

    def test_rejection_step_clamps_to_the_exact_floor_once(self) -> None:
        spec = importlib.util.spec_from_file_location("r26_continuation_floor", DRIVER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.continuation_step_after_rejection(0.0049, 0.0025), 0.0025)
        self.assertIsNone(module.continuation_step_after_rejection(0.0025, 0.0025))

    def test_collision_relevant_sources_are_hash_matched(self) -> None:
        for name, expected in BASELINE_CORE_HASHES.items():
            self.assertEqual(sha256(CODE / name), expected, name)

    def test_driver_manifest_uses_canonical_public_keys(self) -> None:
        spec = importlib.util.spec_from_file_location("r26_continuation_driver", DRIVER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = module.build_source_manifest()
        for name, expected in BASELINE_CORE_HASHES.items():
            self.assertEqual(manifest[f"code/{name}"], expected, name)

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

    def test_driver_requires_reconciliation_for_restart_already_at_target(self) -> None:
        case = jfm_maxwell_cavity_case(5, kn=0.20, grid_stretch_beta=0.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            restart = root / "restart.npz"
            np.savez_compressed(
                restart,
                state=case.equilibrium_state(),
                x=case.x,
                y=case.y,
                lid_velocity=case.lid_velocity,
                kn_input=case.kn,
                kn_convention=case.kn_convention.value,
                beta=case.grid_stretch_beta,
                accepted=True,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(DRIVER),
                    "--nodes", "5",
                    "--case-family", "jfm-maxwell",
                    "--kn-gu", "0.20",
                    "--beta", "0.0",
                    "--initial-state", str(restart),
                    "--output-dir", str(root / "out"),
                ],
                env={**dict(__import__("os").environ), "PYTHONPATH": str(CODE)},
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be revalidated with --reconcile-initial", result.stderr)

    def test_driver_termination_is_fail_closed(self) -> None:
        spec = importlib.util.spec_from_file_location("r26_continuation_exit", DRIVER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.termination_exit_code("target_accepted"), 0)
        for termination in (
            "not_started",
            "grid_reconciliation_rejected",
            "smoke_rejected",
            "minimum_step_rejected",
        ):
            with self.subTest(termination=termination):
                self.assertNotEqual(module.termination_exit_code(termination), 0)

    def test_validator_requires_declared_grid_beta(self) -> None:
        environment = dict(__import__("os").environ)
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, str(VALIDATOR), "--help"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--expected-beta", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
