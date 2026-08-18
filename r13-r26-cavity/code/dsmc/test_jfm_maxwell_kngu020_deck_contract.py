#!/usr/bin/env python3
"""Regression tests for the Maxwell-VSS KnGu=0.20 generated input deck."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from validate_jfm_maxwell_kngu020_ensemble_case import production_dump_paths


HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "generate_jfm_maxwell_kngu020_case.py"
VALIDATOR = HERE / "validate_jfm_maxwell_kngu020_ensemble_case.py"


class GeneratedDeckContractTest(unittest.TestCase):
    def generate(self, root: Path) -> Path:
        case = root / "case"
        subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "--output", str(case),
                "--seed", "104729",
                "--kn-gu", "0.20",
                "--nx", "100",
                "--ppc", "8",
                "--warmup", "2000",
                "--sample", "2000",
                "--stride", "10",
                "--checkpoint", "2000",
                "--block", "2000",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return case

    def test_every_fix_reference_is_defined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = self.generate(Path(temporary))
            deck = (case / "in.cavity").read_text(encoding="utf-8")
            fix_ids = {
                fields[1]
                for line in deck.splitlines()
                if (fields := line.split()) and fields[0] == "fix" and len(fields) > 1
            }
            references = set(re.findall(r"\bf_([A-Za-z0-9_]+)\[", deck))
            self.assertEqual(references - fix_ids, set())
            self.assertNotIn("f_fieldavg[*]", deck)
            self.assertIn("restart              2000", deck)

    def test_validator_rejects_undefined_fix_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = self.generate(Path(temporary))
            deck_path = case / "in.cavity"
            deck_path.write_text(
                deck_path.read_text(encoding="utf-8")
                + "dump broken grid all 2000 broken.* id f_missing[*]\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    str(case),
                    "--nx", "100",
                    "--ppc", "8",
                    "--seed", "104729",
                    "--warmup", "2000",
                    "--sample", "2000",
                    "--stride", "10",
                    "--block", "2000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("undefined fix IDs", completed.stderr + completed.stdout)

    def test_step_zero_dump_is_not_counted_as_production(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            for name in (
                "grid.block.00000000",
                "grid.block.00002000",
                "grid.final.00000000",
                "grid.final.00002000",
            ):
                (case / name).write_text("placeholder\n", encoding="utf-8")

            blocks = production_dump_paths(case, "grid.block.", [2000])
            final = production_dump_paths(case, "grid.final.", [2000])
            self.assertEqual([path.name for path in blocks], ["grid.block.00002000"])
            self.assertEqual([path.name for path in final], ["grid.final.00002000"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
