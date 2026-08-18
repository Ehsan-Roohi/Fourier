#!/usr/bin/env python3
"""Validate one generated or completed Maxwell-VSS JFM cavity case."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATOR_PATH = SCRIPT_DIR / "generate_jfm_maxwell_kngu020_case.py"
SPEC = importlib.util.spec_from_file_location("maxwell_generator", GENERATOR_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError(f"cannot import {GENERATOR_PATH}")
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def close(actual: object, expected: float, label: str) -> None:
    if not math.isclose(float(actual), expected, rel_tol=2.0e-14, abs_tol=0.0):
        raise ValueError(f"{label}: got {actual!r}, expected {expected!r}")


def validate_case(case_dir: Path, expected_kn_gu: float, require_final: bool) -> Path | None:
    metadata_path = case_dir / "case_metadata.json"
    deck_path = case_dir / "in.cavity"
    vss_path = case_dir / "maxwell.vss"
    for path in (metadata_path, deck_path, vss_path, case_dir / "argon.species"):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    exact = {
        "kn_convention": "gu_lambda_over_L",
        "nx": 160,
        "ny": 160,
        "particles_per_cell": 256,
        "warmup_steps": 40_000,
        "sample_steps": 200_000,
        "sample_stride": 10,
        "accumulated_samples_per_cell": 20_000,
        "viscosity_index": 1.0,
        "vss_alpha": 2.14,
        "dump_schema_version": "maxwell_vss_antifourier_v1_15_fields",
        "dump_field_count": 15,
        "dump_columns": generator.DUMP_COLUMNS,
    }
    bad = {
        key: (metadata.get(key), value)
        for key, value in exact.items()
        if metadata.get(key) != value
    }
    if bad:
        raise ValueError(f"metadata contract failed: {bad}")

    close(metadata["kn_gu"], expected_kn_gu, "kn_gu")
    close(metadata["kn_gu_reconstructed"], expected_kn_gu, "kn_gu_reconstructed")
    close(
        metadata["mean_free_path_gu_m"],
        expected_kn_gu * float(metadata["length_m"]),
        "mean_free_path_gu_m",
    )
    close(
        metadata["diameter_vss_input_m"],
        generator.DIAMETER_VSS_INPUT,
        "diameter_vss_input_m",
    )
    close(
        metadata["diameter_vhs_viscosity_equivalent_m"],
        generator.DIAMETER_VHS_EQUIVALENT,
        "diameter_vhs_viscosity_equivalent_m",
    )

    expected_values = generator.physical_parameters(
        expected_kn_gu,
        float(metadata["length_m"]),
        int(metadata["nx"]),
        int(metadata["particles_per_cell"]),
        float(metadata["wall_temperature_K"]),
    )
    for key in ("number_density_m-3", "fnum", "dx_over_lambda_gu"):
        close(metadata[key], float(expected_values[key]), key)

    moment = metadata.get("moment_sampling", {})
    if moment.get("instantaneous_COM_sonine") is not True:
        raise ValueError("sonine/grid instantaneous-COM limitation is not recorded")
    if moment.get("sonine_role") != "diagnostic_only":
        raise ValueError("sonine/grid columns must be labelled diagnostic_only")
    if moment.get("quantitative_R_or_Delta_claim_ready") is not False:
        raise ValueError("raw sonine/grid B1 cannot support an R/Delta claim")
    if moment.get("direct_rank3_moment_m_ijk_available") is not False:
        raise ValueError("upstream SPARTA has no direct full m_ijk sampler")

    words = vss_path.read_text(encoding="utf-8").splitlines()[-1].split()
    if len(words) != 5 or words[0] != "Ar":
        raise ValueError("malformed Maxwell VSS parameter row")
    close(words[1], generator.DIAMETER_VSS_INPUT, "VSS input diameter")
    close(words[2], 1.0, "VSS omega")
    close(words[3], generator.TEMPERATURE_REF, "VSS Tref")
    close(words[4], 2.14, "VSS alpha")

    deck = deck_path.read_text(encoding="utf-8")
    required_deck_lines = (
        "collide              vss gas maxwell.vss",
        "compute              thermal thermal/grid all gas temp",
        "compute              heat eflux/grid all gas heatx heaty",
        "compute              stress pflux/grid all gas momxx momxy momyy momzz",
        "compute              sonine sonine/grid all gas b xx 1 b xy 1 b yy 1 b zz 1",
        "c_flow[*] c_thermal[*] c_heat[*] c_stress[*] c_sonine[*]",
    )
    for required in required_deck_lines:
        if required not in deck:
            raise ValueError(f"deck schema missing: {required}")

    finals = sorted(case_dir.glob("grid.final.*"))
    if not finals:
        if require_final:
            raise FileNotFoundError("grid.final.*")
        return None
    final = finals[-1]
    with final.open(encoding="utf-8") as handle:
        header = next(
            (line.strip() for line in handle if line.startswith("ITEM: CELLS")),
            "",
        )
    expected_fields = [f"f_fieldavg[{index}]" for index in range(1, 16)]
    if header.split() != ["ITEM:", "CELLS", "id", "xc", "yc", *expected_fields]:
        raise ValueError(f"unexpected final-grid schema: {header}")
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--kn-gu", type=float, required=True)
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args()
    final = validate_case(args.case_dir.resolve(), args.kn_gu, args.require_final)
    print(
        f"MAXWELL_KNGU_CASE_VALIDATION_PASS kn_gu={args.kn_gu:.17g} "
        f"final={final if final else 'not-required'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
