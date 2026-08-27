#!/usr/bin/env python3
"""Authenticate and classify the supplied Rana Code_Saturne R13/R26 sources.

This is a source/provenance audit, not a numerical comparison.  The supplied
archive contains no mesh or result files, and its active R26 configuration is
Kn=0.02, Re=50 rather than the present Kn=0.2, 100 m/s Maxwell target.  The
script therefore authorizes architectural reuse only and fails closed on any
hash or required-field mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "r13": {
        "cs_user_modules.f90": "c510471f5f9dcb53c88b66e50aaf69f82fda1263379183c761dd3f1643014811",
        "cs_user_parameters_R13_base.c": "7c888d861916ed699239c5356cafb323da526ff986a4fd013415ed8080588ff3",
        "cs_user_parameters_R13.f90": "1bbfd66d8c3200447bea35eff2de2449463572c5c02ae340cf107b404aa8c872",
        "cs_user_initialization_R13.f90": "ad515ba56af9dc07d46dc2a75c9026658818d31ace4d65d9019de7e6bfe6b91b",
        "cs_user_physical_properties_R13.f90": "de9fa9399347f908e03a81eb3a630df165f36fb36f9ab38fd0c25167e095c3a1",
        "cs_user_source_terms_R13.f90": "84fce21ce8e9a0fe7cc83e735e5e800c9f8dee2586e798234258993c6cda9c04",
        "cs_user_boundary_conditions_R13.f90": "4aee44dff509554078891fca069a02275f463454c321da31551c0d1a42e3d54a",
    },
    "r26": {
        "cs_user_modules.f90": "d92e0142776d90499e2beea4a8b3b37b590597f66b61f43bb49f58ade73a884b",
        "cs_user_parameters_R26_base.c": "f9595eae382014845daa338dcde76c447341c6f395018c22a4c573d6d676a578",
        "cs_user_parameters_R26.f90": "0ce53e0811b00154fc0b3c7cb370cfce92a9382d53741a04758499c8132a13ca",
        "cs_user_initialization_R26.f90": "f1e2f5ca8ec7323bfaebdce5112b21b5e435f17f184324aa807502f379762798",
        "cs_user_physical_properties_R26.f90": "a01d309692acf26093c65aa4c11453afc07f3f98b7be1bb8f2c1ea7ba2e44d5d",
        "cs_user_source_terms_R26.f90": "6e0698bb73d1428572384877a9ac15764731351cb50277d36d5450f1e91af426",
        "cs_user_boundary_conditions_R26.f90": "d65ca2755480e8145ea8b54381ad5c90892d62398c19b121ae9826e3aae0e87c",
    },
}

R26_FIELDS = (
    "stress_xx",
    "stress_yy",
    "stress_xy",
    "stress_xz",
    "stress_yz",
    "heatflux_x",
    "heatflux_y",
    "heatflux_z",
    "mxxx",
    "myyy",
    "mxxy",
    "mxyy",
    "mxyz",
    "mxxz",
    "myyz",
    "rr_xx",
    "rr_yy",
    "rr_xy",
    "rr_xz",
    "rr_yz",
    "delta",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_directory(kind: str, directory: Path) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    files: dict[str, object] = {}
    for name, expected in EXPECTED[kind].items():
        path = directory / name
        actual = sha256(path) if path.is_file() else None
        matches = actual == expected
        files[name] = {"expected_sha256": expected, "actual_sha256": actual, "matches": matches}
        if not matches:
            failures.append(f"{kind}:{name}:sha256")
    return files, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r13-dir", type=Path, required=True)
    parser.add_argument("--r26-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    r13_files, failures = audit_directory("r13", args.r13_dir)
    r26_files, r26_failures = audit_directory("r26", args.r26_dir)
    failures.extend(r26_failures)

    base_source = (args.r26_dir / "cs_user_parameters_R26_base.c").read_text(errors="replace")
    parameters = (args.r26_dir / "cs_user_parameters_R26.f90").read_text(errors="replace")
    properties = (args.r26_dir / "cs_user_physical_properties_R26.f90").read_text(errors="replace")
    missing_fields = [name for name in R26_FIELDS if f'"{name}"' not in base_source]
    failures.extend(f"r26:missing_field:{name}" for name in missing_fields)
    configuration_checks = {
        "kn_0p02": "Kn_num = 0.02d0" in parameters,
        "re_50": "Rey_num = 50.0" in parameters,
        "property_relaxation_2e_minus_4": "rxfp = 0.2d-3" in properties,
        "sutherland_reference_273K": "varam = 273.0d0" in properties,
        "sutherland_constant_144K": "varcm = 144.0d0" in properties,
    }
    failures.extend(
        f"r26:configuration:{name}"
        for name, passed in configuration_checks.items()
        if not passed
    )
    passed = not failures
    report = {
        "status": "RANA_CODE_SATURNE_SOURCE_AUDIT_PASS" if passed else "RANA_CODE_SATURNE_SOURCE_AUDIT_FAIL",
        "r13_files": r13_files,
        "r26_files": r26_files,
        "r26_required_fields": list(R26_FIELDS),
        "r26_missing_fields": missing_fields,
        "active_r26_configuration": {
            "Kn": 0.02,
            "Re": 50.0,
            "property_update_relaxation": 2.0e-4,
            "checks": configuration_checks,
        },
        "architecture_reuse_authorized": passed,
        "numerical_profile_validation_authorized": False,
        "numerical_validation_blockers": [
            "the supplied archive contains source but no mesh or result fields",
            "its active R26 case is Kn=0.02 and Re=50, not Kn=0.2 and a 100 m/s lid",
            "the archive does not explicitly select CUBISTA in its active user parameters",
        ],
        "failures": failures,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
