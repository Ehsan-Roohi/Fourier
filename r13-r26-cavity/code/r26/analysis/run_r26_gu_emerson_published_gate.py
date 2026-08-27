#!/usr/bin/env python3
"""Validate only the numerical algorithm that Gu--Emerson actually publish.

This is a source-locked N8/N16 gate for equations (48)--(63) and the
segregated stage order in JFM 636 section 5.2.  It is deliberately not a
cavity-production solver: numerical controls absent from the papers remain
unresolved and every refined-grid/production authorization is false.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from r26_cases import gu_asme2009_cavity_case
from r26_gu_emerson_algorithm import (
    GU_EMERSON_STAGE_ORDER,
    GuEmersonAlgorithmDisclosure,
    GuEmersonSegregatedOperators,
    advance_gu_emerson_outer_iteration,
    gu_emerson_field_equations,
)
from r26_gu_emerson_variables import (
    gu_emerson_fields_as_planar17,
    gu_emerson_fields_from_planar17,
    gu_emerson_fields_from_state,
    state_from_gu_emerson_fields,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smooth_manufactured_state(case: object) -> np.ndarray:
    y, x = np.meshgrid(case.y, case.x, indexing="ij")
    state = case.equilibrium_state()
    state[..., 0] = 1.0 + 0.01 * np.sin(np.pi * x) * np.sin(np.pi * y)
    state[..., 1] = 0.02 * np.sin(np.pi * x) * np.sin(np.pi * y)
    state[..., 2] = -0.015 * np.cos(np.pi * x) * np.sin(np.pi * y)
    state[..., 3] = 1.0 + 0.008 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)
    for component in range(4, state.shape[-1]):
        state[..., component] = (
            2.0e-4
            / (component + 1.0)
            * np.sin((1 + component % 2) * np.pi * x)
            * np.sin((1 + component % 3) * np.pi * y)
        )
    return state


def identity_driver_check(case: object) -> tuple[float, list[str]]:
    state = case.equilibrium_state()
    fields = gu_emerson_fields_from_state(
        state, x=case.x, y=case.y, mu=case.mu(state[..., 3])
    )
    calls: list[str] = []

    def whole(name: str):
        def solve(current):
            calls.append(name)
            return current
        return solve

    def component(name: str, attribute: str):
        def solve(current):
            calls.append(name)
            return np.asarray(getattr(current, attribute)).copy()
        return solve

    def wall(physical: np.ndarray, current):
        if not np.array_equal(physical, state):
            raise RuntimeError("identity published driver changed equilibrium")
        calls.append("wall_boundary_update")
        return replace(current)

    result = advance_gu_emerson_outer_iteration(
        fields,
        GuEmersonSegregatedOperators(
            solve_velocity=whole("velocity"),
            simple_pressure_correction=whole("simple_pressure_correction"),
            solve_temperature=component("temperature", "theta"),
            solve_g=component("g", "g"),
            solve_h=component("h", "h"),
            solve_omega=component("omega", "omega"),
            solve_gamma=component("gamma", "gamma"),
            solve_chi=component("chi", "chi"),
            update_wall_boundaries=wall,
        ),
        x=case.x,
        y=case.y,
        viscosity=case.mu,
    )
    error = float(np.max(np.abs(result.physical_state - state), initial=0.0))
    expected_calls = list(GU_EMERSON_STAGE_ORDER[:8]) + ["wall_boundary_update"]
    if calls != expected_calls:
        raise RuntimeError(f"segregated stage order changed: {calls!r}")
    return error, calls


def check_grid(nodes: int) -> dict[str, object]:
    case = gu_asme2009_cavity_case(
        nodes,
        kn=0.2,
        lid_speed_m_per_s=100.0,
        wall_temperature_K=273.0,
    )
    state = smooth_manufactured_state(case)
    mu = case.mu(state[..., 3])
    fields = gu_emerson_fields_from_state(state, x=case.x, y=case.y, mu=mu)
    rebuilt = state_from_gu_emerson_fields(fields, x=case.x, y=case.y, mu=mu)
    physical_roundtrip = float(np.max(np.abs(rebuilt - state), initial=0.0))
    packed = gu_emerson_fields_as_planar17(fields)
    unpacked = gu_emerson_fields_from_planar17(packed)
    transformed_roundtrip = max(
        float(np.max(np.abs(np.asarray(getattr(unpacked, name)) - np.asarray(getattr(fields, name))), initial=0.0))
        for name in ("rho", "velocity", "theta", "g", "h", "omega", "gamma", "chi")
    )
    equilibrium = case.equilibrium_state()
    equilibrium_fields = gu_emerson_fields_from_state(
        equilibrium,
        x=case.x,
        y=case.y,
        mu=case.mu(equilibrium[..., 3]),
    )
    equilibrium_non_gradient = max(
        float(np.max(np.abs(value), initial=0.0))
        for value in (
            equilibrium_fields.g,
            equilibrium_fields.h,
            equilibrium_fields.omega,
            equilibrium_fields.gamma,
            equilibrium_fields.chi,
        )
    )
    driver_error, calls = identity_driver_check(case)
    passed = bool(
        physical_roundtrip <= 5.0e-11
        and transformed_roundtrip <= 5.0e-15
        and equilibrium_non_gradient <= 5.0e-14
        and driver_error == 0.0
    )
    return {
        "nodes": nodes,
        "case": asdict(case),
        "physical_roundtrip_linf": physical_roundtrip,
        "transformed_storage_roundtrip_linf": transformed_roundtrip,
        "equilibrium_non_gradient_linf": equilibrium_non_gradient,
        "identity_driver_error_linf": driver_error,
        "executed_stage_calls": calls,
        "passed": passed,
    }


def jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if hasattr(value, "value"):
        return value.value
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not args.source_commit or len(args.source_commit) != 40:
        parser.error("source commit must be an immutable 40-character SHA")
    if args.output.exists():
        parser.error("output record already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    disclosure = GuEmersonAlgorithmDisclosure()
    grids = [check_grid(nodes) for nodes in (8, 16)]
    equations = [asdict(item) for item in gu_emerson_field_equations("asme2009-cavity")]
    passed = bool(
        all(item["passed"] is True for item in grids)
        and disclosure.production_authorized is False
        and tuple(item["equation"] for item in equations) == tuple(range(56, 63))
    )
    record = {
        "status": (
            "R26_GU_EMERSON_PUBLISHED_ALGORITHM_GATE_PASSED"
            if passed
            else "R26_GU_EMERSON_PUBLISHED_ALGORITHM_GATE_FAILED"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "reference": "Gu--Emerson JFM 636 (2009), equations (48)-(63), section 5.2",
        "case_reference": "Gu--John--Tang--Emerson ASME HT2009 driven cavity",
        "published_stage_order": list(GU_EMERSON_STAGE_ORDER),
        "field_equations": equations,
        "grids": grids,
        "unresolved_unpublished_controls": list(disclosure.unresolved_controls),
        "published_algorithm_gate_passed": passed,
        "production_authorized": False,
        "n24_authorized": False,
        "n28_authorized": False,
        "n29_authorized": False,
        "n30_authorized": False,
        "next_required_stage": (
            "source every unpublished numerical control from the original THOR implementation "
            "before constructing or running the concrete segregated cavity solver"
        ),
        "source_manifest": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                ROOT / "r26_gu_emerson_variables.py",
                ROOT / "r26_gu_emerson_algorithm.py",
                Path(__file__).resolve(),
            )
        },
    }
    args.output.write_text(json.dumps(jsonable(record), indent=2, sort_keys=True) + "\n")
    print(json.dumps(jsonable(record), sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
