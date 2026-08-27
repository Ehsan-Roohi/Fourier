from __future__ import annotations

from dataclasses import replace

import numpy as np

from r26_cases import rana_first_case
from r26_gu_emerson_algorithm import (
    GU_EMERSON_STAGE_ORDER,
    GuEmersonAlgorithmDisclosure,
    GuEmersonSegregatedOperators,
    NumericalControlSource,
    advance_gu_emerson_outer_iteration,
    gu_emerson_field_equations,
)
from r26_gu_emerson_variables import (
    gu_emerson_fields_as_planar17,
    gu_emerson_fields_from_state,
    gu_emerson_fields_from_planar17,
    state_from_gu_emerson_fields,
)


def _smooth_state(nodes: int = 7) -> tuple[object, np.ndarray]:
    case = rana_first_case(nodes)
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
    return case, state


def test_gu_emerson_equation48_variable_mapping_round_trips_physical_state() -> None:
    case, state = _smooth_state()
    mu = case.mu(state[..., 3])
    fields = gu_emerson_fields_from_state(state, x=case.x, y=case.y, mu=mu)
    rebuilt = state_from_gu_emerson_fields(fields, x=case.x, y=case.y, mu=mu)
    assert np.allclose(rebuilt, state, rtol=2.0e-12, atol=3.0e-12)


def test_gu_emerson_uniform_equilibrium_has_zero_non_gradient_fields() -> None:
    case = rana_first_case(5)
    state = case.equilibrium_state()
    fields = gu_emerson_fields_from_state(
        state, x=case.x, y=case.y, mu=case.mu(state[..., 3])
    )
    for value in (fields.g, fields.h, fields.omega, fields.gamma, fields.chi):
        assert np.array_equal(value, np.zeros_like(value))


def test_gu_emerson_transformed_planar17_storage_round_trip_is_lossless() -> None:
    case, state = _smooth_state()
    fields = gu_emerson_fields_from_state(
        state, x=case.x, y=case.y, mu=case.mu(state[..., 3])
    )
    packed = gu_emerson_fields_as_planar17(fields)
    rebuilt = gu_emerson_fields_from_planar17(packed)
    for name in ("rho", "velocity", "theta", "g", "h", "omega", "gamma", "chi"):
        assert np.allclose(
            getattr(rebuilt, name), getattr(fields, name), rtol=0.0, atol=2.0e-18
        )


def test_gu_emerson_equations_56_to_63_have_the_printed_diffusion_multipliers() -> None:
    equations = gu_emerson_field_equations("jfm2009")
    assert tuple(item.equation for item in equations) == tuple(range(56, 63))
    assert tuple(item.field for item in equations) == (
        "velocity", "temperature", "g", "h", "omega", "gamma", "chi"
    )
    assert np.allclose(
        tuple(item.diffusion_multiplier for item in equations),
        (1.0, 2.0 / 5.0, 3.0 / 2.0, 5.0 / 6.0, 2.097, 7.0 * 1.698 / 9.0, 3.0 / 7.0),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_gu_emerson_disclosure_fails_closed_on_unpublished_controls() -> None:
    disclosure = GuEmersonAlgorithmDisclosure()
    assert not disclosure.production_authorized
    assert "under_relaxation_factors" in disclosure.unresolved_controls
    try:
        disclosure.require_production_authorization()
    except RuntimeError as exc:
        assert "paper alone" in str(exc)
    else:
        raise AssertionError("paper-unspecified numerical controls must fail closed")

    cited = {
        name: NumericalControlSource(value="declared", provenance="external source")
        for name in disclosure.unresolved_controls
    }
    complete = GuEmersonAlgorithmDisclosure(controls=cited)
    complete.require_production_authorization()
    assert complete.production_authorized


def test_gu_emerson_driver_enforces_the_printed_segregated_stage_order() -> None:
    case = rana_first_case(5)
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

    def wall(physical, current):
        assert physical.shape == state.shape
        calls.append("wall_boundary_update")
        return replace(current)

    operators = GuEmersonSegregatedOperators(
        solve_velocity=whole("velocity"),
        simple_pressure_correction=whole("simple_pressure_correction"),
        solve_temperature=component("temperature", "theta"),
        solve_g=component("g", "g"),
        solve_h=component("h", "h"),
        solve_omega=component("omega", "omega"),
        solve_gamma=component("gamma", "gamma"),
        solve_chi=component("chi", "chi"),
        update_wall_boundaries=wall,
    )
    result = advance_gu_emerson_outer_iteration(
        fields,
        operators,
        x=case.x,
        y=case.y,
        viscosity=case.mu,
    )
    assert calls == list(GU_EMERSON_STAGE_ORDER[:8]) + ["wall_boundary_update"]
    assert result.stage_order == GU_EMERSON_STAGE_ORDER
    assert np.array_equal(result.physical_state, state)


def test_gu_emerson_driver_contract_contains_no_continuation_or_monolithic_solver() -> None:
    text = " ".join(GU_EMERSON_STAGE_ORDER).lower()
    for forbidden in ("homotopy", "arclength", "krylov", "jacobian", "newton"):
        assert forbidden not in text


def test_gu_emerson_published_gate_is_source_locked_and_nonproduction() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    slurm = (root / "hpc" / "r26_gu_emerson_published_gate_n8_n16.slurm").read_text()
    submit = (root / "hpc" / "submit_r26_gu_emerson_published_gate_n8_n16.sh").read_text()
    runner = (root / "analysis" / "run_r26_gu_emerson_published_gate.py").read_text()
    assert "R26_GE_PUBLISHED_REF" in slurm and "checkout --detach FETCH_HEAD" in slurm
    assert "run_tests.py" in slurm
    assert "R26_GE_PUBLISHED_REF" in submit and "raw.githubusercontent.com" in submit
    for authorization in ("production_authorized", "n24_authorized", "n28_authorized", "n29_authorized", "n30_authorized"):
        assert f'"{authorization}": False' in runner
    for forbidden in ("solve_r26_thor_bvp", "root(", "homotopy", "pseudo_arclength"):
        assert forbidden not in runner
    assert runner.index("np.bool_, bool") < runner.index("np.integer, int")
