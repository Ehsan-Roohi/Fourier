from dataclasses import replace

from r26_cases import gu_asme2009_cavity_case, jfm_maxwell_cavity_case
from r26_gu_emerson_jfm_same_grid_gate import (
    JFM_N28_REFERENCE_STATE_SHA256,
    jsonable,
    require_jfm_n28_case,
    run_jfm_n28_same_grid_gate,
)


def _case():
    return jfm_maxwell_cavity_case(
        28,
        kn=0.2,
        lid_speed_m_per_s=100.0,
        wall_temperature_K=300.0,
        grid_stretch_beta=0.0,
    )


def test_jfm_n28_case_lock_accepts_only_the_exact_maxwell_target() -> None:
    require_jfm_n28_case(_case())
    invalid = (
        jfm_maxwell_cavity_case(16, kn=0.2),
        jfm_maxwell_cavity_case(28, kn=0.1),
        jfm_maxwell_cavity_case(28, kn=0.2, grid_stretch_beta=1.0),
        gu_asme2009_cavity_case(28, kn=0.2, lid_speed_m_per_s=100.0),
        replace(_case(), lid_velocity=0.0),
    )
    for case in invalid:
        try:
            require_jfm_n28_case(case)
        except ValueError:
            pass
        else:
            raise AssertionError("off-contract N28 case must be rejected")


def test_jfm_n28_frozen_reference_hash_is_exact() -> None:
    assert (
        JFM_N28_REFERENCE_STATE_SHA256
        == "e1fb8c5696351f0409c3a7cf984bfd4c99a25dbc79f82bd944655cfa21467ff4"
    )


def test_jfm_n28_json_record_preserves_boolean_types() -> None:
    record = jsonable({"passed": True, "blocked": False, "count": 1})
    assert record["passed"] is True
    assert record["blocked"] is False
    assert type(record["count"]) is int


def test_jfm_n28_gate_rejects_a_different_state_before_any_acceptance_work() -> None:
    case = _case()
    state = case.equilibrium_state()
    try:
        run_jfm_n28_same_grid_gate(
            state,
            case.x,
            case.y,
            case=case,
            source_commit="0" * 40,
        )
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("non-reference N28 state must fail closed")
