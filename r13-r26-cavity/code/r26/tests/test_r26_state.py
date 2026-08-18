from __future__ import annotations

import numpy as np

from r26_state import (
    NVAR,
    STATE_ORDER,
    astr_pack_planar_state,
    pack_stf2,
    pack_stf3,
    pack_stf4,
    planar_state_to_tensors,
    tensors_to_planar_state,
    unpack_stf2,
    unpack_stf3,
    unpack_stf4,
)
from r26_tensor_closures import stf2_project, stf3_project, stf4_project


def _assert_symmetric(a: np.ndarray, rank: int, atol: float = 2.0e-13) -> None:
    for i in range(rank - 1):
        assert np.allclose(a, np.swapaxes(a, -(i + 1), -(i + 2)), rtol=0.0, atol=atol)


def test_state_order_is_the_audited_17_component_planar_order() -> None:
    assert NVAR == 17
    assert STATE_ORDER == (
        "rho",
        "vx",
        "vy",
        "theta",
        "qx",
        "qy",
        "sigma_xx",
        "sigma_xy",
        "sigma_yy",
        "R_xx",
        "R_xy",
        "R_yy",
        "m_xxx",
        "m_xxy",
        "m_xyy",
        "m_yyy",
        "Delta",
    )


def test_planar_state_round_trip_and_z_trace_components() -> None:
    u = np.arange(1.0, 18.0)
    u[0] = 1.25
    u[3] = 0.9
    tensors = planar_state_to_tensors(u)
    assert tensors.sigma[2, 2] == -u[6] - u[8]
    assert tensors.R[2, 2] == -u[9] - u[11]
    assert tensors.m[0, 2, 2] == -u[12] - u[14]
    assert tensors.m[1, 2, 2] == -u[13] - u[15]
    assert np.allclose(tensors_to_planar_state(tensors), u, rtol=0.0, atol=0.0)


def test_astr_component_mapping_is_explicit_and_exact() -> None:
    u = np.arange(1.0, 18.0)
    u[0] = 1.0
    u[3] = 4.0
    packed = astr_pack_planar_state(u)
    assert np.array_equal(packed.velocity, [2.0, 3.0, 0.0])
    assert np.array_equal(packed.heat_flux, [5.0, 6.0, 0.0])
    assert np.array_equal(packed.sigma5, [7.0, 8.0, 0.0, 9.0, 0.0])
    assert np.array_equal(packed.R5, [10.0, 11.0, 0.0, 12.0, 0.0])
    assert np.array_equal(packed.m7, [13.0, 14.0, 0.0, 15.0, 16.0, 0.0, 0.0])


def test_full_3d_stf_pack_unpack_rank2_rank3_rank4() -> None:
    rng = np.random.default_rng(72126)
    a2 = stf2_project(rng.normal(size=(4, 3, 3)))
    a3 = stf3_project(rng.normal(size=(4, 3, 3, 3)))
    a4 = stf4_project(rng.normal(size=(4, 3, 3, 3, 3)))

    b2 = unpack_stf2(pack_stf2(a2))
    b3 = unpack_stf3(pack_stf3(a3))
    b4 = unpack_stf4(pack_stf4(a4))
    assert np.allclose(b2, a2, rtol=0.0, atol=8.0e-13)
    assert np.allclose(b3, a3, rtol=0.0, atol=8.0e-13)
    assert np.allclose(b4, a4, rtol=0.0, atol=1.5e-12)

    _assert_symmetric(b2, 2)
    _assert_symmetric(b3, 3)
    _assert_symmetric(b4, 4)
    assert np.max(np.abs(np.einsum("...iik->...k", b3))) < 1.0e-12
    assert np.max(np.abs(np.einsum("...iikl->...kl", b4))) < 2.0e-12

