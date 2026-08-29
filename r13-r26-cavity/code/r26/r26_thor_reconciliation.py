#!/usr/bin/env python3
"""Fail-closed helpers for reconciling the accepted THOR/legacy R26 roots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from r26_thor_audit import state_sha256
from r26_state import validate_planar_state


# File hashes are hashes of the immutable ``.npz`` bytes, not hashes of the
# decoded state array.  N27 is deliberately retained even though its bytes are
# absent from the uploaded diagnostic ZIP: the original Unity directory must
# supply the exact file that the independent validator accepted.
EXPECTED_ROOT_FILE_SHA256 = {
    24: "94924cbf73d367418f32042f25e80fe1fc5d84aa572672776ef1728ed612d411",
    25: "1ab75d87bc21f37d5658d8c2d728eb909ce758bada0f205287beb7d6f2f18a13",
    27: "b64fcadcf8e7c0e17e1f07df348f5cf619f152662ff606024e11404c79010905",
    28: "a28951fed0063f66dac5e0ec481a108f9a4599fdcfd83d6865b2a160ffa4a409",
}

# Independently reproduced hashes of the decoded little-endian float64 state.
# The N27 value is intentionally not guessed; its file hash is the available
# immutable identity and its decoded-state hash is recorded by the new audit.
EXPECTED_ROOT_STATE_SHA256 = {
    24: "0afe39116e8176818b2bdb8e0092091743838b05f9716230f5a22be2cf4faae1",
    25: "cf8c40482f1c54a183724c7a82a37334fa54697faf1e517a1635075c72695452",
    28: "e1fb8c5696351f0409c3a7cf984bfd4c99a25dbc79f82bd944655cfa21467ff4",
}


@dataclass(frozen=True)
class ImmutableRoot:
    nodes: int
    path: Path
    state: np.ndarray
    x: np.ndarray
    y: np.ndarray
    lid_velocity: float
    kn_input: float
    beta: float
    file_sha256: str
    state_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_native(value: object) -> object:
    """Recursively convert NumPy JSON leaves without hiding unknown types."""

    if isinstance(value, np.ndarray):
        return json_native(value.tolist())
    if isinstance(value, np.generic):
        return json_native(value.item())
    if isinstance(value, dict):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported JSON diagnostic type: {type(value).__name__}")


def load_immutable_root(
    path: Path,
    *,
    nodes: int,
    expected_file_sha256: str,
    require_accepted_flag: bool = True,
) -> ImmutableRoot:
    """Load one explicitly accepted, byte-locked R26 root."""

    if not path.is_file():
        raise ValueError(f"state archive missing: {path}")
    actual_file_sha256 = sha256_file(path)
    if actual_file_sha256 != expected_file_sha256:
        raise ValueError(
            f"N{nodes} state file hash mismatch: {actual_file_sha256}"
        )
    with np.load(path, allow_pickle=False) as archive:
        required = {"state", "x", "y", "lid_velocity", "kn_input", "beta"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"N{nodes} state keys missing: {missing}")
        if require_accepted_flag and "accepted" not in archive.files:
            raise ValueError(f"N{nodes} state acceptance flag missing")
        if "accepted" in archive.files:
            accepted = np.asarray(archive["accepted"])
            if accepted.shape != () or accepted.dtype.kind != "b":
                raise ValueError(f"N{nodes} state acceptance flag is not scalar boolean")
            if not bool(accepted.item()):
                raise ValueError(f"N{nodes} state is explicitly rejected")
        state = validate_planar_state(np.asarray(archive["state"], dtype=float))
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        lid_velocity = float(np.asarray(archive["lid_velocity"]).item())
        kn_input = float(np.asarray(archive["kn_input"]).item())
        beta = float(np.asarray(archive["beta"]).item())
    if state.shape != (nodes, nodes, 17):
        raise ValueError(f"N{nodes} state shape mismatch: {state.shape}")
    if x.shape != (nodes,) or y.shape != (nodes,):
        raise ValueError(f"N{nodes} coordinate shape mismatch")
    if not np.isfinite(state).all() or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"N{nodes} state or coordinates contain NaN/Inf")
    return ImmutableRoot(
        nodes=nodes,
        path=path.resolve(),
        state=state,
        x=x,
        y=y,
        lid_velocity=lid_velocity,
        kn_input=kn_input,
        beta=beta,
        file_sha256=actual_file_sha256,
        state_sha256=state_sha256(state),
    )


def n16_n24_profile_envelope(record: dict[str, object]) -> float:
    """Return the observed N16->N24 THOR grid-difference envelope."""

    pairs = record.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("grid-sensitivity pairs are missing")
    matches = [
        row
        for row in pairs
        if isinstance(row, dict)
        and int(row.get("coarse_nodes", -1)) == 16
        and int(row.get("fine_nodes", -1)) == 24
    ]
    if len(matches) != 1:
        raise ValueError("exactly one N16->N24 grid-sensitivity pair is required")
    value = float(matches[0].get("maximum_normalized_rms_difference", float("nan")))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("invalid N16->N24 profile envelope")
    return value


def ladder_comparison_passed(
    report: dict[str, object],
    *,
    maximum_profile_nrms: float,
    maximum_dg_relative_difference: float,
) -> bool:
    """Apply the inherited grid envelope and existing 2% D/G gate."""

    return bool(
        float(report.get("maximum_normalized_rms_difference", float("inf")))
        <= maximum_profile_nrms
        and float(report.get("D_relative_difference", float("inf")))
        <= maximum_dg_relative_difference
        and float(report.get("G_relative_difference", float("inf")))
        <= maximum_dg_relative_difference
    )


def same_grid_cross_solver_passed(
    report: dict[str, object],
    *,
    maximum_profile_nrms: float,
    maximum_line_nrms: float,
    maximum_dg_relative_difference: float,
) -> bool:
    """Apply the already-established N8/N16 cross-solver thresholds."""

    return bool(
        float(report.get("maximum_normalized_rms_difference", float("inf")))
        <= maximum_profile_nrms
        and float(report.get("maximum_line_normalized_rms_difference", float("inf")))
        <= maximum_line_nrms
        and float(report.get("D_relative_difference", float("inf")))
        <= maximum_dg_relative_difference
        and float(report.get("G_relative_difference", float("inf")))
        <= maximum_dg_relative_difference
    )


__all__ = [
    "EXPECTED_ROOT_FILE_SHA256",
    "EXPECTED_ROOT_STATE_SHA256",
    "ImmutableRoot",
    "json_native",
    "ladder_comparison_passed",
    "load_immutable_root",
    "n16_n24_profile_envelope",
    "same_grid_cross_solver_passed",
    "sha256_file",
]
