#!/usr/bin/env python3
"""Private Python reconstruction and archive-derived R13 experiments.

``legacy-border`` faithfully reconstructs the supplied old algorithm.
``conservative-fv`` is an explicitly derived private experiment: it retains
the author's other sixteen equations and wall rows but replaces continuity,
quadrature, and the mass-border column by a conservative shared-face scheme.
It must not be described as the author's original implementation.

Two boundary discretizations are explicit:

``legacy-hobc3``
    Reproduces the three-point high-order boundary rows in the supplied old
    MATLAB program.

``paper-linear``
    Reproduces the two-point linear extrapolation in published Eqs. (15)--(20).
    This switch changes only the boundary stencil; the bulk matrices remain
    the ones in the supplied old program.  It is therefore an audit variant,
    not a claim of a complete paper-exact discretization.

The old program also evaluates the effective wall pressure with the *normal*
stress.  Published Eq. (7) uses the *tangential* stress.  Both are available as
``legacy-normal`` and ``paper-tangential`` so the discrepancy is measured
rather than silently hidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, qmr, splu, spsolve

try:
    from .rana_original_coefficients import (
        NVAR,
        flux_x,
        flux_y,
        production,
        wall_data,
        wall_matrix,
    )
except ImportError:  # Direct script execution.
    from rana_original_coefficients import (
        NVAR,
        flux_x,
        flux_y,
        production,
        wall_data,
        wall_matrix,
    )


STATE_ORDER = (
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
IDX = {name: index for index, name in enumerate(STATE_ORDER)}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one local evidence file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return a SHA-256 digest for the exact bytes that were parsed."""
    return hashlib.sha256(payload).hexdigest()


def json_safe(value: object) -> object:
    """Convert reports to strict RFC-compatible JSON data.

    Python's default encoder writes NaN and Infinity even though neither is
    valid JSON.  Evidence files are fail-closed: every non-finite number is
    represented by ``null`` and serialization uses ``allow_nan=False``.
    """
    if isinstance(value, (bool, np.bool_)) or value is None:
        return bool(value) if value is not None else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, str):
        return value
    raise TypeError(f"unsupported JSON evidence value: {type(value).__name__}")


def _reject_nonfinite_json_constant(token: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {token}")


def strict_json_loads(payload: str) -> object:
    """Parse evidence JSON while rejecting NaN and Infinity tokens."""
    return json.loads(payload, parse_constant=_reject_nonfinite_json_constant)


_CI_VARIABLES = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "TF_BUILD",
    "CIRCLECI",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
)


def _git_worktree_root(path: Path) -> Path | None:
    """Return the nearest containing Git worktree, without invoking Git."""
    candidate = Path(path).resolve()
    if candidate.exists() and candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        marker = parent / ".git"
        directory_marker = marker.is_dir() and (marker / "HEAD").is_file()
        file_marker = False
        if marker.is_file():
            try:
                file_marker = marker.read_text(errors="replace").lstrip().startswith(
                    "gitdir:"
                )
            except OSError:
                file_marker = False
        if directory_marker or file_marker:
            return parent
    return None


def assert_private_local_output(output_path: Path) -> dict[str, object]:
    """Reject CI execution and output paths inside a Git worktree."""
    active_ci = [name for name in _CI_VARIABLES if os.environ.get(name)]
    if active_ci:
        raise RuntimeError(
            "private-local execution refused in CI environment: "
            + ", ".join(active_ci)
        )
    worktree = _git_worktree_root(Path(output_path))
    if worktree is not None:
        raise ValueError(
            "private-local output must be outside every Git worktree"
        )
    return {
        "execution_intent": "private-local",
        "ci_environment_rejected": True,
        "active_ci_variables": [],
        "output_outside_git_worktree": True,
    }


def runtime_environment() -> dict[str, str]:
    """Record the numerical runtime used for one private evidence bundle."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "executable_name": Path(sys.executable).name,
    }


@dataclass(frozen=True)
class RanaOriginalConfig:
    nx: int = 8
    ny: int = 8
    kn: float = 0.5 * np.sqrt(2.0 / np.pi)
    lid_velocity: float = 50.0 / np.sqrt(208.0 * 273.0)
    accommodation: float = 1.0
    rb: float = 1.0
    ra: float = 1.0
    ma: float = 1.0
    boundary_scheme: str = "legacy-hobc3"
    pressure_mode: str = "legacy-normal"
    # ``legacy-border`` reproduces the supplied MATLAB saddle-point system,
    # including its non-zero Lagrange multiplier.  ``conservative-fv``
    # replaces the continuity rows, their mass quadrature, and border column
    # by a telescoping finite-volume balance with a compatible global
    # saddle-point mass constraint.
    continuity_mode: str = "legacy-border"
    conservative_linearization: str = "defect-newton"
    nonlinear_solver: str = "frozen"
    # A direct solve is the safe default for the small verification grids.  The
    # archive used QMR with rtol=1e-4; those controls can be requested on the
    # command line, but they can stop at the unchanged initial guess and must
    # not be confused with a verified nonlinear solution.
    linear_solver: str = "direct"
    qmr_rtol: float = 1.0e-10
    qmr_maxiter: int = 30000
    outer_tolerance: float = 1.0e-4
    max_outer_iterations: int = 21
    outer_relaxation: float = 1.0
    # ``fixed`` preserves the supplied iteration exactly.  The opt-in
    # residual backtracking is restricted to the private conservative
    # defect-Newton derivative, where a physical residual can be evaluated
    # without allowing the mass-border multiplier to hide a bad state.
    outer_globalization: str = "fixed"
    line_search_reduction: float = 0.5
    line_search_min_step: float = 1.0 / 128.0
    line_search_armijo: float = 1.0e-4
    physical_floor: float = 1.0e-12
    jfnk_fd_relative_step: float = float(np.cbrt(np.finfo(float).eps))
    jfnk_max_fd_halvings: int = 10
    jfnk_gmres_restart: int = 40
    jfnk_gmres_max_cycles: int = 10
    jfnk_initial_forcing: float = 1.0e-2
    jfnk_min_forcing: float = 1.0e-6
    jfnk_max_forcing: float = 1.0e-1
    verification_linear_residual_tolerance: float = 1.0e-8
    verification_fixed_point_residual_tolerance: float = 1.0e-6
    verification_mass_tolerance: float = 1.0e-8
    physical_core_residual_tolerance: float = 1.0e-8
    verification_lagrange_tolerance: float = 1.0e-10
    verification_continuity_residual_tolerance: float = 1.0e-8
    verification_global_balance_tolerance: float = 1.0e-10
    verification_border_source_tolerance: float = 1.0e-10

    def validate(self) -> None:
        if self.nx < 4 or self.ny < 4:
            raise ValueError("nx and ny must each be at least 4")
        if not np.isfinite(self.kn) or self.kn <= 0.0:
            raise ValueError("kn must be positive")
        if self.lid_velocity == 0.0:
            raise ValueError("lid_velocity must be nonzero for the cavity benchmark")
        if not np.isfinite(self.lid_velocity):
            raise ValueError("lid_velocity must be finite")
        if not all(np.isfinite(value) for value in (self.rb, self.ra, self.ma)):
            raise ValueError("rb, ra, and ma must be finite")
        if any(value not in (0.0, 1.0) for value in (self.rb, self.ra, self.ma)):
            raise ValueError("rb, ra, and ma are archive switches and must be 0 or 1")
        if not np.isfinite(self.accommodation) or not 0.0 < self.accommodation <= 1.0:
            raise ValueError("accommodation must be in (0, 1]")
        if self.boundary_scheme not in ("legacy-hobc3", "paper-linear"):
            raise ValueError("unknown boundary scheme")
        if self.pressure_mode not in ("legacy-normal", "paper-tangential"):
            raise ValueError("unknown effective-pressure mode")
        if self.continuity_mode not in ("legacy-border", "conservative-fv"):
            raise ValueError("unknown continuity mode")
        if self.conservative_linearization not in (
            "density-picard",
            "defect-newton",
        ):
            raise ValueError("unknown conservative continuity linearization")
        if self.nonlinear_solver not in ("frozen", "jfnk"):
            raise ValueError("unknown nonlinear solver")
        if self.linear_solver not in ("qmr", "direct"):
            raise ValueError("linear_solver must be qmr or direct")
        if (
            not np.isfinite(self.qmr_rtol)
            or self.qmr_rtol <= 0.0
            or self.qmr_maxiter < 1
        ):
            raise ValueError("invalid QMR controls")
        if (
            not np.isfinite(self.outer_tolerance)
            or self.outer_tolerance <= 0.0
            or self.max_outer_iterations < 1
        ):
            raise ValueError("invalid nonlinear controls")
        if (
            not np.isfinite(self.outer_relaxation)
            or not 0.0 < self.outer_relaxation <= 1.0
        ):
            raise ValueError("outer_relaxation must be in (0, 1]")
        if self.outer_globalization not in ("fixed", "residual-backtracking"):
            raise ValueError("unknown outer globalization")
        if self.outer_globalization == "residual-backtracking" and not (
            self.continuity_mode == "conservative-fv"
            and self.conservative_linearization == "defect-newton"
        ):
            raise ValueError(
                "residual-backtracking requires conservative-fv "
                "with defect-newton linearization"
            )
        if self.nonlinear_solver == "jfnk" and not (
            self.continuity_mode == "conservative-fv"
            and self.conservative_linearization == "defect-newton"
            and self.outer_globalization == "residual-backtracking"
        ):
            raise ValueError(
                "jfnk requires conservative-fv, defect-newton, and "
                "residual-backtracking"
            )
        if (
            not np.isfinite(self.line_search_reduction)
            or not 0.0 < self.line_search_reduction < 1.0
        ):
            raise ValueError("line_search_reduction must be in (0, 1)")
        if (
            not np.isfinite(self.line_search_min_step)
            or not 0.0 < self.line_search_min_step <= 1.0
        ):
            raise ValueError("line_search_min_step must be in (0, 1]")
        if (
            not np.isfinite(self.line_search_armijo)
            or not 0.0 < self.line_search_armijo < 1.0
        ):
            raise ValueError("line_search_armijo must be in (0, 1)")
        if not np.isfinite(self.physical_floor) or self.physical_floor <= 0.0:
            raise ValueError("physical_floor must be positive")
        if (
            not np.isfinite(self.jfnk_fd_relative_step)
            or self.jfnk_fd_relative_step <= 0.0
            or self.jfnk_max_fd_halvings < 0
        ):
            raise ValueError("invalid JFNK finite-difference controls")
        if self.jfnk_gmres_restart < 1 or self.jfnk_gmres_max_cycles < 1:
            raise ValueError("invalid JFNK GMRES controls")
        if not (
            np.isfinite(self.jfnk_min_forcing)
            and np.isfinite(self.jfnk_initial_forcing)
            and np.isfinite(self.jfnk_max_forcing)
            and 0.0
            < self.jfnk_min_forcing
            <= self.jfnk_initial_forcing
            <= self.jfnk_max_forcing
            < 1.0
        ):
            raise ValueError("invalid JFNK forcing controls")
        verification_tolerances = (
            self.verification_linear_residual_tolerance,
            self.verification_fixed_point_residual_tolerance,
            self.verification_mass_tolerance,
            self.physical_core_residual_tolerance,
            self.verification_lagrange_tolerance,
            self.verification_continuity_residual_tolerance,
            self.verification_global_balance_tolerance,
            self.verification_border_source_tolerance,
        )
        if not all(
            np.isfinite(value) and value > 0.0
            for value in verification_tolerances
        ):
            raise ValueError("verification tolerances must be positive")

    @property
    def dx(self) -> float:
        return 1.0 / (self.nx + 1)

    @property
    def dy(self) -> float:
        return 1.0 / (self.ny + 1)


def equilibrium(config: RanaOriginalConfig) -> np.ndarray:
    config.validate()
    result = np.zeros((config.ny, config.nx, NVAR), dtype=float)
    result[..., IDX["rho"]] = 1.0
    result[..., IDX["theta"]] = 1.0
    return result


def legacy_mass_weights(config: RanaOriginalConfig) -> np.ndarray:
    """Exact Python equivalent of the supplied ``wightarray.m`` result."""
    config.validate()
    w = 4.0 * np.ones((config.ny, config.nx), dtype=float)
    if config.ny > 4:
        w[2:-2, 1] = 2.0
        w[2:-2, 0] = 8.0
        w[2:-2, -2] = 2.0
        w[2:-2, -1] = 8.0
    if config.nx > 4:
        w[1, 2:-2] = 2.0
        w[0, 2:-2] = 8.0
        w[-2, 2:-2] = 2.0
        w[-1, 2:-2] = 8.0
    w[0, 0] = w[0, -1] = w[-1, 0] = w[-1, -1] = 16.0
    w[1, 1] = w[1, -2] = w[-2, 1] = w[-2, -2] = 1.0
    return (config.dx * config.dy / 4.0) * w.reshape(-1)


def conservative_volume_weights(config: RanaOriginalConfig) -> np.ndarray:
    """Positive vertex-control-volume weights for the interior nodal grid.

    The first and last control volumes extend from the physical wall to the
    midpoint between the first two interior nodes, so their widths are
    ``3*h/2``.  Interior widths are ``h`` and the tensor-product weights sum
    to the unit-square area exactly.
    """
    config.validate()
    wx = np.full(config.nx, config.dx, dtype=float)
    wy = np.full(config.ny, config.dy, dtype=float)
    wx[[0, -1]] = 1.5 * config.dx
    wy[[0, -1]] = 1.5 * config.dy
    return np.outer(wy, wx).reshape(-1)


def conservative_mass_residual(
    state: np.ndarray, config: RanaOriginalConfig
) -> np.ndarray:
    """Evaluate the nonlinear shared-face FV residual ``div(rho*v)``."""
    config.validate()
    state = np.asarray(state, dtype=float)
    if state.shape != (config.ny, config.nx, NVAR):
        raise ValueError("state shape does not match configuration")
    wx = np.full(config.nx, config.dx, dtype=float)
    wy = np.full(config.ny, config.dy, dtype=float)
    wx[[0, -1]] = 1.5 * config.dx
    wy[[0, -1]] = 1.5 * config.dy
    residual = np.zeros((config.ny, config.nx), dtype=float)
    for j in range(config.ny):
        for i in range(config.nx - 1):
            flux = 0.25 * (
                state[j, i, IDX["rho"]]
                + state[j, i + 1, IDX["rho"]]
            ) * (
                state[j, i, IDX["vx"]]
                + state[j, i + 1, IDX["vx"]]
            )
            residual[j, i] += flux / wx[i]
            residual[j, i + 1] -= flux / wx[i + 1]
    for j in range(config.ny - 1):
        for i in range(config.nx):
            flux = 0.25 * (
                state[j, i, IDX["rho"]]
                + state[j + 1, i, IDX["rho"]]
            ) * (
                state[j, i, IDX["vy"]]
                + state[j + 1, i, IDX["vy"]]
            )
            residual[j, i] += flux / wy[j]
            residual[j + 1, i] -= flux / wy[j + 1]
    return residual


def _node(config: RanaOriginalConfig, j: int, i: int) -> int:
    return j * config.nx + i


class _BlockBuilder:
    def __init__(self, size: int):
        self.size = size
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.data: list[float] = []

    def block(self, row_node: int, col_node: int, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        local_rows, local_cols = np.nonzero(values)
        self.rows.extend((row_node * NVAR + local_rows).tolist())
        self.cols.extend((col_node * NVAR + local_cols).tolist())
        self.data.extend(values[local_rows, local_cols].tolist())

    def scalar(self, row: int, col: int, value: float) -> None:
        if value != 0.0:
            self.rows.append(int(row))
            self.cols.append(int(col))
            self.data.append(float(value))

    def matrix(self) -> sparse.csr_matrix:
        result = sparse.coo_matrix(
            (self.data, (self.rows, self.cols)), shape=(self.size, self.size)
        ).tocsr()
        result.sum_duplicates()
        return result


def _wall_terms(
    u: np.ndarray,
    config: RanaOriginalConfig,
    *,
    axis: str,
    normal: int,
    wall_velocity: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = wall_matrix(
        u,
        axis=axis,
        normal=normal,
        accommodation=config.accommodation,
        pressure_mode=config.pressure_mode,
    )
    data = wall_data(
        u,
        axis=axis,
        normal=normal,
        wall_velocity=wall_velocity,
        wall_temperature=1.0,
        accommodation=config.accommodation,
        pressure_mode=config.pressure_mode,
    )
    return matrix, data


def _replace_continuity_rows_with_conservative_fv(
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
    state: np.ndarray,
    config: RanaOriginalConfig,
    mass_weights: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Install a telescoping, density-lagged finite-volume mass balance.

    The supplied code writes continuity in non-conservative split form.  Here
    each internal face has one shared mass flux

        F_f = average(rho_old)_f * average(v_new)_f,

    and every physical-wall normal flux is exactly zero.  Adjacent dual
    control volumes therefore receive equal and opposite face contributions.
    The volume-weighted sum of all continuity rows is identically zero, so the
    bordered mass constraint can fix the uniform-density null mode without a
    non-zero Lagrange multiplier or an artificial mass source.
    """
    nodes = config.nx * config.ny
    unknowns = nodes * NVAR
    result = matrix.tolil(copy=True)
    updated_rhs = np.asarray(rhs, dtype=float).copy()
    wx = np.full(config.nx, config.dx, dtype=float)
    wy = np.full(config.ny, config.dy, dtype=float)
    wx[[0, -1]] = 1.5 * config.dx
    wy[[0, -1]] = 1.5 * config.dy

    def add_face(
        row: int,
        left_node: int,
        right_node: int,
        velocity_index: int,
        rho_face: float,
        velocity_face: float,
        signed_inverse_width: float,
    ) -> None:
        velocity_coefficient = 0.5 * signed_inverse_width * rho_face
        result[row, left_node * NVAR + velocity_index] += velocity_coefficient
        result[row, right_node * NVAR + velocity_index] += velocity_coefficient
        if config.conservative_linearization == "defect-newton":
            density_coefficient = 0.5 * signed_inverse_width * velocity_face
            result[row, left_node * NVAR + IDX["rho"]] += density_coefficient
            result[row, right_node * NVAR + IDX["rho"]] += density_coefficient
            updated_rhs[row] += signed_inverse_width * rho_face * velocity_face

    for j in range(config.ny):
        for i in range(config.nx):
            node = _node(config, j, i)
            row = node * NVAR + IDX["rho"]
            # Remove every contribution from the legacy split continuity row,
            # including its old border entry, before installing the FV row.
            result.rows[row] = []
            result.data[row] = []
            updated_rhs[row] = 0.0

            if i < config.nx - 1:
                east = _node(config, j, i + 1)
                rho_face = 0.5 * (
                    state[j, i, IDX["rho"]]
                    + state[j, i + 1, IDX["rho"]]
                )
                velocity_face = 0.5 * (
                    state[j, i, IDX["vx"]]
                    + state[j, i + 1, IDX["vx"]]
                )
                add_face(
                    row,
                    node,
                    east,
                    IDX["vx"],
                    rho_face,
                    velocity_face,
                    1.0 / wx[i],
                )
            if i > 0:
                west = _node(config, j, i - 1)
                rho_face = 0.5 * (
                    state[j, i - 1, IDX["rho"]]
                    + state[j, i, IDX["rho"]]
                )
                velocity_face = 0.5 * (
                    state[j, i - 1, IDX["vx"]]
                    + state[j, i, IDX["vx"]]
                )
                add_face(
                    row,
                    west,
                    node,
                    IDX["vx"],
                    rho_face,
                    velocity_face,
                    -1.0 / wx[i],
                )
            if j < config.ny - 1:
                north = _node(config, j + 1, i)
                rho_face = 0.5 * (
                    state[j, i, IDX["rho"]]
                    + state[j + 1, i, IDX["rho"]]
                )
                velocity_face = 0.5 * (
                    state[j, i, IDX["vy"]]
                    + state[j + 1, i, IDX["vy"]]
                )
                add_face(
                    row,
                    node,
                    north,
                    IDX["vy"],
                    rho_face,
                    velocity_face,
                    1.0 / wy[j],
                )
            if j > 0:
                south = _node(config, j - 1, i)
                rho_face = 0.5 * (
                    state[j - 1, i, IDX["rho"]]
                    + state[j, i, IDX["rho"]]
                )
                velocity_face = 0.5 * (
                    state[j - 1, i, IDX["vy"]]
                    + state[j, i, IDX["vy"]]
                )
                add_face(
                    row,
                    south,
                    node,
                    IDX["vy"],
                    rho_face,
                    velocity_face,
                    -1.0 / wy[j],
                )

            # A unit entry makes lambda a grid-independent uniform mass-source
            # strength.  Exact telescoping makes it zero to solver roundoff.
            result[row, unknowns] = 1.0

    return result.tocsr(), updated_rhs


def assemble_system(
    state: np.ndarray, config: RanaOriginalConfig
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Assemble M(U_old) X_new = b(U_old), including the mass border."""
    config.validate()
    state = np.asarray(state, dtype=float)
    if state.shape != (config.ny, config.nx, NVAR):
        raise ValueError("state shape does not match configuration")
    if not np.isfinite(state).all():
        raise FloatingPointError("state contains non-finite values")
    if np.min(state[..., IDX["rho"]]) <= 0.0 or np.min(
        state[..., IDX["theta"]]
    ) <= 0.0:
        raise FloatingPointError("state contains non-positive rho or theta")

    nodes = config.nx * config.ny
    unknowns = nodes * NVAR
    size = unknowns + 1
    identity = np.eye(NVAR)
    builder = _BlockBuilder(size)
    rhs = np.zeros(size, dtype=float)

    for j in range(config.ny):
        for i in range(config.nx):
            node = _node(config, j, i)
            u = state[j, i]
            a = flux_x(u)
            b = flux_y(u)
            center = production(
                u, config.kn, rb=config.rb, ra=config.ra, ma=config.ma
            )
            blocks: dict[int, np.ndarray] = {}

            def add(col_node: int, value: np.ndarray) -> None:
                if col_node in blocks:
                    blocks[col_node] = blocks[col_node] + value
                else:
                    blocks[col_node] = np.asarray(value, dtype=float).copy()

            # X derivative or x-wall elimination.
            if 0 < i < config.nx - 1:
                add(_node(config, j, i - 1), -a / (2.0 * config.dx))
                add(_node(config, j, i + 1), a / (2.0 * config.dx))
            elif i == 0:
                x, xd = _wall_terms(
                    u, config, axis="x", normal=1, wall_velocity=0.0
                )
                if config.boundary_scheme == "legacy-hobc3":
                    center = center - a @ x / config.dx - a / (2.0 * config.dx)
                    add(_node(config, j, 1), a @ (identity + x) / config.dx)
                    add(
                        _node(config, j, 2),
                        -a @ x / (3.0 * config.dx) - a / (6.0 * config.dx),
                    )
                    rhs[node * NVAR : (node + 1) * NVAR] += a @ xd / (
                        3.0 * config.dx
                    )
                else:
                    center = center - a @ x / config.dx
                    add(
                        _node(config, j, 1),
                        a @ (identity + x) / (2.0 * config.dx),
                    )
                    rhs[node * NVAR : (node + 1) * NVAR] += a @ xd / (
                        2.0 * config.dx
                    )
            else:
                x, xd = _wall_terms(
                    u, config, axis="x", normal=-1, wall_velocity=0.0
                )
                if config.boundary_scheme == "legacy-hobc3":
                    center = center + a @ x / config.dx + a / (2.0 * config.dx)
                    add(_node(config, j, i - 1), -a @ (identity + x) / config.dx)
                    add(
                        _node(config, j, i - 2),
                        a @ x / (3.0 * config.dx) + a / (6.0 * config.dx),
                    )
                    rhs[node * NVAR : (node + 1) * NVAR] -= a @ xd / (
                        3.0 * config.dx
                    )
                else:
                    center = center + a @ x / config.dx
                    add(
                        _node(config, j, i - 1),
                        -a @ (identity + x) / (2.0 * config.dx),
                    )
                    rhs[node * NVAR : (node + 1) * NVAR] -= a @ xd / (
                        2.0 * config.dx
                    )

            # Y derivative or y-wall elimination.  At a corner this is added
            # to the x contribution in the same block row, matching Eq. (20).
            if 0 < j < config.ny - 1:
                add(_node(config, j - 1, i), -b / (2.0 * config.dy))
                add(_node(config, j + 1, i), b / (2.0 * config.dy))
            elif j == 0:
                y, yd = _wall_terms(
                    u, config, axis="y", normal=1, wall_velocity=0.0
                )
                if config.boundary_scheme == "legacy-hobc3":
                    center = center - b @ y / config.dy - b / (2.0 * config.dy)
                    add(_node(config, 1, i), b @ (identity + y) / config.dy)
                    add(
                        _node(config, 2, i),
                        -b @ y / (3.0 * config.dy) - b / (6.0 * config.dy),
                    )
                    rhs[node * NVAR : (node + 1) * NVAR] += b @ yd / (
                        3.0 * config.dy
                    )
                else:
                    center = center - b @ y / config.dy
                    add(
                        _node(config, 1, i),
                        b @ (identity + y) / (2.0 * config.dy),
                    )
                    rhs[node * NVAR : (node + 1) * NVAR] += b @ yd / (
                        2.0 * config.dy
                    )
            else:
                y, yd = _wall_terms(
                    u,
                    config,
                    axis="y",
                    normal=-1,
                    wall_velocity=config.lid_velocity,
                )
                if config.boundary_scheme == "legacy-hobc3":
                    center = center + b @ y / config.dy + b / (2.0 * config.dy)
                    add(_node(config, j - 1, i), -b @ (identity + y) / config.dy)
                    add(
                        _node(config, j - 2, i),
                        b @ y / (3.0 * config.dy) + b / (6.0 * config.dy),
                    )
                    rhs[node * NVAR : (node + 1) * NVAR] -= b @ yd / (
                        3.0 * config.dy
                    )
                else:
                    center = center + b @ y / config.dy
                    add(
                        _node(config, j - 1, i),
                        -b @ (identity + y) / (2.0 * config.dy),
                    )
                    rhs[node * NVAR : (node + 1) * NVAR] -= b @ yd / (
                        2.0 * config.dy
                    )

            add(node, center)
            for col_node, value in blocks.items():
                builder.block(node, col_node, value)

    weights = (
        legacy_mass_weights(config)
        if config.continuity_mode == "legacy-border"
        else conservative_volume_weights(config)
    )
    for node, weight in enumerate(weights):
        density_row = node * NVAR + IDX["rho"]
        density_col = node * NVAR + IDX["rho"]
        builder.scalar(density_row, unknowns, float(weight))
        builder.scalar(unknowns, density_col, float(weight))
    rhs[unknowns] = 1.0
    matrix = builder.matrix()
    if config.continuity_mode == "conservative-fv":
        matrix, rhs = _replace_continuity_rows_with_conservative_fv(
            matrix, rhs, state, config, weights
        )
    return matrix, rhs


def _solve_linear_system(
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
    initial: np.ndarray,
    config: RanaOriginalConfig,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    iterations = 0
    if config.linear_solver == "direct":
        solution = spsolve(matrix, rhs)
        info = 0
    else:
        identity = LinearOperator(
            matrix.shape,
            matvec=lambda value: value,
            rmatvec=lambda value: value,
            dtype=float,
        )

        def count(_: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        solution, info = qmr(
            matrix,
            rhs,
            x0=initial,
            rtol=config.qmr_rtol,
            atol=0.0,
            maxiter=config.qmr_maxiter,
            M1=identity,
            M2=identity,
            callback=count,
        )
    residual = matrix @ solution - rhs
    relative_residual = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    )
    if info != 0 or not np.isfinite(solution).all():
        raise RuntimeError(
            f"{config.linear_solver} failed: info={info}, residual={relative_residual:.3e}"
        )
    return solution, {
        "method": config.linear_solver,
        "info": int(info),
        "iterations": iterations,
        "relative_residual": relative_residual,
    }


def diagnose_final_state(
    state: np.ndarray,
    config: RanaOriginalConfig,
    *,
    lagrange_multiplier: float,
) -> dict[str, float | bool | None | str]:
    """Recompute raw physical and bordered diagnostics for a saved state."""
    config.validate()
    state = np.asarray(state, dtype=float)
    if state.shape != (config.ny, config.nx, NVAR):
        raise ValueError("state shape does not match configuration")
    final_matrix, final_rhs = assemble_system(state, config)
    unknowns = config.nx * config.ny * NVAR
    current = np.concatenate((state.reshape(-1), [float(lagrange_multiplier)]))
    final_residual = final_matrix @ current - final_rhs
    core_residual = (
        final_matrix[:unknowns, :unknowns] @ current[:unknowns]
        - final_rhs[:unknowns]
    )
    weights = (
        legacy_mass_weights(config)
        if config.continuity_mode == "legacy-border"
        else conservative_volume_weights(config)
    )
    weighted_mass = float(weights @ state[..., IDX["rho"]].reshape(-1))
    residual_fields = core_residual.reshape(-1, NVAR)
    continuity_residual = residual_fields[:, IDX["rho"]]
    noncontinuity_residual = np.delete(residual_fields, IDX["rho"], axis=1)
    border_vector = np.asarray(
        final_matrix[:unknowns, unknowns].toarray()
    ).reshape(-1)
    border_source = float(lagrange_multiplier) * border_vector
    if config.continuity_mode == "conservative-fv":
        bordered_continuity = (
            continuity_residual
            + border_source.reshape(-1, NVAR)[:, IDX["rho"]]
        )
        continuity_lambda_invariant_error: float | None = float(
            weights @ bordered_continuity - float(lagrange_multiplier)
        )
        invariant_status = "applicable"
    else:
        continuity_lambda_invariant_error = None
        invariant_status = "not-applicable-to-legacy-border"
    return {
        "fixed_point_relative_residual": float(
            np.linalg.norm(final_residual)
            / max(np.linalg.norm(final_rhs), np.finfo(float).tiny)
        ),
        "core_relative_residual_without_mass_border": float(
            np.linalg.norm(core_residual)
            / max(np.linalg.norm(final_rhs[:unknowns]), np.finfo(float).tiny)
        ),
        "weighted_mass": weighted_mass,
        "mass_error": weighted_mass - 1.0,
        "lagrange_multiplier": float(lagrange_multiplier),
        "continuity_residual_l2": float(np.linalg.norm(continuity_residual)),
        "continuity_residual_max_abs": float(
            np.max(np.abs(continuity_residual))
        ),
        "volume_weighted_continuity_residual": float(
            weights @ continuity_residual
        ),
        "noncontinuity_residual_l2": float(
            np.linalg.norm(noncontinuity_residual)
        ),
        "mass_border_source_l2": float(np.linalg.norm(border_source)),
        "mass_border_source_max_abs": float(np.max(np.abs(border_source))),
        "continuity_lambda_invariant_error": (
            continuity_lambda_invariant_error
        ),
        "continuity_lambda_invariant_status": invariant_status,
        "rho_min": float(np.min(state[..., IDX["rho"]])),
        "theta_min": float(np.min(state[..., IDX["theta"]])),
        "effective_wall_pressure_min": _effective_wall_pressure_minimum(
            state, config
        ),
        "finite": bool(np.isfinite(state).all()),
    }


def _globalization_merit(
    state: np.ndarray,
    config: RanaOriginalConfig,
    *,
    lagrange_multiplier: float,
    residual_scale: float | None = None,
    assembled: tuple[sparse.csr_matrix, np.ndarray] | None = None,
) -> dict[str, float]:
    """Evaluate the unbordered physical merit used by backtracking.

    The mass-border column is deliberately excluded from the core residual:
    a non-zero multiplier must not be allowed to make an inconsistent
    continuity field look converged.  ``residual_scale`` is fixed for every
    trial in one backtracking sweep so a changing denominator cannot create a
    spurious decrease.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (config.ny, config.nx, NVAR):
        raise ValueError("state shape does not match configuration")
    if assembled is None:
        matrix, rhs = assemble_system(state, config)
    else:
        matrix, rhs = assembled
    unknowns = config.nx * config.ny * NVAR
    core_residual = matrix[:unknowns, :unknowns] @ state.reshape(-1) - rhs[:unknowns]
    if residual_scale is None:
        residual_scale = float(
            max(np.linalg.norm(rhs[:unknowns]), np.finfo(float).tiny)
        )
    elif not np.isfinite(residual_scale) or residual_scale <= 0.0:
        raise ValueError("residual_scale must be finite and positive")
    weights = conservative_volume_weights(config)
    mass_error = float(weights @ state[..., IDX["rho"]].reshape(-1) - 1.0)
    core_relative_residual = float(np.linalg.norm(core_residual) / residual_scale)
    merit = 0.5 * (
        core_relative_residual**2
        + mass_error**2
        + float(lagrange_multiplier) ** 2
    )
    return {
        "merit": float(merit),
        "residual_scale": float(residual_scale),
        "core_relative_residual": core_relative_residual,
        "mass_error": mass_error,
        "lagrange_multiplier": float(lagrange_multiplier),
        "rho_min": float(np.min(state[..., IDX["rho"]])),
        "theta_min": float(np.min(state[..., IDX["theta"]])),
    }


def _positivity_step_cap(
    state: np.ndarray,
    direction: np.ndarray,
    *,
    physical_floor: float,
) -> float:
    """Return a 0.99 fraction-to-boundary cap for rho and theta."""
    cap = 1.0
    for variable in ("rho", "theta"):
        values = state[..., IDX[variable]]
        changes = direction[..., IDX[variable]]
        if np.min(values) <= physical_floor:
            return 0.0
        decreasing = changes < 0.0
        if np.any(decreasing):
            ratios = (values[decreasing] - physical_floor) / (-changes[decreasing])
            cap = min(cap, 0.99 * float(np.min(ratios)))
    return max(0.0, cap)


def bordered_nonlinear_residual(
    unknown: np.ndarray,
    config: RanaOriginalConfig,
) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray]:
    """Return the exact discrete bordered residual ``A(U) z - b(U)``.

    In conservative defect-Newton mode the continuity entries are exactly the
    shared-face mass residual plus the unit mass-border source.  The final
    entry is the conservative volume-weighted density constraint.
    """
    config.validate()
    unknown = np.asarray(unknown, dtype=float)
    size = config.nx * config.ny * NVAR + 1
    if unknown.shape != (size,):
        raise ValueError("bordered unknown has the wrong shape")
    if not np.isfinite(unknown).all():
        raise FloatingPointError("non-finite bordered unknown")
    state = unknown[:-1].reshape(config.ny, config.nx, NVAR)
    if (
        np.min(state[..., IDX["rho"]]) <= config.physical_floor
        or np.min(state[..., IDX["theta"]]) <= config.physical_floor
    ):
        raise FloatingPointError("non-positive density or temperature")
    effective_pressure_min = _effective_wall_pressure_minimum(state, config)
    if (
        not np.isfinite(effective_pressure_min)
        or effective_pressure_min <= config.physical_floor
    ):
        raise FloatingPointError("non-positive effective wall pressure")
    matrix, rhs = assemble_system(state, config)
    residual = np.asarray(matrix @ unknown - rhs)
    if not np.isfinite(matrix.data).all() or not np.isfinite(rhs).all():
        raise FloatingPointError("non-finite assembled nonlinear system")
    if not np.isfinite(residual).all():
        raise FloatingPointError("non-finite bordered residual")
    return residual, matrix, rhs


def _jfnk_variable_and_residual_scaling(
    unknown: np.ndarray,
    residual: np.ndarray,
    frozen_matrix: sparse.csr_matrix,
    config: RanaOriginalConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build deterministic two-sided scaling for one nonlinear solve.

    The caller freezes these scales after the initial residual.  Keeping one
    ``W`` and ``S`` makes merit values comparable across Newton iterations and
    avoids magnifying structurally small rows into finite-difference noise.
    """
    state = unknown[:-1].reshape(config.ny, config.nx, NVAR)
    perturbation_scale = max(abs(config.lid_velocity), 1.0e-3)
    field_scales = np.full(NVAR, perturbation_scale, dtype=float)
    field_scales[IDX["rho"]] = 1.0
    field_scales[IDX["theta"]] = 1.0
    for variable in range(NVAR):
        field_scales[variable] = max(
            field_scales[variable],
            float(np.max(np.abs(state[..., variable]))),
        )
    variable_scale = np.concatenate(
        (np.tile(field_scales, config.nx * config.ny), [1.0])
    )
    absolute_matrix = frozen_matrix.copy()
    absolute_matrix.data = np.abs(absolute_matrix.data)
    row_denominator = np.asarray(absolute_matrix @ variable_scale).reshape(-1)
    row_denominator = np.maximum(row_denominator, np.abs(residual))
    row_denominator = np.maximum(row_denominator, 1.0)
    residual_weight = 1.0 / row_denominator
    return variable_scale, residual_weight, {
        "field_scales": {
            name: float(field_scales[index])
            for index, name in enumerate(STATE_ORDER)
        },
        "lambda_scale": 1.0,
        "row_denominator_min": float(np.min(row_denominator)),
        "row_denominator_max": float(np.max(row_denominator)),
    }


def _effective_wall_pressure_minimum(
    state: np.ndarray,
    config: RanaOriginalConfig,
) -> float:
    """Return the minimum Eq. (7) effective pressure seen by any wall."""

    def pressure(samples: np.ndarray, *, axis: str) -> np.ndarray:
        if axis == "x":
            normal_stress = samples[..., IDX["sigma_xx"]]
            tangential_stress = samples[..., IDX["sigma_yy"]]
            normal_regularized = samples[..., IDX["R_xx"]]
            tangential_regularized = samples[..., IDX["R_yy"]]
        elif axis == "y":
            normal_stress = samples[..., IDX["sigma_yy"]]
            tangential_stress = samples[..., IDX["sigma_xx"]]
            normal_regularized = samples[..., IDX["R_yy"]]
            tangential_regularized = samples[..., IDX["R_xx"]]
        else:  # pragma: no cover - internal caller fixes the two axes.
            raise ValueError("axis must be x or y")
        if config.pressure_mode == "legacy-normal":
            stress = normal_stress
            regularized = normal_regularized
        else:
            stress = tangential_stress
            regularized = tangential_regularized
        theta = samples[..., IDX["theta"]]
        return (
            samples[..., IDX["rho"]] * theta
            + 0.5 * stress
            - regularized / (28.0 * theta)
            - samples[..., IDX["Delta"]] / (120.0 * theta)
        )

    values = np.concatenate(
        (
            pressure(state[:, 0, :], axis="x").reshape(-1),
            pressure(state[:, -1, :], axis="x").reshape(-1),
            pressure(state[0, :, :], axis="y").reshape(-1),
            pressure(state[-1, :, :], axis="y").reshape(-1),
        )
    )
    return float(np.min(values))


def _jfnk_physical_merit(
    unknown: np.ndarray,
    config: RanaOriginalConfig,
    *,
    residual_weight: np.ndarray,
    assembled: tuple[sparse.csr_matrix, np.ndarray] | None = None,
) -> dict[str, object]:
    """Physical Armijo merit that cannot hide the core defect with lambda."""
    state = unknown[:-1].reshape(config.ny, config.nx, NVAR)
    if assembled is None:
        matrix, rhs = assemble_system(state, config)
    else:
        matrix, rhs = assembled
    size = unknown.size
    core_size = size - 1
    core = matrix[:core_size, :core_size] @ unknown[:core_size] - rhs[:core_size]
    border = np.asarray(matrix[:core_size, core_size].toarray()).reshape(-1)
    mass = float(
        np.asarray(
            matrix[core_size, :core_size] @ unknown[:core_size] - rhs[core_size]
        ).reshape(-1)[0]
    )
    lambda_source = border * float(unknown[-1])
    vector = np.concatenate(
        (
            residual_weight[:core_size] * core,
            [residual_weight[core_size] * mass],
            residual_weight[:core_size] * lambda_source,
        )
    )
    weights = conservative_volume_weights(config)
    continuity = core.reshape(-1, NVAR)[:, IDX["rho"]]
    bordered_continuity = (
        continuity
        + lambda_source.reshape(-1, NVAR)[:, IDX["rho"]]
    )
    invariant_error = float(
        weights @ bordered_continuity - float(unknown[-1])
    )
    return {
        "merit": float(0.5 * np.dot(vector, vector)),
        "vector": vector,
        "core_relative_residual": float(
            np.linalg.norm(core)
            / max(np.linalg.norm(rhs[:core_size]), np.finfo(float).tiny)
        ),
        "mass_error": mass,
        "lagrange_multiplier": float(unknown[-1]),
        "continuity_lambda_invariant_error": invariant_error,
        "volume_weighted_physical_continuity_residual": float(
            weights @ continuity
        ),
        "rho_min": float(np.min(state[..., IDX["rho"]])),
        "theta_min": float(np.min(state[..., IDX["theta"]])),
        "effective_wall_pressure_min": _effective_wall_pressure_minimum(
            state, config
        ),
    }


def _jfnk_jacobian_vector(
    unknown: np.ndarray,
    scaled_direction: np.ndarray,
    base_residual: np.ndarray,
    variable_scale: np.ndarray,
    residual_weight: np.ndarray,
    config: RanaOriginalConfig,
    *,
    statistics: dict[str, object] | None = None,
) -> np.ndarray:
    """Central finite-difference action of ``W J S``.

    A cube-root-epsilon central stencil is materially more reliable than the
    former forward square-root-epsilon action on the N40 system.  Both sides
    must be physical; otherwise the step is halved fail-closed.
    """
    direction_norm = float(np.linalg.norm(scaled_direction))
    if direction_norm == 0.0:
        return np.zeros_like(base_residual)
    scaled_unknown = unknown / variable_scale
    step = config.jfnk_fd_relative_step * (
        1.0 + float(np.linalg.norm(scaled_unknown))
    ) / direction_norm
    rejected = 0
    residual_evaluations = 0
    while True:
        try:
            plus = unknown + step * variable_scale * scaled_direction
            residual_evaluations += 1
            plus_residual, _, _ = bordered_nonlinear_residual(plus, config)
            minus = unknown - step * variable_scale * scaled_direction
            residual_evaluations += 1
            minus_residual, _, _ = bordered_nonlinear_residual(minus, config)
        except FloatingPointError:
            if rejected >= config.jfnk_max_fd_halvings:
                raise
            step *= 0.5
            rejected += 1
            continue
        break
    if statistics is not None:
        statistics["calls"] = int(statistics.get("calls", 0)) + 1
        statistics["feasibility_halvings"] = int(
            statistics.get("feasibility_halvings", 0)
        ) + rejected
        statistics["residual_evaluations"] = int(
            statistics.get("residual_evaluations", 0)
        ) + residual_evaluations
        statistics["finite_difference_scheme"] = "central-cuberoot-epsilon"
        steps = statistics.setdefault("steps", [])
        if isinstance(steps, list):
            steps.append(float(step))
        physical_perturbation = step * variable_scale * scaled_direction
        perturbation_l2 = statistics.setdefault(
            "physical_perturbation_l2", []
        )
        if isinstance(perturbation_l2, list):
            perturbation_l2.append(
                float(np.linalg.norm(physical_perturbation))
            )
        perturbation_linf = statistics.setdefault(
            "physical_perturbation_linf", []
        )
        if isinstance(perturbation_linf, list):
            perturbation_linf.append(
                float(np.linalg.norm(physical_perturbation, ord=np.inf))
            )
    return residual_weight * (plus_residual - minus_residual) / (2.0 * step)


def _jfnk_right_preconditioner(
    frozen_matrix: sparse.csr_matrix,
    variable_scale: np.ndarray,
    residual_weight: np.ndarray,
) -> tuple[Callable[[np.ndarray], np.ndarray], object]:
    """Factor and return ``(W P S)^-1`` for right-preconditioned GMRES."""
    factor = splu(frozen_matrix.tocsc())

    def apply(value: np.ndarray) -> np.ndarray:
        physical = factor.solve(np.asarray(value) / residual_weight)
        return physical / variable_scale

    return apply, factor


def _jfnk_forcing_parameter(
    bordered_relative: float,
    previous_bordered_relative: float | None,
    previous_history: dict[str, object] | None,
    config: RanaOriginalConfig,
) -> tuple[float, dict[str, object]]:
    """Return an Eisenstat--Walker forcing with a globalization safeguard.

    A damped Newton step is evidence that the local model was not reliable
    over the full step.  The next linear solve must therefore not become
    looser merely because the damped residual ratio was poor.  Cap the new
    forcing by ``eta_previous * alpha_previous`` in that case.
    """
    if previous_bordered_relative is None:
        raw = config.jfnk_initial_forcing
        ratio: float | None = None
    else:
        ratio = bordered_relative / max(
            previous_bordered_relative, np.finfo(float).tiny
        )
        raw = float(
            np.clip(
                0.9 * ratio**1.5,
                config.jfnk_min_forcing,
                config.jfnk_max_forcing,
            )
        )
    safeguard_cap: float | None = None
    if previous_history is not None:
        previous_alpha = float(
            previous_history.get("accepted_relaxation", 1.0)
        )
        previous_forcing = previous_history.get("forcing")
        if previous_alpha < 1.0 and previous_forcing is not None:
            safeguard_cap = max(
                config.jfnk_min_forcing,
                float(previous_forcing) * previous_alpha,
            )
    forcing = raw if safeguard_cap is None else min(raw, safeguard_cap)
    return forcing, {
        "raw_eisenstat_walker": raw,
        "residual_ratio": ratio,
        "damped_step_cap": safeguard_cap,
        "selected": forcing,
    }


def _jfnk_forcing_trials(
    initial: float,
    minimum: float,
) -> list[float]:
    """Return geometric forcing retries, including the exact minimum."""
    result: list[float] = []
    current = max(initial, minimum)
    while True:
        if current <= minimum * (1.0 + 8.0 * np.finfo(float).eps):
            result.append(float(minimum))
            break
        result.append(float(current))
        next_value = max(minimum, 0.1 * current)
        if next_value >= current:
            break
        current = next_value
    return result


def solve_rana_original(
    config: RanaOriginalConfig,
    *,
    initial: np.ndarray | None = None,
    callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run the selected archive-faithful or corrected nonlinear iteration."""
    config.validate()
    if config.nonlinear_solver == "jfnk":
        return solve_rana_jfnk(config, initial=initial, callback=callback)
    state = equilibrium(config) if initial is None else np.asarray(initial, dtype=float).copy()
    if state.shape != (config.ny, config.nx, NVAR):
        raise ValueError("initial state shape does not match configuration")
    current = np.concatenate((state.reshape(-1), [0.0]))
    history: list[dict[str, object]] = []
    converged = False
    termination_reason = "max_outer_iterations"
    started = time.monotonic()

    for outer in range(1, config.max_outer_iterations + 1):
        matrix, rhs = assemble_system(
            current[:-1].reshape(config.ny, config.nx, NVAR), config
        )
        candidate, linear = _solve_linear_system(matrix, rhs, current, config)
        raw_error = float(np.linalg.norm(candidate - current, ord=np.inf))
        direction = candidate - current

        if config.outer_globalization == "fixed":
            # Keep the historical/default path intentionally unchanged.
            updated = current + config.outer_relaxation * direction
            error = float(np.linalg.norm(updated - current, ord=np.inf))
            record: dict[str, object] = {
                "outer_iteration": outer,
                "infinity_update": error,
                "undamped_infinity_update": raw_error,
                "accepted_relaxation": config.outer_relaxation,
                "linear_iterations": int(linear["iterations"]),
                "linear_relative_residual": float(linear["relative_residual"]),
                "matrix_nnz": int(matrix.nnz),
            }
            history.append(record)
            if callback is not None:
                callback(record)
            current = updated
            state = current[:-1].reshape(config.ny, config.nx, NVAR)
            if (
                not np.isfinite(state).all()
                or np.min(state[..., IDX["rho"]]) <= 0.0
                or np.min(state[..., IDX["theta"]]) <= 0.0
            ):
                raise FloatingPointError(
                    f"non-physical state after outer iteration {outer}"
                )
        else:
            state = current[:-1].reshape(config.ny, config.nx, NVAR)
            merit_before = _globalization_merit(
                state,
                config,
                lagrange_multiplier=float(current[-1]),
                assembled=(matrix, rhs),
            )
            base_record: dict[str, object] = {
                "outer_iteration": outer,
                "undamped_infinity_update": raw_error,
                "linear_iterations": int(linear["iterations"]),
                "linear_relative_residual": float(linear["relative_residual"]),
                "matrix_nnz": int(matrix.nnz),
                "merit_before": merit_before["merit"],
                "core_relative_residual_before": merit_before[
                    "core_relative_residual"
                ],
                "mass_error_before": merit_before["mass_error"],
                "lagrange_multiplier_before": merit_before[
                    "lagrange_multiplier"
                ],
                "rho_min_before": merit_before["rho_min"],
                "theta_min_before": merit_before["theta_min"],
            }
            if (
                float(linear["relative_residual"])
                > config.verification_linear_residual_tolerance
            ):
                record = {
                    **base_record,
                    "infinity_update": 0.0,
                    "accepted_relaxation": 0.0,
                    "line_search_backtracks": 0,
                    "line_search_rejections": ["linear_residual_too_large"],
                    "line_search_status": "linear_residual_failed",
                    "merit_after": merit_before["merit"],
                    "core_relative_residual_after": merit_before[
                        "core_relative_residual"
                    ],
                    "mass_error_after": merit_before["mass_error"],
                    "lagrange_multiplier_after": merit_before[
                        "lagrange_multiplier"
                    ],
                    "rho_min_after": merit_before["rho_min"],
                    "theta_min_after": merit_before["theta_min"],
                }
                history.append(record)
                if callback is not None:
                    callback(record)
                termination_reason = "linear_residual_failed"
                break

            # A genuinely small *undamped* correction is already a valid
            # nonlinear termination test.  Avoid demanding a numerically
            # meaningless Armijo decrease once roundoff dominates.
            if raw_error < config.outer_tolerance:
                record = {
                    **base_record,
                    "infinity_update": 0.0,
                    "accepted_relaxation": 0.0,
                    "line_search_backtracks": 0,
                    "line_search_rejections": [],
                    "line_search_status": "converged_before_step",
                    "merit_after": merit_before["merit"],
                    "core_relative_residual_after": merit_before[
                        "core_relative_residual"
                    ],
                    "mass_error_after": merit_before["mass_error"],
                    "lagrange_multiplier_after": merit_before[
                        "lagrange_multiplier"
                    ],
                    "rho_min_after": merit_before["rho_min"],
                    "theta_min_after": merit_before["theta_min"],
                }
                history.append(record)
                if callback is not None:
                    callback(record)
                converged = True
                termination_reason = "outer_tolerance"
                break

            direction_state = direction[:-1].reshape(config.ny, config.nx, NVAR)
            positivity_cap = _positivity_step_cap(
                state,
                direction_state,
                physical_floor=config.physical_floor,
            )
            alpha = min(config.outer_relaxation, 1.0, positivity_cap)
            accepted: tuple[np.ndarray, dict[str, float], float] | None = None
            backtracks = 0
            rejections: list[str] = []
            while alpha >= config.line_search_min_step:
                trial = current + alpha * direction
                trial_state = trial[:-1].reshape(config.ny, config.nx, NVAR)
                if not np.isfinite(trial).all():
                    rejections.append(f"alpha={alpha:.17g}:nonfinite")
                elif (
                    np.min(trial_state[..., IDX["rho"]])
                    <= config.physical_floor
                    or np.min(trial_state[..., IDX["theta"]])
                    <= config.physical_floor
                ):
                    rejections.append(f"alpha={alpha:.17g}:positivity")
                else:
                    try:
                        merit_after = _globalization_merit(
                            trial_state,
                            config,
                            lagrange_multiplier=float(trial[-1]),
                            residual_scale=merit_before["residual_scale"],
                        )
                    except FloatingPointError as error:
                        rejections.append(
                            f"alpha={alpha:.17g}:infeasible:{error}"
                        )
                    else:
                        armijo_bound = (
                            1.0 - config.line_search_armijo * alpha
                        ) * merit_before["merit"]
                        if merit_after["merit"] <= armijo_bound:
                            accepted = (trial, merit_after, alpha)
                            break
                        rejections.append(f"alpha={alpha:.17g}:armijo")
                alpha *= config.line_search_reduction
                backtracks += 1

            if accepted is None:
                record = {
                    **base_record,
                    "infinity_update": 0.0,
                    "accepted_relaxation": 0.0,
                    "positivity_step_cap": positivity_cap,
                    "line_search_backtracks": backtracks,
                    "line_search_rejections": rejections,
                    "line_search_status": "line_search_failed",
                    "merit_after": merit_before["merit"],
                    "core_relative_residual_after": merit_before[
                        "core_relative_residual"
                    ],
                    "mass_error_after": merit_before["mass_error"],
                    "lagrange_multiplier_after": merit_before[
                        "lagrange_multiplier"
                    ],
                    "rho_min_after": merit_before["rho_min"],
                    "theta_min_after": merit_before["theta_min"],
                }
                history.append(record)
                if callback is not None:
                    callback(record)
                termination_reason = "line_search_failed"
                break

            updated, merit_after, accepted_alpha = accepted
            error = float(np.linalg.norm(updated - current, ord=np.inf))
            record = {
                **base_record,
                "infinity_update": error,
                "accepted_relaxation": accepted_alpha,
                "positivity_step_cap": positivity_cap,
                "line_search_backtracks": backtracks,
                "line_search_rejections": rejections,
                "line_search_status": "accepted",
                "merit_after": merit_after["merit"],
                "core_relative_residual_after": merit_after[
                    "core_relative_residual"
                ],
                "mass_error_after": merit_after["mass_error"],
                "lagrange_multiplier_after": merit_after[
                    "lagrange_multiplier"
                ],
                "rho_min_after": merit_after["rho_min"],
                "theta_min_after": merit_after["theta_min"],
            }
            history.append(record)
            if callback is not None:
                callback(record)
            current = updated
            state = current[:-1].reshape(config.ny, config.nx, NVAR)
        # Relaxation must not make a large raw nonlinear correction look
        # converged.  The accepted update is retained for diagnostics, but the
        # undamped fixed-point/Newton correction controls termination.
        if raw_error < config.outer_tolerance:
            converged = True
            termination_reason = "outer_tolerance"
            break

    diagnostics = diagnose_final_state(
        state,
        config,
        lagrange_multiplier=float(current[-1]),
    )
    report: dict[str, object] = {
        "converged": converged,
        "termination_reason": termination_reason,
        "outer_iterations": len(history),
        "final_infinity_update": float(history[-1]["infinity_update"]),
        "final_undamped_infinity_update": float(
            history[-1]["undamped_infinity_update"]
        ),
        "maximum_linear_relative_residual": float(
            max(item["linear_relative_residual"] for item in history)
        ),
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
        **diagnostics,
    }
    return state, report


def solve_rana_jfnk(
    config: RanaOriginalConfig,
    *,
    initial: np.ndarray | None = None,
    callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve the exact conservative bordered residual with Newton--GMRES."""
    config.validate()
    state = equilibrium(config) if initial is None else np.asarray(initial, dtype=float).copy()
    if state.shape != (config.ny, config.nx, NVAR):
        raise ValueError("initial state shape does not match configuration")
    unknown = np.concatenate((state.reshape(-1), [0.0]))
    history: list[dict[str, object]] = []
    converged = False
    termination_reason = "max_outer_iterations"
    previous_bordered_relative: float | None = None
    all_accepted_linear_solves_met_requested_forcing = True
    selected_linear_solve_count = 0
    total_jacobian_vector_calls = 0
    variable_scale: np.ndarray | None = None
    residual_weight: np.ndarray | None = None
    scaling: dict[str, object] | None = None
    started = time.monotonic()

    for outer in range(1, config.max_outer_iterations + 1):
        residual, frozen_matrix, rhs = bordered_nonlinear_residual(unknown, config)
        if variable_scale is None or residual_weight is None or scaling is None:
            variable_scale, residual_weight, scaling = _jfnk_variable_and_residual_scaling(
                unknown, residual, frozen_matrix, config
            )
        state = unknown[:-1].reshape(config.ny, config.nx, NVAR)
        physical_before = _jfnk_physical_merit(
            unknown,
            config,
            residual_weight=residual_weight,
            assembled=(frozen_matrix, rhs),
        )
        bordered_relative = float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(rhs), np.finfo(float).tiny)
        )
        if (
            bordered_relative <= config.outer_tolerance
            and physical_before["core_relative_residual"]
            <= config.physical_core_residual_tolerance
            and abs(float(physical_before["mass_error"]))
            <= config.verification_mass_tolerance
            and abs(float(physical_before["lagrange_multiplier"]))
            <= config.verification_lagrange_tolerance
            and abs(
                float(physical_before["continuity_lambda_invariant_error"])
            )
            <= config.verification_global_balance_tolerance
            and np.isfinite(physical_before["effective_wall_pressure_min"])
            and float(physical_before["effective_wall_pressure_min"])
            > config.physical_floor
        ):
            final_check = diagnose_final_state(
                state,
                config,
                lagrange_multiplier=float(unknown[-1]),
            )
            if (
                final_check["continuity_residual_max_abs"]
                <= config.verification_continuity_residual_tolerance
                and abs(final_check["volume_weighted_continuity_residual"])
                <= config.verification_global_balance_tolerance
                and final_check["mass_border_source_max_abs"]
                <= config.verification_border_source_tolerance
            ):
                converged = True
                termination_reason = "nonlinear_residual"
                break

        forcing, forcing_update = _jfnk_forcing_parameter(
            bordered_relative,
            previous_bordered_relative,
            history[-1] if history else None,
            config,
        )

        jv_statistics: dict[str, object] = {
            "calls": 0,
            "residual_evaluations": 0,
            "feasibility_halvings": 0,
            "finite_difference_scheme": "central-cuberoot-epsilon",
            "steps": [],
            "physical_perturbation_l2": [],
            "physical_perturbation_linf": [],
        }
        size = unknown.size

        def scaled_jacobian(value: np.ndarray) -> np.ndarray:
            return _jfnk_jacobian_vector(
                unknown,
                value,
                residual,
                variable_scale,
                residual_weight,
                config,
                statistics=jv_statistics,
            )

        jacobian_operator = LinearOperator(
            (size, size), matvec=scaled_jacobian, dtype=float
        )
        try:
            right_precondition, _ = _jfnk_right_preconditioner(
                frozen_matrix, variable_scale, residual_weight
            )
        except RuntimeError as error:
            record = {
                "outer_iteration": outer,
                "infinity_update": 0.0,
                "undamped_infinity_update": 0.0,
                "accepted_relaxation": 0.0,
                "linear_iterations": 0,
                "linear_relative_residual": float("inf"),
                "matrix_nnz": int(frozen_matrix.nnz),
                "line_search_status": "preconditioner_failed",
                "failure": str(error),
                "bordered_relative_residual_before": bordered_relative,
                "merit_before": physical_before["merit"],
                "forcing_update": forcing_update,
                "scaling": scaling,
            }
            history.append(record)
            if callback is not None:
                callback(record)
            all_accepted_linear_solves_met_requested_forcing = False
            termination_reason = "preconditioner_failed"
            break

        def right_preconditioned_matvec(value: np.ndarray) -> np.ndarray:
            return jacobian_operator @ right_precondition(value)

        right_operator = LinearOperator(
            (size, size), matvec=right_preconditioned_matvec, dtype=float
        )
        scaled_rhs = -residual_weight * residual
        scaled_rhs_norm = max(
            float(np.linalg.norm(scaled_rhs)), np.finfo(float).tiny
        )
        linear_attempts: list[dict[str, object]] = []
        selected: dict[str, object] | None = None
        trial_forcings = _jfnk_forcing_trials(
            forcing, config.jfnk_min_forcing
        )

        for attempt_forcing in trial_forcings:
            gmres_residuals: list[float] = []

            def monitor(value: float) -> None:
                gmres_residuals.append(float(value))

            try:
                preconditioned_step, info = gmres(
                    right_operator,
                    scaled_rhs,
                    rtol=attempt_forcing,
                    atol=0.0,
                    restart=config.jfnk_gmres_restart,
                    maxiter=config.jfnk_gmres_max_cycles,
                    callback=monitor,
                    callback_type="pr_norm",
                )
                scaled_step = right_precondition(preconditioned_step)
                linear_defect = jacobian_operator @ scaled_step - scaled_rhs
                true_linear_relative = float(
                    np.linalg.norm(linear_defect) / scaled_rhs_norm
                )
            except (FloatingPointError, RuntimeError) as error:
                linear_attempts.append(
                    {
                        "forcing": attempt_forcing,
                        "info": -1,
                        "iterations": len(gmres_residuals),
                        "true_relative_residual": float("inf"),
                        "failure": str(error),
                        "met_requested_forcing": False,
                        "accepted_linear_tolerance": False,
                        "met_forcing": False,
                        "line_search_status": "linear_solve_failed",
                    }
                )
                continue

            requested_threshold = attempt_forcing * (
                1.0 + 64.0 * np.finfo(float).eps
            )
            linear_ok = bool(
                info == 0
                and np.isfinite(true_linear_relative)
                and true_linear_relative <= requested_threshold
            )
            attempt: dict[str, object] = {
                "forcing": attempt_forcing,
                "info": int(info),
                "iterations": len(gmres_residuals),
                "true_relative_residual": true_linear_relative,
                "reported_residual_last": (
                    gmres_residuals[-1] if gmres_residuals else None
                ),
                "gmres_reported_converged": bool(info == 0),
                "requested_relative_residual": attempt_forcing,
                "met_requested_forcing": linear_ok,
                "accepted_linear_tolerance": linear_ok,
                # Compatibility alias; it now has the strict requested-forcing
                # semantics instead of the former 2x safety-factor semantics.
                "met_forcing": linear_ok,
                "line_search_status": (
                    "not_attempted" if linear_ok else "linear_forcing_failed"
                ),
            }
            linear_attempts.append(attempt)
            if not linear_ok:
                continue

            direction = variable_scale * scaled_step
            direction_state = direction[:-1].reshape(
                config.ny, config.nx, NVAR
            )
            positivity_cap = _positivity_step_cap(
                state,
                direction_state,
                physical_floor=config.physical_floor,
            )
            probe = min(1.0e-4, 0.1 * positivity_cap)
            slope: float | None = None
            probe_rejections: list[str] = []
            while probe >= config.line_search_min_step * 1.0e-3:
                probe_unknown = unknown + probe * direction
                try:
                    _, probe_matrix, probe_rhs = bordered_nonlinear_residual(
                        probe_unknown, config
                    )
                    probe_merit = _jfnk_physical_merit(
                        probe_unknown,
                        config,
                        residual_weight=residual_weight,
                        assembled=(probe_matrix, probe_rhs),
                    )
                except FloatingPointError as error:
                    probe_rejections.append(str(error))
                    probe *= 0.5
                    continue
                slope = float(
                    (float(probe_merit["merit"]) - float(physical_before["merit"]))
                    / probe
                )
                break
            attempt["directional_probe"] = probe
            attempt["directional_slope"] = slope
            attempt["probe_rejections"] = probe_rejections
            if slope is None or not np.isfinite(slope) or slope >= 0.0:
                attempt["descent_direction"] = False
                attempt["line_search_status"] = "no_descent_direction"
                continue
            attempt["descent_direction"] = True
            raw_error = float(np.linalg.norm(direction, ord=np.inf))
            alpha = min(
                config.outer_relaxation,
                1.0,
                positivity_cap,
            )
            backtracks = 0
            rejections: list[str] = []
            accepted: tuple[np.ndarray, dict[str, object], float] | None = None
            while alpha >= config.line_search_min_step:
                trial_unknown = unknown + alpha * direction
                try:
                    _, trial_matrix, trial_rhs = bordered_nonlinear_residual(
                        trial_unknown, config
                    )
                    physical_after = _jfnk_physical_merit(
                        trial_unknown,
                        config,
                        residual_weight=residual_weight,
                        assembled=(trial_matrix, trial_rhs),
                    )
                except FloatingPointError as error:
                    rejections.append(
                        f"alpha={alpha:.17g}:infeasible:{error}"
                    )
                else:
                    armijo_bound = float(physical_before["merit"]) + (
                        config.line_search_armijo * alpha * slope
                    )
                    if float(physical_after["merit"]) <= armijo_bound:
                        accepted = (trial_unknown, physical_after, alpha)
                        break
                    rejections.append(f"alpha={alpha:.17g}:armijo")
                alpha *= config.line_search_reduction
                backtracks += 1

            attempt.update(
                {
                    "undamped_infinity_update": raw_error,
                    "positivity_step_cap": positivity_cap,
                    "line_search_backtracks": backtracks,
                    "line_search_rejections": rejections,
                    "line_search_status": (
                        "accepted" if accepted is not None else "failed"
                    ),
                    "accepted_relaxation": (
                        accepted[2] if accepted is not None else 0.0
                    ),
                }
            )
            if accepted is None:
                # A direction can be descent only in a tiny local probe while
                # remaining too inaccurate over every admissible Armijo step.
                # Rebuild it with a tighter inexact-Newton forcing before
                # declaring a fail-closed globalization failure.
                continue
            selected = {
                "direction": direction,
                "positivity_cap": positivity_cap,
                "slope": slope,
                "forcing": attempt_forcing,
                "linear_iterations": len(gmres_residuals),
                "linear_relative_residual": true_linear_relative,
                "raw_error": raw_error,
                "accepted": accepted,
                "line_search_backtracks": backtracks,
                "line_search_rejections": rejections,
            }
            selected_linear_solve_count += 1
            break

        total_jacobian_vector_calls += int(jv_statistics["calls"])
        if selected is None:
            any_linear_ok = any(
                bool(item.get("met_forcing", False))
                for item in linear_attempts
            )
            any_descent = any(
                bool(item.get("descent_direction", False))
                for item in linear_attempts
            )
            if not any_linear_ok:
                all_accepted_linear_solves_met_requested_forcing = False
            failure_status = (
                "line_search_failed" if any_descent else "jfnk_direction_failed"
            )
            record = {
                "outer_iteration": outer,
                "infinity_update": 0.0,
                "undamped_infinity_update": float(
                    max(
                        (
                            float(item.get("undamped_infinity_update", 0.0))
                            for item in linear_attempts
                        ),
                        default=0.0,
                    )
                ),
                "accepted_relaxation": 0.0,
                "linear_iterations": int(
                    sum(int(item["iterations"]) for item in linear_attempts)
                ),
                "linear_relative_residual": float(
                    min(
                        (
                            float(item["true_relative_residual"])
                            for item in linear_attempts
                        ),
                        default=float("inf"),
                    )
                ),
                "matrix_nnz": int(frozen_matrix.nnz),
                "line_search_status": failure_status,
                "line_search_backtracks": int(
                    sum(
                        int(item.get("line_search_backtracks", 0))
                        for item in linear_attempts
                    )
                ),
                "line_search_rejections": [
                    f"forcing={float(item['forcing']):.17g}:{rejection}"
                    for item in linear_attempts
                    for rejection in item.get("line_search_rejections", [])
                ],
                "bordered_relative_residual_before": bordered_relative,
                "merit_before": physical_before["merit"],
                "forcing_update": forcing_update,
                "forcing_trials": trial_forcings,
                "linear_attempts": linear_attempts,
                "jv_statistics": jv_statistics,
                "scaling": scaling,
            }
            history.append(record)
            if callback is not None:
                callback(record)
            termination_reason = failure_status
            break

        direction = np.asarray(selected["direction"])
        raw_error = float(selected["raw_error"])
        backtracks = int(selected["line_search_backtracks"])
        rejections = list(selected["line_search_rejections"])
        accepted = selected["accepted"]
        if not isinstance(accepted, tuple):  # pragma: no cover - invariant.
            raise RuntimeError("accepted JFNK attempt lost its trial state")
        updated, physical_after, accepted_alpha = accepted
        error = float(np.linalg.norm(updated - unknown, ord=np.inf))
        record = {
            "outer_iteration": outer,
            "infinity_update": error,
            "undamped_infinity_update": raw_error,
            "accepted_relaxation": accepted_alpha,
            "linear_iterations": int(selected["linear_iterations"]),
            "linear_relative_residual": float(
                selected["linear_relative_residual"]
            ),
            "matrix_nnz": int(frozen_matrix.nnz),
            "line_search_status": "accepted",
            "line_search_backtracks": backtracks,
            "line_search_rejections": rejections,
            "directional_slope": selected["slope"],
            "forcing": selected["forcing"],
            "forcing_update": forcing_update,
            "forcing_trials": trial_forcings,
            "forcing_retry_count": int(
                len(linear_attempts) - 1
            ),
            "bordered_relative_residual_before": bordered_relative,
            "merit_before": physical_before["merit"],
            "merit_after": physical_after["merit"],
            "core_relative_residual_before": physical_before[
                "core_relative_residual"
            ],
            "core_relative_residual_after": physical_after[
                "core_relative_residual"
            ],
            "mass_error_after": physical_after["mass_error"],
            "lagrange_multiplier_after": physical_after[
                "lagrange_multiplier"
            ],
            "rho_min_after": physical_after["rho_min"],
            "theta_min_after": physical_after["theta_min"],
            "effective_wall_pressure_min_after": physical_after[
                "effective_wall_pressure_min"
            ],
            "continuity_lambda_invariant_error_after": physical_after[
                "continuity_lambda_invariant_error"
            ],
            "linear_attempts": linear_attempts,
            "jv_statistics": jv_statistics,
            "scaling": scaling,
        }
        history.append(record)
        if callback is not None:
            callback(record)
        previous_bordered_relative = bordered_relative
        unknown = updated
        state = unknown[:-1].reshape(config.ny, config.nx, NVAR)

    state = unknown[:-1].reshape(config.ny, config.nx, NVAR)
    diagnostics = diagnose_final_state(
        state,
        config,
        lagrange_multiplier=float(unknown[-1]),
    )
    final_residual, _, final_rhs = bordered_nonlinear_residual(
        unknown, config
    )
    final_weights = conservative_volume_weights(config)
    core_size = unknown.size - 1
    final_bordered_relative = float(
        np.linalg.norm(final_residual)
        / max(np.linalg.norm(final_rhs), np.finfo(float).tiny)
    )
    final_bordered_continuity = final_residual[:core_size].reshape(
        -1, NVAR
    )[:, IDX["rho"]]
    continuity_lambda_invariant_error = float(
        final_weights @ final_bordered_continuity - float(unknown[-1])
    )
    final_convergence_gates = {
        "finite": bool(diagnostics["finite"]),
        "rho_positive": bool(diagnostics["rho_min"] > config.physical_floor),
        "theta_positive": bool(
            diagnostics["theta_min"] > config.physical_floor
        ),
        "effective_wall_pressure_positive": bool(
            np.isfinite(diagnostics["effective_wall_pressure_min"])
            and diagnostics["effective_wall_pressure_min"]
            > config.physical_floor
        ),
        "bordered_residual": bool(
            final_bordered_relative <= config.outer_tolerance
        ),
        "physical_core_residual": bool(
            diagnostics["core_relative_residual_without_mass_border"]
            <= config.physical_core_residual_tolerance
        ),
        "mass_constraint": bool(
            abs(diagnostics["mass_error"])
            <= config.verification_mass_tolerance
        ),
        "lagrange_multiplier_zero": bool(
            abs(diagnostics["lagrange_multiplier"])
            <= config.verification_lagrange_tolerance
        ),
        "continuity_residual_local": bool(
            diagnostics["continuity_residual_max_abs"]
            <= config.verification_continuity_residual_tolerance
        ),
        "continuity_global_balance": bool(
            abs(diagnostics["volume_weighted_continuity_residual"])
            <= config.verification_global_balance_tolerance
        ),
        "mass_border_source_local": bool(
            diagnostics["mass_border_source_max_abs"]
            <= config.verification_border_source_tolerance
        ),
        "continuity_lambda_invariant": bool(
            abs(continuity_lambda_invariant_error)
            <= config.verification_global_balance_tolerance
        ),
    }
    if (
        not converged
        and termination_reason == "max_outer_iterations"
        and all(final_convergence_gates.values())
    ):
        # The last permitted Newton step may be the one that reaches the root;
        # do not require a nonexistent (max+1)st loop merely to notice it.
        converged = True
        termination_reason = "nonlinear_residual"
    maximum_linear_residual = max(
        (float(item.get("linear_relative_residual", 0.0)) for item in history),
        default=0.0,
    )
    report: dict[str, object] = {
        "converged": converged,
        "termination_reason": termination_reason,
        "outer_iterations": len(history),
        "final_infinity_update": float(
            history[-1]["infinity_update"] if history else 0.0
        ),
        "final_undamped_infinity_update": float(
            history[-1]["undamped_infinity_update"] if history else 0.0
        ),
        "maximum_linear_relative_residual": maximum_linear_residual,
        "selected_linear_solve_count": selected_linear_solve_count,
        "all_accepted_linear_solves_met_requested_forcing": (
            all_accepted_linear_solves_met_requested_forcing
        ),
        # Compatibility alias with strict semantics.
        "all_linear_solves_met_forcing": (
            all_accepted_linear_solves_met_requested_forcing
        ),
        "final_bordered_relative_residual": final_bordered_relative,
        "continuity_lambda_invariant_error": (
            continuity_lambda_invariant_error
        ),
        "final_convergence_gates": final_convergence_gates,
        "scaling_policy": "frozen_from_initial_state",
        "globalization_merit": (
            "scaled_unbordered_core_plus_mass_plus_separate_lambda_source"
        ),
        "preconditioner": (
            "physical_frozen_matrix_LU_with_algebraic_W_S_maps"
        ),
        "jacobian_vector_calls": total_jacobian_vector_calls,
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
        **diagnostics,
    }
    return state, report


def eliminated_wall_state(
    state: np.ndarray,
    config: RanaOriginalConfig,
    *,
    side: str,
) -> np.ndarray:
    """Recover one physical wall from the converged interior state."""
    if side == "left":
        first, second = state[:, 0], state[:, 1]
        axis, normal, velocity = "x", 1, 0.0
    elif side == "right":
        first, second = state[:, -1], state[:, -2]
        axis, normal, velocity = "x", -1, 0.0
    elif side == "bottom":
        first, second = state[0], state[1]
        axis, normal, velocity = "y", 1, 0.0
    elif side == "top":
        first, second = state[-1], state[-2]
        axis, normal, velocity = "y", -1, config.lid_velocity
    else:
        raise ValueError("side must be left, right, bottom, or top")
    result = np.zeros_like(first)
    if config.boundary_scheme == "legacy-hobc3":
        if side == "left":
            third = state[:, 2]
        elif side == "right":
            third = state[:, -3]
        elif side == "bottom":
            third = state[2]
        else:
            third = state[-3]
        # The supplied HOBC3 rows are obtained by inserting the quadratic
        # wall trace 3*U1 - 3*U2 + U3 into the one-sided derivative.
        extrapolated = 3.0 * first - 3.0 * second + third
    else:
        extrapolated = 2.0 * first - second
    for index, (u, extrapolation) in enumerate(zip(first, extrapolated)):
        x, xd = _wall_terms(
            u,
            config,
            axis=axis,
            normal=normal,
            wall_velocity=velocity,
        )
        result[index] = x @ extrapolation + xd
    return result


def cavity_metrics(
    state: np.ndarray, config: RanaOriginalConfig
) -> dict[str, float | str]:
    top = eliminated_wall_state(state, config, side="top")
    sigma_integral = float(
        config.dx * np.sum(top[:, IDX["sigma_xy"]])
    )
    reduction_factor = np.sqrt(2.0) / abs(config.lid_velocity)
    x = np.arange(1, config.nx + 1, dtype=float) * config.dx
    center_velocity = np.asarray(
        [np.interp(0.5, x, row) for row in state[..., IDX["vx"]]], dtype=float
    )
    return {
        "D": abs(reduction_factor * sigma_integral),
        "D_signed": reduction_factor * sigma_integral,
        "D_sigma_over_p0_signed": sigma_integral,
        "D_reduced_stress_factor": float(reduction_factor),
        "G": float(
            config.dy * np.sum(np.abs(center_velocity)) / abs(config.lid_velocity)
        ),
        "rho_min": float(np.min(state[..., IDX["rho"]])),
        "theta_min": float(np.min(state[..., IDX["theta"]])),
        "state_max_abs": float(np.max(np.abs(state))),
        "provenance": (
            "independent Eq. (30) post-processing; the supplied archive does "
            "not contain a D/G routine"
        ),
    }


def build_result(
    config: RanaOriginalConfig,
    state: np.ndarray,
    solver: dict[str, object],
    *,
    state_file: str,
) -> dict[str, object]:
    maximum_linear_residual = float(
        solver["maximum_linear_relative_residual"]
    )
    invariant_value = solver.get("continuity_lambda_invariant_error")
    invariant_available = bool(
        config.continuity_mode == "conservative-fv"
        and not isinstance(invariant_value, (bool, np.bool_))
        and isinstance(invariant_value, (int, float, np.integer, np.floating))
        and np.isfinite(float(invariant_value))
    )
    execution_gates = {
        "finite": bool(solver["finite"] is True),
        "rho_positive": bool(solver["rho_min"] > 0.0),
        "theta_positive": bool(solver["theta_min"] > 0.0),
        "effective_wall_pressure_positive": bool(
            np.isfinite(solver["effective_wall_pressure_min"])
            and solver["effective_wall_pressure_min"]
            > config.physical_floor
        ),
    }
    if config.nonlinear_solver == "jfnk":
        linear_gate = bool(
            solver.get(
                "all_accepted_linear_solves_met_requested_forcing", False
            )
            is True
        )
    else:
        linear_gate = bool(
            solver.get(
                "all_accepted_linear_solves_met_requested_forcing",
                solver.get(
                    "all_linear_solves_met_forcing",
                    maximum_linear_residual
                    <= config.verification_linear_residual_tolerance,
                ),
            )
            is True
        )
    numerical_solution_gates = {
        **execution_gates,
        "outer_iteration_converged": bool(solver["converged"] is True),
        "linear_residual": linear_gate,
        "fixed_point_residual": bool(
            solver["fixed_point_relative_residual"]
            <= config.verification_fixed_point_residual_tolerance
        ),
        "mass_constraint": bool(
            abs(solver["mass_error"]) <= config.verification_mass_tolerance
        ),
    }
    exact_archive_mode = bool(
        config.boundary_scheme == "legacy-hobc3"
        and config.pressure_mode == "legacy-normal"
        and config.continuity_mode == "legacy-border"
    )
    archive_reproduction_gates = {
        **numerical_solution_gates,
        "archive_source_formulation_selected": exact_archive_mode,
    }
    physical_consistency_gates = {
        **numerical_solution_gates,
        "mass_border_inactive": bool(
            solver["core_relative_residual_without_mass_border"]
            <= config.physical_core_residual_tolerance
        ),
        "lagrange_multiplier_zero": bool(
            abs(solver["lagrange_multiplier"])
            <= config.verification_lagrange_tolerance
        ),
        "continuity_residual_local": bool(
            solver["continuity_residual_max_abs"]
            <= config.verification_continuity_residual_tolerance
        ),
        "continuity_global_balance": bool(
            abs(solver["volume_weighted_continuity_residual"])
            <= config.verification_global_balance_tolerance
        ),
        "mass_border_source_local": bool(
            solver["mass_border_source_max_abs"]
            <= config.verification_border_source_tolerance
        ),
        "continuity_lambda_invariant": bool(
            invariant_available
            and abs(float(invariant_value))
            <= config.verification_global_balance_tolerance
        ),
    }
    model_name = (
        "Rana supplied old-code 17-state R13 source-form reconstruction"
        if exact_archive_mode
        else (
            "Private experimental mass-conservative archive-derived R13 formulation"
            if config.continuity_mode == "conservative-fv"
            else "Private Rana-matrix wall-audit derivative"
        )
    )
    metrics_gate_passed = bool(all(physical_consistency_gates.values()))
    metrics: dict[str, object] = cavity_metrics(state, config)
    metrics.update(
        {
            "valid": metrics_gate_passed,
            "status": (
                "accepted private numerical solution"
                if metrics_gate_passed
                else "nonconverged diagnostic only; D/G must not be cited"
            ),
        }
    )
    result: dict[str, object] = {
        "evidence_schema_version": 3,
        "model": model_name,
        "reference": (
            "Rana, Torrilhon & Struchtrup, Journal of Computational Physics "
            "236 (2013) 169--186"
        ),
        "configuration": asdict(config),
        "formulation": (
            "archive-source-form-selection-under-missing-driver-assumptions"
            if exact_archive_mode
            else (
                "experimental-mass-conservative-archive-derived-formulation"
                if config.continuity_mode == "conservative-fv"
                else "author-bulk-matrices-with-controlled-wall-audit-variants"
            )
        ),
        "solver": solver,
        "metrics": metrics,
        "metrics_valid": metrics_gate_passed,
        "metrics_status": metrics["status"],
        "state_file": state_file,
        "state_semantics": (
            "numerically_converged_pending_provenance"
            if metrics_gate_passed
            else "last_valid_nonconverged_state"
        ),
        "execution_gates": execution_gates,
        "numerical_solution_gates": numerical_solution_gates,
        "archive_reproduction_gates": archive_reproduction_gates,
        "physical_consistency_gates": physical_consistency_gates,
        "passed_execution_gates": bool(all(execution_gates.values())),
        "passed_archive_reproduction_gates": bool(
            all(archive_reproduction_gates.values())
        ),
        "passed_physical_consistency_gates": bool(
            all(physical_consistency_gates.values())
        ),
        # Backward-compatible names.  A finite file alone is not an
        # operationally successful solve; paper verification additionally
        # requires the mass border to be inactive.
        "passed_operational_gates": bool(
            all(numerical_solution_gates.values())
        ),
        "passed_verification_gates": bool(
            all(physical_consistency_gates.values())
        ),
        "publication_grade": False,
        "scientific_status": (
            "private verification implementation; independent MATLAB-output "
            "comparison, unresolved paper/archive coefficient choices, and "
            "grid convergence remain separate publication gates"
        ),
    }
    return result


def run(
    config: RanaOriginalConfig,
    output_dir: Path,
    *,
    initial: np.ndarray | None = None,
    initial_state_path: Path | None = None,
    continuation_reason: str | None = None,
) -> dict[str, object]:
    """Solve locally and emit a self-hashed private evidence bundle."""
    output_dir = Path(output_dir)
    privacy_checks = assert_private_local_output(output_dir)
    if initial_state_path is not None and (
        Path(initial_state_path).resolve() == (output_dir / "state.npy").resolve()
    ):
        raise ValueError("output directory must not overwrite the initial state")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(__file__).resolve()
    coefficient_path = source_path.with_name("rana_original_coefficients.py")
    source_hash_start = sha256_file(source_path)
    coefficient_hash_start = sha256_file(coefficient_path)

    initial_provided = initial is not None or initial_state_path is not None
    initial_provenance: dict[str, object] = {"provided": initial_provided}
    if initial_provided:
        if initial_state_path is None:
            initial_provenance.update(
                {
                    "tracking": "in-memory state; source path and hash unavailable",
                    "file_stable_while_loaded": False,
                    "pairing_verified": False,
                    "source_report_passed_private_run_gates": False,
                    "source_report_schema_accepted": False,
                }
            )
        else:
            source_state = Path(initial_state_path).resolve()
            source_digest_before_load = sha256_file(source_state)
            loaded_initial = np.load(source_state, allow_pickle=False)
            source_digest = sha256_file(source_state)
            if source_digest_before_load != source_digest:
                raise RuntimeError("initial state changed while it was being loaded")
            if initial is not None and not np.array_equal(
                np.asarray(initial), loaded_initial, equal_nan=True
            ):
                raise ValueError(
                    "initial array does not match the hashed initial-state file"
                )
            initial = loaded_initial
            source_report = source_state.parent / "report.json"
            initial_provenance.update(
                {
                    "tracking": "local file",
                    "path": source_state.name,
                    "sha256": source_digest,
                    "file_stable_while_loaded": True,
                    "continuation_reason": continuation_reason
                    or "user-specified continuation",
                    "source_report": None,
                    "source_report_sha256": None,
                    "pairing_verified": False,
                    "source_report_passed_private_run_gates": False,
                    "source_report_schema_accepted": False,
                    "configuration_changes": None,
                }
            )
            if source_report.exists():
                source_report_bytes = source_report.read_bytes()
                previous = strict_json_loads(source_report_bytes.decode("utf-8"))
                if not isinstance(previous, dict):
                    raise ValueError("sibling report must contain a JSON object")
                previous_provenance = previous.get("execution_provenance", {})
                expected_digest = previous_provenance.get("output_state_sha256")
                if expected_digest is not None and expected_digest != source_digest:
                    raise ValueError(
                        "initial state SHA-256 does not match its sibling report"
                    )
                previous_configuration = previous.get("configuration")
                changes = None
                if isinstance(previous_configuration, dict):
                    current_configuration = asdict(config)
                    changes = {
                        key: {
                            "source": previous_configuration.get(key),
                            "current": value,
                        }
                        for key, value in current_configuration.items()
                        if previous_configuration.get(key) != value
                    }
                initial_provenance.update(
                    {
                        "source_report": source_report.name,
                        "source_report_sha256": sha256_bytes(source_report_bytes),
                        "pairing_verified": expected_digest is not None,
                        "source_report_passed_private_run_gates": bool(
                            previous.get("passed_private_run_gates", False) is True
                        ),
                        "source_report_schema_accepted": bool(
                            previous.get("evidence_schema_version") == 3
                            and previous.get("metrics_valid") is True
                        ),
                        "configuration_changes": changes,
                    }
                )
    state, solver = solve_rana_original(
        config,
        initial=initial,
        callback=lambda item: print(
            json.dumps(json_safe(item), sort_keys=True, allow_nan=False),
            flush=True,
        ),
    )
    source_hash_end = sha256_file(source_path)
    coefficient_hash_end = sha256_file(coefficient_path)
    source_unchanged = bool(
        source_hash_start == source_hash_end
        and coefficient_hash_start == coefficient_hash_end
    )
    state_path = output_dir / "state.npy"
    temporary_state_path = output_dir / ".state.tmp.npy"
    np.save(temporary_state_path, state)
    temporary_state_path.replace(state_path)
    result = build_result(config, state, solver, state_file=state_path.name)
    result["execution_provenance"] = {
        **privacy_checks,
        "solver_source": source_path.name,
        "solver_source_sha256_at_start": source_hash_start,
        "solver_source_sha256_at_end": source_hash_end,
        "coefficient_source": coefficient_path.name,
        "coefficient_source_sha256_at_start": coefficient_hash_start,
        "coefficient_source_sha256_at_end": coefficient_hash_end,
        "source_unchanged_during_execution": source_unchanged,
        "output_state": state_path.name,
        "output_state_sha256": sha256_file(state_path),
        "initial_state": initial_provenance,
        "runtime_environment": runtime_environment(),
    }
    provenance_gates = {
        "source_unchanged_during_execution": source_unchanged,
        "initial_state_traceable": bool(
            not initial_provided
            or (
                initial_provenance.get("file_stable_while_loaded", False)
                and initial_provenance.get("pairing_verified", False)
                and initial_provenance.get(
                    "source_report_passed_private_run_gates", False
                )
                and initial_provenance.get(
                    "source_report_schema_accepted", False
                )
            )
        ),
    }
    result["provenance_gates"] = provenance_gates
    result["passed_provenance_gates"] = bool(all(provenance_gates.values()))
    result["passed_verification_gates"] = bool(
        result["passed_physical_consistency_gates"]
        and result["passed_provenance_gates"]
    )
    result["passed_private_run_gates"] = bool(
        (
            result["passed_physical_consistency_gates"]
            if config.continuity_mode == "conservative-fv"
            else result["passed_archive_reproduction_gates"]
        )
        and result["passed_provenance_gates"]
    )
    physical_metrics_valid = bool(
        result["passed_physical_consistency_gates"]
        and result["passed_provenance_gates"]
    )
    result["metrics"]["valid"] = physical_metrics_valid
    result["metrics"]["status"] = (
        "accepted private numerical solution with verified local provenance"
        if physical_metrics_valid
        else "nonconverged or unverified diagnostic only; D/G must not be cited"
    )
    result["metrics_valid"] = physical_metrics_valid
    result["metrics_status"] = result["metrics"]["status"]
    if physical_metrics_valid:
        result["state_semantics"] = "accepted_private_physical_solution"
    elif result["passed_archive_reproduction_gates"] and result[
        "passed_provenance_gates"
    ]:
        result["state_semantics"] = "accepted_archive_replay_diagnostic_state"
    else:
        result["state_semantics"] = "last_valid_nonconverged_or_unverified_state"
    result = json_safe(result)
    if not isinstance(result, dict):  # pragma: no cover - structural invariant.
        raise TypeError("sanitized report is not a mapping")
    temporary_report_path = output_dir / ".report.tmp.json"
    temporary_report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary_report_path.replace(output_dir / "report.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--kn", type=float, default=0.5 * np.sqrt(2.0 / np.pi))
    parser.add_argument(
        "--lid-velocity", type=float, default=50.0 / np.sqrt(208.0 * 273.0)
    )
    parser.add_argument(
        "--rb",
        type=float,
        default=1.0,
        help="archive switch/coefficient Rb; the supplied bundle contains no driver",
    )
    parser.add_argument(
        "--ra",
        type=float,
        default=1.0,
        help="archive switch/coefficient Ra; the supplied bundle contains no driver",
    )
    parser.add_argument(
        "--ma",
        type=float,
        default=1.0,
        help="archive switch/coefficient Ma; the supplied bundle contains no driver",
    )
    parser.add_argument(
        "--boundary-scheme",
        choices=("legacy-hobc3", "paper-linear"),
        default="legacy-hobc3",
    )
    parser.add_argument(
        "--pressure-mode",
        choices=("legacy-normal", "paper-tangential"),
        default="legacy-normal",
    )
    parser.add_argument(
        "--continuity-mode",
        choices=("legacy-border", "conservative-fv"),
        default="legacy-border",
    )
    parser.add_argument(
        "--conservative-linearization",
        choices=("density-picard", "defect-newton"),
        default="defect-newton",
    )
    parser.add_argument(
        "--nonlinear-solver", choices=("frozen", "jfnk"), default="frozen"
    )
    parser.add_argument("--linear-solver", choices=("qmr", "direct"), default="direct")
    parser.add_argument("--qmr-rtol", type=float, default=1.0e-10)
    parser.add_argument("--qmr-maxiter", type=int, default=30000)
    parser.add_argument("--outer-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--max-outer-iterations", type=int, default=21)
    parser.add_argument("--outer-relaxation", type=float, default=1.0)
    parser.add_argument(
        "--outer-globalization",
        choices=("fixed", "residual-backtracking"),
        default="fixed",
    )
    parser.add_argument("--line-search-reduction", type=float, default=0.5)
    parser.add_argument("--line-search-min-step", type=float, default=1.0 / 128.0)
    parser.add_argument("--line-search-armijo", type=float, default=1.0e-4)
    parser.add_argument("--physical-floor", type=float, default=1.0e-12)
    parser.add_argument(
        "--jfnk-fd-relative-step",
        type=float,
        default=float(np.cbrt(np.finfo(float).eps)),
    )
    parser.add_argument("--jfnk-max-fd-halvings", type=int, default=10)
    parser.add_argument("--jfnk-gmres-restart", type=int, default=40)
    parser.add_argument("--jfnk-gmres-max-cycles", type=int, default=10)
    parser.add_argument("--jfnk-initial-forcing", type=float, default=1.0e-2)
    parser.add_argument("--jfnk-min-forcing", type=float, default=1.0e-6)
    parser.add_argument("--jfnk-max-forcing", type=float, default=1.0e-1)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument(
        "--continuation-reason",
        help="private provenance note recorded when --initial-state is used",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = RanaOriginalConfig(
        nx=args.nx,
        ny=args.ny,
        kn=args.kn,
        lid_velocity=args.lid_velocity,
        rb=args.rb,
        ra=args.ra,
        ma=args.ma,
        boundary_scheme=args.boundary_scheme,
        pressure_mode=args.pressure_mode,
        continuity_mode=args.continuity_mode,
        conservative_linearization=args.conservative_linearization,
        nonlinear_solver=args.nonlinear_solver,
        linear_solver=args.linear_solver,
        qmr_rtol=args.qmr_rtol,
        qmr_maxiter=args.qmr_maxiter,
        outer_tolerance=args.outer_tolerance,
        max_outer_iterations=args.max_outer_iterations,
        outer_relaxation=args.outer_relaxation,
        outer_globalization=args.outer_globalization,
        line_search_reduction=args.line_search_reduction,
        line_search_min_step=args.line_search_min_step,
        line_search_armijo=args.line_search_armijo,
        physical_floor=args.physical_floor,
        jfnk_fd_relative_step=args.jfnk_fd_relative_step,
        jfnk_max_fd_halvings=args.jfnk_max_fd_halvings,
        jfnk_gmres_restart=args.jfnk_gmres_restart,
        jfnk_gmres_max_cycles=args.jfnk_gmres_max_cycles,
        jfnk_initial_forcing=args.jfnk_initial_forcing,
        jfnk_min_forcing=args.jfnk_min_forcing,
        jfnk_max_forcing=args.jfnk_max_forcing,
    )
    result = run(
        config,
        args.output_dir,
        initial=None,
        initial_state_path=args.initial_state,
        continuation_reason=args.continuation_reason,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    # A faithful archive replay can be a successful executable even when its
    # independent physical-consistency gate exposes a defect in the archive.
    return 0 if result["passed_private_run_gates"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
