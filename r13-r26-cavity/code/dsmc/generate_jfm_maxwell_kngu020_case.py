#!/usr/bin/env python3
"""Generate one transport-matched Maxwell-VSS SPARTA cavity case.

The collision law is the standard VSS approximation to inverse-power-law
Maxwell molecules: omega=1 and alpha=2.140.  The input VSS diameter and the
viscosity-equivalent VHS diameter are deliberately kept separate.  The latter
is the diameter that enters the Gu viscosity-based mean-free-path definition.

The production dump includes the seven lower-order fields used previously,
four COM-subtracted momentum-flux components, and four standard SPARTA
``sonine/grid`` B1 components.  The latter are raw fourth-order moments; they
must not be relabelled as a complete R26 tensor without the documented
post-processing transformation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


K_B = 1.380649e-23
MASS = 6.6335e-26
TEMPERATURE_REF = 273.0
WALL_TEMPERATURE = 300.0
LID_VELOCITY = 100.0

# Reference argon VHS calibration used by the earlier omega=0.81 case.
BASE_DIAMETER_REF = 4.17e-10
BASE_VISCOSITY_INDEX = 0.81

# Koura--Matsumoto VSS representation of IPL Maxwell molecules.
VISCOSITY_INDEX = 1.0
VSS_ALPHA = 2.140

DUMP_COLUMNS = [
    "nrho",
    "u",
    "v",
    "w",
    "T",
    "qx",
    "qy",
    "Pxx",
    "Pxy",
    "Pyy",
    "Pzz",
    "B1xx",
    "B1xy",
    "B1yy",
    "B1zz",
]


def viscosity_denominator(omega: float) -> float:
    return (5.0 - 2.0 * omega) * (7.0 - 2.0 * omega)


VSS_AREA_FACTOR = (
    (1.0 + VSS_ALPHA) * (2.0 + VSS_ALPHA) / (6.0 * VSS_ALPHA)
)

# First preserve the equilibrium viscosity at 300 K when changing omega from
# 0.81 to 1.0.  Then enlarge the total-collision VSS diameter so that its
# momentum-transfer cross section equals that of the viscosity-equivalent VHS
# model.  SPARTA reads DIAMETER_VSS_INPUT; Kn_Gu uses DIAMETER_VHS_EQUIVALENT.
DIAMETER_VHS_EQUIVALENT = BASE_DIAMETER_REF * math.sqrt(
    viscosity_denominator(BASE_VISCOSITY_INDEX)
    / viscosity_denominator(VISCOSITY_INDEX)
    * (WALL_TEMPERATURE / TEMPERATURE_REF)
    ** (VISCOSITY_INDEX - BASE_VISCOSITY_INDEX)
)
DIAMETER_VSS_INPUT = DIAMETER_VHS_EQUIVALENT * math.sqrt(VSS_AREA_FACTOR)


def gu_mean_free_path(
    number_density: float,
    temperature: float = WALL_TEMPERATURE,
) -> float:
    """Gu viscosity-based mean free path for the equivalent VHS diameter."""

    return 15.0 / (
        2.0
        * math.sqrt(2.0)
        * viscosity_denominator(VISCOSITY_INDEX)
        * DIAMETER_VHS_EQUIVALENT**2
        * number_density
    ) * (temperature / TEMPERATURE_REF) ** (VISCOSITY_INDEX - 0.5)


def physical_parameters(
    kn_gu: float,
    length: float,
    nx: int,
    ppc: int,
    temperature: float,
) -> dict[str, float | int]:
    target_lambda_gu = kn_gu * length
    number_density = 15.0 / (
        2.0
        * math.sqrt(2.0)
        * viscosity_denominator(VISCOSITY_INDEX)
        * DIAMETER_VHS_EQUIVALENT**2
        * target_lambda_gu
    ) * (temperature / TEMPERATURE_REF) ** (VISCOSITY_INDEX - 0.5)

    nparticles = nx * nx * ppc
    # SPARTA 2-D uses unit depth, so the represented molecule count is n*L^2.
    fnum = number_density * length**2 / nparticles
    most_probable_speed = math.sqrt(2.0 * K_B * temperature / MASS)
    dx = length / nx
    dt = 0.25 * dx / (4.0 * most_probable_speed + LID_VELOCITY)

    # This collision-diameter value controls the conservative timestep check;
    # it is not the viscosity-based Gu mean free path.
    lambda_collision = 1.0 / (
        math.sqrt(2.0)
        * math.pi
        * DIAMETER_VSS_INPUT**2
        * number_density
    ) * (temperature / TEMPERATURE_REF) ** (VISCOSITY_INDEX - 0.5)
    lambda_gu = gu_mean_free_path(number_density, temperature)
    collision_time = lambda_collision / most_probable_speed

    reconstructed_kn_gu = lambda_gu / length
    if not math.isclose(lambda_gu, target_lambda_gu, rel_tol=2.0e-15):
        raise RuntimeError("Maxwell-VSS Kn_Gu contract failed")
    if not math.isclose(reconstructed_kn_gu, kn_gu, rel_tol=2.0e-15):
        raise RuntimeError("Maxwell-VSS reconstructed Kn_Gu contract failed")

    return {
        "mean_free_path_gu_m": lambda_gu,
        "mean_free_path_collision_diameter_m": lambda_collision,
        "kn_gu_reconstructed": reconstructed_kn_gu,
        "kn_collision_diameter": lambda_collision / length,
        "number_density_m-3": number_density,
        "initial_simulator_particles": nparticles,
        "fnum": fnum,
        "dx_m": dx,
        "dx_over_lambda_gu": dx / lambda_gu,
        "dx_over_lambda_collision_diameter": dx / lambda_collision,
        "dt_s": dt,
        "mean_collision_time_s": collision_time,
        "dt_over_collision_time": dt / collision_time,
    }


def write_case(
    output: Path,
    *,
    seed: int,
    kn_gu: float = 0.20,
    length: float = 1.0e-6,
    nx: int = 160,
    ppc: int = 256,
    warmup_steps: int = 40_000,
    sample_steps: int = 200_000,
    sample_stride: int = 10,
    checkpoint_steps: int = 40_000,
) -> dict[str, object]:
    if min(kn_gu, length, nx, ppc, sample_steps, sample_stride) <= 0:
        raise ValueError("Kn, length, grid, PPC, sample steps, and stride must be positive")
    if warmup_steps < 0 or checkpoint_steps < 0:
        raise ValueError("warmup and checkpoint steps cannot be negative")
    if sample_steps % sample_stride:
        raise ValueError("sample_steps must be divisible by sample_stride")
    if checkpoint_steps and checkpoint_steps % sample_stride:
        raise ValueError("checkpoint_steps must be divisible by sample_stride")

    output.mkdir(parents=True, exist_ok=False)
    values = physical_parameters(kn_gu, length, nx, ppc, WALL_TEMPERATURE)
    samples_per_cell = sample_steps // sample_stride

    (output / "argon.species").write_text(
        "# ID molwt(amu) mass(kg) rotdof rotrel vibdof vibrel vibtemp(K) weight charge\n"
        f"Ar 40.00 {MASS:.10e} 0 0.0 0 0.0 0.0 1.0 0.0\n",
        encoding="utf-8",
    )
    (output / "maxwell.vss").write_text(
        "# ID diameter_VSS(m) omega Tref(K) alpha\n"
        f"Ar {DIAMETER_VSS_INPUT:.16e} {VISCOSITY_INDEX:.8g} "
        f"{TEMPERATURE_REF:.8g} {VSS_ALPHA:.8g}\n",
        encoding="utf-8",
    )

    kn_tag = f"{round(100.0 * kn_gu):03d}"
    checkpoint_block = ""
    if checkpoint_steps:
        checkpoint_block = f"""dump                 checkpoints grid all {checkpoint_steps} grid.checkpoint.* id xc yc f_fieldavg[*]
dump_modify          checkpoints pad 8
restart              {checkpoint_steps} restart.maxwell{kn_tag}.1 restart.maxwell{kn_tag}.2
"""

    deck = f"""# Generated by generate_jfm_maxwell_kngu020_case.py; do not hand-edit.
# Transport-matched VSS approximation to IPL Maxwell molecules.
# Exact operating point: Kn_Gu={kn_gu:.16g}, omega=1, alpha={VSS_ALPHA:.8g}.

units                si
seed                 {seed}
dimension            2
boundary             s s p

create_box           0.0 {length:.16e} 0.0 {length:.16e} -0.5 0.5
create_grid          {nx} {nx} 1

global               nrho {values['number_density_m-3']:.16e} fnum {values['fnum']:.16e} temp {WALL_TEMPERATURE:.8g}
species              argon.species Ar
mixture              gas Ar nrho {values['number_density_m-3']:.16e} temp {WALL_TEMPERATURE:.8g} vstream 0.0 0.0 0.0

surf_collide         fixed diffuse {WALL_TEMPERATURE:.8g} 1.0
surf_collide         lid diffuse {WALL_TEMPERATURE:.8g} 1.0 translate {LID_VELOCITY:.8g} 0.0 0.0
bound_modify         xlo xhi ylo collide fixed
bound_modify         yhi collide lid

collide              vss gas maxwell.vss
create_particles     gas n 0
timestep             {values['dt_s']:.16e}

stats                1000
stats_style          step cpu np nattempt ncoll nscoll
run                  {warmup_steps}

reset_timestep       0
compute              flow grid all gas nrho u v w
compute              thermal thermal/grid all gas temp
compute              heat eflux/grid all gas heatx heaty
# Momentum flux gives direct Pxx, Pxy, Pyy and Pzz.  Pxy is also the
# off-diagonal stress.  Standard upstream SPARTA does not directly output the
# complete rank-three R26 m_ijk tensor.
compute              stress pflux/grid all gas momxx momxy momyy momzz
# B1ij = <C_i C_j C^2> is the standard raw fourth-order Sonine moment.
compute              sonine sonine/grid all gas b xx 1 b xy 1 b yy 1 b zz 1
fix                  fieldavg ave/grid all {sample_stride} 1 {sample_stride} c_flow[*] c_thermal[*] c_heat[*] c_stress[*] c_sonine[*] ave running
{checkpoint_block}dump                 final grid all {sample_steps} grid.final.* id xc yc f_fieldavg[*]
dump_modify          final pad 8
run                  {sample_steps}
"""
    (output / "in.cavity").write_text(deck, encoding="utf-8")

    metadata: dict[str, object] = {
        "case": "JFM Maxwell-VSS single-realisation lid-driven cavity",
        "molecular_model": "VSS transport approximation to IPL Maxwell molecules",
        "collision_model_scope": (
            "omega=1 and alpha=2.140 reproduce the Maxwell-molecule transport "
            "class within the VSS approximation; this is not the exact IPL angular kernel"
        ),
        "kn": kn_gu,
        "kn_gu": kn_gu,
        "kn_convention": "gu_lambda_over_L",
        "kn_definition": (
            "lambda_Gu/L, evaluated with the viscosity-equivalent VHS diameter; "
            "lambda_Gu=15/[2*sqrt(2)*(5-2*omega)*(7-2*omega)*"
            "d_eq^2*n]*(T/Tref)^(omega-1/2)"
        ),
        "length_m": length,
        "nx": nx,
        "ny": nx,
        "particles_per_cell": ppc,
        "wall_temperature_K": WALL_TEMPERATURE,
        "lid_velocity_m_per_s": LID_VELOCITY,
        "argon_mass_kg": MASS,
        "diameter_vss_input_m": DIAMETER_VSS_INPUT,
        "diameter_vhs_viscosity_equivalent_m": DIAMETER_VHS_EQUIVALENT,
        "vss_area_factor": VSS_AREA_FACTOR,
        "temperature_ref_K": TEMPERATURE_REF,
        "viscosity_index": VISCOSITY_INDEX,
        "vss_alpha": VSS_ALPHA,
        "viscosity_calibration": (
            "mu(300 K) matched to the prior d_ref=4.17e-10 m, omega=0.81 VHS operating point"
        ),
        "warmup_steps": warmup_steps,
        "sample_steps": sample_steps,
        "sample_stride": sample_stride,
        "accumulated_samples_per_cell": samples_per_cell,
        "checkpoint_frequency_steps": checkpoint_steps,
        "seed": seed,
        "wall_model": "fully diffuse, full thermal accommodation",
        "temperature_observable": "thermal/grid COM-subtracted translational temperature",
        "heat_flux_observable": "eflux/grid COM-subtracted heat-flux density",
        "dump_schema_version": "maxwell_vss_antifourier_v1_15_fields",
        "dump_field_count": len(DUMP_COLUMNS),
        "dump_columns": DUMP_COLUMNS,
        "moment_sampling": {
            "momentum_flux_compute": "pflux/grid momxx momxy momyy momzz",
            "momentum_flux_columns": ["Pxx", "Pxy", "Pyy", "Pzz"],
            "off_diagonal_stress_xy": "Pxy (direct COM-subtracted momentum-flux density)",
            "sonine_compute": "sonine/grid b xx 1 b xy 1 b yy 1 b zz 1",
            "sonine_columns": ["B1xx", "B1xy", "B1yy", "B1zz"],
            "sonine_definition": "B1ij=<C_i*C_j*C^2>, mass-weighted per-particle mean",
            "sonine_com_averaging": "cell COM is subtracted separately at each sampled timestep",
            "instantaneous_COM_sonine": True,
            "sonine_role": "diagnostic_only",
            "pooled_COM_fourth_moment_available": False,
            "quantitative_R_or_Delta_claim_ready": False,
            "direct_rank3_moment_m_ijk_available": False,
            "full_r26_higher_moment_claim": False,
        },
        "evidence_level": "single_realisation_model_audit",
        **values,
    }
    (output / "case_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--kn-gu", type=float, default=0.20)
    parser.add_argument("--length", type=float, default=1.0e-6)
    parser.add_argument("--nx", type=int, default=160)
    parser.add_argument("--ppc", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=40_000)
    parser.add_argument("--sample", type=int, default=200_000)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--checkpoint", type=int, default=40_000)
    args = parser.parse_args()
    metadata = write_case(
        args.output.resolve(),
        seed=args.seed,
        kn_gu=args.kn_gu,
        length=args.length,
        nx=args.nx,
        ppc=args.ppc,
        warmup_steps=args.warmup,
        sample_steps=args.sample,
        sample_stride=args.stride,
        checkpoint_steps=args.checkpoint,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
