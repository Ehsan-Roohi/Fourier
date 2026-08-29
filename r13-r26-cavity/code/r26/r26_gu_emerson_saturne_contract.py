#!/usr/bin/env python3
"""Fail-closed source contract for the Gu--Emerson/Code_Saturne carrier.

The 2009 JFM paper publishes the segregated R26 field order but not enough
numerical controls to reproduce a calculation.  The supplied Rana routines
and the exact Code_Saturne 5.0.3 core close part of that gap.  This module
records only facts that can be traced to those sources and keeps the missing
historical case files, thermal limitation, and high-speed authorization
machine readable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final


CODE_SATURNE_TAG: Final[str] = "v5.0.3"
CODE_SATURNE_COMMIT: Final[str] = "e17068ce692ad2d90c694d375b7c098043b16969"
CODE_SATURNE_CORE_SHA256: Final[dict[str, str]] = {
    "src/base/iniini.f90": "d3aac378a08b420933bf78ab73311f9050c67d4f123f64006b5f953799e27993",
    "src/base/modini.f90": "d262eddfa634c43755ea4291f7c38ff44a83205223b5678d45b54e2bc0353477",
    "src/base/resopv.f90": "adda23ddd1e93e7b0b0a387e56890efdd26f29a766363aecae0da4297d438261",
    "src/base/navstv.f90": "ff5a8813ac977bc36353389e7a80c27478f2911d852a6eda3e388b011d226802",
    "src/base/cs_stokes_model.c": "c6261d7e689d92522f5c1fa71671478ae1611a6a9e67e21fc759d63c4e72ba85",
}
RANA_USER_SOURCE_SHA256: Final[dict[str, str]] = {
    "cs_user_parameters_R26.f90": "0ce53e0811b00154fc0b3c7cb370cfce92a9382d53741a04758499c8132a13ca",
    "cs_user_physical_properties_R26.f90": "a01d309692acf26093c65aa4c11453afc07f3f98b7be1bb8f2c1ea7ba2e44d5d",
    "cs_user_modules.f90": "d92e0142776d90499e2beea4a8b3b37b590597f66b61f43bb49f58ade73a884b",
}
MNHMT2024_PDF_SHA256: Final[str] = (
    "332669decfc8f58a1229b9f4141c0413b28782bfe8cee7f76b46cb932fc894ce"
)


@dataclass(frozen=True)
class SaturneCarrierEvidence:
    code_saturne_tag: str = CODE_SATURNE_TAG
    code_saturne_commit: str = CODE_SATURNE_COMMIT
    steady_field_relaxation: float = 0.7
    steady_pressure_relaxation: float = 0.3
    rhie_chow_default_factor: float = 1.0
    density_update: str = "rho_new=0.0002*p_total/(R*T)+0.9998*rho_old"
    pressure_update: str = "independent gauge pressure; relaxed stored pressure"
    velocity_flux_update: str = "full pressure-increment correction"
    published_cavity_lid_speed_m_s: float = 10.0
    published_cavity_solver: str = "Code_Saturne incompressible carrier"
    thermal_high_mach_limitation: str = (
        "MNHMT 2024 states that compressible use requires solving the energy "
        "equation and developing new energy wall boundary conditions"
    )
    historical_case_inputs_available: bool = False
    historical_reproduction_authorized: bool = False
    high_speed_authorized: bool = False
    n24_authorized: bool = False
    n28_authorized: bool = False
    n29_authorized: bool = False
    n30_authorized: bool = False

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["code_saturne_core_sha256"] = dict(CODE_SATURNE_CORE_SHA256)
        record["rana_user_source_sha256"] = dict(RANA_USER_SOURCE_SHA256)
        record["mnhmt2024_pdf_sha256"] = MNHMT2024_PDF_SHA256
        record["missing_historical_inputs"] = [
            "setup.xml",
            "run.cfg",
            "mesh",
            "listing/convergence history",
        ]
        return record


def saturne_carrier_evidence() -> dict[str, object]:
    """Return the immutable, deliberately non-authorizing source record."""

    return SaturneCarrierEvidence().as_record()


__all__ = [
    "CODE_SATURNE_COMMIT",
    "CODE_SATURNE_CORE_SHA256",
    "CODE_SATURNE_TAG",
    "MNHMT2024_PDF_SHA256",
    "RANA_USER_SOURCE_SHA256",
    "SaturneCarrierEvidence",
    "saturne_carrier_evidence",
]
