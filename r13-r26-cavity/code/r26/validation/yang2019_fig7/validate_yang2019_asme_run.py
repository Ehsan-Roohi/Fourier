#!/usr/bin/env python3
"""Fail-closed validator for the source-matched ASME R26 audit."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"YANG2019_ASME_R26_RUN_FAILED: {message}")

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("output_dir",type=Path); p.add_argument("--expected-nodes",type=int,required=True); p.add_argument("--raw-tolerance",type=float,default=1e-8); a=p.parse_args()
    summary_path=a.output_dir/"run_summary.json"; state_path=a.output_dir/"last_accepted_state.npz"
    require(summary_path.is_file(),"run_summary.json missing"); require(state_path.is_file(),"last_accepted_state.npz missing")
    s=json.loads(summary_path.read_text()); c=s.get("case",{})
    require(s.get("termination")=="target_accepted","target root not accepted")
    require(c.get("family")=="gu-asme2009","wrong case family")
    require(c.get("kn_convention")=="gu_lambda_over_L","wrong Kn convention")
    require(math.isclose(float(c.get("kn_input")),.1,abs_tol=2e-15),"Kn is not 0.1")
    require(int(c.get("nodes"))==a.expected_nodes,"wrong grid size")
    require(c.get("closure_mode")=="asme2009-cavity","wrong complete closure set")
    require(c.get("viscosity_kind")=="gu_sutherland","wrong viscosity law")
    require(math.isclose(float(c.get("wall_accommodation")),1.,abs_tol=2e-15),"wall not diffuse")
    require(math.isclose(float(c.get("wall_temperature_K")),273.,abs_tol=2e-12),"wrong wall temperature")
    require(math.isclose(float(c.get("lid_speed_m_per_s")),10.,abs_tol=2e-12),"wrong lid speed")
    require(math.isclose(float(c.get("mu_equilibrium")),.1*math.sqrt(2/math.pi),abs_tol=2e-15),"Gu Kn conversion mismatch")
    final=s.get("attempts",[])[-1]; require(bool(final.get("accepted")),"last attempt rejected"); require(float(final.get("raw_acceptance_gate"))<=a.raw_tolerance,"raw residual gate failed")
    d=final.get("diagnostics",{}); require(float(d.get("min_density"))>0,"non-positive density"); require(float(d.get("min_temperature"))>0,"non-positive temperature")
    b=final.get("global_balances",{}); require(float(b.get("wall_effective_pressure_min"))>0,"non-positive wall pressure"); require(float(b.get("momentum_boundary_flux_linf"))<=10*a.raw_tolerance,"momentum balance failed"); require(abs(float(b.get("internal_energy_balance_error")))<=10*a.raw_tolerance,"energy balance failed")
    with np.load(state_path,allow_pickle=False) as z:
        state=np.asarray(z["state"],float); require(state.shape==(a.expected_nodes,a.expected_nodes,17),"wrong state shape"); require(np.isfinite(state).all(),"non-finite state")
    print(json.dumps({"status":"YANG2019_ASME_R26_RUN_PASS","nodes":a.expected_nodes,"state_file_sha256":hashlib.sha256(state_path.read_bytes()).hexdigest()},sort_keys=True))
if __name__=="__main__": main()
