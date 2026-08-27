# THOR-style pressure-based R26 path

This path is a new numerical solver for the existing audited nonlinear R26
equations. It is not another lid-continuation or pseudo-arclength rescue.

## What is reused

| Source | Reused | Not reused |
|---|---|---|
| Supplied Rana R13 Code_Saturne | segregated field layout, under-relaxed defect-correction idea, pressure-based CFD architecture | R13 coefficient matrices and 13-moment closure |
| Supplied Rana R26 Code_Saturne | explicit stress, heat-flux, `m`, `R`, and `Delta` field families; source/wall organization | active `Kn=0.02`, `Re=50` case and its `2e-4` property relaxation |
| Gu--Emerson THOR method | collocated finite volume, CUBISTA convection, SIMPLE, Rhie--Chow | unpublished/private THOR implementation details |
| Current Python R26 | all Gu--Emerson bulk equations, closures, Maxwell walls, positivity transform, mass border and raw acceptance gates | the failed N30 fixed-lid/arclength path as a solver strategy |

`r26_fv_backend.py` contains the bounded CUBISTA normalized-variable face
interpolation and the shared Rhie--Chow coefficient. `r26_thor_solver.py`
uses a conservative SIMPLE pressure-correction operator and one frozen,
colored CUBISTA defect Jacobian with incomplete LU. The frozen Jacobian is
built once; it is not rebuilt and factorized at every outer iteration.

The outer function always evaluates the complete audited R26 BVP. Therefore
the preconditioner cannot change the physical root. A run is a validation
candidate only when the unscaled residual, held continuity, mass, positivity,
effective wall pressure, momentum balance and energy balance all pass.

## Local fixed-case validation

```bash
cd r13-r26-cavity/code/r26
export PYTHONPATH="$PWD"

python3 analysis/run_r26_thor_validation.py \
  --case-family jfm-maxwell \
  --nodes 8 \
  --kn-gu 0.2 \
  --lid-speed-m-s 100 \
  --output-dir /tmp/r26-thor-n8
```

The output record deliberately keeps `production_accepted=false`. A physical
candidate still needs an independent final numerical-Jacobian rank check and
cross-solver/profile comparison.

## Unity N8/N16 gate

Fetch `hpc/submit_r26_kn020_thor_gate_n8_n16.sh` from an immutable commit and
set `R26_THOR_REF` to that exact 40-character SHA. The gate:

1. runs the complete unit/contract suite;
2. solves N8 directly from analytic equilibrium at 100 m/s;
3. interpolates the accepted N8 candidate to N16 and reconciles it using the
   new CUBISTA operator;
4. checks all raw physical and global-balance gates; and
5. packages both states while leaving N28, N30 and production acceptance
   explicitly unauthorized.

No N28/N30 job is part of this commit. The next stage after a passed N8/N16
gate is an independent final-Jacobian and profile comparison, not another
continuation run.

## Auditing the supplied Code_Saturne sources

The legacy source ZIP itself is not redistributed. After extracting its R13
and R26 directories, authenticate it with:

```bash
python3 tools/audit_rana_code_saturne_sources.py \
  --r13-dir /path/to/SRCR13_22nd_NOV \
  --r26-dir /path/to/SRCR26_22nd_NOV
```

This authorizes architecture reuse only. The supplied archive has no mesh or
result fields and is not a numerical reference for the present 100 m/s case.
