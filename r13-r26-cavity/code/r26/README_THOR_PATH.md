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

## Numerical-rank/cross-solver audit and conditional N24

After the immutable N8/N16 THOR gate passes,
`r26_kn020_thor_cross_solver_n24.slurm` performs the declared next stage. It
independently revalidates the historical SER--PTC N8/N16 gate, recomputes both
solvers' raw residuals with their own spatial operators, builds the final
colored numerical THOR Jacobians, and obtains their actual dense singular
spectra after deterministic row/column scaling. A structural-rank declaration
or successful ILU is not treated as numerical full-rank evidence.

All 17 planar R26 components, including the four independent `m_ijk`
components, are compared on a common grid and on the vertical centreline and
the `y=0.9` horizontal line. Rana D/G differences are recorded separately.
Only if both N8 and N16 pass the raw and numerical-rank gates, all-component
normalized RMS disagreement stays below 5%, line disagreement below 15%, and
the Rana D/G disagreement below 2%, is one N24 fixed-case solve started from
the accepted THOR N16 state. N28 and N30 remain unauthorized, and even an
accepted N24 record keeps `production_accepted=false`.

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

## Source-linearized Gu--Emerson N16 resume

The standalone N8/N16 ladder established two independent accepted N8 roots,
but its first N16 path diverged because all nonlinear source terms were
included in the numerical field matrix.  The supplied Rana R26 module instead
keeps collision and diffusion implicit while applying fixed history factors to
the nonlinear right-hand sides: 0.01 for stress and heat flux, 0.5 for `m` and
`R`, and 0.1 for `Delta`.  The source-linearized resume reproduces that split,
reassembles every field matrix on every outer sweep, independently revalidates
the two accepted N8 states, and ignores every failed N16 checkpoint.

The resume remains a validation gate.  N24 is authorized only when both N16
paths pass the complete raw physical residual and agree within the declared
root tolerance; N28, N29, N30 and production acceptance remain blocked.
