# Gu--Emerson / ASME-2009 transformed-variable R26 path

This branch separates two things that earlier development had conflated.

1. `r26_gu_emerson_variables.py` implements the exact algebraic changes of
   variables in Gu & Emerson, JFM 636 (2009), equations (48)--(55).
2. `r26_gu_emerson_transformed_fv.py` now discretizes the transformed PDE
   (63) itself.  The unknowns advanced by the segregated blocks are
   `u,T,g,h,omega,gamma,chi`; the block defect is no longer selected from the
   physical-moment finite-volume BVP.

## Paper-fixed numerical structure

The direct backend uses the structure printed in section 5.2:

- collocated finite volumes;
- CUBISTA convection of each transformed field;
- central face diffusion with the printed `mu/Gamma_Phi` coefficients;
- central source evaluation from the audited R26 tensor equations;
- SIMPLE pressure--velocity correction and Rhie--Chow mass fluxes;
- the printed segregated order `u -> SIMPLE -> T -> g -> h -> omega -> gamma
  -> chi -> physical moments -> wall conditions`.

The long right-hand sides of equations (58)--(62) are not copied by hand.
For auditability the source is formed by the exact central identity

`S_Phi = LHS_central(Phi) - R_physical_point(Phi)`.

Here `R_physical_point` is the independently tested pointwise R26 equation,
not the physical finite-volume defect.  Consequently the actual solved
residual is `LHS_FV_equation63 - S_Phi`, and the transformed block can be
tested to prove that it never calls the physical FV balance operator.

## Literal ASME cavity contract

`gu_asme2009_published_cavity_case` admits only the values printed in Gu,
John, Tang & Emerson, ASME MNHMT2009-18236:

- argon, `R=208 J/(kg K)`, `T0=Tw=273 K`;
- `mu0=21.25e-6 Pa s`, Sutherland temperature `144 K`;
- diffuse walls (`alpha=1`);
- uniform `100 x 100` grid points;
- lid speed `10` or `100 m/s`;
- `Kn=0.05,0.1,0.2,0.5` with the paper's mean-free-path convention;
- the preliminary ASME R26 coefficient set, never silently replaced by the
  final JFM coefficient set.

The paper leaves the dimensional cavity length `L` symbolic.  The project's
`50 micrometre` THOR comparison length is recorded separately and is not
attributed to the paper.

## Acceptance and current limitation

For the direct backend, convergence is fail-closed and simultaneous:

- transformed equation-(63) interior residual;
- independent complete raw physical R26 residual;
- scaled physical residual;
- held-out continuity, total mass, density positivity and temperature
  positivity.

The paper does not publish its linear solver, under-relaxation factors,
source-term linearisation, exact Rhie--Chow coefficient, sharp-corner rule or
residual thresholds.  The direct profile therefore discloses every external
control.  It does not claim to reproduce the unavailable private THOR code.

A local N8 diagnostic at `Kn=0.1`, `U=10 m/s` showed that applying the public
Code_Saturne steady defaults (`0.7` fields, `0.3` pressure) together with an
unrelaxed post-sweep wall update is not stable for this independent solver:
the physical raw residual grew from `1.47e-1` to `1.52` in eight sweeps.  A
wall-relaxation sensitivity was diagnostic only and is not promoted as a
paper control.  Therefore neither the literal `100 x 100` case nor the
`100 m/s` case is authorized by this branch until a source-backed treatment
of the unpublished controls passes reduced-grid transformed and physical
gates.
