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
- segregated Picard linearisation: source, viscosity and mass flux are frozen
  at the entry to each field block and rebuilt before the next printed field;
- the dissipative linear collision sinks printed in equations (58)--(62) are
  retained on the implicit block diagonal;
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

The numerical block Jacobian previously differentiated the right-hand side
together with the transported field, cancelling much of the intended implicit
convection--diffusion operator.  The direct backend now freezes the remaining
right-hand side and transport coefficients within each printed field block,
retains the dissipative collision terms on the block diagonal, and obtains
SIMPLE continuity from the direct transformed backend rather than the physical
finite-volume acceptance operator.

The corrected local N8 diagnostic at `Kn=0.1`, `U=10 m/s` still fails closed.
With the public Code_Saturne steady field/pressure defaults (`0.7`, `0.3`) and
an unrelaxed post-sweep wall update, the complete physical raw gate grows from
`1.46877e-1` to `4.53504` in eight sweeps.  A declared local wall correction of
`0.25` reduces the gate from `9.44524e-2` to a best `1.92932e-2` at sweep 9,
but the envelope subsequently grows to `8.95927e-2` at sweep 20.  Stage-wise
instrumentation identifies the explicit `T -> chi` source coupling as the
largest transient defect.  These controls are diagnostic only and are not
promoted as paper values.  N16 and all production grids remain unauthorized
until both the transformed and complete physical N8 gates contract to their
declared tolerances.

The next bounded N8 profile adds three explicit non-paper safeguards without
changing equations (56)--(63) or their printed stage order: nonlinear
equation-(63) non-increase within each scalar block, depth-one Anderson affine
mixing with raw-Picard fallback, and a ten-sweep nonmonotone outer window with
backtracking down to `1/4096`.  It retains the best accepted state rather than
returning a later oscillatory state.  Because equation (62) remains the
dominant bulk row, the chi block uses a full nominal correction (`1.0`); every
other transported field and pressure keeps the declared Code_Saturne values.

On N8 at `Kn=0.1`, `U=10 m/s`, this profile improves the simultaneous maximum
of the physical and transformed gates to `1.070565e-2` at sweep 14.  Later
sweeps reduce the transformed residual to `3.459903e-3` while the independent
physical gate rises, so the run is still rejected and the sweep-14 state is
returned only as a non-accepted diagnostic checkpoint.  This separation shows
that the next defect to resolve is the transformed-to-physical reconstruction
consistency near the interior chi/velocity rows, not wall enforcement.  N16
remains blocked.

Every sweep record now audits that separation directly on interior rows as

`R_63 = R_physical,point + (L_FV - L_central)`.

For sweep 21, the three unscaled infinity norms are respectively
`3.459903e-3`, `1.250344e-2`, and `1.250860e-2`; the identity closes to
`3.47e-18`.  The small transformed residual therefore contains a resolved
cancellation between the physical point residual and the FV/central
transport-discretization defect.  This is not evidence of a faulty variable
inverse and it cannot pass the independent physical gate.  The new record
fields expose both defects, their dominant planar-17 slots, and the identity
roundoff so subsequent source-discretization work cannot hide that
cancellation behind a scalar transformed norm.
