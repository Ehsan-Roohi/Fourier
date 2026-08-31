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
- direct conservative RHS fluxes for equations (56)--(62), formed on faces
  with central transported values and two-point normal gradients, with
  collision/nonlinear local terms evaluated at cell centres;
- segregated Picard linearisation: source, viscosity and mass flux are frozen
  at the entry to each field block and rebuilt before the next printed field;
- the dissipative linear collision sinks printed in equations (58)--(62) are
  retained on the implicit block diagonal;
- SIMPLE pressure--velocity correction and Rhie--Chow mass fluxes;
- the printed segregated order `u -> SIMPLE -> T -> g -> h -> omega -> gamma
  -> chi -> physical moments -> wall conditions`.

The long right-hand sides of equations (56)--(62) are represented in their
printed conservative flux form.  The transformed and reconstructed physical
fluxes are each built directly on a control-volume face and then subtracted.
CUBISTA is therefore confined to the transported field on the left-hand side
and cannot leak into a source term.  The former compatible CUBISTA-difference
source is retained in the term record only as an audit baseline.  The backend
never calls the physical BVP residual from a transformed field block.  A
manufactured non-equilibrium test confirms the expected algebraic cancellation
in momentum/temperature source transport, a genuine central-versus-CUBISTA
difference in the higher-moment rows, and equilibrium as an exact conservative
fixed point.

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

Before the RHS flux discretization was made compatible, the corrected local
N8 diagnostic at `Kn=0.1`, `U=10 m/s` failed closed.
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

Every sweep record audits that separation directly on interior rows as

`R_63 = R_physical,point + (L_FV - L_central) - (S_FV - S_central)`.

For sweep 21, the three unscaled infinity norms are respectively
`3.459903e-3`, `1.250344e-2`, and `1.250860e-2`; the identity closes to
`3.47e-18`.  The small transformed residual therefore contains a resolved
cancellation between the physical point residual and the FV/central
transport-discretization defect.  This is not evidence of a faulty variable
inverse and it cannot pass the independent physical gate.  The new record
fields expose the transport and source discretization defects, their dominant
planar-17 slots, and the identity
roundoff so subsequent source-discretization work cannot hide that
cancellation behind a scalar transformed norm.

The compatible RHS stage resolves that diagnosed defect.  In the new N8 run,
the transformed and complete physical raw residuals become identical to
roundoff once the interior bulk row dominates: at sweep 24 both are
`6.349235e-3`, down from the initial physical gate `9.445238e-2`, and the best
accepted state is the final sweep rather than an earlier cancellation point.
The run remains fail-closed because `1e-8` has not been reached.  The next
bounded step was therefore a longer N8 work-budget run using these same
equations.  At sweep 80 the three transformed, complete physical raw and
scaled gates are all `3.748628e-3`; the best checkpoint is sweep 79 at
`3.721863e-3`.  This is continued contraction, but far too slow to justify a
blind increase toward the `1e-8` gate.

Sweep records now name the physical gate region and the dominant transformed
planar-17 slot and segregated block.  The N8 audit shows that the wall region
controls through sweep 12, after which the bulk controls and slot 16 (`chi`)
is dominant at sweeps 16, 20 and 24.  The next bounded development stage is
therefore the equation-(62)/`chi` source-coupling linearisation on N8, not a
larger grid or another undiagnosed work-budget increase.  N16 remains
unauthorized.

The first equation-(62) source-fidelity stage now follows the paper's stated
central source discretization directly.  It leaves the N8 history unchanged
through sweep 8.  At sweep 20 the raw gate is `9.283571e-3`, slightly below
the compatible baseline `9.695921e-3`, but the 24-sweep gate is
`1.640931e-2`, above the compatible baseline `6.349235e-3`; the best checkpoint
is sweep 23.  This commit therefore records a source-discretization correction,
not a convergence claim.  N8 still fails closed and N16 remains blocked.  The
next bounded source stage is to centralize the remaining printed RHS fluxes
consistently rather than mix the direct equation-(62) source with compatible
CUBISTA-difference sources in equations (56)--(61).

That all-equation central-source stage is now implemented.  On N8 it improves
the sweep-8 raw gate from the compatible baseline `3.028342e-2` to
`2.851913e-2`.  At sweep 24 the transformed residual reaches `5.080048e-3`,
but the independent physical raw gate is `1.395834e-2`; this is better than
the equation-(62)-only stage (`1.640931e-2`) and worse than the compatible
baseline (`6.349235e-3`).  The best simultaneous checkpoint occurs at sweep
17, after which the nonmonotone safeguard permits diagnostic sweeps without
promoting them.  This is a complete central-source fidelity correction, not a
convergence claim: N8 still fails closed and N16 remains blocked.  The next
bounded work is the measured transformed/physical discretization mismatch at
that N8 checkpoint, not relaxation tuning or grid refinement.

That mismatch is now recorded directly in every sweep.  The discrete identity
`R63 - Rphysical,FV + (Scentral - Scompatible) = 0` closes to roundoff; at the
former best sweep 17 it exposed a `1.031240e-2` source-scheme mismatch caused
by forming the central source from cell-centred gradients and only then
interpolating it.  The corrected stencil constructs both sides of each source
flux directly at the face.  On the same N8 run the raw gate is
`3.028981e-2` at sweep 8 and `5.013962e-3` at sweep 24, where the transformed
and independent compatible physical FV maxima agree.  The 24-sweep value is
below both the compatible baseline (`6.349235e-3`) and the former
cell-centred-source result (`1.395834e-2`); the source-scheme mismatch has
fallen to `7.900630e-5`.  Sweep 24 is the best simultaneous checkpoint.  N8
still fails the `1e-8` gate, so N16 remains blocked; the next bounded stage
must continue on this N8 face-consistent system.

The outer safeguard now preserves the acceptance baseline, both full-step
merits, and the best attempted kind/step/merit when a sweep is rejected.  A
48-sweep request stops fail-closed at sweep 34: the best accepted checkpoint
is sweep 29 with raw merit `4.473728e-3`.  At the stopping state, even the best
raw trial at the minimum `1/4096` step raises the normalized merit from
`5.107024e5` to `5.107252e5`, proving that the fixed-point direction is locally
ascending rather than merely failing a sufficient-decrease margin.  Restarting
once from sweep 29 with an empty Anderson history improves the raw gate to
`4.433743e-3`; a second clean restart has no descending trial and its minimum
step gives `4.433763e-3`.  Repeated restarts are therefore rejected as the next
method.

A stage-by-stage replay at the `4.433743e-3` checkpoint localizes the ascent.
The velocity stage first raises the merit to `4.584644e-3`; SIMPLE and the
successive interior blocks then reduce it through `4.429066e-3`,
`4.416446e-3`, and finally `4.401497e-3` after `h`.  The `omega`, `gamma`, and
`chi` blocks preserve that value.  Only the final wall reconstruction raises
the merit to `5.080962e-3` and changes the dominant transformed row from slot
3 (`temperature`) to slot 16 (`chi`).  Scaling one already-computed wall
correction shows a smooth increase: zero wall correction gives
`4.401497e-3`, one quarter of the correction gives `4.429170e-3`, and the full
declared wall correction gives `5.080962e-3`.

Two bounded globalization responses were tested and rejected.  Applying a
wall safeguard throughout the run worsened the best 24--36-sweep gates to
`9.430828e-3` and `9.948198e-3`, depending on its reference merit.  Activating
wall-only reduction solely after the original outer rejection carried the run
through sweep 48 but did not improve the `4.473728e-3` best checkpoint; after
a clean best-state restart, 24 more sweeps still did not improve the first
`4.433743e-3` checkpoint.  No wall-globalization experiment is retained in
the solver.  The next N8 stage is therefore the wall-reconstruction/closure
coupling that injects the slot-16 defect, not more relaxation, restarts, or
grid refinement.  N16 remains blocked.

The wall reconstruction ownership audit found that the physical-to-transformed
conversion at the end of a sweep rewrote every interior primary field.  This
was especially consequential for `chi`, whose inverse contains a heat-flux
divergence: changing wall heat flux changes the adjacent gradient even though
the interior `chi` block has already been solved.  The wall stage now owns only
boundary values of the eight Gu--Emerson primary fields.  A bounded boundary
fit reconstructs the relaxed physical wall target while preserving every
interior transformed value exactly.

All 170 tests pass.  On the fresh N8 gate the best 24-sweep raw residual is
`5.364705e-3` at sweep 23 (the final sweep is `1.082937e-2`).  This removes the
unintended interior overwrite and is close to, but does not improve on, the
previous restart checkpoint `4.433743e-3`.  It is therefore an algorithmic
ownership correction rather than a convergence claim.  N8 still fails the
`1e-8` gate and N16 remains blocked.

A separate N8-only monolithic oracle now tests whether replacing the
segregated iteration by Newton--Krylov resolves the stall.  Its square raw
system uses equation (63) on every interior row, retains the physical
wall/corner equations, and preserves the global mass border.  Logarithmic
density and temperature coordinates enforce positivity.  It is diagnostic
only and cannot authorize N16 or production output.

Starting from the best 24-sweep transformed checkpoint, the raw monolithic
objective is `5.362940e-3`.  Eight bounded JFNK outer iterations use 277
residual evaluations; every trial is worse, rising first to `6.268060e-3` and
ending at `6.516780e-3`.  The fail-closed oracle therefore returns its initial
checkpoint (transformed residual `5.362940e-3`, complete physical raw gate
`5.364705e-3`, held continuity `-1.865038e-5`, mass error at roundoff).  This
rules out both more fixed-point sweeps and an unpreconditioned monolithic JFNK
swap as useful next steps.  It does not prove that the discrete system has no
root; the remaining defensible work is an equation/boundary audit or a
physics-based block preconditioner, still on N8 only.

That block-preconditioner test is now complete.  It reuses the seven frozen
equation-(63) interior matrices from the segregated solver while deliberately
leaving density/SIMPLE, wall, corner and mass rows as identity rows.  On the
same sweep-23 N8 checkpoint, seven block factorizations and eight Krylov outer
iterations use 310 residual evaluations.  The first trial raises the objective
from `5.362940e-3` to `5.648190e-3`; subsequent trials are approximately
`3.1e-2`.  The fail-closed oracle therefore returns the initial checkpoint.
This rejects an interior-field block diagonal as the missing ingredient and
localizes the remaining algebraic problem to the omitted coupled carrier and
boundary rows.  A larger work budget, N16, or another interior relaxation
experiment is not authorized.

The complete N8 transformed Jacobian has now been built with the repository's
radius-two colored sparsity and absolute finite-difference floor.  It contains
1088 unknowns and 334123 numerical nonzeros and is numerically full rank
(`1088/1088`).  After deterministic row/column max-norm scaling, its smallest
singular value is `4.427686e-3` and its reciprocal condition estimate is
`4.609959e-4`.  The stall is therefore not a pressure null space or algebraic
rank defect.  The weakest left singular vector is 92.72% wall-supported.  Its
dominant boundary row is slot 10, which the boundary assembly maps to
`C8_Delta` (not the bulk slot-10 name `gamma_xy`).  The weakest right vector
is 56.30% wall-supported and 48.26% `gamma_xy`.  This independently identifies
a weak wall-`C8_Delta`/unknown-`gamma_xy` coupling rather than a missing
interior block solve.

A bounded full colored-Jacobian Newton oracle then tests whether exact linear
solves can exploit that full rank.  Three Jacobian builds use 1439 objective
evaluations and accept only steps `1/64`, `1/128`, and `1/512`.  The objective
falls from `5.362940e-3` to `5.357921e-3`, with no invalid evaluations, but the
complete physical raw gate remains `5.360998e-3`.  This proves that a descent
direction exists while also rejecting a blind extension of full Newton: its
local model is valid only at vanishing step lengths and the measured progress
is many orders too slow for the `1e-8` gate.  The next N8 audit must resolve
the wall `gamma_xy` rows by side and boundary equation before changing the
solver again.  N16 remains blocked.

Correct interpretation of the boundary row layout is essential here: slots
0--10 are the ordered smooth-wall equations and slots 11--16 are free-moment
extrapolations.  A side-resolved repeat finds 86.97% of the weakest equation
energy on the left wall, including 41.02% in `C8_Delta`, while the weakest
unknown energy lies mainly on the moving top wall (32.49%) and left wall
(22.66%) and remains dominated by transformed `gamma_xy`.  The remaining
audit target is therefore the top-left wall/corner stencil, not a generic
domain-wide conditioning problem.
