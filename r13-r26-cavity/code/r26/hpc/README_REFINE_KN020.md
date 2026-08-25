# R26 KnGu=0.20 SER-PTC gate and fresh N30 production

The first commit on this development line deliberately contained **no N30
launcher**.  The old interpolation and fresh-N30 workflows were removed so
that a refined-grid job could not be submitted before the modern solver passed
its small-grid gate.  Unity subsequently passed that gate at immutable commit
`8cbd874eea68dd475faa3f5e3fb318b49cc0c665`: N8 and N16 both reached the full
100 m/s target in 11 accepted attempts, with zero rejections and no more than
two Jacobian builds per attempt.

`r26_kn020_ser_ptc_gate_n8_n16.slurm` performs one fail-closed sequence:

1. clone an immutable 40-character commit;
2. run the complete R26 unit/contract suite;
3. solve N8 from its analytic equilibrium to the full 100 m/s target;
4. independently validate the N8 root;
5. only then solve N16 from its own analytic equilibrium to the same target;
6. independently validate N16 and write `N16_GATE_PASSED.json` plus checksums.

Both solves require the analytic global-mass Jacobian row, mass-preserving
encoded secant prediction, and bulk-only SER pseudo-transient globalization.
Final acceptance remains the unchanged raw residual gate `1e-8`, positivity,
effective-wall-pressure, momentum-balance, and internal-energy-balance checks.
Each continuation attempt is capped at five colored Jacobian builds.

The historical gate record correctly leaves `n30_authorized` false because
that commit exposed no N30 launcher.  The later, separate production commit
adds `r26_kn020_ser_ptc_fresh_n30.slurm` and authorizes exactly one bounded N30
workflow only after it independently revalidates the immutable gate.

The N30 production workflow:

1. verifies the five gate artifacts and their checksums;
2. independently recomputes the N8 and N16 physical residual gates;
3. runs the complete unit and contract suite from the immutable production
   commit;
4. starts N30 from its own analytic equilibrium (never an interpolated N16 or
   N28 state);
5. uses the same analytic-mass, mass-preserving secant and bulk-only SER-PTC
   settings that passed N16;
6. caps every continuation attempt at five colored Jacobian builds and 4000
   objective evaluations;
7. independently validates the accepted N30 root; and
8. creates a portable success or failure ZIP and SHA-256 file automatically.

The helper `submit_r26_kn020_ser_ptc_fresh_n30.sh` is intended to be fetched
from an immutable production commit.  Set `R26_N16_GATE_DIR` to the original
Unity directory containing `N16_GATE_PASSED.json`, and set `R26_N30_REF` to
the full production commit SHA.  `R26_N30_OUT` is optional; when omitted the
helper creates a timestamped directory under `CavityColdToHotIdentify`.

Passing N30 remains an algebraic acceptance result.  Grid convergence and
external agreement with DSMC are deliberately not claimed by this workflow.

## Source-locked N30 pseudo-arclength rescue

The fresh production job at commit
`312ee29799e5fdb4340d1146af5c408d72563d49` established a more precise
failure boundary.  It accepted N30 from equilibrium through
`lid=0.36935558170895255`, with positive density, temperature, and effective
wall pressure and no invalid evaluations, but fixed-parameter continuation
could not correct `lid=0.37185558170895255` within five Jacobian builds.  Its
raw gate remained `1.98608e-2`.  This is a numerical continuation stall; it is
not evidence of a physical blow-up and it is not an accepted 100 m/s result.

`r26_kn020_n30_pseudo_arclength_rescue.slurm` addresses that specific failure
without starting another arbitrary small-step ladder:

1. source-locks and independently rechecks the last two accepted N30 roots
   from the failed production directory;
2. promotes lid velocity to an unknown and solves the full bordered
   pseudo-arclength system, which remains regular at a simple branch fold;
3. retains the analytic global-mass Jacobian row, positivity-preserving log
   coordinates, and bulk-only SER pseudo-transient globalization;
4. caps each arclength attempt at seven Jacobian builds and 6000 objective
   evaluations, with at most 24 attempts and a declared step floor;
5. accepts every intermediate point only after the unchanged raw residual,
   mass, positivity, wall-pressure, momentum, and energy gates pass;
6. uses the first target-bracketing segment only to construct a predictor at
   exactly 100 m/s; and
7. performs one ordinary fixed-lid correction and independent validation at
   the target before any success record is written.

Thus pseudo-arclength changes the path parameterization, not the R26 equations
or their final acceptance criteria.  If the target is not bracketed within the
bounded budget, or the final fixed-lid correction fails, the job packages a
diagnostic failure and makes no N30 claim.  N32 remains unauthorized until
this N30 target gate passes.

## Evidence-based chord-reuse resume

The first arclength job at commit
`380a5cb05f0620813c255a086ca899b4051ea2ae` accepted a new raw-gated N30 root
at `lid=0.37021571065841696` (`raw=3.7241093542306203e-9`).  Density,
temperature, and wall pressure remained positive, there were no invalid
evaluations, and the arclength constraint was satisfied to roundoff.  The
failure therefore did not establish a physical singularity or a failed
bordered formulation.

The diagnostic records instead exposed a work-accounting error: the
corrector's loop was bounded by `maximum_jacobians`, so seven permitted
Jacobian builds also meant only seven nonlinear updates.  The audited
fixed-parameter SER-PTC solver already reuses a colored Jacobian for controlled
chord steps.  The corrected arclength implementation now uses the same
separation: at most 80 nonlinear updates, seven Jacobian builds, twelve PTC
chord updates per build, three Newton chord updates per build, and 6000 total
objective evaluations per attempt.  Every nonlinear update is recorded in an
iteration trace.

`r26_kn020_n30_arclength_chord_resume.slurm` is source-locked to both failed
archives.  It independently revalidates the accepted `0.37021571065841696`
root, resumes from it, retains the previous absolute minimum arclength step
`0.010751760973081607`, and caps the new maximum step at the previously
accepted `0.021503521946163215`.  It neither restarts the earlier ladder nor
authorizes N32.  A successful arclength bracket still must pass one ordinary
fixed-lid correction and the unchanged raw Maxwell validator at exactly
100 m/s.

## Balanced-metric correction after the chord-resume diagnostic

The chord-resume job at commit
`d494980776d3ef204158782a4176336566d06969` proved that Jacobian chord reuse
was active (16--17 nonlinear iterations for seven Jacobian builds), but it
accepted no new point.  Its full trace exposed the actual defect: with
`parameter_scale=0.04`, the last accepted N30 secant assigned
`0.9999648576` of the squared arclength norm to lid speed and only
`0.0000351424` to all 15,300 encoded state unknowns.  The bordered hyperplane
therefore constrained lid speed almost exactly and the supposed
pseudo-arclength corrector was numerically equivalent to fixed-lid
continuation.  The roundoff-level arclength residual was not evidence that a
fold had been traversed.

The corrected implementation computes the parameter scale from two accepted
roots so that the mesh-independent RMS state increment and the lid increment
each contribute one half of the squared secant norm.  A requested metric
outside the fail-closed `[0.1, 0.9]` parameter-fraction interval is rejected.
Absolute step lengths from the old metric are never reused after rebalancing;
the step schedule is rebuilt from the accepted secant in the new metric.

This change is deliberately gated before another N30 attempt.
`r26_kn020_balanced_metric_gate_n8_n16.slurm` independently validates the
immutable historical N8/N16 SER-PTC gate, replays the final known branch
segment on both grids with the balanced bordered corrector, and independently
checks the raw physical residual.  Its pass record explicitly leaves
`n30_authorized` false.  No further N30 rescue should be prepared until this
new N16 metric gate passes.

## Gate-approved N30 balanced-arclength stage

Job `63544304` completed the source-locked N8/N16 gate at commit
`93fd4b55b8932bcfedb36f1c66e90b443e7744e2`.  Both replays assigned exactly
one half of the squared secant norm to the state and one half to lid speed.
The independent raw gates were `9.177797410941935e-12` on N8 and
`4.175583750293255e-11` on N16.  The gate therefore validates the metric and
corrector combination; it does not itself claim an N30 solution.

`r26_kn020_n30_balanced_arclength_rescue.slurm` is the next bounded stage.  It
requires that exact gate, the failed fresh-N30 record, and the independently
accepted N30 root at `lid=0.37021571065841696`.  It does not reuse any
absolute step from the old degenerate metric.  The initial, minimum, and
maximum steps are rebuilt as `1`, `1/8`, and `2` times the accepted secant in
the balanced metric.  Each attempt remains capped at seven Jacobian builds,
80 nonlinear updates, and 6000 objective evaluations, with no more than 24
attempts.  Success still requires a target bracket, exactly one fixed-lid
correction at 100 m/s, and the unchanged independent Maxwell raw gate.

## Fold-continuation resume after job 63549978

Job `63549978` passed all 117 source tests and the independent N8/N16 metric
gate.  It then accepted two additional N30 roots at lid parameters
`0.3703661695101804` and `0.37038324507064474`, with raw gates
`4.027990163635309e-10` and `2.0262469480059053e-10`.  The subsequent secant
assigned only `0.00152901971` of its squared norm to lid speed, and the old
runtime guard stopped before the next nonlinear solve.

That small fraction is valid fold geometry, not a singular metric.  A
positive-definite arclength metric must be calibrated on a non-degenerate
secant and then held fixed while the parameter component of the tangent is
allowed to pass through zero.  Rebalancing the metric at every step would
instead suppress the very fold that pseudo-arclength is meant to traverse.

`r26_kn020_n30_fold_continuation_resume.slurm` implements this correction.  It
source-locks and independently revalidates the two accepted roots from job
`63549978`, keeps the original 50/50 calibrated metric fixed, rebuilds the
bounded step schedule from their physical secant, and preserves the existing
24-attempt, seven-Jacobian, 80-update, and 6000-objective caps.  A target claim
still requires target bracketing, one ordinary fixed-lid correction, and the
unchanged independent Maxwell raw gate.  N32 remains unauthorized.
