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
