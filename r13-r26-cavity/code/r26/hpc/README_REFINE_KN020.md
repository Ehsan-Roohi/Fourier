# R26 KnGu=0.20 nonlinear-globalization gate

This branch deliberately contains **no N30 launcher**.  The old interpolation
and fresh-N30 workflows are removed here so that a refined-grid production job
cannot be submitted before the new solver passes its small-grid gate.

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

The gate record explicitly leaves `n30_authorized` false.  An N30 workflow, if
scientifically justified after inspection of the N8/N16 results, must be added
later in a separate reviewed commit.
