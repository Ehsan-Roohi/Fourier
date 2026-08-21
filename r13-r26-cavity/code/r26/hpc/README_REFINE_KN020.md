# R26 KnGu=0.20 refined-grid support runs

The legacy `r26_kn020_refine_20_25_30.slurm` job refines an already accepted
`N20` state through `N25` and then `N30`.
Each grid is reconciled at the target lid speed and must pass the strict raw
residual, positivity, balance, molecular-model, Knudsen-convention and source
hash checks before the next grid is attempted. A failed `N25` validation stops
the job, so an unaccepted state is never promoted as the `N30` restart.

This workflow reduces continuation risk but cannot mathematically guarantee
non-divergence. The publication status of `N30` remains conditional on the
validator reporting `R26_MAXWELL_VALIDATION_PASS` and on a subsequent grid
comparison; solver completion alone is not acceptance.

## Recommended post-submission path: N28 -> N30 -> N32

The accepted production state is `N28`.  The earlier `N29` attempt was not an
accepted root: its residual stalled near `1.7e-2`, line searches repeatedly
failed, and the old workflow accumulated about 13,800 objective evaluations.
This was a numerical stall, not a demonstrated physical instability and not a
residual blow-up.  A separate earlier `N30` job failed a source-hash validation
check after its preceding stage had been accepted; that event was a provenance
failure rather than evidence of numerical divergence.

Use `r26_kn020_refine_28_30_32.slurm` for new support calculations.  It:

- validates the supplied `N28` directory before using its state;
- runs `N30` before `N32`, with each accepted state acting as a hard gate;
- uses colored Newton only, without an automatic trust-region fallback;
- limits each reconciliation to 48 nonlinear iterations and 2,500 actual
  objective evaluations, so a stalled line search fails closed;
- records the exact Git commit and accepted-state checksums; and
- executes the complete R26 test suite before production work.

The even-grid sequence is a risk-reduction experiment because it avoids
repeating the failed `N29` transition; it is not proof that grid parity caused
the failure.

One-line Unity submission from the repository root is:

```bash
R26_N28_DIR=/absolute/path/to/accepted_N28 R26_REFINE_OUT=/project/pi_roohie_umass_edu/R26/review_support_N28_N30_N32_$(date +%Y%m%d_%H%M%S) sbatch r13-r26-cavity/code/r26/hpc/r26_kn020_refine_28_30_32.slurm
```

Do not use an unvalidated interpolated state, optimizer success alone, or a
rejected `N30` directory as the `N32` seed.
