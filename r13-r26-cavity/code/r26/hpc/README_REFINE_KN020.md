# R26 KnGu=0.20 refinement to N30

The job refines an already accepted `N20` state through `N25` and then `N30`.
Each grid is reconciled at the target lid speed and must pass the strict raw
residual, positivity, balance, molecular-model, Knudsen-convention and source
hash checks before the next grid is attempted. A failed `N25` validation stops
the job, so an unaccepted state is never promoted as the `N30` restart.

This workflow reduces continuation risk but cannot mathematically guarantee
non-divergence. The publication status of `N30` remains conditional on the
validator reporting `R26_MAXWELL_VALIDATION_PASS` and on a subsequent grid
comparison; solver completion alone is not acceptance.

