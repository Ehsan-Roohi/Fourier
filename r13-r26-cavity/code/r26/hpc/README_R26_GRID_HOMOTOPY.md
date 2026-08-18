# R26 KnGu=0.20 N28-to-N30 grid-homotopy recovery

The previous direct N28-to-N29 grid reconciliation stopped after 13,800
residual evaluations: its final candidate remained positive but the sparse
Newton direction crossed the effective-wall-pressure domain and the Armijo
search could not reduce the localized bulk residual.  It was correctly
rejected and is not a physical R26 prediction.

This recovery keeps the equations, Maxwell closure, Knudsen convention, wall
model and final raw acceptance gate unchanged.  It introduces only a numerical
path for grid reconciliation:

`F_N(u) = (1-lambda) F_N(u_interpolated)`, with `lambda: 0 -> 1`.

The interpolated seed is therefore exact at `lambda=0`; only `lambda=1` solves
the physical R26 system.  Intermediate path states are saved with
`accepted=False`.  Each failed stage rolls back to the last accepted path
state, halves the homotopy step, and can fall back from colored sparse Newton
to bounded trust-region least squares.  N30 is not attempted unless N29 passes
the original fail-closed Maxwell validator.

The campaign then reports N28-to-N29, N29-to-N30 and N28-to-N30 relative L2
changes for every R26 moment group.  These are described as grid sensitivity,
not asymptotic grid convergence unless monotonicity is actually demonstrated.

## One-line Unity submission

```bash
R26_GIT_REF=agent/r26-grid-homotopy-recovery bash <(curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/Fourier/agent/r26-grid-homotopy-recovery/r13-r26-cavity/code/r26/hpc/bootstrap_unity_r26_kn020_homotopy_recovery.sh)
```

The bootstrap locks the accepted N28 seed by SHA-256, records the exact Git
commit, runs all 92 R26 and Maxwell-contract tests, and always packages success or
failure diagnostics into the project `deliverables` directory.
