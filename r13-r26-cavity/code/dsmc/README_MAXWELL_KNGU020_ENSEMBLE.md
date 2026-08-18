# Publication-grade Maxwell-VSS DSMC campaign at KnGu=0.20

This campaign removes the VHS-omega=0.81 versus Maxwell-moment mismatch in the
transition case and measures sampling, grid and particle-loading uncertainty.
It does not silently reinterpret VSS as the exact IPL angular kernel:
`omega=1, alpha=2.140` is explicitly labelled as the transport-class VSS
representation of Maxwell molecules.

## Design

- Primary: `N160, PPC256`, eight independent seeds.
- Grid sensitivity: `N120` and `N200` at `PPC256`, four seeds each.
- Particle sensitivity: `PPC128` and `PPC512` at `N160`, four seeds each.
- Every run: 200,000 warm-up steps, 2,000,000 production steps, sampling every
  100 steps, 20,000 accumulated samples per cell, and ten non-overlapping
  200,000-step block averages.
- The fixed SPARTA source commit is
  `912c9e163c38ea5c3562d039e65215f6e2a4f3f8`.
- Each case is rejected unless its Knudsen convention, VSS parameters, grid,
  particle loading, timestep, dump schema, positive density/temperature,
  particle-number balance, completed block series and source commit pass.

The dependent gate computes ensemble standard errors, split-half drift,
grid/PPC differences and seed-to-ensemble anti-Fourier Jaccard stability. It
creates an archive only when all predeclared publication thresholds pass.

The standard SPARTA `sonine/grid` output is retained as a fourth-order
diagnostic. It is not relabelled as independently resolved R26 `R_ij` and
`Delta`; a full observable/hidden projection claim requires a documented
moment transformation or an extended particle tally and must pass the same
ensemble gate.

## One-line Unity submission

```bash
curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/Fourier/main/r13-r26-cavity/code/dsmc/bootstrap_unity_maxwell_kngu020_ensemble.sh | bash
```

The command prints the array job ID, dependent gate job ID and immutable
campaign directory. Existing results are never overwritten.

