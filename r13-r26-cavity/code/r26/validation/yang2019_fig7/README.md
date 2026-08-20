# External R26 validation: Yang, Tang & Yang (2019), Figure 7

This directory defines a fail-closed reproduction of the published R26
centreline profiles in Figure 7 of:

W. Yang, S. Tang and H. Yang, *Analysis of the Moment Method and the Discrete
Velocity Method in Modeling Non-Equilibrium Rarefied Gas Flows: A Comparative
Study*, **Applied Sciences 9** (2019) 2733,
<https://doi.org/10.3390/app9132733>.

## What this validates

The target is the paper's square lid-driven cavity at `Kn=0.1`, `T0=273 K`,
and `U0=10 m/s`, with fully diffuse walls.  The present solver is run with the
final Gu--Emerson JFM-2009 Maxwell-molecule R26 bulk, closure and wall model.
Its four centreline profiles are scored against the *published R26 markers*:

1. `ux` on the vertical centreline;
2. `uy` on the horizontal centreline;
3. `qx` on the vertical centreline; and
4. `qy` on the horizontal centreline.

A pass is external implementation validation of our R26 solver against a
published R26 result.  It is not an independent validation of Yang et al.'s
DVM implementation, and it does not establish R26 accuracy for every flow.
The Yang paper provides the DVM context and reports close velocity-profile
agreement at `Kn=0.1`.

## Source and digitisation lock

The publisher PDF is not redistributed.  Its SHA-256 is
`9ac8c50c10e4801d053b73ee4c387086368aa54824d003a2ecec739d36887d43`.
`extract_yang2019_fig7_vector.py` converts PDF page 12 to SVG with Poppler,
selects the blue triangular R26 markers, and applies vector tick-coordinate
calibrations.  The committed CSV has SHA-256
`4172749582a1448ce460a79e7cc62d1373be9235f88b5e1c40d68494a0211547`.
No fitted curve or manually smoothed target is used.

The paper normalises velocity with `sqrt(2 R T0)` and heat flux with
`rho0 [sqrt(2 R T0)]^3`; the solver uses `sqrt(R T0)`.  The scorer therefore
divides solver velocities by `sqrt(2)` and heat fluxes by `2 sqrt(2)`.
Published centred coordinates `[-0.5,0.5]` are mapped to the solver domain
`[0,1]`.

## Predeclared numerical gates

The job runs boundary-clustered `17x17`, `21x21`, and `25x25` grids.  Every
state must satisfy the algebraic residual and global-balance gates before it
can be scored.  On the finest grid, relative L2 error must be at most 15% for
each velocity profile and 30% for each heat-flux profile.  The last-two-grid
profile change must be at most 10% for velocity and 15% for heat flux.  All
eight conditions must pass.  A failed gate remains a useful diagnostic and is
packaged; it must not be described as validation.

## Outputs

The Unity job writes `yang2019_fig7_validation.json`, publication-quality PDF
and PNG overlays, accepted states and summaries, SHA-256 hashes, and a curated
ZIP under the campaign root.  The one-line bootstrap printed with the release
pins the exact Git commit and prints the Slurm job ID and campaign path.
