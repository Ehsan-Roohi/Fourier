# R13 and R26 implementation provenance

This note records the implementation lineage used in the manuscript. No
coefficient in either moment calculation was fitted to the DSMC fields.

## R13

The R13 work began from a supplied 17-field MATLAB cavity code whose source
record associates its formulation with Rana, Torrilhon and Struchtrup,
*Journal of Computational Physics* 236 (2013), 169--186,
DOI `10.1016/j.jcp.2012.11.023`. The coefficient matrices and saddle-point
structure were transcribed to Python. The reported branch then introduced:

- a conservative shared-face finite-volume continuity balance;
- a compatible global mass constraint;
- defect-Newton nonlinear iteration;
- the printed two-point wall extrapolation; and
- the tangential effective-pressure wall convention.

The original choices remain separately selectable in the archived modules.
The final Maxwell branch additionally uses the Appendix-A production operator,
temperature-dependent viscosity for the declared model, JFNK globalisation,
source-file hash locking and explicit physical/provenance gates.
The manuscript describes the documented formulation used here; it does not
assert exact reproduction of every discretisation detail in the 2013 paper.

Key `Kn_Gu=0.20` identities:

- `state.npy`: `a1aaff3820dcc20b5aa4c5d338590b1a2d9fe9f5eea1a944fdf94bc85ade42ae`
- `report.json`: `ab2c73dfee2369b8413331d2686e9ec99fa88c9d23ba5073df7149de004aad59`
- `rana_original_coefficients.py`: `08caba3895db19c72cc69fe8c9be4b41fb5676b47e140ce123798df548a0b6fd`
- `rana_original_reference_solver.py`: `9b10862a3582ae59e91303292865ef142eabb022c103a4b18abbe0956f7f2e24`

## R26

The R26 solver is an independent implementation developed from the nonlinear
bulk equations, regularised closures and smooth-wall conditions of Gu and
Emerson, *Journal of Fluid Mechanics* 636 (2009), 177--216,
DOI `10.1017/S002211200900768X`. It reconstructs the full three-dimensional
symmetric trace-free tensors before applying the planar 17-variable state
layout. Development of the reported implementation included:

- correction of the `A_psi1` coefficient to 1.698;
- restoration of the nonlinear source and `phi`, `psi` and `Omega` terms;
- quotient gradients for density-normalised moments;
- removal of an undocumented scaled-divergence term; and
- a fixed geometric frame for every wall face.

The implementation has no R13 fallback. The submission package distributes
the R26 bulk equations, tensor closures, wall conditions, state layout,
solvers, post-processing routines and unit tests under `code/r26/`, together
with the accepted numerical states and run records used by the comparison.

## Public reproducibility record

The source and reduced-data record is mirrored at
`https://github.com/Ehsan-Roohi/Fourier/tree/main/r13-r26-cavity`. The public
record is not presented as an archival DOI. File-level SHA-256 hashes in the
submission ZIP identify the exact version used for the paper.

Key `Kn_Gu=0.20` identities:

- `last_accepted_state.npz`: `d752c14200a77dad9bdbb0436a9f58a0f62f56d0dc6729ed1587b2ddecebfc04`
- `attempt.json`: `792e511bca9578aa1c690de30d2995ea8eee1035b1407852916c0eb3905448b8`
- `run_summary.json`: `c383c1d2c99de5ca668533fbe0790b52b7ef26b14b4ac26b1998e50e608d30c3`

## Matched transition comparison

DSMC, R13 and R26 are compared at the same `Kn_Gu=0.20`, wall temperature,
lid speed, coordinates and nondimensional basis. The R13 native value obeys
`Kn_Gu = sqrt(pi/2) Kn_Rana`; R26 stores `Kn_Gu` directly. The DSMC number
density is computed from the Gu viscosity-based mean free path.

- DSMC final field: `8292e718c18a0cbdeecf22281759ba01c9e0a90ffb48391cd57d09a5345fe770`
- DSMC metadata: `c38c20380e0e63c69b0e90fe665fd0bb806f28a1d1e9324346353adfd1a3ec50`

The comparison script verifies these identities before computing any metric.
