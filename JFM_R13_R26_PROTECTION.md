# JFM R13/R26 protection policy

The complete `r13-r26-cavity/` tree supports an active/final JFM submission and
is read-only during repository organization.

1. Do not reformat, rename, relocate, squash, delete, or opportunistically merge
   article code or data.
2. Preserve the R13 lineage, independent R26 implementation, DSMC setup,
   nondimensional basis, boundary conditions, closure coefficients, grids,
   restart history, reference states, result labels, and recorded hashes.
3. A scientific change requires a dedicated pull request with the exact
   baseline commit, affected equation/numerical contract, physical
   justification, reproduction command, before/after metrics, and updated
   manifest hashes.
4. Never relabel a diagnostic, held, failed, or incomplete result as accepted.
5. Documentation outside the protected tree may improve navigation but must not
   imply a new scientific qualification.

The authoritative implementation identities and scope are in
`r13-r26-cavity/CODE_PROVENANCE.md`.

