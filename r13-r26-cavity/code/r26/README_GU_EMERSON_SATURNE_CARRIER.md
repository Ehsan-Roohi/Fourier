# Gu--Emerson / Code_Saturne carrier audit

This path is a low-speed method reconstruction and is not a high-speed R26
production claim.

## What the publications and sources actually fix

- Gu and Emerson, JFM 636 (2009), section 5.2 fixes the collocated finite
  volume architecture, CUBISTA convection, central diffusion/source terms,
  SIMPLE, Rhie--Chow, and the segregated field order.
- The supplied Rana sources identify Code_Saturne 5.0, temperature mode
  `itherm=1`, variable density/viscosity, the moment source histories, and the
  thermophysical update
  `rho_new = 0.0002*p_total/(R*T) + 0.9998*rho_old`.
- The official Code_Saturne 5.0.3 core at commit
  `e17068ce692ad2d90c694d375b7c098043b16969` keeps gauge pressure independent,
  applies the full final pressure-increment flux correction, and uses steady
  defaults 0.7 for fields and 0.3 for pressure.
- The 2024 Code_Saturne module paper validates the cavity at a 10 m/s lid.  It
  states that the module is coupled to the incompressible solver and that
  compressible use requires the energy equation and new energy wall boundary
  conditions.

## Missing historical evidence

The supplied archive has no `setup.xml`, `run.cfg`, mesh, or listing.  Its
active user-source constants are Kn=0.02 and Re=50 rather than the published
cavity case.  The official upstream tags v8.0.0 and v9.2.0 contain no R13/R26
module, so the promised core merge cannot be used to recover those inputs.

Consequently the exact historical steady/unsteady selection, time step,
non-orthogonal corrections, scalar schemes, convergence controls, and cavity
mesh are unavailable.  Architecture reuse is authorized; historical profile
reproduction and high-speed authorization fail closed.

## Implementation boundary

`GuEmersonReconstructionOptions.code_saturne_v5_rana_diagnostic()` adds a
non-authorizing carrier diagnostic without changing the legacy reconstruction:

1. pressure is retained independently of `rho*theta`;
2. SIMPLE uses the full velocity correction and relaxes only stored pressure;
3. density responds to total pressure at the beginning of the next outer
   iteration with the exact Rana factor `2e-4`;
4. no mass projection, clipping, continuation, pseudo-arclength, or adaptive
   damping is introduced;
5. `r26_gu_emerson_saturne_contract.py` records the source hashes and keeps all
   fine-grid/high-speed authorizations false.

## Local numerical stop rule

The complete unit/contract suite passes.  However, a one-sweep fixed-point
test on an independently accepted Maxwell N8 root changed the state by
`7.44e-4` and moved the raw R26 gate from `2.55e-12` to `1.72e-2`.  Thus the
source-default momentum diagonal and the repository's independently declared
physical Rhie--Chow face coefficient do not define the same discrete root.

The independent Sutherland/ASME 10 m/s N8 start remained positive for five
sweeps, but its raw gate was not monotone:

| sweep | raw gate | minimum density |
|---:|---:|---:|
| 1 | 1.4688e-1 | 1.0000 |
| 2 | 1.3696e-1 | 0.99954 |
| 3 | 1.2355e-1 | 0.99925 |
| 4 | 2.1302e-1 | 0.99845 |
| 5 | 4.1068e-1 | 0.99911 |

This removes the previous immediate non-positive-density failure but neither
preserves the accepted fixed point nor establishes standalone convergence.
No N16 or Unity production job is authorized from this result.
The missing historical case controls must be recovered from the authors, or a
new energy-capable high-speed carrier must be documented as new work rather
than attributed to Gu--Emerson or Rana.
