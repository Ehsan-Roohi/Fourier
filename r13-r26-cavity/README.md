# JFM final submission package

This package supports the revised manuscript

> *What does anti-Fourier heat-flux agreement validate? DSMC sensitivity and
> R13--R26 diagnostics in a rarefied cavity*

It contains the article, response to all 20 recorded referee comments, cover
letter, numerical evidence, and scripts used for the final figures. The
lower-rarefaction study uses five eight-realisation DSMC designs at
`Kn_Gu = 0.05`. The transition comparison places DSMC, R13 and R26 at the
same `Kn_Gu = 0.20`, wall state, coordinates and nondimensional basis.

## Submission documents

- `manuscript/main.pdf` and `main.tex`: final article with enlarged landscape figures.
- `manuscript/revision_memorandum.pdf` and `.tex`: seven-page point-by-point
  response with 20 recorded comments and 20 implemented responses.
- `manuscript/cover_letter.pdf` and `.tex`: one-page JFM cover letter.
- `manuscript/figures/`: the eight figures used by the article, including the
  enlarged landscape comparisons.

The article cites 32 references, including 27 external works and five
self-citations. All 32 bibliography entries are cited in the text.

## Numerical evidence

- `data/dsmc_sensitivity/`: reduced `Kn_Gu = 0.05` grid and particle-loading
  sensitivity tables.
- `data/kn005_models/`: low-rarefaction R13 and R26 states and reduced
  resolution/comparison evidence.
- `data/matched_final/`: completed exactly matched Maxwell-molecule
  DSMC/R13/R26 inputs at `Kn_Gu = 0.05`.
- `data/kn020_sparta/raw/`: completed DSMC input, metadata, log and final
  `160 x 160` averaged field at exact `Kn_Gu = 0.20`.
- `data/kn020_models/`: accepted R13 and R26 transition states and their
  documented numerical records. The grids are stated separately in the
  manuscript rather than appended to the method names.
- `data/kn020_sparta/matched_kn020_*`: common-grid fields, centreline data,
  full-field metrics and processing-sensitivity tables.

`CODE_PROVENANCE.md` records the R13 starting lineage, the documented changes,
the independent Gu--Emerson R26 development, state hashes, and the scope of
the distributed source.

The complete reusable source is under `code/`: R13, R26, SPARTA case setup,
validation tests, and the common-grid analysis/plotting workflow. It is also
mirrored at <https://github.com/Ehsan-Roohi/Fourier/tree/main/r13-r26-cavity>.

## Reproduction

The transition analysis requires Python 3 with NumPy, SciPy and Matplotlib:

```bash
python3 -m pip install -r requirements.txt

python3 analysis/kn020/analyze_sparta_kngu020.py \
  --input-dir data/kn020_sparta/raw \
  --output-dir reproduced/kn020_dsmc

KN020_OUTPUT_DIR="$PWD/reproduced/kn020_matched" \
  python3 analysis/kn020/analyze_matched_kn020.py
```

The presentation-only restyling of the archived lower-rarefaction spatial
figures can be reproduced with:

```bash
python3 figures_work/restyle_lowkn_publication_figures.py

python3 analysis/matched_models/make_matched_model_figures.py \
  --data data/matched_final \
  --out manuscript/figures/matched_final
```

Compile the documents with:

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error revision_memorandum.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error cover_letter.tex
```

The top-level `MANIFEST.sha256` authenticates every distributed file. The
archive checksum is supplied beside the final ZIP.
