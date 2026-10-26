# Deck v3: consistency re-pass (CSO plan, 22 Oct 2026)

Goal: one narrative that introduces all nine representations together and compares them everywhere. The new models must not read as an appendix. A reader should be able to diagnose each model and choose one to work with.

## The nine representations (codes used everywhere)
| code | name | family | output per entry | cell embedding |
|---|---|---|---|---|
| R1 | processed counts (log-norm, HVG, PCA) | heuristic | none | PCA-50 |
| R2 | binary Y (PCA) | heuristic | none | PCA-50 of Y |
| R3 | BernFac logit, UVᵀ | low-rank logit | σ(uᵀv) | SVD of UVᵀ (k=40) |
| R4 | BernFac logit + offsets | low-rank logit | σ(a+b+uᵀv) | SVD of UVᵀ |
| R5 | BernFac cloglog + offsets | low-rank cloglog | 1−exp(−e^{a+b+uᵀv}) | SVD of UVᵀ |
| R6 | NB count factorization | counts | 1−(1+μ/φ)^−φ | SVD of UVᵀ (k=120) |
| R7 | mixture of local factors (K=1, d=40) | structured logit | Σ r E[σ(ψ)] | logit-mean SVD |
| R8 | neural manifold (affine head, d=20) | structured logit | σ(g(z)) | z |
| R9 | latent block model (200×200) | structured logit | R σ(B) Cᵀ | R·B |
| (Poisson counts) | reported in prediction tables only | counts | 1−e^{−μ} | |

## Storyline (sections)
0. Title, the question, and the data.
1. Preamble: a BioRender map of methods, the shared notation, one LaTeX slide per family with its BioRender schematic, inference, and the evaluation metrics in math with a BioRender evaluation-design schematic.
2. Synthetic ground truth for all methods: one shared benchmark (worker S).
3. Model selection on real data: all selection curves in one figure, then the held-out test comparison.
4. Representations: a 3×3 UMAP gallery coloured by hemilineage, log UMI, library and sex, plus zoom slides (worker U), then biology metrics.
5. Heatmap snippets: cells × genes for every model, with probabilities where available (worker H).
6. Diagnostics: convergence, calibration, identifiability and runtime for all methods (worker D, plus calibration from worker H).
7. Scorecard and recommendation; next steps.
8. Appendix: the 1000×100 pilot (old slides 9–20), detailed synthetic BernFac CV (old 21–29), and per-model detail figures.

## Workers (outputs in v3/figures unless noted)
- S: shared synthetic benchmark → `v3/synthetic/`, `fig_syn_*`.
- H: heatmap snippets and calibration → `fig_heat_*`, `fig_calibration_all`.
- U: UMAP gallery and zooms → `fig_umap_*`.
- D: composite selection, convergence, runtime, scorecard → `fig_select_all`, `fig_converge_all`, `fig_runtime`, `fig_scorecard`.
- CSO: LaTeX preamble (`v3/equations`), BioRender overview and evaluation schematics (`v3/biorender`), humanized text, deck assembly and upload as a NEW Google Slides deck.

## Rules
- Python only. Limit each worker to ≤4 threads and ~20 GB RAM. 19 GB of disk is free; never save n×m arrays.
- Figures follow the house style (`figstyle.py`, mathtext cm), PNG at 200 dpi under 1.5 MB, plus PDF.
- Use the same method codes, names and colours everywhere. The palette lives in `v3/style.py`.

## Progress log
- 22 Oct: plan written, workers launched.
- 22 Oct (H): loaders v3/predictors (cm_minimal/experiments/bernfac/v3/predictors.py) validated vs comparison.csv (tables/loader_validation.csv); fig_heat_{overview,markers,depth,diff}, fig_calibration_all, fig_test_by_depth; tables heat_metrics, calibration_*, test_nll_by_*.
- 22 Oct 15:40 (CSO): workers U, D and H are done (figures in `figures/`, tables in `tables/`); all test suites re-run and passing.
  - BioRender: `biorender/v3_methods_map.png` (R5 formula, heatmap axis labels, colour-bar max and the R6 dispersion symbol patched in PIL; raw kept as `_raw`) and `biorender/v3_evaluation_design.png` (OCR clean).
  - LaTeX: `equations/v3_recommendation.png` added. Extra figures `fig_test_all` and `fig_depth_regressed` (v3 palette) from `cm_minimal/experiments/bernfac/v3/make_extra_figs_v3.py`.
  - Deck spec: `cm_minimal/experiments/bernfac/v3/make_slides_v3.py` writes `deck/slides_main.json` and `deck/slides_appendix.json`. There are two PPTX files so each stays under the 10 MB browser-upload limit; both get imported into one Google Slides deck. The Keynote render check passed after fixes.
  - Waiting on worker S (synthetic benchmark); the deck is otherwise ready.
- 22 Oct 18:51: synthetic benchmark finished (133 tasks, 0 failures); `collect` and `syn_figures.py` ran, and the KL axis ticks were fixed.
- 22 Oct 19:00: deck uploaded as Google Slides `NineRepresentations_v3_22Oct2026` (1Jc-fiWr1VZ0Vra8R5fhztL7XCrJmZxUNeBuQAF6wNFk, 90 slides, verified with the Drive reader). The two helper PPTX uploads were moved to the Drive trash. DONE.
- 23 Oct (round 3b): user asked for tuning transparency and depth-standardized probabilities woven into the narrative.
  - Worker Sy made `fig_syn_select`. Worker Hd made `standardized()` and the std figures.
  - CSO added LaTeX `v3_depth` and `v3_hyperparams`, clarified the metrics panels, and made the BioRender `v3_depth_schematic` (R1 box patched in PIL; raw kept as `_raw`).
  - Deck reordered and rebuilt (98 slides). Live deck content replaced: new slides imported, old 90 deleted, verified with the Drive reader. Helper PPTX uploads moved to Drive trash. DONE.
