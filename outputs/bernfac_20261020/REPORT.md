# BernFac study — 20 Oct 2026

**Goal.** Infer continuous latent Bernoulli probabilities `P = σ(UVᵀ)` from a binary
cell × gene detection matrix. Then find out which (λ, k) setting represents the
data most faithfully, and whether it can be chosen from the data alone.

The code is in `cm_minimal/experiments/bernfac/` (see its README). The real-data
results are in `real/`, the synthetic results in `synthetic/`, and the
schematics in `schematics/`. Each folder has `figures/` and `tables/`
(or `results/`).

## Data

- Source: motor preMN, 24 h APF, raw UMI counts.
- 1000 cells. The design is balanced: 4 libraries (2 F, 2 M) × 5 hemilineages
  (14A, 04B, 09A, 12B, 03A) × 50 cells.
- 100 top HVGs with detection between 5 and 95 %.
- `Y = 1[count ≥ 1]`, density 0.487.
- Sensitivity variant: `Y = 1[count ≥ 2]`, density 0.373.

## Solver validation

- All 11 handoff tests pass.
- J was monotone in every one of the 45 real fits and 504 synthetic grid points.

## Real data (λ ∈ 0.03–10, k ∈ 1–20; CV extension to λ = 100, k = 40)

**Held-out entry CV.** Five folds, each holding out 10 % of entries.

| model | NLL | AUC | Brier |
|---|---|---|---|
| gene rate | 0.548 | 0.796 | 0.184 |
| additive cell + gene logistic | 0.544 | 0.800 | 0.183 |
| BernFac λ=1, k=5 | 0.506 | 0.839 | 0.164 |
| **BernFac λ=10, k=20** | **0.479** | **0.858** | **0.155** |
| BernFac λ=10, k=40 | 0.478 | 0.860 | 0.154 |
| BernFac λ=0.03, k=20 (overfit) | 3.67 | 0.742 | 0.246 |
| BernFac λ=100 (U=V=0, P≡0.5) | 0.693 | – | 0.250 |

**Information content.**
- Within detected entries, the median gene-wise Spearman ρ(P, counts) is 0.30.
- With depth partialled out it is 0.28, against 0.03 for a depth × abundance
  reference. So P carries expression-level information beyond depth, even though
  it only saw binary data.

**Biology vs batch.**
- Hemilineage decoding is 0.97–0.99 at k ≥ 5. The silhouette is 0.29 at
  λ=1, k=5, against 0.06 for raw Y.
- Library and sex decode near chance in every representation, raw counts
  included. This subset therefore cannot test batch removal.
- BernFac has no row offset, so a sequencing-depth factor appears from k = 5
  (ρ with total UMI ≈ 0.56).

**Identifiability.**
- At λ=1, k=5, five seeds agree after Procrustes alignment (difference 1e-5), and
  P agrees to 3e-5.
- At λ=0.3, k=10 the seeds reach different optima (ΔJ up to 69, |ΔP| up to 0.98).
- Gauge check: an orthogonal R leaves J unchanged. A general invertible A lowers
  J, so the prior leaves only the O(k) rotation ambiguity.

**Convergence.**
- Fits with λ ≥ 3 converge in 10–190 iterations.
- 12 of 45 fits (all with λ ≤ 1) hit the 500-iteration cap.

**Recommendation.**
- For prediction or imputation, use λ ≈ 10, k ≈ 20.
- For a compact, reproducible embedding, use λ ≈ 1–3, k ≈ 5.

## Synthetic ground truth (4 scenarios × 3 seeds, 42-point grid each)

**Scenarios.** All are calibrated to the real Y.
- S1: low-rank logistic truth with rank 3 or 8.
- S2: Poisson counts with cluster structure, `p = 1 − e^{−μ}`, which is not
  low-rank in logit space.
- S3: S2 plus library batch effects, with the batch-free target `P_bio`.

**Exact oracle-argmin hits by each criterion.**

| criterion | S1_k3 | S1_k8 | S2 | S3 |
|---|---|---|---|---|
| C1 entry-wise CV | 3/3 | 3/3 | 3/3 | 2/3 (regret +1.5 %) |
| C2 count splitting (Poisson thinning) | – | – | 3/3 | 3/3 |
| C3 independent replicate (oracle-like) | 3/3 | 3/3 | 3/3 | 3/3 |
| C4 cell split + fold-in | 3/3 | 3/3 | 3/3 | 3/3 |
| BIC / AIC, k(n+m) parameters | 0/3 | 0/3 | 0/3 | 0/3 |

**What the criteria tell us.**
- C1 matches the replicate experiment C3 almost exactly (r ≥ 0.998), shifted by a
  constant 0.010–0.015 nats. Masked-entry CV is therefore the practical stand-in
  for "evaluate on a technical replicate".
- Count splitting picks the right model but scores the half-depth problem. Use it
  for selection only, then refit on the full data. At the selected point:
  - the half-data P, uncorrected, has KL 0.105;
  - after the `1 − (1 − p)²` correction, KL 0.050;
  - after the full refit, KL 0.038.
- Overspecifying the rank is costly: with true rank 3, a k = 8 fit has 3.5× the
  KL. Gaussian priors do not prune unused rank.

**Batch (S3).**
- The selected model still decodes library at 0.70–0.74 (chance 0.25).
- Driving library decoding toward chance requires k ≤ 3, which also destroys
  hemilineage decoding.
- So neither shrinkage nor rank removes batch. It needs explicit covariates.

## Conclusions

1. PG-EM BernFac works as specified. It is monotone and stable, and identifiable
   up to rotation when well regularised.
2. Choose (λ, k) by entry-wise masked CV, cross-checked with count splitting when
   counts exist. BIC and AIC are unusable here.
3. The continuous P carries real expression-level and cell-type information
   beyond the binary data.
4. It does not remove depth or batch. The next model to add is
   `Ψ = a1ᵀ + 1bᵀ + X_batch Γ + UVᵀ`, with row and column offsets plus batch
   covariates.

## Caveats

- The λ grid is coarse. The real-data optimum was near λ = 10, the synthetic
  optimum at λ = 3; neither had a finer grid around it.
- There is a single real subset and a single CV permutation.
- The S3 batch is stylised and larger than the real library differences.
- Count splitting was tested only under Poisson counts.
- "Converged" is a stagnation test, not a stationarity test.

## Figures and deck

- The schematics in `biorender/` (model, PG-EM inference, real-data design, synthetic CV design) were drafted with BioRender's Leo AI in Chrome and exported as flat 300-dpi PNGs. AI-rendered misspellings were patched by `fix_biorender_typos.py`; the unpatched originals are kept as `*_raw.png`.
- The equation panels in `equations/` are typeset in LaTeX with tectonic by `make_equations.py`.
- All data figures use the house style of the scientific-figure-making skill (`figstyle.py`), with math in Computer Modern mathtext.
- The deck is `deck/BernFac_PG_logistic_MF_20Oct2026_v2_BioRender.pptx` (52 slides), built by `build_deck.py` from `deck/slides.json`. The Google Slides copy is at https://docs.google.com/presentation/d/1sgXlWFZ-JcWVNErPBt53k6pSFE1Ra6K8BaDnzyaiTz4/edit

## Full dataset (21 Oct 2026)

All 16,460 cells × 19,543 detected genes (1,507 never-detected genes dropped), 322 M entries, density 0.181. Fit with the memory-bounded PG-EM in `bernfac_large.py`, which matches the dense solver to float32 precision. A fixed random 5% of entries (16.1 M) was held out.

- **CV grid:** k ∈ {10, 20, 40, 80} × λ ∈ {300, 100, 30, 10}, plus λ ∈ {3, 1} for k = 20 and 40. Each λ path was warm-started. J was monotone in every fit.
- **Best model:** k = 40, λ = 10. Held-out NLL 0.1778, AUC 0.965, Brier 0.0555. Baselines:

  | baseline | NLL | AUC |
  |---|---|---|
  | cell + gene logistic | 0.1854 | 0.962 |
  | gene rate | 0.1975 | 0.957 |

  The refit on all entries converged in 188 iterations (`full/fits/best_full.npz`).
- **Representations:**
  - R1: processed counts (normalize_total → log1p → 2,000 seurat_v3 HVGs → scale → PCA 50).
  - R2: binary Y (PCA 50).
  - R3: BernFac P̂, as the rotation-invariant 40-d embedding of logit P̂.

| metric | R1 counts | R2 binary | R3 P̂ | R3 P̂, depth regressed |
|---|---|---|---|---|
| hemilineage kNN balanced acc | **0.977** | 0.899 | 0.873 | 0.914 |
| hemilineage Leiden ARI (best res) | **0.649** | 0.514 | 0.490 | 0.609 |
| hemilineage cosine silhouette | **0.296** | −0.034 | −0.168 | 0.115 |
| 15-NN hemilineage purity | **0.974** | 0.853 | 0.776 | 0.843 |
| library kNN balanced acc (chance 0.048; lower is better) | **0.127** | 0.234 | 0.269 | 0.258 |
| \|ρ\| of top-2 components vs log depth | 0.557 | 0.967 | 0.976 | 0.107 |

**Verdict:** processed counts reflect biology best. BernFac P̂ is the better *predictive* model of detection, but as a cell embedding it carries a strong sequencing-depth axis, because the model has no cell offset. Removing that axis recovers most of the gap, but not all of it.

The next model should add explicit cell and gene offsets: Ψ = a1ᵀ + 1bᵀ + UVᵀ.

Figures are in `full/figures/` and `full/umap/figures/`. The deck is 36 slides (slides 30–35 are the full-data section).

## Offset model: Ψ = a1ᵀ + 1bᵀ + UVᵀ (21 Oct 2026)

`bernfac_large.fit_offsets` adds per-cell (a) and per-gene (b) offsets with a near-flat prior (λ₀ = 1e-3). It uses the same PG-EM with augmented k+1 blocks; the other side's offset moves to the right-hand side of each solve. On synthetic data J is monotone and the fit recovers the offsets (r = 0.98) and P (r = 0.97).

- **CV** (same 5% held-out entries), k ∈ {20, 40, 80} × λ ∈ {30, 10, 3}:
  - The best is k = 40, λ = 10: NLL 0.17767, AUC 0.9653. Without offsets: 0.17779 / 0.9653.
  - The fit improvement is tiny, but convergence is twice as fast: 81 iterations for the final refit vs 188.
  - Fit: `full/fits/best_full_offsets.npz`.
- **Cell offset:** â absorbs depth, with ρ(â, log UMI) = 0.95.
- **R4** is the embedding of UVᵀ only, without the offsets:

| metric | R1 counts | R3 P̂ | R4 offsets | R4, depth regressed |
|---|---|---|---|---|
| hemilineage kNN balanced acc | **0.977** | 0.873 | 0.923 | 0.940 |
| hemilineage Leiden ARI | **0.649** | 0.490 | 0.538 | 0.648 |
| hemilineage cosine silhouette | **0.296** | −0.168 | 0.080 | 0.138 |
| 15-NN hemilineage purity | **0.974** | 0.776 | 0.858 | 0.877 |
| \|ρ\| of top-2 components vs log depth | 0.557 | 0.976 | 0.800 | 0.105 |

**Verdict:** offsets clearly improve BernFac's cell representation. With the residual depth axis removed, R4 matches processed counts on clustering (ARI), but processed counts still lead on kNN accuracy and neighbour purity.

The residual depth axis persists because, under p = 1 − exp(−s μⱼ), depth shifts the logit of rare genes more than abundant ones. That depth × gene interaction cannot be absorbed by any additive offset. The natural next model is a complementary log-log link, log(−log(1−p)) = log s + log μ, where depth enters additively.

The deck is now 37 slides; slides 30–36 are the full-data section, including the offset slide (34).

## Complementary log-log link (21 Oct 2026)

**Model:** log(−log(1−p)) = aᵢ + bⱼ + uᵢᵀvⱼ, i.e. p = 1 − exp(−μ) with μ = e^{aᵢ} e^{bⱼ} e^{uᵢᵀvⱼ}. This is a Poisson-threshold model in which depth multiplies every gene's rate.

**Inference:** the Pólya–Gamma identity does not apply to this link, so `bernfac_large.fit_cloglog` uses block Fisher scoring with the same k+1 augmented row and column solves.
- Score g = yμ/(e^μ−1) − (1−y)μ; Fisher weight w = μ²/(e^μ−1).
- The log-likelihood is concave in η, and a per-row backtracking line search makes J nondecreasing.
- Synthetic check: the score matches finite differences to 1e-9, J is monotone, and â tracks log depth (r = 0.97). Held-out NLL is slightly better than logit + offsets.

**CV** (same 5% held-out entries), k ∈ {20, 40, 80} × λ ∈ {100, 30, 10, 3}:
- Best is k = 40, λ = 30, with held-out NLL 0.17772 and AUC 0.9653.

  | model | held-out NLL |
  |---|---|
  | logit + offsets | 0.17767 |
  | logit, no offsets | 0.17779 |
  | cloglog, cell + gene offsets only | 0.1845 |
  | logistic, cell + gene offsets only | 0.1854 |

- The final refit converged in 71 iterations (`full/fits/best_full_cloglog.npz`). ρ(â, log UMI) = 0.96.

| metric | R1 counts | R4 logit + offsets | **R5 cloglog + offsets** | R5, depth regressed |
|---|---|---|---|---|
| hemilineage kNN balanced acc | **0.977** | 0.923 | 0.932 | 0.939 |
| hemilineage Leiden ARI | **0.649** | 0.538 | 0.546 | 0.615 |
| hemilineage cosine silhouette | **0.296** | 0.080 | 0.118 | 0.141 |
| 15-NN hemilineage purity | **0.974** | 0.858 | 0.886 | 0.888 |
| library kNN balanced acc (lower is better) | **0.127** | 0.273 | 0.228 | 0.224 |
| library mixing (1 = ideal) | **0.937** | 0.868 | 0.898 | 0.900 |
| \|ρ\| of top-2 components vs log depth | 0.557 | 0.800 | **0.411** | 0.097 |

**Verdict:**
- All three links tie on held-out prediction.
- Cloglog gives the best BernFac cell representation: it halves the residual depth axis and improves kNN accuracy, purity, silhouette and library mixing.
- Processed counts still reflect biology best on every hemilineage metric.
- The remaining gap is plausibly information lost by binarising: detection keeps only whether a count is above zero, not its level. The natural next step is to factorize the counts directly (Poisson/NB with a log-depth offset).

The deck is now 38 slides; slide 35 is the cloglog slide.

## Round 2: count models and three structured Bernoulli models (21–22 Oct 2026)

All code is Python and lives in `cm_minimal/experiments/bernfac/models/{counts,mfa,neural,lbm}/`, each with tests and a README. Outputs are in `models/`.

**Split.** The 5% held-out entries were split into a validation half (model selection) and a test half, scored once.

| | entries |
|---|---|
| train | 305.6 M |
| validation | 8.05 M |
| test | 8.04 M |

The earlier BernFac fits were rescored on this split.

### Held-out detection prediction (test, each model selected on validation)

| model | selected setting | test NLL | test AUC |
|---|---|---|---|
| negative-binomial counts | k=120, λ=10 | **0.17660** | **0.9657** |
| mixture of local factors | K=1, d=40, no residual | 0.17750 | 0.9653 |
| BernFac logit + offsets | k=40, λ=10 | 0.17770 | 0.9653 |
| neural manifold | affine head, d=20, 64×64 | 0.17772 | 0.9652 |
| BernFac cloglog + offsets | k=40, λ=30 | 0.17774 | 0.9652 |
| BernFac logit | k=40, λ=10 | 0.17780 | 0.9652 |
| latent block model | 200 × 200 groups | 0.18300 | 0.9631 |
| Poisson counts | k=80, λ=10 | 0.18390 | 0.9650 |
| gene rate | | 0.19763 | 0.9572 |

### Cell representation (16,460 cells, 23 hemilineages)

| representation | kNN bal. acc. | Leiden ARI | purity | silhouette | library kNN (↓) | depth \|ρ\| |
|---|---|---|---|---|---|---|
| processed counts | **0.977** | 0.649 | **0.974** | **0.296** | **0.127** | 0.557 |
| NB counts | 0.970 | **0.674** | 0.954 | 0.130 | 0.213 | **0.303** |
| neural manifold (z) | 0.941 | 0.549 | 0.928 | 0.247 | 0.139 | 0.909 |
| cloglog + offsets | 0.932 | 0.546 | 0.886 | 0.118 | 0.228 | 0.411 |
| mixture of local factors | 0.866 | 0.518 | 0.772 | −0.039 | 0.272 | 0.972 |
| latent blocks (R·B) | 0.418 | 0.065 | 0.437 | −0.109 | 0.104 | 0.766 |

### What this shows

- Modelling the counts with a negative binomial gives the best detection predictions, and the best clustering of hemilineages of any representation, processed counts included.
- Among models that see only binary detection, five very different structures tie to within 0.0003 nats: low-rank logit, offsets, cloglog, mixture with K=1, and neural decoder. Extra mixture components and co-clustering do not help on this dataset.
- The neural manifold is the binary model whose cells mix best across libraries.

### Tests and fitting

- Tests pass for all four models:

  | model | tests passed |
  |---|---|
  | counts | 24 configurations, all tests |
  | mixture | 21/21 |
  | neural | 14/14 |
  | block | 12/12 |

- Every monitored objective or bound was monotone. The LBM's float32 ELBO had dips under 2e-6 relative; its float64 tests are strict.
- PG-GEM beat direct BCE on the full data at equal wall time: validation NLL 0.1776 vs 0.1786.

### Caveats

- The neural and mixture fits were time-limited.
- The NB rank was extended until it stopped improving (k=160 is worse than k=120).
- The mixture's d was extended to 80; d=40 is an interior optimum.
- The latent block model is best at its 200 × 200 corner and was not extended; it is far behind the other models.
- There is only one dataset.

The deck has 52 slides. Slides 36–52 cover this round: a BioRender schematic, LaTeX panel and selection figure for each model, then the comparisons and conclusions.

## Round 3: one narrative for all nine representations (22 Oct 2026)

Google Slides deck v3: `NineRepresentations_v3_22Oct2026` (Drive id 1Jc-fiWr1VZ0Vra8R5fhztL7XCrJmZxUNeBuQAF6wNFk, 90 slides). The v2 deck is unchanged. Everything for this round is in `v3/` (plan and log: `v3/PLAN.md`); code is in `cm_minimal/experiments/bernfac/v3/`.

Order: methods and metrics for every family (two new BioRender schematics and five LaTeX panels) → shared synthetic benchmark → held-out prediction → cell maps → cells × genes snippets → diagnostics → scorecard and recommendation → appendix (pilot, BernFac synthetic CV, per-model detail).

New results in this round:
- **Synthetic benchmark** (`v3/synthetic/`): five worlds G1–G5, n=2000, m=300, 3 seeds; every method selected on validation entries, as on real data.
  - Lowest KL(P_true ‖ P̂): the local-factor mixture in G1–G3 (G2 is the NB world: 0.022 vs 0.026 for NB counts). The neural manifold wins G4 (0.031) and latent blocks win G5 (0.0007).
  - Poisson is close to the cell + gene baseline on overdispersed counts.
  - Recovery of the true cell groups: NB counts has the best G3 kNN accuracy (0.95), processed counts the best G3 Leiden ARI (0.97), and BernFac logit only 0.63.
  - Grids are small and there is one restart per configuration. R8 hits its iteration cap.
- **Cell maps** (`v3/figures/fig_umap_*`): depth is the main axis of binary PCA, BernFac logit and the mixture (|ρ| 0.97–1.00).
  - The deepest decile is 84% primary neurons, so part of that axis is biology.
  - Hemilineage kNN errors concentrate at the depth extremes (R3: 28% and 26%, against 3–4% in the middle deciles).
  - Regressing out depth lifts BernFac logit's Leiden ARI from 0.49 to 0.61. NB counts stays best at 0.69.
- **Heatmaps** (`fig_heat_*`): NB counts gives the sharpest marker blocks and is closest to the empirical stratum rates (mean |diff| 0.073). Latent blocks are off by whole blocks.
- **Calibration**: every model is calibrated to within 0.7 percentage points on test entries except Poisson (up to 2.3).
  - Poisson's val–test gap comes from 136 test entries with NLL above 20.
  - The selected NB fit has median φ of about 6×10⁵, which is effectively Poisson.
- **Recommendation**:
  - Use NB counts (k=120) as the primary representation, with processed counts as the reference map.
  - Use cloglog + offsets if a binary-detection model is required.
  - Rerun the neural manifold with a longer budget before judging it.

## Round 3b: tuning transparency and depth-standardized probabilities (23 Oct 2026)

The same Google Slides deck (1Jc-fiWr1VZ0Vra8R5fhztL7XCrJmZxUNeBuQAF6wNFk) was rebuilt as `NineRepresentations_v3_23Oct2026`, 98 slides. Depth now runs through the narrative: the question, a BioRender schematic of where each model puts depth (`v3/biorender/v3_depth_schematic.png`), a LaTeX decomposition (`v3/equations/v3_depth.png`), a tuning table (`v3_hyperparams.png`), the synthetic tuning figure, and Part 5 with observed and depth-standardized heatmaps.

- **Tuning:**
  - Every model with hyperparameters is selected on validation entries. This is one random 95/2.5/2.5 split of entries, not K-fold.
  - Processed counts and binary PCA have fixed settings.
  - The Leiden resolution is the one setting chosen against the labels.
  - Synthetic benchmark (`v3/tables/syn_selection.csv`, `fig_syn_select`): the 3 seeds agree on the config in 34 of 40 method × world cells. R9 chose its largest K (40) in 11 of 15 runs and R8 its strongest λ_V (100) in 12 of 15, so those grids may be too small.
- **Depth-standardized probabilities** for R4–R6:
  - Definition: the cell offset c_i is replaced by its median, giving p̃_ij = g⁻¹(c̄ + b_j + u_iᵀv_j). The code is `predictors.standardized()`; checks are in `v3/check_standardized.py`.
  - Observed p̂ keeps each cell's own depth and is what is scored. p̃ is never scored by NLL.
  - Depth dependence of p (median within-hemilineage |ρ| with log UMI) falls from 0.82/0.90/0.79 to 0.29/0.19/0.11 for R4/R5/R6. Models without an offset stay at 0.50–0.83.
  - Marker genes keep 84–90% of their between-hemilineage variance.
  - R4 keeps the most depth, because depth leaks into UVᵀ (10-PC R² for log UMI is 0.90).
  - Figures: `fig_heat_std_overview|depth|diff`, `fig_std_quant`. Table: `v3/tables/depth_std_metrics.csv`.
- **Recommendation updated:** R6's p̃ is the cleanest depth-free probability; R5's p̃ is the binary-model option.

## Round 4: strict review, revised experiments, deck v4 and summary deck (23–25 Oct 2026)

Decks (Google Slides; v3 and v2 kept):
- Full v4: `NineRepresentations_v4_25Oct2026`, id 1_SVQuqGqOUm85xniWjNvMFhf0b9iSmAGJUkPSRdVgHM, 86 slides (44 main, 42 appendix).
- Summary: `NineRepresentations_v4_summary`, id 1oWG6Uomf3l-XsDZJcwZT6nEZWK2U0OwmfbyXjBlO5v0, 10 slides.

**Review.** Three independent reviews: statistics, code, claims (`v3/review/review_{A,B,C}*.md`; triage in `revision_plan.md`).
- No leakage or scoring bug: held-out entries are excluded from every fit, and NLL was recomputed independently to 1e-9.
- Several conclusions were wrong or over-stated. They were re-tested by five workers; code is in `cm_minimal/experiments/bernfac/v4/`, outputs in `outputs/bernfac_20261020/v4/`.

**Prediction.** Paired two-way (cell × gene) bootstrap, 1000 reps, all 8.04 M test entries.
- R6 NB beats every model by 0.9–1.2e-3 nats; every CI excludes 0.
- R7 < R4 < R3 are resolvable but within 0.2e-3; R4, R5 and R8 are tied.
- About 230 bursty wiring and TF genes carry half of R6's gain.

**NB vs Poisson.**
- NB wins at every matched (k, λ), by 4–9e-3. The gain is on zeros of frequently detected genes with φ < 10.
- The median φ of about 6e5 comes from rare genes where φ is unidentifiable. The v3 "rank, not overdispersion" claim is withdrawn.
- Poisson outliers are misspecification, not overflow.

**Refits.**
- Poisson k=120 (best λ=30) still trails NB by 2.8e-3.
- NB k=160 overfits; cold and warm starts agree.
- R4 edges are worse.
- R8 with a 250-iteration budget (d=40, λV=10): test 0.17754, tied with R7. It is at a grid corner and not converged.

**Selection audit.**
- Validation-only re-selection gives the same R3–R5 configs. The original grids used val∪test.
- Test NLL had been computed for all configs.

**Embeddings.** One protocol: centred SVD, d ∈ {20, 40}, whitened or not, linear probe, Leiden at 23 clusters over 5 seeds. Pearson residuals and scVI added as comparators.
- The probe reads hemilineage at 0.97–0.99 for every model, so v3's "binary models lose signal" was coordinate scaling.
- Clustering: R6 is best at d=40 whitened (ARI 0.74 ± 0.03); R1 leads at d=20.
- Depth within hemilineage × birth time (R² from the top-10 axes): R10 0.10, R11 0.20, R6 0.37, R1 0.42; binary models 0.87–0.97. R8's earlier "first axis is depth-free" was misleading.
- Sex: the label is almost exactly roX expression (AUC 0.998). Within mixed-sex libraries only the all-gene factor models decode sex.

**Depth-adjusted p̃.**
- Not depth-free: R² of log UMI is 0.41 (R6), 0.66 (R5), 0.87 (R4), against 0.61 for log counts.
- Marker variance is retained at 0.98–1.02 (v3's 84–90% was η²).
- The held-out depth-gene gap falls from 0.40 to 0.23 / 0.15 / 0.13; about a third of the gap is birth time.
- On synthetic data p̃ recovers the true adjusted P.
- On real data it also removes birth-time variance, so over-correction cannot be ruled out.

**Synthetic.** Grids contain the true rank; every config is scored against the truth; R8 runs under an iteration budget.
- Validation regret is about 0.
- Each family wins its own world (R6 wins the NB world).
- R1's clustering advantage came from seeing all entries.
- Count models trailed in G1 only because of how its counts were constructed.

**Equations fixed.**
- The offsets slide had the depth direction backwards: abundant genes shift most.
- Notation unified.
- The tuning table is now exact.
- Errata and recommendation panels added.

**BioRender.** New evaluation design (scattered entries), new BernFac-family schematic, corrected depth schematic, summary overview.

**Open.** Refit and split variance; NB k=160 with stronger priors; R8 beyond its grid corner; a depth model that separates birth time from sequencing depth.
