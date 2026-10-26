# Review C: claims, equations, schematics, narrative (v3 deck, 23 Oct 2026)

Scope: `v3/deck/slides_main.json` (M#), `slides_appendix.json` (A#), equation sources, BioRender PNGs, figures and tables. All checks were read-only. Numbers I recomputed come from the tables cited, plus `cells.csv.gz`, `knn_predictions.csv.gz`, `cell_depth.csv.gz`, `full/umap/{metrics,leiden_by_resolution,metrics_depth_regressed}.csv` and `models/counts/{tables/selection.csv,configs/*.json}`. Where `leiden_extra.csv` (reviewer A's artefact in this folder) is cited, I did not re-verify it.

---

## CRITICAL

**C1. M65 Recommendation: "its edge over Poisson probably comes from the higher rank (120 vs 80)" is contradicted by the deck's own table.**
- Evidence: `models/counts/tables/selection.csv` at the same k=80, λ=10 gives NB val NLL 0.17656 and Poisson 0.18063, a gap of 0.0041. NB k=40 (0.17738) already beats the best Poisson (0.18063). NB k=80→120 gains only 0.0001.
- The premise "dispersion runs to the Poisson limit" is a rank/prior artefact. The fitted φ median by config is 39 (k=20), 65 (k=40), 230 (k=80, λ=10), 4.2e2 (k=120, λ=30) and 5.9e5 (k=120, λ=10). It also reaches 4e5 at k=80, λ=3. High rank with a weak prior lets the factors absorb the overdispersion. The scorecard note "φ→∞ (≈Poisson)" is the same misreading.
- Fix: delete the sentence. Say that NB beats Poisson at matched rank, and that the apparent φ→∞ at k=120 means the factors absorb extra-Poisson variance. Report the φ distribution at the selected config, not just the median. Consider reporting NB k=80 as the primary model, since its NLL ties k=120 and its dimension is closer to the others.

**C2. M9 (eq_offsets panel): the depth × gene interaction is stated backwards.**
- The panel says depth "shifts the logit of rare genes more than abundant ones". Under p = 1 − e^{−sμ}, logit p = log(e^{sμ} − 1), so d logit/d log s = x·e^x/(e^x − 1) with x = sμ. This is ≈1 for rare genes and ≈x for abundant ones, so depth shifts the logit of **abundant** genes more.
- The deck's own data agree: in `fig_std_quant` B, R4's p̃ keeps depth (|ρ| 0.61) only in the top detection-rate decile.
- Fix: correct the sentence. Then use it as *the* explanation for R4's residual depth axis and for R5 beating R4. This also lets you drop the vaguer "depth leaks into its factors" (M40, M58, M65).

**C3. M36 Calibration: "R5 and R6 also produce confident predictions (p̂ below 0.01 or above 0.99) that the logit models rarely give" is false at the stated thresholds.**
- Evidence (`calibration_summary.csv`): the fraction with p̂ < 0.01 is 0.521 (R3), 0.526 (R4), 0.516 (R5), 0.528 (R6) and 0.523 (R7). The fraction above 0.99 is 0.031–0.033 for the logit models against 0.037–0.038 for R5/R6.
- The real difference in panel c is in the extreme tails (|logit| > ~10, i.e. p̂ < 1e-4 or > 1 − 1e-4).
- The same caption calls the Poisson error "mid-range". It actually peaks at p̂ ≈ 0.89 (−2.34 pp, bin 18) and the model over-predicts in every bin above ~0.1.
- Fix: restate with the correct thresholds and location.

**C4. "Best Leiden ARI" (M48, M64, M65) is a headline reason for choosing R6, and it is not robust.**
- The metric is chosen against the labels as the best of 6 resolutions. R1 and R6 both peak at res = 0.25, the lowest value in the grid (`leiden_by_resolution.csv`).
- The ranking depends on resolution. At res 1.0: R5 0.546 > R6 0.532 > R7 0.518 > R3 0.490 > R1 0.480. At res 0.5: R6 0.589, R1 0.572, R8 0.549.
- Reviewer A's `leiden_extra.csv` (5 seeds) gives R6 0.668 vs R1 0.650 at res 0.25. R1 reaches 0.645 at res 0.1. The gap is within about one seed SD.
- R1/R6 produce 33–42 clusters against 23 hemilineages at their best resolution. Birth time splits hemilineages, so ARI against hemilineage penalises real structure.
- Fix: report ARI/AMI at a fixed number of clusters (for example 23, found by bisection on resolution), with seed spread. Do not claim "best Leiden ARI (0.67 vs 0.65)" as a differentiator.

---

## MAJOR

**M-a. Embedding comparison is not matched (M39–M49, M64, M65).**
- Dimensions differ: R1/R2 50, R3–R5/R7 40, R6 120, R8 20, R9 200 (`metrics.csv` column `dim`).
- Gene sets differ: R1 uses 2,000 HVGs and scaling, while all factor models use all 19,543 genes.
- Distances are raw Euclidean on singular-value-scaled coordinates. R3's first singular value is 89,025 against 9,829 for the second (`run_info.json`), so one axis dominates kNN.
- A linear probe (`hl_logreg_balacc`, already computed but not shown) gives 0.97–0.996 for R1–R7. Hemilineage information is present in every embedding; the kNN differences reflect geometry and scaling.
- Reviewer A's whitened kNN gives R3 0.87→0.96 and R7 0.87→0.96, against R1 0.98 and R6 0.98.
- Fix: show logreg and whitened/cosine kNN next to raw kNN. Rerun the metrics at matched d (for example 40 for all, or NB k=40/80). Run R1 on all genes, or the factor models on the HVGs. Rewrite "binary models lose cell-type signal / embeddings dominated by depth" as "their leading axis is depth; after whitening they recover hemilineage nearly as well".

**M-b. Depth is partly biology, and the "depth-free" metrics reward removing it (M45, M49, M56–M58, M65).**
- Birth time explains 24% of the variance in log10 UMI. Library explains 8% and sex 1% (η², recomputed from `cells.csv.gz` + `cell_depth.csv.gz`). The top depth decile is 84% primary neurons, against 16.7% overall (confirmed).
- So "median depth correlation 0.11 = cleanest" (M65) and "regressing depth helps" (M49) are not unambiguous virtues. Lower is not better once the biology tracks depth.
- Fix: compute the depth correlations within hemilineage × birth-time strata, and regress depth within birth-time class. At minimum, state that the ideal value is not 0.

**M-c. M45 "most hemilineage errors sit at the two depth extremes" conflates depth with birth time.**
- R3 15-NN error rates (recomputed from `knn_predictions.csv.gz`): primary neurons 23.5% overall, secondary 5.3%, precursors 58%.
- In the middle deciles, R3 primary error is still 8–16% against 2% for secondary. In the top decile it is 27% primary and 19% secondary.
- In the bottom decile the errors include precursors and primaries (primary 58%).
- Fix: show error by depth × birth time. The fair reading is that birth time and depth both contribute in R3, while R1/R6 have little of either (R6 primary 5%).

**M-d. M42 and the Next steps on M65: the sex question can be answered now, and the answer is sex genes, not single-sex libraries.**
- `cells.csv.gz` shows 14 of 21 libraries contain both sexes, holding 84.6% of cells. The 7 single-sex libraries are B2, C2, D2, D9, F2, G2 and H9.
- Sex kNN restricted to mixed-sex libraries is barely lower: R1 0.804 vs 0.815, R3 0.813 vs 0.830, R4 0.845 vs 0.863, R6 0.800 vs 0.814.
- roX1 is detected in 99.7% of male vs 38% of female cells; roX2 in 98.6% vs 14%.
- Also confirm how `Sex` was annotated. If it was called from roX expression, the kNN test is circular.
- Fix: replace "we have not checked" with this result and drop the item from Next steps. "About 0.82 in every method" is also wrong: R8 0.78, R9 0.54.

**M-e. Synthetic benchmark does not validate the recommended model, and its design favours the binary models (M28–M32, M65).**
- In G2 the generator *is* R6's model (rank 8, NB, φ ~ logN(0,1)), yet R6 is third: KL 0.0259±0.0016 vs R7 0.0215 and R4 0.0246. This points to a deficiency in fitting, dispersion estimation (φ recovery is not reported) or the grid (λ ∈ {1, 10, 30} at n = 2000, m = 300). It should be investigated, not reported as "mid-pack".
- In G1, G4 and G5 the counts are artificial: X = Y·(1+Poisson(1)) (`syn_generators.py` docstring). The positive part is noise, so R6/Poisson are handicapped by construction. M28 says the same split is used for every method but never discloses this.
- The R7 synthetic grid is 2×2 (K ∈ {1, 4}, d ∈ {5, 10}). Every R7 pick in all 15 world-seed runs is on a grid edge. M29 names only R8 and R9 as edge-limited.
- Fix: disclose the count construction. Report R6's φ recovery on G2 and rerun G2 with a wider λ grid. Flag R7's edge picks. Rewrite the M65 R6 line as "fails to win its own world (G2); under investigation".

**M-f. Missing standard comparators.**
- R6 is NB GLM-PCA (Townes et al. 2019) with a ridge prior, and R3 is logistic PCA. Both should be named as such.
- An expert will ask for: scVI (with library as covariate, latent 10–30; it gives both a held-out likelihood and an embedding); analytic Pearson residuals + PCA (Lause et al. 2021) or sctransform as the stronger R1; GLM-PCA itself; and possibly ZINB-WaVE or NewWave.
- Without at least scVI and Pearson residuals, "R6 primary choice" is a claim about an in-house method against a weak baseline set.

**M-g. Uncertainty is absent from every figure.**
- There is one data split, one fit per config, one UMAP/Leiden seed and no CIs.
- Reviewer A's `boot.json` shows R6−R4 = −0.0011 (two-way CI −0.00125 to −0.00091). That gap is robust and consistent across all 10 depth deciles in `test_nll_by_depth_decile.csv`. R8−R4 (CI spans 0) and R5−R4 (+0.00005) are not separable.
- Fix: put the CIs on M35, call R3–R8 "tied", and add seed spread to every representation metric.

**M-h. Three different depth metrics are used interchangeably.**
- They are: comp-1 |ρ| (M47 "R6 0.24", M65 "R8 depth off its first axis"), max over PC1–2 (scorecard and `v3_metrics_rep`, R6 0.30) and max over the top 5 (M47 figure, R6 0.52).
- The ordering changes with the metric: R8 is best on comp 1 (0.05) but 0.91 on comp 2.
- R8's z coordinates and R9's RB are not PCA-ordered, so "first axis" means nothing for them.
- Fix: use one metric (max over the top-5 PCs of a PCA-rotated embedding, or a CCA/R² of log depth on the full embedding) everywhere, and drop the "depth off its first axis" argument for R8.

**M-i. Numbers that do not match their sources.**
- M48: "low-rank detection models trail by 0.04–0.10 on hemilineage metrics". Against R6/R1 the trails are kNN 0.04–0.10, but ARI 0.13–0.18 (R3 0.490 vs 0.674) and purity 0.09–0.20 (R3 0.776 vs 0.974).
- M47: "Depth dominates R2, R3, R7 (|ρ| 0.97–1.00)". Depth |ρ| is 0.967–0.976; the 1.00 is the detection-rate column. "R6 is cleanest (0.24 on its first component)" mixes metrics (see M-h).
- M37: "NB best in every gene-frequency decile". In decile 1, R7 (0.00184), R9 (0.00185) and R8 (0.00187) are below R6 (about 0.0019). Trivial in size, but the claim is literally false.
- M30: "Poisson is close to the cell + gene baseline (G2, G3)". In G3 Poisson is *worse* than the baseline (0.063 vs 0.048) and even than gene-rate (0.056).
- M31: "Poisson over-predicts everywhere above 0.3". It over-predicts from the lowest bin (pooled 0.052 vs truth 0.034). R5's top bin is 0.96 vs truth 0.87; only R6 matches "0.97 vs 0.85".
- M32: "G5: latent blocks recover their own blocks almost exactly". True for kNN/k-means, but on Leiden R7 is best and R9 is about 0.67.
- M34: "LBM and Poisson are still improving at the edge". R8 is also at the edge (d = 20, still falling, all runs time-limited).
- M41: "0.21–0.27 for the detection models". This includes R6, which is a count model.
- M43: birth time is described as "primary, secondary, precursor", but there are 4 classes (earlysecondary too, chance 0.25 in the figure).
- M57: "Genes that are off stay off, so the hemilineage pattern is unchanged". In R4/R5 the deep 05B/06B cells lose up to 0.4 on their own markers (lbl, vg, Vsx1, sens-2). The pattern is kept, but its amplitude is not.
- M9/M10 panels quote appendix-CV results (5% hold-out: 0.17767/0.17772; "81 vs 188 iterations"), while `runtime.csv` says 45 vs 99 iterations and the main-deck test NLLs differ. Use v3 numbers only, or label the source.
- A20 refers to "equations, slide 80", which no longer exists. A2 says "36 (λ, k) fits" but A11 and eq_validation say 45.

**M-j. M65 R8 line ("Promising: library kNN 0.14, depth off its first axis").**
- Its depth |ρ| over PC1–2 is 0.91, worse than R4. Its d = 20 is at the grid edge and every run hit the time budget. Low library kNN can also come from a noisier embedding: its sex kNN is the lowest of R1–R8 (0.78).
- Fix: "inconclusive" rather than "promising".

**M-k. Scorecard (M64) rewards uninformative embeddings.**
- Min–max colouring boxes R9 as best for library kNN (0.104), and R9 also scores well on depth, while it has hemilineage kNN 0.42.
- Nuisance scores need to be conditioned on biology: library predictability within hemilineage, iLISI/kBET per hemilineage, or library kNN minus the value expected from composition.
- Library composition is unbalanced (7 single-sex libraries; library explains 8% of depth variance), so the raw library kNN partly reflects sex and depth.
- Identifiability "UVᵀ up to GL(k)" contradicts eq_mstep and the PG-EM schematic ("O(k)"). With a Frobenius prior at MAP only rotations remain (A12 says so too).

---

## MINOR

- **Notation:**
  - ψ is "logit" in v3_notation but the linear predictor is η in v3_depth.
  - c_i means the cell offset (v3_depth), the mixture label (eq_mfa) and the LBM row cluster (eq_lbm).
  - a_i, b_i are reused in the silhouette formula.
  - v3_models says the R7 representation is "mean logits E[ψ_i]" (19,543-dim), but the embedding used is 40-dim.
  - R3 has no b_j, yet v3_depth and the depth schematic show b_j for "R3, R7, R8, R9".
- **Equations checked and correct:** PG identity and E[ω]; R9 block update; cloglog score/Fisher weight (μ/(e^μ−1), μ²/(e^μ−1)); NB score/weight; NB Pr(X ≥ 1); KL = E[NLL] − H; nuclear-norm identity; offset block solve; count-split mapping. The R6 offset log s_i + a_i with a near-flat prior on a_i is redundant (a_i can undo s_i). This is harmless, but should be stated.
- **Tail behaviour:** the Poisson test NLL is driven by 136 entries with NLL > 20 (2.1% of the total; max 25,257 nats, i.e. μ ≈ 2.5e4 predicted for an observed zero). This is an extrapolation blow-up and should be diagnosed (clip η or check the gene) rather than read as a model property. R4 has 28 entries > 20 under the near-flat offset prior λ0 = 1e-3.
- **Schematics:**
  - v3_evaluation_design: validation and test are drawn as whole bottom rows, which reads as a cell hold-out. The split is random entries; redraw as scattered cells. The NLL and Brier icons (histogram, scatter) are meaningless.
  - v3_depth_schematic: R3 is drawn with b_j (wrong). "R1: depth divided out" ignores R1's remaining |ρ| = 0.56. "All rows same brightness" overstates what p̃ does.
  - v3_methods_map:
    - The R5 cloglog curve is drawn symmetric, identical to the logistic.
    - R9 is drawn with dendrograms, but an LBM is not hierarchical.
    - The R6 "dispersion vs mean" curve implies a fitted φ–mean trend that does not exist.
    - The output panel implies p̂ for R1/R2.
  - bernfac_1 (used on M7 for the full data): it says "1000 cells × 100 genes" (pilot), the sigmoid passes through 0 at ψ = 0, and it is titled "Bayesian" although the fit is MAP.
  - bernfac_5: the right panel is titled "Detection probability for binary models"; it should say "from count models".
  - bernfac_8: σ(B) is 2×4 while the matrix has 4×4 blocks.
  - bernfac_6: the JJ/PG bound is drawn as crossing straight lines instead of a quadratic upper bound.
- **Interpretation:** "depth leaks into its factors" (M40, M58, M65) should be replaced by the link-misspecification explanation (C2). M56: the mitochondrial-ribosomal depth genes (mRpS17, mRpS33, ND-B14.5A) are also classic quality/size markers. Say that both readings are open.
- **Heat slices (M51–M57):** 36 genes and 222 held-out entries; `heldout_nll` on 222 entries is noise. Present them as illustrations. Use only whole-matrix numbers (M58) for claims. "Sharpest marker blocks" (M52) rewards near-0/1 probabilities; sharpness is not accuracy.
- **M60:** "its numbers are a lower bound on what it can do". Use "pessimistic estimate". Longer training with λ_V at the grid edge could also overfit.
- **M61:** fit times mix warm-started (time excludes predecessor) and cold-started runs; the scorecard boxes R7 (cold, 4.4 min) as best. Not comparable.

---

## Narrative / structure

1. M6–M18: one table, then 6 models × (schematic + equation) = 13 slides before any result. Keep M6 plus one combined schematic/equation slide per family, and move the detailed equations (M8, M12, M14, M16, M18, M22) to the appendix. Saves about 8 slides.
2. M39–M43 (5 UMAP grids): merge into 2 (hemilineage + depth; library/sex/birth time as small multiples). M47 and M48 overlap; keep one table that includes the logreg and whitened columns.
3. M51–M57 (7 heatmap slides): merge to 3 (observed overview + markers; depth genes observed vs standardized; std diff). M58 carries the quantitative claim.
4. A31–A33 are v2 duplicates of Part 4 with slightly different framing ("binary models keep a strong depth axis"). Delete them or mark them superseded.
5. Missing logic step: the deck never says why a detection-probability matrix is the target for ConnectionMiner, or which downstream task decides the choice. The recommendation optimises held-out NLL and hemilineage ARI, which may not be the downstream objective.
6. Recommendation slide, which an expert would rewrite as follows:
   - R6's prediction gain (0.0011 nats, CI excludes 0, every depth decile) is the one robust real-data result. It uses count information that the binary models do not see.
   - The embedding advantage over R1/R4/R5 is not established (C4, M-a).
   - The synthetic study does not support R6 (M-e).
   - The depth-standardized p̃ is useful but removes primary-neuron biology (M-b).
   - Do not recommend a single model until a matched-dimension, matched-gene, fixed-cluster-number comparison including scVI and Pearson residuals is done.

## Controls to run before collaborators see it (in priority order)
1. NB vs Poisson at matched k, and φ distribution by config (data already exist; just report them).
2. Fixed-cluster-number ARI/AMI with seed spread. Linear probe and whitened kNN at matched d (40) and on a matched gene set.
3. scVI and analytic Pearson residuals PCA under the same metrics. scVI also on the held-out entries if feasible (mask → impute).
4. Depth metrics within hemilineage × birth time; depth regression within birth-time class.
5. G2 re-run with a wider λ grid and φ-recovery check. Disclose the artificial counts in G1/G4/G5. Widen R7's synthetic grid.
6. Longer R8 budget with d > 20. Poisson grid extended to k = 120 for completeness. Investigate the k = 160 NB jump (0.1765 → 0.1783), which may be an optimisation issue.
7. Bootstrap CIs on all test-NLL differences (reviewer A has them) and on representation metrics.
