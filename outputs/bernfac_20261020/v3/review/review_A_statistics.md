# Review A: statistical rigor of the nine-representation comparison (v3 deck, 23 Oct 2026)

Scope: the claims in `v3/deck/slides_main.json`, the PANELS in `cm_minimal/experiments/bernfac/v3/make_equations_v3.py` and `REPORT.md`, checked against the tables. No existing file was edited. I ran three new analyses. Their scripts and outputs are in this folder:
- `boot.py` → `boot.json`: paired bootstrap of test NLL differences.
- `leiden.py` → `leiden_extra.csv`: Leiden seeds and resolutions, whitened kNN, and sex decoding within each library.

## 0. New numbers

### 0a. Paired test-NLL differences

These use all 8,044,655 test entries and the cached per-entry NLLs (`v3/cache/test_nll_*.npy`). Each bootstrap has 400 Poisson-weight replicates. The two-way bootstrap is a cell × gene pigeonhole bootstrap (Owen 2007). Δ = NLL(a) − NLL(b), so a negative Δ means a is better.

| a − b | Δ | iid SE | cell SE | gene SE | two-way SE | two-way 95% CI | cells where a is better | genes where a is better |
|---|---|---|---|---|---|---|---|---|
| R6 − R4 | −1.10e-3 | 4.0e-5 | 4.1e-5 | 6.9e-5 | 9.0e-5 | [−1.25e-3, −0.91e-3] | 62% | **51%** |
| R4 − R3 | −1.09e-4 | 1.6e-5 | 1.6e-5 | 1.6e-5 | 2.7e-5 | [−1.59e-4, −0.55e-4] | 59% | 64% |
| R7 − R4 | −1.95e-4 | 1.3e-5 | 1.4e-5 | 1.3e-5 | 2.3e-5 | [−2.42e-4, −1.51e-4] | 54% | 40% |
| R5 − R4 | +4.5e-5 | 1.7e-5 | 1.8e-5 | 2.0e-5 | 3.2e-5 | [−2.1e-5, +1.0e-4] | 48% | 47% |
| R8 − R4 | +2.0e-5 | 2.8e-5 | 3.3e-5 | 3.8e-5 | 5.7e-5 | [−9.7e-5, +1.26e-4] | 49% | 32% |
| R7 − R6 | +9.0e-4 | 4.0e-5 | 4.0e-5 | 6.5e-5 | 9.0e-5 | [+7.1e-4, +1.07e-3] | 39% | 33% |

What the bootstrap shows:
- **Dependence matters.** The two-way SE is 1.7–2.2× the iid SE.
- **R6's lead is real.** R6 beats R4 by 12 SE. The gain is 5.5% of the improvement over the gene-rate baseline (0.0200 nats).
- **The gain is concentrated.** R6 is better on only 51% of genes, and 1% of genes (about 195) carry 45% of its total gain. The R7 − R4 gain is concentrated the same way: R7 is better on only 40% of genes, and the top 1% carry 56% of the gain.
- **The binary models are not a clean tie.** R7 beats R4 (8 SE) and R4 beats R3 (4 SE), although both gaps are ≤0.2 milli-nats, which is practically negligible. R4, R5 and R8 cannot be ordered.
- **R8 vs R4 flips between splits.** R8 is better than R4 on validation (0.177606 vs 0.177639) and worse on test.
- **These SEs leave out refit variance.** They condition on the fitted models and ignore seed and split variance. The seed spread I could find is about 1e-5 (R7 seeds s1/s2: 0.17850/0.17849; R8 r0/r1: 0.177597/0.177606).

### 0b. Leiden ARI over 5 seeds

Hemilineage ARI, mean ± SD over Leiden seeds 0–4, with number of clusters in brackets (23 hemilineages exist):

| | res 0.05 | res 0.10 | res 0.25 | res 0.50 |
|---|---|---|---|---|
| R1 | 0.554 ± 0.014 (24) | **0.645 ± 0.006** (31) | 0.650 ± 0.007 (41) | 0.573 (52) |
| R6 | 0.487 ± 0.024 (16) | 0.626 ± 0.004 (24) | 0.668 ± 0.014 (33; range 0.654–0.689) | 0.595 (43) |
| R8 | 0.401 ± 0.042 | 0.542 ± 0.037 | 0.524 ± 0.022 | 0.549 |

- At res 0.10, R1 is ahead of R6, so the ranking depends on resolution.
- The deck's 0.6745 vs 0.6492 is one seed, taken at the lowest resolution tried (0.25, the grid edge).

### 0c. Whitened kNN

Hemilineage kNN balanced accuracy, with each embedding column z-scored first (as-is → whitened):

| | as-is | whitened |
|---|---|---|
| R1 | 0.977 | 0.980 |
| R3 | 0.873 | 0.957 |
| R4 | 0.923 | 0.962 |
| R5 | 0.932 | 0.961 |
| R6 | 0.970 | 0.976 |
| R7 | 0.866 | 0.962 |
| R8 | 0.941 | 0.937 |

### 0d. Sex decoding within each library

Sex kNN balanced accuracy within each of the 14 mixed-sex libraries (≥50 cells of each sex), averaged; chance is 0.5:

| | R1 | R3 | R4 | R5 | R6 | R7 | R8 |
|---|---|---|---|---|---|---|---|
| balanced acc. | **0.585** | 0.744 | 0.768 | 0.738 | 0.807 | 0.732 | **0.575** |

15 of 21 libraries contain both sexes (`cells.csv.gz`, library × Sex).

---

## 1. Critical

**C1. The recommendation's explanation of R6 > Poisson is contradicted by the study's own tables.**
- **Claim.** `v3_recommendation` says: "fitted dispersion runs to the Poisson limit, so its edge over Poisson probably comes from the higher rank (120 vs 80)". REPORT says "median φ ≈ 6×10⁵ … effectively Poisson".
- **Evidence on rank.** At equal rank and λ (k = 80, λ = 10), NB has validation NLL 0.17656 and Poisson 0.18063, a gap of 4.1e-3 (`models/counts/tables/selection.csv`). Going from rank 80 to rank 120 within NB gains only 1.0e-4.
- **Evidence on dispersion.** φ is bimodal (`models/counts/fits/best_nb.npz`):
  - 10,127 genes have φ > 1e4. These are rare genes (median detection 0.14%), where φ cannot be identified and runs to the bound.
  - 6,955 genes (36%) have φ < 10.
  - The 25th percentile of φ is 1.4.
  - The median is therefore a summary of the unidentifiable genes.
- **Fix.**
  - Delete the sentence.
  - Report the φ distribution stratified by detection rate.
  - Attribute the gain correctly by scoring NB − Poisson per gene, split by φ bin, using cached per-entry NLL. This takes minutes.

**C2. The embedding comparisons (Part 4, scorecard, "embeddings dominated by depth") mostly reflect an arbitrary scaling choice.**
- **How the embeddings are built.** For R3–R5, e = Q_u W S (`umap_compare.embed_r3`): it is uncentred and weighted by singular values. For R3, s₁/s₂ = 89,024/9,829, so Euclidean kNN, Leiden and silhouette are driven by component 1, which is depth. kNN uses raw Euclidean distance on embeddings of 20–200 dimensions with different spectra.
- **Whitening closes most of the gap.** Per-component standardization shrinks the R1 − R3 kNN gap from 0.104 to 0.022, and the R1 − R7 gap from 0.112 to 0.017 (§0c).
- **The information is already there.** The linear probe in `full/umap/metrics.csv` (`hl_logreg_balacc`) is 0.978–0.996 for R1–R7, so hemilineage information is essentially equal. The kNN, ARI, purity and silhouette differences measure geometry, which is an analyst choice, not the model.
- **What does not follow.** "R3, R4, R7 … embeddings are dominated by depth" and "low-rank detection models trail by 0.04–0.10" do not follow as statements about the models.
- **Fix.**
  - Fix one embedding convention for all models (centred, then whitened or unit-variance top-d), with the same d for everyone (e.g. d = 20 and d = 40, so R6 gets no advantage from k = 120 and R9 none from 200).
  - Report the linear probe alongside kNN.
  - Recompute every Part 4 metric.
  - Compute: about 20 minutes on 4 threads.

## 2. Major

**M1. The test set is not untouched for R3–R5, and test is evaluated on every grid point.**
- **R3–R5 were selected on test entries.** In Round 1 their grids, λ paths and winners (k = 40, λ = 10; offsets/cloglog k = 40) were chosen on the full 5% held-out set (REPORT "Full dataset" and "Offset model"; `run_full.py` scores `heldout.npy`). `val.npy` and `test.npy` were cut from that same mask later (`make_splits.py`, 21 Oct). So test entries were used to design and select these three models. The `v3_metrics_pred` line "report once on 𝒯" and slide 33 "test entries were scored once" are false for these models.
- **Every model's test NLL was computed for every configuration.** Every `selection.csv` (counts 27 rows, mfa 36, lbm 51, neural 9, rescored 44) stores `test_nll` for every configuration.
- **The extensions may have been informed by test.** Grid extensions (NB k = 120 and 160, MFA d = 80 and `_it150`) were made after those tables existed, and nothing documents that test was not consulted.
- **The bias is probably small** (validation and test agree to about 1e-4). But the binary-model claims sit at the 1e-4 level, which is exactly where this bias lives.
- **Fix, minimum.** Reword both statements and drop `test_nll` from the selection tables and figures.
- **Fix, proper.**
  - Carve a fresh 2.5% test set from the current train entries.
  - Refit only the 8 selected configurations, warm-started, on the reduced train set.
  - Score once on the new test set.
  - Compute: about 3–4 h total (R6 about 35 min, R8 25 min, the others 5–20 min each).
  - This also gives one estimate of split-to-split variance, which no current number includes.

**M2. No uncertainty is reported for any real-data comparison, and some of the comparative wording is wrong.**
- **What is wrong.** The deck shows no CI anywhere on real data. With CIs (§0a):
  - "Five binary models sit within 0.0003" is numerically true but hides an ordering that is statistically resolvable: R7 < R4 < R3.
  - "Tie" holds only for R4, R5 and R8.
- **The AUC differences have no CI either.** The deck reports AUCs of 0.9652–0.9657 without one.
- **Fix.**
  - Add the two-way bootstrap CI to `fig_test_all` and the scorecard, using `boot.py` (5 min).
  - Report effect sizes relative to baseline.
  - Say explicitly that the R6 gain is carried by about 1% of genes, and name them.

**M3. Several selected settings sit at a grid edge that the deck does not flag.**
- **R4 logit + offsets, k = 80.** Validation NLL keeps falling as λ grows: 0.17974 → 0.17843 → 0.17774 at λ = 3, 10, 30 (`rescored/selection.csv`). λ = 100 was never run, and k = 80 at λ = 30 is only 1.0e-4 behind the winner. Slide 34's "low-rank models settle at rank 40" is unsupported for R4.
- **R6 NB counts.** k = 160 was run at λ = 10 only. The best λ grows with k (k = 40: best λ 3 or 10; k = 120: λ 10 and 30 within 9e-5). The k = 160 point is anomalous: validation 0.17832, worse than k = 120 by 1.9e-3, and test 0.17788 < validation, unlike every other row. That suggests a fitting problem, not saturation. So "extended until it stopped improving" is unsupported.
- **R8 neural manifold.** It sits at a grid corner: d = 20 = max and λ_V = 1 = min. `v3_hyperparams` and slide 34 do not flag this; only Poisson and R9 are marked "edge".
- **Poisson.** k = 80 = max. This is flagged, but Poisson was then used as a comparator in C1.
- **The tuning table misdescribes the grids.**
  - R3: λ ∈ {1, 3} was run only for k = 20 and 40.
  - R5: "k ∈ {0, …} × λ ∈ {1 … 100}", but k = 0 was run only at λ = 1, and k > 0 only at λ ≥ 3.
  - R6: k = 120 at λ ∈ {10, 30} only; k = 160 at λ = 10 only.
- **Fix, runs.**
  - R4 at k = 80, λ ∈ {100, 300}: about 30 min, warm-started.
  - NB at k = 160, λ ∈ {30, 100}: about 2 × 50 min.
  - Poisson at k = 120: about 15 min.
- **Fix, text.** Mark every edge in the tuning table, and show the grids as they were actually run.

**M4. Compute and convergence budgets differ, so several conclusions are budget artifacts.**
- **R8.** Every run stopped at the 1,500 s time limit (`termination=time_limit` in all 9 rows). The nonlinear head got 34 outer iterations against 87 for the affine head; width 128 got 38; direct BCE got 45. So the claims "affine beats nonlinear" and "PG-GEM beats direct BCE at equal wall time" compare unequally converged fits.
- **R7.** Nearly every K > 1 fit is non-converged at 30 iterations, or 15 when warm-started. The K = 4, d = 40 run, at 150 iterations, is still not converged. So "extra mixture components do not help" is untested.
- **R7's d.** "d = 40 is an interior optimum" rests on 0.17746 vs 0.17748 (d = 80), a difference of 2e-5. That is below one paired SE (§0a: about 1.3–2.3e-5 for close models).
- **R6.** NB got about 8 h of grid time, against 9 × 25 min for R8.
- **Fix.**
  - Run to a fixed relative-objective tolerance, or give equal wall time per configuration across families.
  - Rerun R8 (affine and nonlinear, d ∈ {20, 40}, λ_V ∈ {0.3, 1}) at 2–3 h each, about 10 h in total.
  - Run R7 with K ∈ {2, 4} at d = 40 to convergence, about 3 h.

**M5. "R6 wins Leiden ARI (0.67 vs 0.65)" is not robust (slides 48–49, recommendation).**
- **The resolution was label-tuned and the winner sits at the grid edge.** The best resolution is 0.25, the lowest one tried, for R1, R6 and R9 (`full/umap/leiden_by_resolution.csv`).
- **One seed was used, and the gap is within seed noise.** Seed SD is 0.007–0.014 for R1/R6 and up to 0.04 for R8. The gap at res 0.25 is 0.018, about 1.5 seed-SD, and the ranking reverses at res 0.10 (§0b).
- **Choosing the best of 6 resolutions against the labels inflates every ARI,** and inflates it by different amounts, because the cluster counts at a given resolution differ by model (R3: 15 clusters at 0.25; R1: 41).
- **Fix.**
  - Report ARI at a matched number of clusters (bisect resolution to 23, and also report 30 and 40), mean ± SD over ≥10 seeds.
  - Keep NMI, where R1 0.830 vs R6 0.828 already shows a tie.
  - Compute: about 15 min.

**M6. The depth-standardized probability is not depth-free, and "cleanest" is judged by a metric with no reference value (`v3_recommendation`, REPORT 3b).**
- **Depth is still linearly recoverable from p̃** (`v3/cache/std_embed.json`). R² of log UMI from the 10-PC embedding of p̃:
  - R6: 0.41 (0.38 within hemilineage)
  - R5: 0.66
  - R4: 0.87
  - So "depth-free probability" is wrong. At most it is the "least depth-dependent" one.
- **The "median within-HL |ρ|" (0.11/0.19/0.29) has no reference.** It is computed on smooth p, where any shared cell factor gives large |ρ|. The data's own value is 0.045, because binary noise attenuates it, so it is not comparable either. There is no "correct" value: depth genuinely co-varies with biology (the deepest decile is 84% primary neurons; mitochondrial-ribosomal genes are among the depth genes).
- **Low ρ could mean over-correction.** Driving ρ to 0 could remove real biology. Nothing in the study separates the two.
- **The median reference is a choice.** `ref = median` of c_i is arbitrary. Results should be shown to be invariant to the reference, e.g. 25th and 75th percentiles. Rank-based ρ is invariant, but the "gap" and heat-map statements are not.
- **Fix.**
  - Validate p̃ where the truth is known. Synthetic G3 has cell types plus depth: compute KL(P_bio ‖ p̃) for R4–R6 against the depth-free truth, and compare with the ρ metric. Reuse the G3 generator and refit R4–R6 at the selected configurations: about 30–60 min.
  - On real data, report residual depth with hemilineage × birth-time strata held fixed.
  - Rename the claim.

**M7. The synthetic benchmark does not support "R7 recovers probabilities best", or generalization to real data.**
- **R7 was at a grid edge in 15/15 runs.** Its grid is K ∈ {1, 4} × d ∈ {5, 10}, only 4 configurations against 9 for the others (`v3/tables/syn_selection.csv`; G1/G2 picked d = max). Slide 29 names only R8 and R9 as edge-limited.
- **Other methods were also at edges:** Poisson 9/15, R3 6/15, R6 6/15 (k = min in G4/G5), R5 5/15.
- **The oracle was never computed.** `oracle_config_by_test_kl` is empty for every row, so selection regret was never measured, although the column exists.
- **The "built to favour" design failed in 3 of 5 worlds.** Only G4 (R8) and G5 (R9) are won by their own family.
  - In the "NB world" G2, R6 (KL 0.0259) loses to R4 (0.0246) and R7 (0.0215).
  - In the "low-rank logit world" G1, R7 (0.0229) beats R3 (0.0317) and R4 (0.0281).
  - Either the generators do not match the model families, or the inference differs.
- **There is a likely inference confound.** R7 predicts with a posterior predictive that integrates over q(z_i). R3–R6 and R8 are MAP plug-ins. With m = 300 genes per cell, per-cell posterior uncertainty is large, so averaging wins KL. At the real m = 19,543 that advantage nearly vanishes: real-data R7 − R4 is only −2e-4.
- **The cell-type results do not generalize either.**
  - Slide 32 and the recommendation say R6 "recovers cell types well". That holds only in G3. In G5, R6 has the second-worst kNN (0.855 vs 0.98–0.99 for R7 and R9) and a low Leiden ARI (0.81).
  - Three seeds give unstable cell-type results: R5 G3 kNN 0.67 ± 0.17, R9 G3 0.56 ± 0.14, R8 G3 ARI 0.37 ± 0.27.
- **Fix.**
  - Run a well-specified check per family (simulate from that family's own fitted real-data model at reduced m ≥ 2,000).
  - Isolate the inference effect: add a MAP plug-in R7, or a Laplace posterior-predictive R3.
  - Widen the grids until no pick is at an edge.
  - Compute the oracle configuration and the regret.
  - Use ≥10 seeds.
  - Compute: about 3–4 h without R8; R8 adds about 33 min per fit (`seconds_mean` 1,982 s), so restrict R8 to its selected configuration per world.

**M8. The sex and library interpretation is wrong or confounded (slides 41–42, next steps).**
- **Slide 42 is wrong for R1 and R8.** "Female and male cells separate within each strand in every method (kNN about 0.82)" fails for them: within mixed-sex libraries their sex decoding is 0.585 and 0.575, near chance (§0d). Their overall 0.82 comes mostly from single-sex libraries. R3–R7 decode sex within libraries (0.73–0.81), plausibly from a few sex genes outside R1's 2,000 HVGs.
- **The "next step" can be answered today.** 15 of 21 libraries are mixed-sex, so per-cell sex was called from expression, and the requested check is cheap.
- **Library kNN is not a clean batch metric.** Library composition differs by hemilineage (χ² = 618, p = 4e-8). Above-chance library kNN is therefore partly biology.
- **Fix.**
  - Report library mixing within hemilineage (kBET or iLISI per hemilineage).
  - Report sex decoding within library, with and without roX1/roX2 and Y-linked genes.
  - Compute: about 15 min.

## 3. Minor

1. **Numerical claims contradicted by the tables:**
   - Slide 37, "NB best in every gene-frequency decile": false in decile 1. R7 0.00184 < R9 0.00185 < R8 0.00187 < R6 0.00195 (`v3/tables/test_nll_by_gene_rate_decile.csv`).
   - Slide 48, "low-rank detection models trail by 0.04–0.10 on hemilineage metrics": the purity gaps to R1 are 0.09–0.20 (R3 0.776, R7 0.772 vs 0.974), and the ARI gaps to R6 are 0.13–0.18.
   - Slide 47, "R6 is cleanest (0.24 on its first component)": on PC1, R8 (0.054) and R9 (0.13) are lower (`umap_drivers.csv`). R6 is cleanest only on the top-5 measure.
   - REPORT 3b, "Marker genes keep 84–90% of their between-hemilineage variance": 0.84–0.90 is η² (between/total), not retention. Retention p̃ vs p̂ is 98–102% (`depth_std_metrics.csv`, `median_eta2_hl_20markers`).
2. **"Half the depth axis" for R5 depends on the metric.** It holds for top-2 |ρ| (0.41 vs 0.80). On all components (0.59 vs 0.80) or 10-PC R² (0.70 vs 0.90) it is about a quarter less. Use one rotation-invariant measure for all models: R² of log UMI on the whitened top-d embedding.
3. **The depth-gene results are partly circular.** The 8 "depth-sensitive genes" were selected on the same data for the largest depth effect (`heat_select.py`). The gaps "0.43 → 0.26 / 0.17" are inflated by selection, with regression to the mean. Select on half the cells and evaluate on the other half (minutes).
4. **The kNN-error and depth claim is confounded.** "Errors concentrate at depth extremes (28% / 26% vs 3–4%)" ignores that the extremes are enriched for primary neurons and possibly for small hemilineages. Stratify by birth time and hemilineage size.
5. **Regressing out depth also removes biology.** "Depth regressed" metrics take out the primary-neuron signal along with depth (acknowledged in next steps). Regress within birth-time class, or residualize against depth orthogonal to hemilineage × birth time.
6. **The fits used are inconsistent across models.** R3–R5 embeddings come from refits on all entries (`best_full*.npz`, including test). R6–R9 embeddings and all p̂/p̃ come from train-only fits. Use one convention throughout.
7. **NLL is fragile for Poisson.** Its validation–test gap of 3.3e-3 comes from 136 entries with NLL > 20. Report a clipped NLL (p clipped to [1e-6, 1 − 1e-6]) as a sensitivity check.
8. **NB shows a mild winner's curse.** Its validation–test gap is 1.4e-4, against about 5e-5 for the others, consistent with selection from the largest grid. It is small relative to R6 − R4.
9. **One split, one UMAP seed, one kNN k (15).** Give sensitivity at k ∈ {5, 30}, with UMAP labelled as illustration only (already partly stated).
10. **The R9 comparison is not like-for-like.** R9's representation (R·B, n × 200, nearly one-hot) is not comparable as a metric space. Exclude it from kNN/ARI rankings, or compare on hard labels only.

## 4. Minimum before the deck is shown externally (about 1 working day of compute)

1. Fix C1: delete the rank sentence and add the φ-by-detection-rate plot. Minutes.
2. Fix C2: one embedding convention, recompute Part 4 and the scorecard. About 20 min.
3. Add the two-way bootstrap CIs and correct the "tie" wording (M2). Done here; about 5 min to rerun.
4. Leiden at matched cluster number, ≥10 seeds (M5). About 15 min.
5. Fix the sex and library slides using the within-library analysis (M8). About 15 min.
6. Correct the four false numerical statements and the tuning table (Minor 1, M3).
7. Fresh test split with refits of the selected configurations (M1): about 4 h. Edge-extension runs (M3): about 2.5 h. Synthetic p̃ validation (M6): about 1 h.
8. R8 and R7 budget reruns and the synthetic redesign (M4, M7): 1–2 days. Until they are done, state the R7/R8 conclusions as provisional.
