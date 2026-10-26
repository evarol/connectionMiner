# Review B: code correctness audit (bernfac v3)

Scope: cm_minimal/experiments/bernfac (fitting, scoring, selection, representation metrics, standardization, synthetic benchmark) plus spot checks of the deck. I edited no existing files. Scratch scripts are in the session scratchpad (nllcheck.py, repcheck.py, leid.py, g2.py, r8pc.py). pytest is not installed in the venv, so I did not re-run the test suites.

## Prioritized findings

### MAJOR 1. "depth |ρ| (PC1–2)" is not PC1–2 for R8 and R9. Slides 46/47 rest on this artifact.
- File: umap_compare.py:227-229 (`spearmanr(E[:, 0])`, `E[:, 1]`). These are raw coordinates 0 and 1. That is only right for embeddings that are already SVD-ordered (R1–R7). R8's embedding is the raw latent Z (models/neural/best.json: "fitted Z"), whose coordinates are not variance-ordered (variances 0.385, 1.35, ...; the top-variance coordinates are 7 and 1). R9's embedding is R·B with 200 unordered columns. v3_umap_drivers.py:58-62 ranks coordinates by per-coordinate variance. That is still not PCA for a non-orthogonal Z.
- What I ran and saw (centred PCA of the cached embeddings vs log10 UMI):
  - R8: PC1..5 |ρ| = 0.078, 0.150, 0.073, 0.427, 0.399. The scorecard says 0.909 (from raw coordinate 0, a low-variance remnant of the SVD init). The drivers table says pc1 0.054 and top5 0.909.
  - R9: PC1–2 = 0.956. The scorecard says 0.766.
  - R6: PC1..5 = 0.238, 0.301, 0.108, 0.517, 0.089. R6 is already SVD-ordered, so its numbers are correct.
- Consequences:
  - Slide 46 says "R6 is cleanest (0.24 on its first component). R8 keeps depth off its first axis but not out of the top five." Both clauses are wrong: R8 PC1 = 0.08 and R8 top-5 max = 0.43, against 0.24 and 0.52 for R6.
  - Slide 47 says R6 "has the weakest depth axis". Under PCA, R8 is weaker (PC1–2 0.15 vs 0.30).
  - Scorecard column `depth |ρ| (PC1–2)` for R8 (0.909) and R9 (0.766) is wrong.
- Fix: in metrics_for and v3_umap_drivers, centre E and project onto its principal components (SVD) before taking the depth correlations, for every representation. Then regenerate scorecard.csv, umap_drivers_full.csv, metrics_depth_regressed.csv and slides 39/46/47.

### MAJOR 2. Synthetic R8 fits are truncated by compute, not converged, and truncation varies by seed and configuration.
- File: v3/syn_methods.py:168-199 (`max_outer=300`, `max_seconds=240`).
- Evidence: in synthetic/results.csv, 135 of 135 R8 grid fits have converged=False. Iteration counts per configuration vary widely: G3 seeds 0–1 run 300 iterations everywhere (max_outer), while G3 seed 2 and G4 seeds 0–2 give d=5/10 only 6–15 outer iterations (wall 420–1100 s, past the 240 s budget). Selection is by val NLL, so selection picks whichever d happened to get iterations (for example d=2 in G4, d=5 with 14 iterations in G3 seed 2).
- Consequences: the R8 synthetic KL, calibration and embedding metrics (slide 31 "neural manifold included"; G3 Leiden 0.37) are confounded by under-fitting. The real-data R8 fits are also all `time_limit` (neural/tables/selection.csv), and best d=20 sits at the edge of the grid. The impact there is smaller, because R8 is within 0.0003 of the other binary models.
- Fix: give R8 a fixed iteration budget (or tol) instead of a wall-clock limit. Run it without pool contention, or at least make the budget equal across configurations and seeds. Report convergence on the slide.

### MAJOR 3. The synthetic NB world (G2) result "R7 beats NB counts in their own world" depends on the rank grid.
- File: v3/syn_methods.py:66 (`BF_GRID k ∈ {5,10,20}`). The true rank in G1/G2 is 8.
- Evidence (G2 seed 0, countfac.fit NB on the same split, my run):

  | config | KL | val NLL |
  |---|---|---|
  | k=8, λ=10 | 0.0215 | 0.3674 |
  | k=8, λ=3 | 0.0222 | 0.3677 |
  | k=10, λ=10 (the benchmark's selected config) | 0.0272 | 0.3727 |

  The benchmark's G2 R6 KL is 0.0270/0.0240/0.0267 over seeds, against R7 0.0214–0.0219. With k=8, NB matches R7 (0.0215 vs 0.0214), and validation would have chosen it.
- Consequences: slide 29 ("KL 0.022 vs 0.026 for NB counts") reflects a grid-granularity artifact of the MAP count model: it overfits at k=10, whereas R7 integrates over z. It is not a property of the model class.
- Fix: use a finer rank grid (for example 4, 6, 8, 10, 12, 16, 20) or a larger λ range for the count models. At minimum, note the sensitivity on the slide.

### MAJOR 4. Synthetic R1/R2 embeddings use 100% of entries; every model uses only the 70% train entries.
- File: v3/syn_methods.py:243-258 (`embed_R1` and `embed_R2` use D["X"] and D["Y"] unmasked). Compare syn_methods.py:71 and the other model runners, which fit on `split==0` only.
- Consequences: slide 31, "processed counts the best clustering (Leiden ARI 0.97)", compares R1, which sees all the data, with models that see 70%. The real-data analogue has the same bias at 100% vs 95%, which is minor there.
- Fix: either refit every model's embedding on all entries after selection, or state the asymmetry on the slide.

### MINOR 5. R3/R4/R5 embeddings come from the "final" refits on ALL entries (incl. val/test); their NLL rows come from train-only fits.
- File: umap_compare.py:46-48 and 340-345 (`best_full*.npz`, run_full.final* with heldout=None). R6–R9 embeddings come from train-only fits.
- Evidence (my run): recomputing R3/R4/R5 embeddings from cv_k40_lam10 / cv_off_k40_lam10 / cv_cll_k40_lam30 gives:

  | rep | HL kNN, train-only fit | HL kNN, cached full refit | lib kNN, train-only | depth top-2, train-only |
  |---|---|---|---|---|
  | R3 | 0.8698 | 0.8732 | 0.270 | 0.976 |
  | R4 | 0.9258 | 0.9231 | 0.273 | 0.796 |
  | R5 | 0.9286 | 0.9318 | 0.226 | 0.417 |

  The R3/R4/R5 ordering and all conclusions are unchanged. No held-out label leakage: the embeddings are unsupervised.
- Fix: build R3–R5 embeddings from the same train-only fits that were scored, for consistency with R6–R9, or state the difference.

### MINOR 6. Leiden "best ARI" is chosen by maximum ARI against the evaluation labels, and R1/R6 peak at the grid edge.
- File: umap_compare.py:160-213. The resolution grid is 0.25–3.0, and best = max ARI vs hemilineage, so selection is done against the truth. This applies equally to every representation, but R1 and R6 are both maximised at res 0.25 (the lower edge).
- Evidence (my run, lower resolutions):
  - R1: 0.570 / 0.654 / 0.653 / 0.626 / 0.649 at 0.05 / 0.1 / 0.15 / 0.2 / 0.25
  - R6: max 0.674 at 0.25
  - R8: 0.565 at 0.15 (vs 0.549 reported)
  - The slide 47 claim "R6 wins Leiden ARI (0.67 vs 0.65)" survives (0.674 vs 0.654), with a margin of 0.02 from a single Leiden seed.
- Fix: extend the grid below 0.25, average over several seeds, and call it oracle-resolution ARI in the caption.

### MINOR 7. The Poisson test NLL is driven by a few extreme entries, which explains its val/test gap.
- countfac.heldout_scores is exact: log p0 = −mu, with eta clipped to 15, so mu can reach 3.3e6.
- Evidence: in v3/tables/calibration_summary.csv, PO has max entry NLL 25,257, 136 entries above 20, and 2.1% of its test NLL from those entries. Selected PO: val 0.1806 vs test 0.1839, while every other model has a val/test gap of 0.0001 or less.
- Consequences: the Poisson ranking stands (about 0.1800 without the tail, still last among the count/binary factor models), but its selection over λ and k was tail-driven.
- Fix: report a trimmed or median NLL alongside the mean, or note it.

### MINOR 8. Calibration "over-confident at the top" (slide 30) is partly an artifact of binning on p̂.
- Even the correctly specified NB model (G2, k=8) puts p̂ ≥ 0.9 at mean 0.969 where P_true = 0.888 (my run). This is regression to the mean when bins are formed on a noisy p̂. Models that rarely predict above 0.9 (R4: about 900 entries in G1/G2) avoid the effect.
- Fix: also bin on P_true (or use quantile bins) before calling a model over-confident.

### MINOR 9. Search budgets are asymmetric across families.
- NB was extended to k=120/160 (run_counts.extend); the other factor families stop at k=80 with interior optima.
- MFA d=40/80 was added on 22 Oct after the d=20 edge (run_full.py:213-219, `in_grid` redefined).
- LBM is selected at the 200×200 grid edge.
- Neural d=20 is at the grid edge and time-limited.
- All of these are selected on val, and none is test-driven. The margins between the five binary models (0.0003) are within what such budget differences could move.

### MINOR 10. R3's "PC" embedding is an uncentred SVD.
- File: umap_compare.py:125-136. Column 0 has mean 693, and the leading singular value of 89,024 is the intercept/depth direction. kNN and Spearman are shift-invariant, so the metrics are unaffected; only the "PC1–2" label is loose.

## Checked and correct
- Split: val∩test = ∅ and val∪test = heldout (verified on 600 rows; asserted in lbm/full_common.py:23). All eight probability models and Poisson exclude heldout in every likelihood, init and SVD step: bernfac_large `_block_terms`/`svd_init`/`additive_init`/`cloglog_init`; countfac CountData (X·T, depth = train UMI); mfa `excluded=H`, `training_pcs`; neural `MaskedBinary(~H)`, `svd_init`; lbm train arrays, `spectral_svd`. The neural val pass is diagnostic only (fit.py:323, never used for stopping).
- R6 depth offset o_i = log(sum of TRAIN counts) (countfac.py:126-137). No leak. The `depth_total` variant (leaky, documented) is not used in comparison.csv.
- Scoring: every model uses the same test mask (n = 8,044,655). Logit uses exact logaddexp; cloglog is clipped to [−30, 4.4] in both fit and score; NB/PO have eta clipped to [−40, 15], exact log p0, and log p1 floored at 1e-300. My independent minimal recomputation on 293,547 test entries (600 random rows) matched predictors.logpairs to ≤ 3e-17 for R3, R4, R5, R6, PO and R9. Subset means: R3 0.17601, R4 0.17584, R5 0.17594, R6 0.17479, PO 0.17821, R9 0.18144. loader_validation.csv recomputation of the full test set matches the reported values to within 1e-9 (R7 1e-6, from the re-inferred local state).
- Selection: every family's comparison.csv row is the validation-NLL argmin: rescored/selection.csv, counts/tables/selection.csv, mfa (chosen restart by training bound), neural (restart by F), lbm (restart by ELBO). Where test would have chosen differently (cloglog λ=10 over 30; Poisson k=80 λ=100), val was used. No family was selected on test.
- kNN balanced-accuracy CV: StratifiedKFold(5, shuffle, seed 0) with cross_val_predict. Correct, no leakage.
- Depth standardization (predictors.standardized): replaces c_i (a_i for R4/R5; o_i + a_i for R6, already summed in Counts.a) with median(c), keeping b_j and u_iᵀv_j. check_standardized (a), (b) and (c) pass (std_offsets.csv). Residual depth in p̃ is depth absorbed into UVᵀ (gauge between a_i and a near-constant V direction). The slides state this; it is not a bug.
- Synthetic generators: NB draw (Gamma(φ, μ/φ) then Poisson) and P_true = 1 − (1 + μ/φ)^(−φ) are correct. X = Y(1 + Poisson(1)) gives Pr(X ≥ 1) = P exactly. The density calibration is correct. The split is shared across methods and seeded independently. KL(P_true‖P̂) uses xlogy with q clipped to 1e-12, which is correct. The oracle row is trivially 0.
- Slide spot checks against tables:
  - Slide 34 (0.1766; 0.1775–0.1778 within 0.0003; 0.1830; 0.1839; 0.1976): correct.
  - Slide 47 (Leiden 0.67 vs 0.65): correct as computed (see 6).
  - Slide 48 (0.49 → 0.61; R6 0.69): correct.
  - Slide 40 (library 0.10–0.14; 0.21–0.27): correct.
  - Slide 57 (0.29 / 0.19 / 0.11; markers 84–90%): correct.
  - Slide 29 (0.022 vs 0.026): matches the table (but see 3).
  - Slide 46: wrong (see 1).
  - Slide 47 "R6 weakest depth axis": wrong (see 1).
