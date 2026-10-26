# v4 synthetic benchmark: how the counts were made, and what changed from v3

## How counts are built in each world

The benchmark scores every method on the same binary data Y = 1[X ≥ 1]. The count models (R6 NB, Poisson) and R1 (processed counts) also see the counts X. The counts are built two ways.

- **G2 and G3: genuine counts.** X ~ NB(μ_ij, φ_j) is drawn from a mean that carries the low-rank (G2) or cell-type (G3) structure and a cell depth. Then P_true = Pr(X ≥ 1) = 1 − (1 + μ/φ)^(−φ). The positive counts carry the same information as the zeros, which is what a count model assumes.
- **G1, G4 and G5: counts attached to a binary world.** P_true is defined first: a logit (G1, G4) or a block probability (G5). Then Y ~ Bernoulli(P_true), and X = Y·(1 + Poisson(1)).
  - Pr(X ≥ 1) = P_true exactly, so the binary data are correct.
  - The positive part is pure noise. Given Y = 1, X − 1 ~ Poisson(1) for every entry, whatever P_true is, and it is independent of the cell and the gene.

### What this means for count models in G1, G4 and G5

- R6 and Poisson model E[X] = μ_ij and read their detection probability off μ, through Pr(X ≥ 1) = 1 − e^(−μ) (Poisson) or the NB analogue. Here μ_ij ≈ P_ij · 2, so the model is fitted to a mean that is not linked to P the way its own likelihood assumes.
- The positive counts add noise rather than signal. A count model therefore pays an information cost and a link-misspecification cost that the binary models do not.
- So in v3, "R6/Poisson trail in G1, G4 and G5" partly measures this construction and not only the model class. These three worlds can test the binary models fairly. They cannot fairly test the count models.

### Control run in v4: G1 with genuine counts at the same P_true

Two variants keep G1's P_true, Y and split exactly and change only the positive counts:

- **G1zP:** given Y = 1, X is drawn from a zero-truncated Poisson(μ), with μ = −log(1 − P).
- **G1zNB:** given Y = 1, X is drawn from a zero-truncated NB(μ, φ_j), with μ = φ((1 − P)^(−1/φ) − 1) and φ_j ~ logN(log 2, 0.5²) clipped to [0.5, 50]. This is milder than G2's φ range, because with φ = 0.05 and P near 1 the implied mean blows up.

In both variants Pr(X ≥ 1) = P_true, and the counts now follow a count law with that zero probability. The binary models see identical data, so their G1 results carry over unchanged. Only R6 and Poisson were refitted. Results: `v4/tables/syn_v4_count_variants.csv` and the right-hand panel of `fig_syn_kl_v4`.

## Protocol changes in v4 (same 15 datasets and the same 70/15/15 split as v3)

| item | v3 | v4 |
|---|---|---|
| R3, R4, R5, R6, Poisson grid | k ∈ {5, 10, 20} × λ ∈ {1, 10, 30} | k ∈ {4, 8, 16} × λ ∈ {3, 10, 30, 100}; true rank in G1/G2 is 8; iteration caps 500 (BernFac) / 400 (counts) |
| R7 grid | K ∈ {1, 4} × d ∈ {5, 10}, 150 iterations | K ∈ {1, 2, 4} × d ∈ {4, 8, 16}, 300 iterations |
| R8 | d ∈ {2, 5, 10} × λ_V ∈ {1, 10, 100}, 300 outer iterations **or** 240 s wall clock, whichever came first; truncation varied by seed and config | d ∈ {2, 5, 10} × λ_V ∈ {10, 100, 1000}, **fixed 250 outer iterations** (tol 1e-6) for every config, seed and world; no wall-clock cap (a 2 h safety stop never fired unless stated in `syn_v4_r8_budget.csv`) |
| R9 grid | K ∈ {5, 10, 20, 40} | K ∈ {5, 10, 20, 40, 80} |
| test scoring | selected config only | every config: val NLL, test NLL, test KL, RMSE, AUC; oracle config = lowest test KL (reported, never used for selection) |
| R1 / R2 embedding | all entries | both: all entries (as v3) and train-only (below) |
| depth standardization | not validated | ground-truth check in G1–G4 (below) |

### R1/R2 on train entries only

- **R1 train-only:** every held-out entry (split ≠ 0, 30% of entries) is set to 0 before `normalize_total`. Library sizes, log1p, gene scaling and PCA-50 therefore use train entries only. The mask is uniform at random, so this shrinks every library by about 30% and adds random false zeros. No imputation is done.
- **R2 train-only:** Y is centred by each gene's train-entry detection rate. Held-out entries are set to 0 after centring, which is mean imputation, and then PCA-50 is taken.
- The factor models are fitted on train entries only, as in v3.

### Depth-standardization ground truth (task 5)

- G1 (a_i), G2 and G3 (log depth s_i) and G4 (d_i) each have an explicit cell offset in the generator. The true depth-standardized probability P_std is the generator's P with every cell's offset set to the median offset. The data were regenerated from the seed, and the regenerated P matches the stored P to 3e-8.
- For R4, R5, R6 and Poisson, p̃ follows `v3/predictors.standardized`: the fitted cell offset c_i (a_i, or o_i + a_i for the count models) is replaced by its median, keeping b_j and u_iᵀv_j.
- Reported over all n × m entries:
  - KL(P_std ‖ p̃)
  - KL(P_std ‖ p̂), which is what you get if you do not standardize
  - KL(P_true ‖ p̂)
  - the Spearman correlation between the true offset and each cell's mean logit (residual depth)

### Embedding metrics

Same definitions as v3: kNN balanced accuracy, k-means ARI, and the best Leiden ARI over 4 resolutions. They are computed on the selected and on the oracle config, with v3's embedding convention (SVD of UVᵀ). Review C2's whitening and matched-dimension protocol is applied to the real data by W3, not here.

## Run record (24 Oct 2026)

- All 263 tasks completed: 15 datasets × 8 methods plus baselines, all 135 R8 configs, and the G1zP/G1zNB variants. None failed and nothing was cut.
- R8: all 135 fits ran exactly 250 outer iterations. None reached the 1e-6 tolerance, and the safety stop never fired. Fits took about 230–330 s each.
- Selection regret is about 0. The validation-selected config equals the test-KL oracle in almost every method × world × seed, and the KL difference is ≤ 2e-4 where they differ.
- **Ridge collapse.** For R3/R4/R6 at λ ≥ 30, k = 8 and k = 16 give identical fits. The ridge zeroes the surplus factors, so at large λ the effective rank is set by λ, not by k.
- **Results with the G1 count controls.**
  - R6 KL falls from 0.0356 (as built) to 0.0209 (Poisson counts) and 0.0214 (NB counts).
  - Poisson KL falls from 0.0513 to 0.0209 and 0.0333.
  - For comparison, R4 is 0.0247 and R7 is 0.0232 on the same binary data.
