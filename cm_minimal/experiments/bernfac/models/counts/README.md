# Count factorization (Poisson and negative binomial)

Worker A of the model comparison in `outputs/bernfac_20261020/models/PLAN.md`.
These models factorize the UMI counts directly instead of the binary detection matrix Y.
Their predictions are turned into detection probabilities so they can be scored like every other model.

## Model

For cell i and gene j:

    log mu_ij = o_i + a_i + b_j + u_i'v_j,     o_i = log s_i

    Poisson:  x_ij ~ Poisson(mu_ij)
    NB2:      x_ij ~ NB(mean mu_ij, variance mu_ij + mu_ij^2 / phi_j)

    u_i, v_j ~ N(0, I/lambda),   a_i, b_j ~ N(0, 1/lambda0),   lambda0 = 1e-3

- The depth offset s_i is the cell's total count over its train entries. This is the primary, leak-free choice.
- The usual library size, the total over all genes, also counts the held-out entries, which is a small leak.
  - It is available as `depth="total"` (`run_counts.py depth_total`).
  - Because a_i is nearly unpenalized, the offset mostly sets the starting point and the scale of a_i, so the choice barely matters.
- Only train entries (`~heldout`) enter the likelihood. Validation and test are the two disjoint halves of `heldout`.

## Fitting

The fit is MAP by alternating block Fisher scoring, the same scheme as `bernfac_large.fit_cloglog`.

1. Row sweep: update `[u_i; a_i]` with `(V, b)` fixed.
2. Column sweep: update `[v_j; b_j]` with `(U, a)` fixed.

Write eta = log mu and L = log(1 + mu/phi). The score and the Fisher weight with respect to eta are:

    Poisson:  g = x - mu,                w = mu
    NB2:      g = (x - mu)/(1 + mu/phi),  w = mu/(1 + mu/phi)

- Both log-likelihoods are concave in eta, so every row problem is concave.
- A per-row backtracking line search accepts a step only if that row's penalized objective does not drop. This makes the objective J monotone, and every fit asserts it.
- For NB, phi_j is updated after each factor sweep, with mu held fixed.
  - Each update takes two safeguarded Newton steps on log phi_j, using digamma and trigamma terms over the train nonzeros. The nonzeros are stored as unique (gene, count) pairs.
  - Step halving is used, and a new phi_j is kept only if it raises gene j's log-likelihood.
  - phi_j is limited to [1e-3, 1e6].
- Precision: the weighted Gram products run in float32. Every log-likelihood value is computed in float64, so the monotonicity checks are not swamped by float32 rounding.

## Predictions and scores

    P(x_ij >= 1) = 1 - exp(-mu_ij)                    (Poisson)
                 = 1 - (1 + mu_ij/phi_j)^(-phi_j)     (NB)

- Scores on validation and test: Bernoulli NLL, Brier, AUC on a 2 M subsample (seed 0), and 10-bin calibration. They are computed the same way as `bernfac_large.heldout_scores_cloglog`.
- Secondary score: the held-out count log-likelihood per entry.

## Selection

- Grid: k ∈ {20, 40, 80} × λ ∈ {100, 30, 10, 3} for each family.
- Each k runs a λ path from strong to weak shrinkage, and each fit is warm-started from the previous one.
- Convergence: relative change in J below 1e-5, with at most 60 iterations for Poisson and 50 for NB. The budget was cut from 1e-6 and 250 iterations after timing showed that the full grid would otherwise take more than 8 h. Warm-started λ steps stop after 4–15 iterations.
- The selected configuration has the lowest validation NLL. It is not refit, so test stays clean.

## Files

| file | content |
|---|---|
| `countfac.py` | model, fitting, dispersion update, scoring, SVD embedding |
| `run_counts.py` | `bench`, `cv`, `queue` (resumable grid), `depth_total`, `cloglog_ref`, `collect` |
| `make_figures.py` | selection curves, calibration, and dispersion figures |
| `tests/test_counts.py` | `python tests/test_counts.py` |

Tests:
- Finite-difference checks of the score, Fisher weight and dispersion derivatives.
- Monotone objective.
- Recovery of known factors, depth and dispersions (n = 500, m = 300).
- Changing held-out counts leaves the fit bit-identical.
- The NB objective approaches the Poisson objective as 1/phi.

Outputs are in `outputs/bernfac_20261020/models/counts/`:
- `tables/selection.csv`, `tables/selection_full.csv`, `tables/calibration.csv` and `tables/cloglog_reference.csv`.
- `best_<family>.json`.
- `fits/best_<family>.npz` with U, V, a, b, o and phi.
- `embedding_<family>.npy`, where E = A S from the thin SVD of U V'. It is computed through QR(U) and QR(V), so the n × m matrix is never formed.
- Figures in `figures/`.

Run with `OMP_NUM_THREADS ≤ 3`.

## Results (21–22 Oct 2026)

Held-out Bernoulli scores on val and test (each about 8.0 M entries):

| model | k | λ | val NLL | test NLL | val AUC | test AUC | test Brier |
|---|---|---|---|---|---|---|---|
| NB (selected) | 80 | 10 | 0.17656 | 0.17665 | 0.96559 | 0.96565 | 0.05517 |
| Poisson (selected) | 80 | 10 | 0.18063 | 0.18390 | 0.96496 | 0.96502 | 0.05575 |
| cloglog BernFac (CSO rescoring, train-mask fit) | 40 | 30 | 0.17769 | 0.17774 | 0.96515 | 0.96520 | 0.05547 |
| logit + offsets (CSO rescoring) | 40 | 10 | 0.17764 | 0.17770 | 0.96523 | 0.96525 | 0.05550 |

What the results show:
- NB is the best detection model so far. It beats cloglog BernFac by 0.0011 nats per entry on test.
  - NB improves with k at every λ, so k = 80 sits at the edge of the grid and larger k may help.
- Poisson is clearly worse.
  - Its test NLL is also unstable. A single test entry contributes NLL ≈ 25,000: a gene predicted at μ ≈ 2.5e4 where the count is 0.
  - Without the 100 worst entries, the Poisson val and test NLL are both 0.1800.
  - Poisson's thin tail makes confident wrong zeros very costly. NB's worst entry is 118.
- Held-out count log-likelihood per entry, on test: NB −0.4956, Poisson −0.5150.
- Dispersion: 34% of genes reach the cap of φ = 10^6, which means they are Poisson-like. These are mostly rare genes.
  - In the remaining genes, φ increases with abundance. Rare genes are very bursty, with φ < 1.
- Depth offset: refitting the selected NB with the total-UMI offset changes val NLL by 2e-6 (`tables/depth_total.csv`). As expected, a_i absorbs the difference.
- Every fit was monotone and met its tolerance.
- Compute: 1.7 h for Poisson and 3.4 h for NB.
  - Poisson ran on 1 thread and NB on 2, at the same time. The second NB pass (λ = 3 and k = 80) used 3 threads.
