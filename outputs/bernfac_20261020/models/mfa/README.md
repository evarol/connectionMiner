# Worker B: Bernoulli mixture of local factor models (MFA on logits)

Code: `cm_minimal/experiments/bernfac/models/mfa/` (`mfa.py`, `run_full.py`, `synthetic_demo.py`, `make_figures.py`, `tests/test_mfa.py`). Python only.

## What was implemented
This is handoff section 3, implemented as structured variational EM:
- Each row/component pair has its own Gaussian q(z, psi) and its own PG parameters xi.
- xi is updated as sqrt(E[psi^2]), using the mean and the variance.
- The integrated bound includes logdet A, log a and c(xi).
- Responsibilities are a softmax.
- The M-step solves (G + t Lambda) theta = g, including the S b_old / a term. Residual mode `none` has its own M-step.
- pi gets a Dirichlet update. t can optionally be learned under an inverse-gamma prior.
- Held-out cells are scored with the missing-cell predictive N(mu + b'm, t + b'Sb), averaged over components with 20-node Gauss–Hermite quadrature.

The monitored objective is a **variational lower bound**, not the marginal likelihood.

### Departures from the textbook schedule, all checked for monotonicity
1. **One data pass per outer iteration.**
   - Isotropic mode: the xi update and the M-step are taken from the same frozen q. Given q, the bound separates into a xi part and a Theta part, so this is a valid coordinate step.
   - `none` mode: the M-step keeps xi frozen. The next xi is then computed exactly from the stored low-dimensional moments (m, S) and the new Theta, so no n x m xi array is stored in this mode.
2. **xi storage on the full data.** xi is a K x n x m **float16** array held in RAM: 10.3 GB at K = 16. The bound is always evaluated exactly at the stored xi, so rounding only makes the xi step slightly less than optimal. The O(cm) products run in float32. The d x d algebra and all sums run in float64.
3. **Optional parameter-expansion (PX) step after the M-step**, used on the full data and not part of the handoff baseline. It maps the coordinate prior back to N(0, I). The logit distributions do not change (tested). The step is accepted only when the gain it computes exactly is positive. On a subset it roughly doubled early progress.
4. **k-means initialisation on the full data.** The clusters come from 30 training-only PCs. Each cluster's mean is its smoothed column logit. Each cluster's loadings come from an SVD of that cluster's logit proxy. This affects initialisation only.

## Tests
`tests_output.txt` shows 21/21 passing. The suite covers:
- handoff 7.1: PG mean, quadratic bound, stable likelihood, masks, dense/sparse parity, invalid inputs, missing rows/columns, probability averaging, serialization;
- handoff 7.2 A–H: dense Gaussian reference, bound ≤ 1-D evidence, moment-sensitive xi, M-step gradient zero plus the inverse-gamma check, monotonicity across seeds, masks, both residual modes, learned t and 1/3 local sweeps, empty components and missing-cell variance, quadrature at 20 vs 40 nodes vs Monte Carlo, gauge invariance;
- extra checks: chunked pass vs the per-row reference, the float32 kernel, the PX step, new-row inference.

At s² = 25, 20 and 40 nodes differ by 3.5e-3. The test reports this as a note; it is not hidden.

## Synthetic demo (handoff 8.1)
Results are in `synthetic/` and shown in `figures/synthetic_demo`. Setup: n = 250, m = 100, true K = 3, d = 2, t = 0.25, 3 data seeds.
- Balanced case, test NLL: K = 1 gives about 0.604 under every residual mode. K = 3 gives 0.527 (fixed t), 0.531 (learned t) and 0.530 (none). The true-probability NLL is 0.453.
- Component recovery with fixed t and K = 3: ARI 0.96.
- Sparse case: K = 3 gives 0.383, K = 1 gives 0.402, and ARI is about 0.45.
- Learned t shrinks to about 0.05, below the true 0.25. A single binary observation per cell barely identifies t.
- All bounds were monotone.

## Full-data selection
The data are 16,460 x 19,543. Selection uses validation and the test split is reported. In `tables/selection.csv`, `chosen_restart` marks the restart with the best training bound.

Most fits ran for 30 iterations, and **none of the 30-iteration fits converged**: the relative bound gain was still 1e-4 to 1e-3 per iteration. The ranking is therefore a ranking at a fixed compute budget. The residual mode `none` converges much faster than isotropic mode, both here and on synthetic data, and that partly explains why it wins.

| config | val NLL | test NLL | test AUC | test Brier |
|---|---|---|---|---|
| **selected: none, K=1, d=20, λ=1 (30 + 41 it., converged at tol 1e-5)** | 0.17847 | 0.17856 | 0.9649 | 0.05579 |
| best K>1: none, K=4, d=20 (warm start from iso, 45 it.) | 0.17871 | 0.17885 | 0.9647 | 0.05578 |
| best isotropic: K=4, d=20, t=0.25 (100 it.) | 0.17889 | 0.17904 | 0.9646 | 0.05584 |
| isotropic K=1, d=20, t=0.25 | 0.17980 | 0.17989 | 0.9644 | 0.05597 |
| out-of-grid check: none, K=1, **d=40** (30 it.) | 0.17749 | 0.17754 | 0.9653 | 0.05546 |
| reference: logit + offsets k=40 (rescored) | 0.17764 | 0.17770 | 0.9653 | 0.05550 |

What the grid shows:
- **d is the main lever.** Validation NLL improves from d = 5 to 10 to 20, and d = 40, which lies outside the grid, improves it further.
- **K helps only slightly, and only with t > 0.** For isotropic mode with d = 20, validation NLL goes from 0.17980 at K = 1 to 0.17947 at K = 4. With `none`, K ≥ 8 overfits: the training bound rises while validation NLL worsens, reaching 0.1840 at K = 16.
- **t, learned t and λ matter little** compared with d.
- Calibration on the test split is close to the diagonal; see `best.json`.
- The responsibilities of the best K > 1 fit are essentially hard (max r ≈ 1), and occupancy stays near the k-means start at 0.42 / 0.38 / 0.15 / 0.05.

## Files
- `best.json`: the selected configuration and its scores, the K = 1, K > 1, isotropic and none comparisons, and the out-of-grid row.
- `fits/best.npz`: pi, mu, B and t for the selected model. xi is not saved.
- `responsibilities.npy`: n x 1 for the selected model, which has K = 1.
- `responsibilities_best_K_gt_1.npy`: n x 4. This is the one to use for clustering versus hemilineage.
- `coordinate_means.npy`
- `embedding.npy`: n x 20. It is the cell-wise expected logit, sum_l r_il (mu_l + B_l m_il), centred and projected by an exact SVD of its rank ≤ K(d+1) factorisation. For K = 1 and d = 20 the rank is 20, so this file has 20 columns, not 50. It is rotation-invariant.
- Figures (PNG at 200 dpi, plus PDF): `figures/{synthetic_demo, full_selection, full_convergence, full_occupancy}`

## Runtime and memory
- The whole grid took about 5.2 h of wall time on a shared machine (load around 18), using 3 threads.
- One outer iteration took about 6 s at K = 1, 23–27 s at K = 4, 52 s at K = 8 and 103 s at K = 16, all for isotropic mode at d = 10–20.
- Peak RSS was 13.6 GB, at isotropic K = 16. Outputs take 221 MB of disk.

## Limitations
- No fit, apart from the selected `none` K = 1 chain, reached tol 1e-5.
- The d grid stops at 20, but d = 40 is better.
- Restarts: 2 for the best base configurations, 1 for the others.
- Variants at K = 4 were warm-started, with a matched control.
- New-row prediction was implemented and tested (`infer_rows`) but not scored on the full-data new-row split.
