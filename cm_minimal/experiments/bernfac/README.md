# BernFac — Bayesian logistic matrix factorization by Pólya–Gamma EM

BernFac turns a binary matrix `Y ∈ {0,1}^{n×m}` (e.g. gene detected / not
detected per cell) into a continuous matrix of latent Bernoulli probabilities
`P = σ(U Vᵀ) ∈ [0,1]^{n×m}`.

```
y_ij | u_i, v_j ~ Bernoulli(σ(u_iᵀ v_j)),   u_i ~ N(0, I/λ_u),  v_j ~ N(0, I/λ_v)
```

It is fit by deterministic PG-EM (Polson, Scott & Windle 2013, JASA 108:1339):

1. **E-step**: `W = tanh(ψ/2)/(2ψ)` at `ψ = U Vᵀ` (the PG(1, ψ) mean), frozen.
2. **M-step A**: for each row, solve `[λ_u I + Σ_j w_ij v_j v_jᵀ] u_i = Σ_j (y_ij − ½) v_j`.
3. **M-step B**: for each column, the same with the new `U` and the same `W`.
4. Monitor the **original** MAP objective `J`. It never decreases.

With `λ_u = λ_v = λ` the MAP penalty equals `λ‖UVᵀ‖_*` (the nuclear norm), so λ is
the main complexity knob. `k` caps the rank. The problem is nonconvex, and the
factors are identified only up to a common rotation, so compare `P`, not `U`.

## Quick start

```python
import numpy as np
from bernfac import fit, predict
Y = (np.random.rand(200, 150) < 0.4).astype(float)    # NaN = missing allowed
res = fit(Y, k=5, lambda_u=3, lambda_v=3, init="svd", n_restarts=3, seed=1)
P = res.P                     # latent Bernoulli probabilities
res.obj, res.loglik           # MAP objective and log-likelihood per iteration
```

## Files

| file | purpose |
|---|---|
| `bernfac.py` | solver: `fit`, `predict`, `objective` (+ gradients), `loglik`, `pg_mean`, `update_rows` |
| `test_bernfac.py` | 11 validation tests from the handoff spec (PG mean, gradients, block optimality, monotonicity, missing data, empty rows, seeds, prior shrinkage, zero-init trap, synthetic recovery) |
| `make_subset.py` | 1000 × 100 preMN 24h subset: 4 libraries × 5 hemilineages × 50 cells, Y = 1[count ≥ 1] |
| `run_grid.py`, `run_cv.py`, `run_cv_extension.py`, `evaluate.py`, `effective_rank.py`, `run_identifiability.py`, `make_figures.py`, `common.py` | real-data grid, held-out CV, evaluations and figures |
| `synthetic_scenarios.py`, `selection_study.py`, `selection_figures.py` | synthetic ground-truth model-selection study |
| `make_schematics.py`, `build_deck.py` | schematics and the slide deck |

Run with `outputs/connectionminer_ready_inputs/.venv/bin/python` (set
`OMP_NUM_THREADS=1` when running in parallel). Results are in
`outputs/bernfac_20261020/` (see `REPORT.md` there).
