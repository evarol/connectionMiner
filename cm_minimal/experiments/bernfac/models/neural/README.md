# Neural-manifold auto-decoder for a binary matrix (model 2 of the handoff)

Package: `bernoulli_neural/` (config, data, decoder, objectives, pg, fit, predict, io).
Tests: `tests/test_neural.py` (run with `python tests/test_neural.py`; no pytest needed).
Scripts: `synthetic_demo.py` (handoff 8.2), `run_full.py` (full-data selection), `make_figures.py`.
Python: `outputs/connectionminer_ready_inputs/.venv/bin/python` (torch 2.14).

## Model

```
z_i ~ N(0, I_d / lambda_z)                     row coordinate (an optimised variable, no encoder)
h_i = tanh(W2 tanh(W1 z_i + c1) + c2)          hidden network, widths (64, 64) by default
t_i = b + V h_i                                V: m x H, b: m
psi_i = t_i                    (head_mode='affine')
psi_i = t_i + rho tanh(t_i)    (head_mode='nonlinear', rho = 1)
Y_ij ~ Bernoulli(sigmoid(psi_ij))  on training cells only
```

Gaussian priors with separate precisions on Z (`lambda_z`), the hidden weights and biases
(`lambda_hidden`), V (`lambda_output`) and b (`lambda_bias`); each group is penalised exactly once.

**What is estimated.** Joint MAP over (Z, hidden, V, b), minimising

```
F = sum_train [softplus(psi) - y psi] + lambda_z/2 |Z|^2 + lambda_hidden/2 |eta|^2 + lambda_output/2 |V|^2 + lambda_bias/2 |b|^2
```

with `binary_cross_entropy_with_logits(reduction='none')` summed over training cells. This is
not a VAE and not a marginal likelihood: the row coordinates are point estimates, not integrated
out. Predictions `sigmoid(g(z_hat_i)_j)` are conditional plug-in probabilities
(`prediction_mode='map_plugin'`), not posterior predictive probabilities. Z is not identifiable:
any invertible reparametrisation absorbed by the first layer (rotations in particular), network
symmetries (hidden-unit permutations and sign flips of tanh units) and other ambiguities leave
the logits unchanged. Compare logits or probabilities, not raw coordinates.

**Rank bound of the affine head.** With `head_mode='affine'`, `Psi = H_Z V' + 1 b'`, so
`rank(Psi) <= min(n, m, H + 1)` whatever the latent dimension d. A width-64 head therefore keeps
every logit row in a 65-dimensional affine span; inside that span the d-dimensional manifold can
be curved. The nonlinear head removes this particular bound (test F constructs an example with
rank > H + 1) but loses the exact head solve. Neither head makes the map from z injective or fixes
the intrinsic dimension at d.

## Optimisers

`optimizer_mode='pg_gem'` (primary). Each outer iteration freezes PG weights
`w_ij = tanh(psi_ij/2)/(2 psi_ij)` from one parameter snapshot (stored, detached, never
differentiated) and decreases the quadratic surrogate
`S = sum_train [w psi^2/2 - kappa psi] + the same priors`, `kappa = y - 1/2`, which upper-bounds F
up to a constant and touches it at the snapshot. The surrogate is quadratic in the logits only,
not in the network parameters.

* affine head: exact per-column (H+1) x (H+1) ridge solve of S over (b_j, v_j) by Cholesky
  (right-hand side uses kappa, not y); then checked gradient steps for Z (per-row Armijo: rows of S
  are independent given the decoder) and for the hidden parameters (global Armijo); then the head
  is solved again. The Z / hidden steps use the exact sufficient-statistic form
  `S_i = h_i' M_i h_i / 2 + r_i' h_i + c_i` with `M_i = sum_j w_ij v_j v_j'` and
  `r_i = sum_j (w_ij b_j - kappa_ij) v_j` (an identity, not an approximation), so they cost
  O(n H^2) instead of O(n m H).
* nonlinear head: per-column Armijo head steps, per-row Z steps, global hidden steps, all on S.
* after the inner steps: `S_end <= S_start` and `F_new <= F_old` are checked with a
  precision-aware tolerance (`monotone_rtol`, 1e-10 in float64 and 1e-7 in float32). A violation
  rolls the whole outer iteration back and stops with a diagnostic.

`optimizer_mode='direct_bce'`: the same blocks and line searches applied to F itself
(full batch, monotone). No Adam mode is used anywhere in the reported results.

Line-search settings: initial step 1, then twice the last accepted step per unit; backtracking
factor 0.5; at most `max_trials` halvings; a unit with no acceptable step is restored. The
Armijo coefficient default is 0.25, not the handoff's suggested 1e-4: with the doubling initial
step, 1e-4 accepts nearly oscillating steps on stiff rows (seen in test E). Both are monotone
(test D runs 1e-4 as well).

## Configuration (defaults)

```
latent_dim (required), hidden_widths=(64, 64), activation='tanh', head_mode='affine',
nonlinear_head_rho=1, optimizer_mode='pg_gem', dtype='float64', device='cpu',
lambda_z=1, lambda_hidden=0.01, lambda_output=1, lambda_bias=0.01,
max_outer=200, tol=1e-6 (relative F decrease), inner_steps=5, inner_rounds=1, head_steps=5,
resolve_head=True, seed=1, n_restarts=3, z_init_scale=0.1, bias_init='column_logit'
```

These are starting values, not dimension-invariant optimal priors. The restart with the lowest
final training objective F is returned; held-out scores never choose a restart.

## API

```python
from bernoulli_neural import NeuralConfig, fit_bernoulli_neural_manifold, predict_bernoulli_neural_manifold, \
    infer_neural_rows, neural_map_objective, save_model, load_model, heldout_scores
cfg = NeuralConfig(latent_dim=2, hidden_widths=(64, 64), head_mode='nonlinear', optimizer_mode='pg_gem',
                   observed_mask=train_mask, dtype='float64', seed=1, n_restarts=3)
model, history = fit_bernoulli_neural_manifold(Y, latent_dim=2, config=cfg)
P = predict_bernoulli_neural_manifold(model, rows=None, cols=None)     # map_plugin probabilities
Z_new, diag = infer_neural_rows(model, Y_new, observed_mask=context, method='pg_gem', n_starts=5)
parts = neural_map_objective(model, Y, observed_mask=train_mask)
```

New-row inference freezes every decoder parameter (bitwise unchanged, test G) and uses only the
context cells. A row with no context gets z = 0, the prior mode; `sigmoid(g(0))` is a mode
plug-in, not the prior predictive mean. `prior_predictive_mc` gives the Monte Carlo prior
predictive with standard errors.

Checkpoints (`save_model`) are plain dicts: config JSON, `state_dict` (includes Z), ids, dtype,
seed, histories; `torch.load(weights_only=True)` reads them.

## Precision

float64 is the reference (tests, synthetic demo). The full-data runs (16,460 x 19,543) use
float32 tensors with all objective sums accumulated in float64; the Z / hidden inner steps of the
affine PG mode run on float64 copies. Everything is streamed over row chunks, and the PG weights
of one outer iteration are a single stored tensor computed from one snapshot.
