# Worker C: neural-manifold auto-decoder (handoff model 2)

Code: `cm_minimal/experiments/bernfac/models/neural/` (package `bernoulli_neural`, tests, scripts; see its README).

## Files

| path | content |
|---|---|
| `tables/selection.csv` | every full-data fit: validation and test NLL / AUC / Brier, final F, iterations, accepted and rejected line-search units, head solves, monotone flag, seconds, peak RSS. `chosen_restart` marks the restart with the lowest training F within a configuration |
| `best.json` | selected configuration and its scores (selected on validation NLL only) |
| `fits/best.pt` | selected model: config JSON, `state_dict` (includes Z), histories. Load with `bernoulli_neural.load_model` |
| `fits/<config>.pt` | every fit, same format |
| `embedding.npy` | fitted Z of the selected model, 16,460 x 20 float32, rows in the order of `full/data/Y.npy` |
| `embedding_hidden.npy` | h(Z), the last hidden layer, 16,460 x 64 float32. The logits are b + V h, so this is what the output layer reads |
| `tables/best_calibration_{val,test}.csv` | 10 equal-width bins |
| `tables/logit_spectra.json` | singular values of fitted logits, 2,000 random cells x all genes, affine vs nonlinear head |
| `results/newrow.json` | new-row experiment (if present) |
| `synthetic/` | handoff 8.2 demo: data, all fits, summary over 5 seeds, seed-0 spectra, embeddings, optimizer traces |
| `figures/` | `fig_neural_synthetic`, `fig_neural_selection`, `fig_neural_traces`, `fig_neural_spectrum` (PNG dpi 200 + PDF) |
| `logs/` | per-fit logs, timing log |

## How to read the embeddings

Z and h(Z) are joint-MAP point estimates. They are not posterior means, and Z is not identified:
rotations and other reparametrisations absorbed by the first layer, and hidden-unit permutations or
sign flips, give the same logits. Distances and neighbourhoods (UMAP, kNN) are the meaningful use;
individual coordinates are not. The fitted Z are not a sample from the N(0, I) prior.

## Protocol in brief

Train = ~heldout. Validation picks every hyperparameter; test is scored once per finished fit and
never used for a decision. Each full-data fit got the same 25-minute optimisation budget (float32,
3 threads), so selection compares configurations at equal compute; none of them met the relative
tolerance 2e-5 inside that budget (termination `time_limit`). Prediction is `map_plugin`.
