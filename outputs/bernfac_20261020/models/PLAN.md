# Model comparison round 2 (started 21 Oct 2026). CSO plan, kept up to date for resume

All code is Python (user instruction, 21 Oct). Data: full fly preMN 24 h, 16,460 cells × 19,543 detected genes.

## Shared contract (done)
- `full/data/Y.npy`: binary detection (uint8).
- `full/data/heldout.npy`: the 5% held-out entries every earlier fit excluded.
- `full/data/val.npy` and `full/data/test.npy`: disjoint halves of heldout (seed 7).
  - Validation selects models and hyperparameters.
  - Test is used for final reporting only.
- `full/data/newrow_test_rows.npy` and `newrow_context.npy`: 1,646 new-row test cells (10% per hemilineage) with a 50% context column mask. These are optional, for the new-row task.
- Counts: `multisystem_data_audit/bundles/motor_premn_24h/G_prime.npz`, restricted to the columns in `genes_kept.npy`.
- Scores on validation and test: Bernoulli NLL, Brier, AUC on a 2 M subsample, and 10-bin calibration.

## Workers
| id | model | code dir | output dir | status |
|---|---|---|---|---|
| A | Poisson and NB count factorization with a log-depth offset | models/counts | models/counts | launched |
| B | Bernoulli mixture of local factor models (MFA) | models/mfa | models/mfa | launched |
| C | Neural manifold auto-decoder (PG-GEM and direct BCE) | models/neural | models/neural | launched |
| D | Bernoulli latent block model (LBM) | models/lbm | models/lbm | launched |
| E | Rescore earlier models on validation/test (logit, offsets, cloglog, baselines) | CSO | models/rescored | pending |

Each worker writes:
- `tables/selection.csv`: all configurations with validation and test scores.
- `best.json`: the selected configuration.
- `embedding.npy`: the cell representation of the selected model, used for UMAP.
- Figures.
- A report text in its final reply.

## Resource rules
- Limit each process to 3 threads.
- Keep each worker under about 20 GB of RAM and 2 GB of disk. Disk had 19 GB free at the start.
- Never save full n × m probability matrices.

## After the workers finish (CSO)
1. Rescore everything on validation/test and build one comparison table.
2. Run the UMAP and biology metrics for all representations.
3. Write LaTeX panels for every method.
4. Make BioRender schematics in Chrome for every method.
5. Redo the figures in house style.
6. Write the text in plain language.
7. Update the Google Slides deck by importing slides 30 onward, and update REPORT.md.

## Progress log
- 21 Oct (CSO): splits done. LaTeX panels eq_counts, eq_mfa, eq_neural, eq_lbm rendered (`equations/`).
- BioRender schematics exported to `biorender/`, flat 300 dpi, OCR-checked with no typos found:
  - `bernfac_5_count_models`
  - `bernfac_6_mixture_local_factors`
  - `bernfac_7_neural_manifold`
  - `bernfac_8_latent_block_model`
- Rescore of the earlier models is running (`rescore_val_test.py` → `models/rescored/`).
- Workers A–D are running. Worker D has launched its full-data background jobs.
- 21 Oct 21:52: all 4 workers hit the usage limit and were resumed by SendMessage.
  - Rescore is done (`rescored/best_per_model.csv`). The best earlier models on test NLL:

    | model | test NLL |
    |---|---|
    | logit + offsets | 0.17770 |
    | cloglog + offsets | 0.17774 |
    | logit | 0.17780 |
    | gene rate | 0.19763 |

  - LBM grid finished (best Kr=10, Kc=10, λ=1, val NLL 0.1874).
  - MFA run_full_grid.py (pid 72585) and neural run_stages.py (pid 74578) are still running.
  - Counts queues had stopped; worker A was told to restart them.
- Worker C (neural) is DONE: 14/14 tests pass.
  - Selected: affine head, d=20, w64, λV=1. Test NLL 0.17772, which ties logit + offsets (0.17770).
  - PG beat direct BCE on full data at the same wall time.
  - Embeddings: `neural/embedding.npy` (Z) and `neural/embedding_hidden.npy`.
- Worker D (LBM) is DONE: 12/12 tests pass.
  - Best: Kr=200, Kc=200, λ=0.1. Test NLL 0.18300, AUC 0.9631.
  - This is at the grid corner. It is NOT extended (documented as a limitation).
  - Hemilineage ARI is 0.063 and library ARI is 0.005; the row groups follow depth.
  - Embedding: `lbm/embedding.npy`, R·B with n×200.
- Worker B (MFA): 21/21 tests pass.
  - Initial selection: none, K=1, d=20, test NLL 0.17856. d=20 is at the grid edge.
  - Asked B to extend d to {40, 80} for K ∈ {1, 4} and re-select.
  - (My earlier note was wrong: pid 72585 was LBM's job.)
- Worker A (counts): DONE, tests pass.
  - NB k=80, λ=10: test NLL 0.17665, the best so far.
  - Poisson k=80: test NLL 0.18390, unstable because of a few outlier entries.
  - k=80 is at the grid edge, so A is extending NB to k ∈ {120, 160}.
- 22 Oct 02:55 (CSO, after a 2nd usage-limit/auth stop): the worker sessions ended, so the CSO finalizes them directly.
  - MFA re-selected over d ∈ {5, 10, 20, 40, 80}: none, K=1, d=40. Val 0.17746, test 0.17750; d=80 gives 0.17748, an interior optimum.
  - NB k=120 λ=10: val 0.17646, test 0.17660.
  - NB k=160 λ=10 is running (`counts/logs/extend_160.out`) and then calls collect().
- 22 Oct 04:00: DONE.
  - NB k=160 is worse than k=120, so k=120 stays selected.
  - UMAP and metrics are computed for R1–R9, including the depth-regressed versions.
  - Comparison figures and humanized conclusions are written.
  - Google Slides has 52 slides (30–52 imported; old 53–61 deleted). REPORT.md is updated.
