# Revision plan (CSO triage of reviews A, B and C, 23 Oct 2026)

Robust result to keep: R6's held-out prediction gain over the binary models. Two-way bootstrap Δ ≈ −1.1e-3 nats, CI excludes 0. Everything else gets re-tested or re-worded.

## Workers (code in cm_minimal/experiments/bernfac/v4/, outputs in outputs/bernfac_20261020/v4/{figures,tables})

- **W1 prediction statistics**
  - Two-way (cell × gene) bootstrap CIs for all pairwise test-NLL differences, overall and by depth decile and gene-rate decile.
  - NB vs Poisson at matched rank, and the φ distribution per config.
  - Calibration tail facts; diagnosis of the Poisson outlier entries.
  - Verify that selection used validation only, and record where test NLL was computed for every config.
- **W2 real-data refits** (sequential, heavy)
  - Poisson at k ∈ {120, 160}.
  - Check the NB k=160 anomaly.
  - R4 edge: (k=80, λ=100) and (k=120, λ=30).
  - R8 with an iteration budget instead of a time budget, d ∈ {20, 40}.
  - Rebuild the comparison table and the selection figure with grid-edge flags.
- **W3 representations, fair protocol**
  - Centre and SVD every embedding. Compare at matched dimension (20, 40) and with whitened coordinates.
  - Metrics: linear probe, kNN, Leiden ARI/AMI at a fixed 23 clusters over 5 seeds, consistent depth-axis metrics.
  - Depth within hemilineage × birth time. Library within hemilineage. Sex within mixed-sex libraries (roX genes).
  - R3–R5 embeddings from the train-only fits.
  - Add comparators: Pearson-residual PCA and scVI (scVI only if it installs and trains within about 1.5 h).
  - Regenerate the drivers, metrics, depth-regression and UMAP figures for any added comparators.
- **W4 synthetic rerun**
  - Wider grids that include the true rank: k ∈ {4, 8, 16} × λ ∈ {3, 10, 30, 100}; R7 K ∈ {1, 2, 4} × d ∈ {4, 8, 16}; R9 up to 80.
  - R8 with an iteration budget on a reduced grid, or excluded from winner claims.
  - Store per-config test KL so the oracle config can be shown.
  - R1/R2 on train entries (masked) as well as on full data.
  - Disclose how the counts in G1, G4 and G5 were constructed.
  - Ground-truth check of depth standardization on a depth world.
- **W5 depth standardization rigor (real data)**
  - R² of log UMI from p̃.
  - Metrics within hemilineage × birth time.
  - Depth-gene selection on held-out cells, to remove the circularity.
  - Sensitivity to the reference depth.
  - Rename "depth-free" to "depth-adjusted".

## CSO
- Equations:
  - Fix the offsets claim: abundant genes shift more under depth.
  - Fix notation.
  - Make the tuning table exact.
  - Rewrite the recommendation.
- BioRender:
  - Redraw the evaluation design with a scattered entry split.
  - Fix the depth schematic: R3 has no b_j; R1 only approximately removes depth.
  - New full-data BernFac-family schematic, replacing bernfac_1's pilot labels.
  - Patch the small errors in the others.
- Deck v4:
  - Condense the per-model slides; move detailed equations to the appendix.
  - Merge the UMAP grids and the heatmap slides.
  - Add an uncertainty slide and an errata slide.
  - Drop the v2 duplicates from the appendix.
- A summary deck of 10 slides or fewer, with a BioRender takeaway figure.
- Scorecard rebuilt from the v4 tables, with no "best" box for degenerate embeddings.
