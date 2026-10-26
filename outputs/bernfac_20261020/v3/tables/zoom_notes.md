# UMAP zoom slides (worker U, 22 Oct 2026)

Each zoom figure shows the same cells in all nine representations (3×3, R1–R9). The other cells are drawn in light grey behind them. Scripts: `cm_minimal/experiments/bernfac/v3_umap_drivers.py` (kNN predictions, drivers table) and `v3_umap_figs.py` (all figures).

## fig_umap_zoom_hemilineage_pairs: 05B, 06B, 13A, 19A
We picked these from 15-NN hemilineage confusion (5-fold) as the two pairs whose confusion depends most on the method. 05B↔06B confusion is 0.6% in R1 and 1.0% in R6, but 4–7% in R2, R3 and R7. 13A↔19A is 0.4–0.7% in R1/R6 and 4–6% in R2, R3 and R7. R1, R6 and R8 keep all four as separate strands. In R2, R3, R4 and R7 a share of each lineage drops into a shared mixed blob, which is the depth/primary blob described below. 19A forms a compact island in every method except R9.

## fig_umap_zoom_depth_extremes: lowest vs highest 10% total UMI
The highest-depth decile is 84% primary neurons (only 17% of all cells are primary). In R2, R3, R4 and R7 these cells collapse into one mixed-lineage mass. R5, R6 and R8 keep a smaller version of that mass, and in R1 it breaks up into lineage-specific islands. The lowest-depth decile forms its own blob in R2, R3, R7 and partly R4/R5. In R1, R6 and R8 it spreads along the lineage strands. These two extremes account for most hemilineage kNN errors: R3's error rate is 28% in the bottom decile and 26% in the top decile, against 3–4% in the middle deciles.

## fig_umap_zoom_14A_birthtime: hemilineage 14A split by birth time
14A is well separated in all methods except R9 (kNN recall ≥0.95). Inside it, primary neurons (335) and secondary neurons (1,072) form two separate clusters in every method. In the binary models the primary 14A cells sit inside the shared high-depth mass. The 19 precursor cells form a small, separate cluster. Within 14A, kNN birth-time balanced accuracy is 0.91–0.96 for R1–R8 and 0.82 for R9.

## Choice of the extra colouring variable (fig_umap_grid_birthtime)
soma_neuromere is constant (T1). maturation flags only 189 precursor cells. Neurotransmitter is almost fully set by hemilineage: within-hemilineage kNN balanced accuracy is about 0.44, near chance. Birth time stays predictable within hemilineage, at 0.78 balanced accuracy for R1/R6 against about 0.35 chance, so we used it as the fifth colouring.
