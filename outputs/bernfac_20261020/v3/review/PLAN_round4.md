# Round 4: strict review, revisions, regenerated deck and a summary deck (CSO plan, 23 Oct 2026)

User request: check correctness, rigor and conclusions with strict criticism. Revise experiments if needed. Redo figures and schematics (BioRender via Chrome, scientific-figure-making). Regenerate the full deck. Add a summary deck of 10 slides or fewer. The CSO runs subagents and resumes after interruptions until done.

## Phase 1: independent reviews (read-only), reports in this folder
- A: statistical rigor of claims (`review_A_statistics.md`)
- B: code correctness audit (`review_B_code.md`)
- C: claims, equations, schematics and narrative (`review_C_claims.md`)

## Phase 2: CSO triage into `revision_plan.md` (fix list with owners)
## Phase 3: revision workers (experiments and figures); code in `cm_minimal/experiments/bernfac/v4/`, outputs in `outputs/bernfac_20261020/v4/`
## Phase 4: deck v4 (full) + summary deck (≤10 slides); upload as Google Slides
- Replace the content of deck 1Jc-fiWr1VZ0Vra8R5fhztL7XCrJmZxUNeBuQAF6wNFk, or make a new deck; the summary deck is a new deck.
## Rules: Python only; ≤4 threads per process; never save n×m arrays; disk ~19 GB free.

## Progress log
- 23 Oct: reviews A, B and C launched.
- 23 Oct: reviews A/B/C done; revision_plan.md written; workers W1-W5 launched.
- 24 Oct: after the usage-limit stop, workers resumed. CSO: eq_offsets direction fixed, notation fixed (g_i/h_j for groups, psi in v3_depth); BioRender v4_evaluation_design, v4_bernfac_family, v4_depth_schematic in v4/biorender.
- 24 Oct: W5 DONE.
  - p̃ is "depth-adjusted", not depth-free. R² of log UMI from the top-10 PCs: R6 0.41, R5 0.66, R4 0.87; log-norm counts 0.61; partial R² beyond strata for R6 is 0.21. Within-stratum |ρ|: 0.07 / 0.13 / 0.25 (floor 0.01).
  - Marker retention is 0.98–1.02 (variance ratio); profile r ≥ 0.995. The 84–90% was η².
  - Held-out depth-gene gap: 0.40 → 0.23 / 0.15 / 0.13; about a third of it is birth time.
  - p̃ removes 75–80% of within-hemilineage birth-time variance on random genes, so correction cannot be told apart from over-correction on real data (W4 synthetic check).
  - Figures: v4/figures/fig_std_rigor, fig_std_depthgenes_heldout, fig_std_reference.
- 24 Oct: W4 DONE (263 tasks).
  - R6 now wins its own NB world G2 at the true rank; v3's loss came from the grid.
  - R7 wins G1 and G3 but often sits at a grid edge. R8 wins G4 (iteration-matched, not converged). R9 wins G5.
  - R1's best clustering holds only with all entries; on train entries the count models lead.
  - With genuine counts in G1, R6 is best (count models had trailed only because of how G1's counts were built).
  - p̃ recovers the true depth-adjusted P about as well as p̂ recovers P. Residual depth: R4 lowest; R5/R6 small; Poisson leaves depth in NB worlds; R6 and Poisson over-correct in G4.
  - Selection regret is about 0.
  - Figures: v4/figures/fig_syn_kl_v4, fig_syn_select_v4, fig_syn_embed_v4, fig_syn_depthstd.
- 25 Oct: W2 DONE.
  - R8 with an iteration budget: d=40, λV=10, test 0.17754, tied with R7 (0.17750); it sits at a grid corner and is not converged.
  - Poisson best is now k=120, λ=30 (test 0.17937), still 0.003 behind NB; Poisson's val/test gap disappears.
  - NB k=160 overfits (cold and warm starts agree). The R4 grid edges don't change its selection.
  - comparison_v4.csv, fig_select_all_v4, predictors_v4.py.
- 25 Oct: W1 DONE.
  - Bootstrap: R6 beats all models by about 1e-3. R7 < R4 < R3 are resolvable; R4, R5 and R8 are tied.
  - NB beats Poisson at matched rank; the gain comes from zeros of bursty frequent genes with φ<10.
  - Poisson outliers are misspecification, not a numerical blow-up.
  - Calibration-tail wording fixed. The test set was used in R3–R5's round-1 selection, but validation-only reselection picks the same configs.
  - About 230 genes carry half of R6's gain (wiring and TF genes).
- 25 Oct: DONE. Full deck v4 (86 slides, 1_SVQuqGqOUm85xniWjNvMFhf0b9iSmAGJUkPSRdVgHM) and summary deck (10 slides, 1oWG6Uomf3l-XsDZJcwZT6nEZWK2U0OwmfbyXjBlO5v0) uploaded and checked with the Drive reader. Helper PPTX uploads moved to the Drive trash. REPORT.md updated.
