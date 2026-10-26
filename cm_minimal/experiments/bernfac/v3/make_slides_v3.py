"""Slide spec for deck v3: one narrative for all nine representations.

Writes outputs/bernfac_20261020/v3/deck/slides.json, which build_deck.py turns into a PPTX.
Old slides (pilot, BernFac synthetic CV, per-model detail) are pulled from the v2 spec
into the appendix by number, so their captions stay identical.
"""
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[4] / "outputs/bernfac_20261020"
OUT = STUDY / "v3/deck"
OLD = json.loads((STUDY / "deck/slides.json").read_text())["slides"]


def old(n, title=None):
    s = dict(OLD[n - 1])
    if title:
        s["title"] = title
    return s


def S(title, image=None, bullets=(), **kw):
    d = {"title": title, "bullets": list(bullets)}
    if image:
        d["image"] = image
    d.update(kw)
    return d


def section(part, title, bullets=()):
    return {"kind": "section", "part": part, "title": title, "bullets": list(bullets)}


F = "v3/figures/"
SLIDES = [
    {"kind": "title",
     "title": "Nine ways to represent a single-cell matrix",
     "bullets": ["Detection probabilities, count models and cell embeddings, compared on one dataset with one set of metrics",
                 "Fly premotor neurons, 24 h APF: 16,460 cells × 19,543 genes",
                 "ConnectionMiner · 23 Oct 2026"]},
    S("The question", bullets=[
        "We want one representation of the cells × genes matrix to build on.",
        "Two jobs: (1) predict which genes a cell detects, entry by entry, and (2) place cells so that hemilineage and birth time are near, while library and sequencing depth are not.",
        "We fit nine representations under the same split, the same metrics and the same plots. Seven of them output a probability for every entry; two (processed counts, binary PCA) are the reference pipelines.",
        "Sequencing depth is the main nuisance. The models differ in where they put it: in an explicit cell offset (R4–R6), inside their latent factors (R3, R7–R9), or normalised away beforehand (R1). An explicit offset lets us report two probabilities per entry: the observed one, at the cell's own depth, and a depth-standardized one, at a common depth.",
        "Order: methods, depth handling, tuning and metrics → synthetic data with known truth → held-out prediction → cell maps → cells × genes, observed and depth-standardized → diagnostics → recommendation.",
    ]),
    # ---------------------------------------------------------------- preamble
    section("Part 1", "Methods and metrics",
            ["Every method on one page, the math for each family, where each puts sequencing depth, how it is fitted, how it is tuned and how it is scored."]),
    S("Nine representations at a glance", "v3/biorender/v3_methods_map.png", [
        "Colour code used for the rest of the deck: blue = reference pipelines, orange = low-rank detection models, teal = counts, purple = structured detection models.",
    ]),
    S("Shared notation", "v3/equations/v3_notation.png"),
    S("Every model in one table", "v3/equations/v3_models.png"),
    S("R3–R5: low-rank detection models (BernFac)", "biorender/bernfac_1_model_schematic.png", [
        "Each entry is a coin flip with its own probability. The log-odds are a sum of a cell factor times a gene factor; R4 adds a cell and a gene offset, R5 swaps the logistic link for the complementary log-log.",
    ]),
    S("BernFac objective", "equations/eq_model.png"),
    S("R4: cell and gene offsets", "equations/eq_offsets.png", [
        "The cell offset a_i absorbs sequencing depth and the gene offset b_j absorbs abundance, so UVᵀ is free to describe cell state. Because a_i is explicit, it can later be set to one common value for every cell.",
    ]),
    S("R5: the complementary log-log link", "equations/eq_cloglog.png", [
        "Under cloglog, 1 − p = exp(−rate), so depth enters as a multiplicative rate, the way it does when molecules are sampled at random.",
    ]),
    S("R6: factorize the counts themselves", "biorender/bernfac_5_count_models.png", [
        "Negative-binomial (and Poisson) factorization of raw UMI counts with each cell's library size as an offset. Detection probability is Pr(count ≥ 1), so it is scored on the same entries as the binary models.",
    ]),
    S("Count factorization", "equations/eq_counts.png"),
    S("R7: mixture of local factor models", "biorender/bernfac_6_mixture_local_factors.png", [
        "Each cell's logits lie near one of K local affine subspaces. With K = 1 this is a probabilistic version of BernFac with a Gaussian prior on cell coordinates.",
    ]),
    S("Mixture of local factors", "equations/eq_mfa.png"),
    S("R8: a nonlinear manifold with a neural decoder", "biorender/bernfac_7_neural_manifold.png", [
        "Each cell gets a coordinate z; a shared network maps z to one logit per gene. There is no encoder: z and the network are fitted together.",
    ]),
    S("Neural manifold", "equations/eq_neural.png"),
    S("R9: latent block model", "biorender/bernfac_8_latent_block_model.png", [
        "Cells and genes are clustered at the same time; each block of cell group × gene group has one detection probability.",
    ]),
    S("Latent block model", "equations/eq_lbm.png"),
    "DEPTH",
    S("Fitting: one Pólya–Gamma step reused in four models", "biorender/bernfac_2_pgem_inference.png", [
        "The Pólya–Gamma identity turns the logistic likelihood into a Gaussian one given auxiliary weights, so each update is a weighted least-squares solve.",
    ]),
    S("Inference for every model", "v3/equations/v3_inference.png"),
    S("How every model is scored", "v3/biorender/v3_evaluation_design.png", [
        "Three tiers, the same for all nine: held-out entries on real data, synthetic data where the true probabilities are known, and checks on the cell representation.",
    ]),
    S("Prediction metrics", "v3/equations/v3_metrics_pred.png"),
    S("Representation metrics", "v3/equations/v3_metrics_rep.png"),
    S("What is tuned, on which grid, and what was chosen", "v3/equations/v3_hyperparams.png", [
        "Every hyperparameter is chosen on validation entries, on real and synthetic data alike. The selection curves follow in Parts 2 and 3.",
    ]),
    # ---------------------------------------------------------------- synthetic
    section("Part 2", "Synthetic data: which model recovers the truth?",
            ["Five simulated worlds, each built to favour a different model. All methods fitted and selected exactly as on real data."]),
    "SYN",
    # ---------------------------------------------------------------- real: prediction
    section("Part 3", "Real data: held-out prediction",
            ["Hyperparameters chosen on 8.05 M validation entries; the 8.04 M test entries were scored once."]),
    S("Model selection for every family", F + "fig_select_all.png", [
        "Validation NLL over each family's grid (grids and choices are listed in the tuning table, Part 1). Low-rank models settle at rank 40, NB counts at rank 120, the mixture at one component with d = 40. The latent block model and Poisson are still improving at the edge of their grids.",
    ]),
    S("All models on the same test entries", F + "fig_test_all.png", [
        "NB counts predict best (test NLL 0.1766). Five binary models sit within 0.0003 of each other at 0.1775–0.1778. Latent blocks (0.1830) and Poisson (0.1839) trail; a gene-rate-only baseline scores 0.1976.",
    ]),
    "CAL",
    # ---------------------------------------------------------------- real: representation
    section("Part 4", "Cell maps: what drives the clustering?",
            ["The same UMAP settings for all nine embeddings, coloured by biology (hemilineage, birth time, sex) and by nuisance (depth, library). Metrics are computed in the embedding, not in 2-D."]),
    S("Coloured by hemilineage", F + "fig_umap_grid_hemilineage.png", [
        "Processed counts (R1), NB counts (R6) and the neural manifold (R8) give clean strands per hemilineage. R2, R3, R4 and R7 keep a mixed central mass; R9 breaks cells into about 200 mixed fragments.",
    ]),
    S("Coloured by sequencing depth", F + "fig_umap_grid_depth.png", [
        "In binary PCA, BernFac logit and the mixture, the main gradient across the map is depth: a low-UMI blob and a high-UMI blob. R4–R6 embed UVᵀ without their cell offset, which shrinks the blob, but R4 still keeps a depth axis (|ρ| 0.80), so some depth leaks into its factors. NB counts and R1 spread depth along the strands.",
    ]),
    S("Coloured by library", F + "fig_umap_grid_library.png", [
        "No method clusters by library. Library is still predictable above chance (0.05) from the neighbours: 0.10–0.14 for R9, R1 and R8, 0.21–0.27 for the detection models.",
    ]),
    S("Coloured by sex", F + "fig_umap_grid_sex.png", [
        "Female and male cells separate within each strand in every method (kNN accuracy about 0.82). We have not checked whether this comes from sex-specific genes or from single-sex libraries.",
    ]),
    S("Coloured by birth time", F + "fig_umap_grid_birthtime.png", [
        "Birth time (primary, secondary, precursor) is still predictable within a hemilineage: 0.78 for R1 and R6 against about 0.35 by chance. It is the second biological axis after hemilineage.",
    ]),
    S("Zoom: hemilineage pairs that depend on the method", F + "fig_umap_zoom_hemilineage_pairs.png", [
        "05B/06B and 13A/19A: 15-NN confusion is under 1% in R1 and R6 and 4–7% in R2, R3 and R7. The confused cells are the ones that fall into the shared high-depth mass.",
    ]),
    S("Zoom: the deepest and shallowest 10% of cells", F + "fig_umap_zoom_depth_extremes.png", [
        "The top-depth decile is 84% primary neurons (17% overall), so part of the 'depth' blob is biology. Still, most hemilineage errors sit at the two depth extremes: 28% and 26% for R3, against 3–4% in the middle deciles.",
    ]),
    S("Zoom: hemilineage 14A split by birth time", F + "fig_umap_zoom_14A_birthtime.png", [
        "Primary (335) and secondary (1,072) 14A neurons form separate clusters in every method; within 14A, birth time is decoded at 0.91–0.96 for R1–R8. In the binary models the primary cells sit inside the shared high-depth mass.",
    ]),
    S("What each embedding encodes", F + "fig_umap_drivers.png", [
        "Blue: kNN accuracy for biological and technical labels. Red: how strongly depth lines up with the top components. Depth dominates R2, R3 and R7 (|ρ| 0.97–1.00); R6 is cleanest (0.24 on its first component). R8 keeps depth off its first axis but not out of the top five.",
    ]),
    S("Representation metrics for all nine", F + "fig_metrics_all.png", [
        "R1 wins kNN accuracy and purity; R6 wins Leiden ARI (0.67 vs 0.65) and has the weakest depth axis. The low-rank detection models trail by 0.04–0.10 on hemilineage metrics.",
    ]),
    S("Regressing depth out of each embedding", F + "fig_depth_regressed.png", [
        "Removing log depth helps the binary models most: BernFac logit's Leiden ARI goes from 0.49 to 0.61. R1 and R6 barely move because depth was never their main axis. R6 still has the highest Leiden ARI after the correction (0.69).",
    ]),
    # ---------------------------------------------------------------- heatmaps
    section("Part 5", "Cells × genes, observed and depth-standardized",
            ["Same cells and genes in every panel: five hemilineages, low- and high-depth cells, marker, depth-sensitive, rare and ubiquitous genes. First what each model predicts at each cell's own depth, then what R4–R6 predict once every cell is set to a common depth."]),
    "HEAT",
    # ---------------------------------------------------------------- diagnostics
    section("Part 6", "Diagnostics", ["Convergence, run time and tests for every method."]),
    S("Convergence", F + "fig_converge_all.png", [
        "Every objective went one way only. The neural manifold never converged: all of its runs stopped at the 1,500 s budget, so its numbers are a lower bound on what it can do.",
    ]),
    S("Run time", F + "fig_runtime.png", [
        "Selected fits take 4 min (mixture, K = 1) to 34 min (NB, k = 120); full grids took 1.7–8 h each on this machine.",
    ]),
    S("Tests", F + "fig_tests.png", [
        "All suites pass: BernFac 11/11, counts 5/5, mixture 21/21, neural 14/14, latent blocks 12/12. The offsets and cloglog paths (R4, R5) have no unit tests of their own; they are only guarded by the run-time monotonicity check.",
    ]),
    # ---------------------------------------------------------------- conclusion
    section("Part 7", "Which one to work with?"),
    S("Scorecard", F + "fig_scorecard.png", [
        "Darker is better within each column; black boxes mark the best value.",
    ]),
    "REC",
    # ---------------------------------------------------------------- appendix
    section("Appendix", "Earlier results and per-model detail",
            ["A: 1000 × 100 pilot (BernFac only). B: BernFac synthetic model-selection study. C: per-model selection detail and the v2 comparison figures."]),
    *[old(n, "A · " + OLD[n - 1]["title"]) for n in range(9, 21)],
    *[old(n, "B · " + OLD[n - 1]["title"]) for n in range(21, 30)],
    *[old(n, "C · " + OLD[n - 1]["title"]) for n in (30, 31, 32, 33, 38, 41, 44, 47, 49, 50, 51)],
]


def fill(slides, blocks):
    out = []
    for s in slides:
        out.extend(blocks.get(s, [])) if isinstance(s, str) else out.append(s)
    return out


BLOCKS = {
    "SYN": [
        S("Five synthetic worlds with known probabilities", F + "fig_syn_design.png", [
            "2,000 cells × 300 genes, mean detection 0.18, three seeds each. Each world is built in the image of one model family: low-rank logit (R3–R5), NB counts (R6), cell types with depth, a 2-D nonlinear manifold (R8) and blocks (R9). The same 70/15/15 entry split and validation selection is used for every method.",
        ]),
        S("How each method was tuned on the synthetic data", F + "fig_syn_select.png", ["Each cell is validation NLL relative to the best setting in that world; boxes mark what was chosen. The three seeds agree in 34 of 40 cases. Low-rank and count models mostly pick k = 10, λ = 10. Orange boxes are picks at the edge of the tested range, mostly the most blocks for latent blocks and the strongest penalty for the neural manifold, so those two may be held back by their grids."]),
        S("Distance to the true probabilities", F + "fig_syn_kl.png", [
            "The local-factor mixture (R7) gets closest to the truth in G1–G3, including the NB world (KL 0.022 vs 0.026 for NB counts). The neural manifold wins its own world (G4), latent blocks win theirs (G5). Poisson is close to the cell + gene baseline whenever counts are overdispersed (G2, G3).",
        ]),
        S("Calibration against the truth", F + "fig_syn_calibration.png", [
            "Left: p̂ against observed detection; right: against the true probability. The logit models sit on the diagonal. NB counts and cloglog are over-confident at the top (p̂ 0.97 where the truth is about 0.85); Poisson over-predicts everywhere above 0.3.",
        ]),
        S("Do the embeddings recover the true cell groups?", F + "fig_syn_embed.png", [
            "G3 cell types: NB counts has the best kNN accuracy (0.95) and processed counts the best clustering (Leiden ARI 0.97); BernFac logit reaches only 0.63. G4: no method recovers regions of the 2-D latent well (ARI at most 0.45), the neural manifold included. G5: latent blocks recover their own blocks almost exactly.",
        ]),
    ],
    "CAL": [
        S("Calibration on test entries", F + "fig_calibration_all.png", [
            "All seven models are calibrated to within 0.7 percentage points in every bin; Poisson over-predicts mid-range probabilities by up to 2.3 points. R5 and R6 also produce confident predictions (p̂ below 0.01 or above 0.99) that the logit models rarely give.",
        ]),
        S("Where the errors are", F + "fig_test_by_depth.png", [
            "NB counts are best in every depth decile and every gene-frequency decile. The gaps open on frequently detected genes, where the latent block model loses most. Poisson fails in the deepest 10% of cells.",
        ]),
    ],
    "HEAT": [
        S("Overview: one slice through every model", F + "fig_heat_overview.png", [
            "All seven models reproduce the hemilineage marker blocks of the data and fill dropouts within them. R9 blurs them, because one probability is shared by a whole block of cells and genes.",
        ]),
        S("Marker genes, including the confusable 05B/06B", F + "fig_heat_markers.png", [
            "NB counts (R6) gives the sharpest marker blocks; for example toy is called in 05B and absent in 06B. The low-rank binary models smear 05B and 06B markers into each other, matching the kNN confusion on the cell maps.",
        ]),
        S("Depth-sensitive genes", F + "fig_heat_depth.png", [
            "For the eight genes whose detection rises most with depth, every model reproduces the jump from low- to high-depth cells (right). R9 overestimates the shallow cells. R6 shows the expected count μ̂ behind its probabilities.",
        ]),
        S("Differences from the empirical rate", F + "fig_heat_diff.png", [
            "Each panel is p̂ minus the training detection rate in the same hemilineage × depth stratum. R6 is closest (mean |diff| 0.073) while keeping the most cell-to-cell spread. R9 is off by whole blocks, for example 05B and 06B markers.",
        ]),
        S("Depth-standardized probabilities next to every model", F + "fig_heat_std_overview.png", ["Same cells and genes as the overview, with R4–R6 shown twice: observed (each cell at its own depth) and standardized (every cell at the median depth). After standardizing, low- and high-depth cells of one hemilineage look alike on the depth-sensitive genes, and every hemilineage keeps its markers. R3 and R7–R9 have no depth term to standardize."]),
        S("Depth-sensitive genes, observed and standardized", F + "fig_heat_std_depth.png", ["On the eight genes whose detection rises most with depth, every observed p̂ reproduces the data's jump from shallow to deep cells (gap about 0.43). Standardizing cuts the gap to 0.26 for R4 and 0.17–0.18 for R5 and R6. Some of the remaining gap may be real biology: these genes include mitochondrial ribosomal genes, and the deepest cells are mostly primary neurons."]),
        S("What standardization changes", F + "fig_heat_std_diff.png", ["p̃ − p̂: standardizing moves whole cells, raising shallow cells and lowering deep ones (top strip: each cell's offset from the median). Genes that are off stay off, so the hemilineage pattern is unchanged. The mean absolute shift is 0.06–0.07."]),
        S("Depth dependence and biology across the whole matrix", F + "fig_std_quant.png", ["All 16,460 cells × 3,000 random genes. (A) The median correlation of p with depth falls from about 0.8 to 0.29 (R4), 0.19 (R5) and 0.11 (R6); models without a depth term stay at 0.5–0.8. (B) R4 keeps depth in frequently detected genes, where depth leaked into its factors. (C) Markers keep 84–90% of their variance between hemilineages."]),
    ],
    "DEPTH": [
        S("Where sequencing depth goes in each model", "v3/biorender/v3_depth_schematic.png", [
            "R4–R6 give depth its own term, a cell offset. R3, R7, R8 and R9 have no such term, so depth ends up in their latent factors. R1 divides it out before PCA.",
        ]),
        S("Observed and depth-standardized probabilities", "v3/equations/v3_depth.png"),
    ],
    "REC": [
        S("Recommendation", "v3/equations/v3_recommendation.png"),
    ],
}

if __name__ == "__main__":
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    slides = fill(SLIDES, BLOCKS)
    # v2 cross-reference "(equations, slide 22)" -> wherever that equations slide lands now
    eq = 1 + next(i for i, s in enumerate(slides) if s.get("title") == "B · " + OLD[21]["title"])
    for sl in slides:
        sl["bullets"] = [b.replace("(equations, slide 22)", f"(equations, slide {eq})") for b in sl.get("bullets", [])]
    if "--draft" in sys.argv:  # skip figures that do not exist yet
        slides = [s for s in slides if all((STUDY / i).exists() for i in ([s["image"]] if "image" in s else s.get("images", [])))]
    cut = next(i for i, s in enumerate(slides) if s.get("part") == "Appendix")
    jpeg = {"max_w": 1800, "quality": 76}
    # two files so each stays under the 10 MB browser-upload limit; imported into one Google Slides deck
    for name, part, start in (("main", slides[:cut], 1), ("appendix", slides[cut:], cut + 1)):
        spec = {"out": str(OUT / f"NineRepresentations_v3_{name}.pptx"), "start": start, "jpeg": jpeg, "slides": part}
        (OUT / f"slides_{name}.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    missing = [i for s in slides for i in ([s["image"]] if "image" in s else s.get("images", []))
               if not (STUDY / i).exists()]
    print(len(slides), "slides (main", cut, "); missing images:", missing)
