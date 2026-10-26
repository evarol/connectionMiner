"""Slide specs for deck v4 (after the strict review) and the ≤10-slide summary deck.

Writes outputs/bernfac_20261020/v4/deck/slides_{main,appendix,summary}.json for build_deck.py.
Narrative changes from v3: errata up front; one slide per model family (detailed equations in the
appendix); uncertainty on every real-data comparison; one fair protocol for embeddings; depth
handling with its limits; v2 duplicates dropped.
"""
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[4] / "outputs/bernfac_20261020"
OUT = STUDY / "v4/deck"
OLD = json.loads((STUDY / "deck/slides.json").read_text())["slides"]   # v2 spec (pilot, BernFac synthetic CV)
F3, F4, E3, E4, B3, B4 = "v3/figures/", "v4/figures/", "v3/equations/", "v4/equations/", "v3/biorender/", "v4/biorender/"


def S(title, image=None, bullets=(), **kw):
    d = {"title": title, "bullets": list(bullets)}
    if image:
        d["image"] = image
    d.update(kw)
    return d


def section(part, title, bullets=()):
    return {"kind": "section", "part": part, "title": title, "bullets": list(bullets)}


def old(n, prefix):
    s = dict(OLD[n - 1]); s["title"] = f"{prefix} · {s['title']}"; return s


def fix_refs(main, appendix):
    """The v2 caption cites its equations slide as "slide 22"; point it at its v4 position."""
    eq = len(main) + 1 + next(i for i, s in enumerate(appendix) if s["title"] == "E · " + OLD[21]["title"])
    for sl in appendix:
        sl["bullets"] = [b.replace("(equations, slide 22)", f"(equations, slide {eq})") for b in sl.get("bullets", [])]


MAIN = [
    {"kind": "title", "title": "Nine ways to represent a single-cell matrix",
     "bullets": ["Detection probabilities, count models and cell embeddings, compared under one protocol, with uncertainty",
                 "Fly premotor neurons, 24 h APF: 16,460 cells × 19,543 genes",
                 "ConnectionMiner · v4 after a strict internal review · 25 Oct 2026"]},
    S("The question", bullets=[
        "We want one representation of the cells × genes matrix to build on. It has two jobs: predict which genes each cell detects, and place cells so that hemilineage and birth time are near while library and sequencing depth are not.",
        "Nine representations (R1–R9) plus two standard comparators for the embedding part (Pearson residuals, scVI). Seven models output a probability for every entry.",
        "Sequencing depth is the main nuisance, and it is partly biology: hemilineage and birth time explain a third of its variance.",
    ]),
    S("What changed after the review", E4 + "v4_errata.png", [
        "Three independent reviews (statistics, code, claims) found no leakage or scoring bug, but several conclusions in v3 did not hold. This deck reruns the affected analyses.",
    ]),
    # ------------------------------------------------------------ methods
    section("Part 1", "Methods and how they are scored",
            ["One page per model family, where each puts sequencing depth, and one evaluation protocol for all. Detailed derivations are in the appendix."]),
    S("Nine representations at a glance", B3 + "v3_methods_map.png", [
        "Colours used throughout: blue = reference pipelines, red/teal/purple = low-rank detection models, orange = NB counts, green/pink/light blue = structured detection models.",
    ]),
    S("Every model in one table", E3 + "v3_models.png"),
    S("R3–R5: low-rank detection models (BernFac family)", B4 + "v4_bernfac_family.png", [
        "R3 is a logistic PCA. R4 adds a cell and a gene offset. R5 swaps the logistic link for the complementary log-log, under which p = 0.63 at ψ = 0 and depth acts as a rate.",
    ]),
    S("R6: factorize the counts themselves", "biorender/bernfac_5_count_models.png", [
        "Negative-binomial factorization of raw UMI counts with log library size as an offset; essentially an NB GLM-PCA. Detection probability is Pr(count ≥ 1), scored on the same entries as the binary models. Poisson is fitted the same way.",
    ]),
    S("R7–R9: structured detection models", B4 + "v4_structured_models.png", [
        "R7: each cell's logits lie near one of K local subspaces (K = 1 chosen). R8: a learned coordinate z per cell and a shared decoder. R9: cells and genes are co-clustered, one probability per block.",
    ]),
    S("Where sequencing depth goes in each model", B4 + "v4_depth_schematic.png", [
        "R4–R6 give depth its own cell offset, so their probabilities can be recomputed at a common depth. R3 and R7–R9 absorb depth into their latent factors. R1 divides by library size, which removes depth only partly.",
    ]),
    S("Observed and depth-adjusted probabilities", E3 + "v3_depth.png", [
        "p̂ (each cell at its own depth) is what is scored. p̃ (cell offset at the median) is a depth-adjusted view; it is not a prediction and is never scored by NLL.",
    ]),
    S("Fitting every model", E3 + "v3_inference.png"),
    S("How every model is evaluated", B4 + "v4_evaluation_design.png", [
        "Held-out entries scattered over the whole matrix, chosen on validation, scored on test with a paired bootstrap over cells and genes. Synthetic worlds with known truth. One embedding protocol for everyone.",
    ]),
    S("Prediction metrics and uncertainty", E4 + "v4_metrics_pred.png"),
    S("Representation metrics", E4 + "v4_metrics_rep.png"),
    S("What is tuned and what was chosen", E4 + "v4_hyperparams.png", [
        "R8, Poisson and R9 were chosen at the edge of their grids and are flagged wherever they appear.",
    ]),
    # ------------------------------------------------------------ synthetic
    section("Part 2", "Synthetic data: which model recovers the truth?",
            ["Five worlds, each built in the image of one model family, 3 seeds each. Every grid contains the true rank, and every configuration was scored against the truth."]),
    S("Five synthetic worlds with known probabilities", F3 + "fig_syn_design.png", [
        "2,000 cells × 300 genes, mean detection 0.18. In G1, G4 and G5 the counts behind Y are artificial (Y × (1 + Poisson(1))); a control with genuine counts is shown on the next slides.",
    ]),
    S("Tuning on the synthetic worlds", F4 + "fig_syn_select_v4.png", [
        "Validation loss for every configuration, the chosen one boxed (orange at a grid edge) and the best in hindsight starred. The factor models land on the true rank; R7, R8 and R9 often pick an edge value.",
    ]),
    S("Distance to the true probabilities", F4 + "fig_syn_kl_v4.png", [
        "Each family wins the world built for it: NB counts in G2, the neural manifold in G4, latent blocks in G5; the mixture leads G1 and G3. Validation picks the best-in-hindsight setting almost everywhere. With genuine counts in G1, NB counts become the best method there.",
    ]),
    S("Do the embeddings recover the true cell groups?", F4 + "fig_syn_embed_v4.png", [
        "Processed counts look best only when they see every entry. Given the same 70% of entries as the models, the count models lead on cell types and the mixture leads on blocks.",
    ]),
    S("Does depth adjustment recover the truth?", F4 + "fig_syn_depthstd.png", [
        "Setting each cell's offset to the median recovers the true depth-adjusted probabilities about as well as the model recovers the observed ones. R4 leaves the least depth; Poisson leaves depth in NB worlds; R6 and Poisson over-correct in G4.",
    ]),
    # ------------------------------------------------------------ prediction
    section("Part 3", "Real data: held-out prediction",
            ["Settings chosen on 8.05 M validation entries; 8.04 M test entries; paired two-way bootstrap over cells and genes."]),
    S("Model selection for every family", F4 + "fig_select_all_v4.png", [
        "Validation NLL across each grid after the refits, with grid edges marked. NB k = 160 is worse than k = 120 because it overfits (warm and cold starts agree). Larger R4 configurations get worse. Poisson improves to k = 120 but stays far behind NB.",
    ]),
    S("Held-out prediction with 95% intervals", F4 + "fig_test_ci.png", [
        "NB counts beat every other model by about 1 × 10⁻³ nats per entry. Among the binary models R7 < R4 < R3 are resolvable but within 0.2 × 10⁻³; R4, R5 and R8 cannot be separated.",
    ]),
    S("Where NB's lead comes from", F4 + "fig_test_ci_deciles.png", [
        "The lead over R4 holds at every sequencing depth and comes from frequently detected genes; about 230 bursty wiring and TF genes (beat, DIP, dpr families, zld, RunxA) carry half of it. In the rarest genes the mixture and neural models are marginally better.",
    ]),
    S("Refitted grid edges", F4 + "fig_test_ci_v4refits_nll.png", [
        "Extended grids change no conclusion. With an iteration budget instead of a time limit, the neural manifold at d = 40 moves ahead of R4 and ties the mixture; its setting is at a grid corner and not converged.",
    ]),
    S("Why NB beats Poisson", F4 + "fig_nb_vs_poisson.png", [
        "At the same rank NB beats Poisson by 4 to 9 × 10⁻³ nats. The gain is on observed zeros of bursty genes: Poisson ties detection to the mean count. The fitted dispersion runs to its bound mainly for rare genes, where it cannot be estimated.",
    ]),
    S("Calibration on test entries", F3 + "fig_calibration_all.png", [
        "All models except Poisson are calibrated to within 0.7 percentage points. Poisson over-predicts above p̂ ≈ 0.05, most near 0.89 (+2.3 points). Models differ only in the extreme tails (p̂ < 10⁻⁴ or > 0.999), not in the share of p̂ below 0.01 (51–53% for all).",
    ]),
    # ------------------------------------------------------------ representations
    section("Part 4", "Cell maps under one protocol",
            ["Every embedding centred, rotated to principal axes and compared at 20 and 40 dimensions, unwhitened and whitened. Clustering at a fixed 23 clusters over 5 seeds."]),
    S("Coloured by hemilineage", F4 + "fig_umap_grid_hemilineage_v4.png", [
        "UMAPs of all eleven representations with the same settings; R3–R5 from the train-only fits. For illustration only: the quantitative comparison is on the next slides.",
    ]),
    S("Coloured by sequencing depth", F4 + "fig_umap_grid_depth_v4.png", [
        "A high-depth region is visible in R3 and R7 and weaker in R1, R6, R10 and R11. The deepest cells are mostly primary neurons, so part of this is biology.",
    ]),
    S("Hemilineage under a fair protocol", F4 + "fig_rep_fair.png", [
        "Hemilineage information is nearly equal across models (linear probe 0.97–0.99 at d = 40); whitening closes most of the kNN and clustering gap that v3 attributed to the models. At 40 whitened dimensions NB counts gives the best 23-cluster agreement (ARI 0.74); at 20, processed counts leads.",
    ]),
    S("How much depth each embedding carries", F4 + "fig_depth_axes_v4.png", [
        "Depth measured the same way for all on principal axes. A low first-axis correlation can hide depth: R8's top axes predict depth with R² ≈ 0.9. Within hemilineage × birth time, Pearson residuals, scVI and NB counts keep the least depth.",
    ]),
    S("Library, sex and birth time", F4 + "fig_confounds.png", [
        "Library signal within hemilineage is near chance for gene-subset methods (R1, R10, R11) and about 0.2 for all-gene factor models. The sex label is almost exactly roX expression (AUC 0.998); within mixed-sex libraries only the all-gene factor models decode it.",
    ]),
    # ------------------------------------------------------------ cells x genes
    section("Part 5", "Cells × genes: observed and depth-adjusted",
            ["Five hemilineages, shallow and deep cells, marker, depth-sensitive, rare and ubiquitous genes. These slides illustrate; the quantitative checks follow them."]),
    S("What each model predicts", F3 + "fig_heat_overview.png", [
        "All probability models reproduce the hemilineage marker blocks and fill dropouts within them. R9 blurs them because one probability is shared by a whole block.",
    ]),
    S("Observed and depth-adjusted probabilities", F3 + "fig_heat_std_overview.png", [
        "R4–R6 shown observed and with every cell at the median depth. Shallow and deep cells of one hemilineage look alike after adjustment, and markers stay. R3 and R7–R9 have no depth term to adjust.",
    ]),
    S("Depth-sensitive genes, chosen on held-out cells", F4 + "fig_std_depthgenes_heldout.png", [
        "Genes chosen on one half of the cells and scored on the other give the same depth gap, so the v3 result was not a selection artefact. Adjustment cuts the gap from 0.40 to 0.23 (R4), 0.15 (R5) and 0.13 (R6). About a third of the gap is birth time.",
    ]),
    S("How much depth the adjusted probabilities keep", F4 + "fig_std_rigor.png", [
        "p̃ is depth-adjusted, not depth-free: depth R² from its top axes is 0.41 (R6), 0.66 (R5) and 0.87 (R4), against 0.61 for log-normalised counts. Marker structure is kept (variance ratio 0.98–1.02). p̃ also removes most within-hemilineage birth-time variance on typical genes, so over-correction cannot be ruled out on real data.",
    ]),
    S("Choice of the reference depth", F4 + "fig_std_reference.png", [
        "Moving the reference from the 10th to the 90th depth percentile shifts the level of p̃ (mean 0.15 to 0.23); gene-level depth correlations, hemilineage profiles and the ordering R6 < R5 < R4 do not change.",
    ]),
    # ------------------------------------------------------------ conclusion
    section("Part 6", "Which one to work with?"),
    S("Scorecard", F4 + "fig_scorecard_v4.png", [
        "One definition per column, all from the v4 analyses. Prediction differences carry 95% bootstrap intervals; Leiden values are mean ± sd over 5 seeds.",
    ]),
    S("Recommendation", E4 + "v4_recommendation.png"),
]

APPENDIX = [
    section("Appendix", "Derivations, diagnostics and earlier results",
            ["A: model equations. B: diagnostics. C: v3 views kept for reference. D: 1000 × 100 pilot. E: BernFac synthetic model-selection study."]),
    S("A · Shared notation", E3 + "v3_notation.png"),
    S("A · BernFac objective", "equations/eq_model.png"),
    S("A · Offsets (R4)", "equations/eq_offsets.png"),
    S("A · Complementary log-log (R5)", "equations/eq_cloglog.png"),
    S("A · Count factorization (R6, Poisson)", "equations/eq_counts.png"),
    S("A · Mixture of local factors (R7)", "equations/eq_mfa.png"),
    S("A · Neural manifold (R8)", "equations/eq_neural.png"),
    S("A · Latent block model (R9)", "equations/eq_lbm.png"),
    S("A · Pólya–Gamma EM", "biorender/bernfac_2_pgem_inference.png"),
    S("B · Convergence (v3 fits)", F3 + "fig_converge_all.png", [
        "Every objective went one way only. The v3 neural fits stopped at a 1,500 s limit; the v4 refits use a 250-iteration budget.",
    ]),
    S("B · Run time (v3 fits)", F3 + "fig_runtime.png", [
        "Warm-started and cold-started fits are mixed, so times are indicative only.",
    ]),
    S("B · Tests", F3 + "fig_tests.png", [
        "All suites pass. The offsets and cloglog fits (R4, R5) have no unit tests of their own.",
    ]),
    S("B · Synthetic tuning, v3 grids", F3 + "fig_syn_select.png", [
        "Superseded by the v4 grids, which contain the true rank. Kept to show why the v3 synthetic conclusions changed.",
    ]),
    S("C · Marker genes", F3 + "fig_heat_markers.png"),
    S("C · Depth-sensitive genes (in-sample selection)", F3 + "fig_heat_depth.png"),
    S("C · Difference from the empirical rate", F3 + "fig_heat_diff.png"),
    S("C · What depth adjustment changes", F3 + "fig_heat_std_diff.png"),
    S("C · UMAP coloured by birth time", F3 + "fig_umap_grid_birthtime.png"),
    S("C · UMAP coloured by library", F3 + "fig_umap_grid_library.png"),
    S("C · Zoom: hemilineage pairs", F3 + "fig_umap_zoom_hemilineage_pairs.png", [
        "v3 view on unwhitened coordinates; the v4 fair-protocol slide supersedes the error rates quoted in v3.",
    ]),
    *[old(n, "D") for n in range(9, 21)],
    *[old(n, "E") for n in range(21, 30)],
]

SUMMARY = [
    {"kind": "title", "title": "Nine ways to represent a single-cell matrix: the takeaways",
     "bullets": ["Fly premotor neurons, 24 h APF · 16,460 cells × 19,543 genes",
                 "Summary of the v4 deck (after a strict review) · ConnectionMiner · 25 Oct 2026"]},
    S("The question and the models", B4 + "v4_summary_overview.png", [
        "Nine representations of one matrix, compared on held-out prediction, synthetic ground truth and cell maps under one protocol, with uncertainty.",
    ]),
    S("1. NB counts predict held-out detection best, and the lead is real", F4 + "fig_test_ci.png", [
        "About 1 × 10⁻³ nats per entry ahead of every other model, with 95% intervals that exclude zero. The binary models are within 0.2 × 10⁻³ of each other.",
    ]),
    S("2. The lead comes from bursty genes that Poisson cannot fit", F4 + "fig_nb_vs_poisson.png", [
        "NB beats Poisson at every matched rank. The gain is on observed zeros of frequently detected wiring and TF genes, where overdispersion lets a high mean coexist with a zero.",
    ]),
    S("3. On synthetic data, each family wins the world built for it", F4 + "fig_syn_kl_v4.png", [
        "With grids that contain the true rank, validation picks the best setting almost everywhere. NB counts win the NB world; with genuine counts they also win the low-rank logit world.",
    ]),
    S("4. All embeddings carry hemilineage; geometry is what differs", F4 + "fig_rep_fair.png", [
        "A linear probe reads hemilineage at 0.97–0.99 from every model. Under whitening at 40 dimensions NB counts clusters best (ARI 0.74); processed counts lead at 20.",
    ]),
    S("5. Depth is partly biology, and it hides in different places", F4 + "fig_depth_axes_v4.png", [
        "A third of depth variance is hemilineage and birth time. Measured within those strata, scVI, Pearson residuals and NB counts keep the least depth; the binary models keep the most.",
    ]),
    S("6. Depth-adjusted probabilities help, within limits", F4 + "fig_std_rigor.png", [
        "Setting the cell offset to the median keeps marker structure and recovers the truth on synthetic data. On real data NB's p̃ still carries some depth and also removes birth-time variation.",
    ]),
    S("Scorecard", F4 + "fig_scorecard_v4.png"),
    S("Recommendation and open questions", E4 + "v4_recommendation.png"),
]


def check(slides):
    imgs = [i for s in slides for i in ([s["image"]] if "image" in s else s.get("images", []))]
    return [i for i in imgs if not (STUDY / i).exists()]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jpeg = {"max_w": 1800, "quality": 76}
    n_main = len(MAIN)
    fix_refs(MAIN, APPENDIX)
    for name, slides, start in (("main", MAIN, 1), ("appendix", APPENDIX, n_main + 1), ("summary", SUMMARY, 1)):
        spec = {"out": str(OUT / f"NineRepresentations_v4_{name}.pptx"), "start": start, "jpeg": jpeg, "slides": slides}
        (OUT / f"slides_{name}.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False))
        print(name, len(slides), "slides; missing:", check(slides))
    if len(SUMMARY) > 10:
        sys.exit("summary deck exceeds 10 slides")
