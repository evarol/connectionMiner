"""Which representation reflects biology best? Grouped bars for R1/R2/R3, as is and with
log-depth regressed out of the embedding (umap/metrics_depth_regressed.csv)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figstyle import PALETTE, apply_publication_style, finalize_figure

OUT = Path(__file__).resolve().parents[3] / "outputs/bernfac_20261020/full"
NAMES = {"R1": "processed counts\n(log-norm, HVG, PCA)", "R2": "binary $Y$\n(PCA)", "R3": r"BernFac $\hat P$" + "\n" + r"($\lambda=10,\ k=40$)",
         "R4": r"BernFac + offsets" + "\n" + r"(logit link)",
         "R5": r"BernFac + offsets" + "\n" + r"(cloglog link)",
         "R6": "NB count\nfactorization", "R7": "mixture of\nlocal factors", "R8": "neural\nmanifold",
         "R9": "latent block\nmodel"}
METRICS = [("hl_knn_balacc", "hemilineage\nkNN bal. acc. ↑"), ("hl_leiden_best_ARI", "hemilineage\nLeiden ARI ↑"),
           ("hl_silhouette_cos", "hemilineage\nsilhouette ↑"), ("hl_nn_purity", "15-NN hemilineage\npurity ↑"),
           ("lib_knn_balacc", "library kNN bal. acc.\n(chance 0.05) ↓"), ("depth_max_abs_top2", r"$|\rho|$ top-2 comps vs" + "\nlog depth ↓")]


def main():
    d = pd.read_csv(OUT / "umap/metrics_depth_regressed.csv")
    apply_publication_style()
    fig, axes = plt.subplots(1, len(METRICS), figsize=(22, 5.0))
    cols = {"R1": PALETTE["blue_main"], "R2": PALETTE["neutral"], "R3": PALETTE["red_strong"], "R4": PALETTE["teal"], "R5": PALETTE["violet"],
            "R6": "#E69F00", "R7": "#009E73", "R8": "#CC79A7", "R9": "#56B4E9"}
    x = np.arange(len(NAMES))
    for ax, (m, lab) in zip(axes, METRICS):
        for j, var in enumerate(["as is", "depth regressed"]):
            v = [d[(d.rep == r) & (d.variant == var)][m].item() for r in NAMES]
            ax.bar(x + (j - 0.5) * 0.4, v, 0.38, color=[cols[r] for r in NAMES],
                   alpha=1.0 if j == 0 else 0.45, edgecolor="k", lw=0.6,
                   hatch=None if j == 0 else "///")
        ax.set_xticks(x, [r[1:] for r in NAMES], fontsize=10); ax.set_title(lab, fontsize=11)
        ax.axhline(0, color="k", lw=0.8)
    from matplotlib.patches import Patch
    handles = [Patch(color=cols[r], label=NAMES[r].replace("\n", " ")) for r in NAMES]
    handles += [Patch(facecolor="white", edgecolor="k", label="as is"),
                Patch(facecolor="white", edgecolor="k", hatch="///", label="log-depth regressed out")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.17))
    fig.suptitle("Which representation reflects biology best? (16,460 cells, 23 hemilineages, 21 libraries)",
                 fontsize=14, y=1.03)
    finalize_figure(fig, OUT / "figures/fig_full_rep_comparison")


if __name__ == "__main__":
    main()
