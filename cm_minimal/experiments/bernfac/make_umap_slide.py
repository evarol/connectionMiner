"""Slide-shaped UMAP figure: columns = representation (R1-R4), rows = colour by hemilineage / log depth.
Reads cached UMAP coordinates written by umap_compare.py."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import umap_compare as uc
from figstyle import apply_publication_style, finalize_figure

TITLES = {"R1": "R1 processed counts\n(log-norm, HVG, PCA)", "R2": "R2 binary $Y$ (PCA)",
          "R3": r"R3 BernFac $\hat P$", "R4": "R4 + offsets\n(logit link)", "R5": "R5 + offsets\n(cloglog link)",
          "R6": "R6 NB counts\n(k=120)", "R7": "R7 mixture of\nlocal factors (d=40)", "R8": "R8 neural\nmanifold (z, d=20)",
          "R9": "R9 latent blocks\n(R B, 200 groups)"}


def main(reps=("R1", "R2", "R3", "R4", "R5"), name="fig_full_umap_slide"):
    X, cells, _ = uc.load_data()
    logumi = np.log10(np.asarray(X.sum(1)).ravel())
    reps = list(reps)
    um = {r: np.load(uc.CACHE / f"{r}_umap.npy") for r in reps}
    hl = sorted(cells.hemilineage.unique())
    col = dict(zip(hl, uc.QUAL23))
    c_hl = cells.hemilineage.map(col).to_numpy()
    apply_publication_style()
    fig, axes = plt.subplots(2, len(reps), figsize=(4.2 * len(reps), 9.0))
    order = np.random.default_rng(0).permutation(len(cells))
    for j, r in enumerate(reps):
        Z = um[r][order]
        axes[0, j].scatter(Z[:, 0], Z[:, 1], c=c_hl[order], s=0.6, lw=0, rasterized=True)
        sc = axes[1, j].scatter(Z[:, 0], Z[:, 1], c=logumi[order], s=0.6, lw=0, cmap="viridis",
                                vmin=np.percentile(logumi, 1), vmax=np.percentile(logumi, 99), rasterized=True)
        axes[0, j].set_title(TITLES[r], fontsize=13)
        for ax in axes[:, j]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    axes[0, 0].set_ylabel("hemilineage", fontsize=13, weight="bold")
    axes[1, 0].set_ylabel(r"$\log_{10}$ total UMI", fontsize=13, weight="bold")
    cb = fig.colorbar(sc, ax=axes[1, :], fraction=0.015, pad=0.01)
    cb.set_label(r"$\log_{10}$ total UMI")
    handles = [Line2D([], [], marker="o", ls="", ms=6, color=col[h], label=h) for h in hl]
    fig.legend(handles=handles, loc="lower center", ncol=12, frameon=False, fontsize=10,
               bbox_to_anchor=(0.45, -0.06), title="hemilineage")
    finalize_figure(fig, uc.FULL / f"figures/{name}")


if __name__ == "__main__":
    main()
    main(("R1", "R5", "R6", "R7", "R8", "R9"), "fig_full_umap_slide_new_models")
