"""Side-by-side view of the fly preMN data: log1p(counts) | binary Y | fitted
Bernoulli probabilities P = sigma(U V'), for every one of the 1000 x 100 entries.

fig10_side_by_side       : full matrices (same cell / gene order as Fig. 1)
fig11_side_by_side_zoom  : a 60-cell x 40-gene window, so single entries are visible

The probability panels use the CV-best fit (lambda=10, k=20) and the compact,
reproducible fit (lambda=1, k=5).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from common import (HEMI_COLORS, HEMILINEAGES, LIB_COLORS, LIBRARIES, apply_publication_style,
                    display_order, finalize_figure, load_data, load_fit)

FITS = [(10.0, 20, "CV-best"), (1.0, 5, "compact")]


def annotation_bars(fig, ax_h, ax_l, cells):
    for ax, col, names, colors, label in [(ax_h, "hemilineage", HEMILINEAGES, HEMI_COLORS, "hemilineage"),
                                          (ax_l, "library", LIBRARIES, LIB_COLORS, "library")]:
        idx = cells[col].map({n: i for i, n in enumerate(names)}).to_numpy()[:, None]
        ax.imshow(idx, aspect="auto", interpolation="nearest",
                  cmap=ListedColormap([colors[n] for n in names]), vmin=-0.5, vmax=len(names) - 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label[:4] + ".", rotation=90, fontsize=10, loc="center", pad=4)
        for s in ax.spines.values():
            s.set_visible(False)


def panels(cells, C, Y, Ps, title_suffix, fname, figsize, hline_every=None):
    apply_publication_style()
    fig = plt.figure(figsize=figsize)
    n_main = 2 + len(Ps)
    gs = fig.add_gridspec(2, 2 + n_main, width_ratios=[0.035, 0.035] + [1] * n_main,
                          height_ratios=[1, 0.035], wspace=0.13, hspace=0.12)
    annotation_bars(fig, fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), cells)
    mats = [(np.log1p(C), "viridis", r"$\log(1+\mathrm{UMI\ count})$", (0, np.log1p(C).max()),
             "log-transformed counts"),
            (Y, ListedColormap(["white", "black"]), r"$y_{ij}=\mathbf{1}[\mathrm{count}\geq 1]$", (0, 1),
             "binary matrix $Y$")]
    mats += [(P, "magma", r"$\hat p_{ij}=\sigma(\hat{\mathbf{u}}_i^{\top}\hat{\mathbf{v}}_j)$", (0, 1),
              rf"$\hat P$, $\lambda={lam:g},\ k={k}$ ({tag})") for (lam, k, tag), P in Ps]
    for j, (M, cmap, cbar_label, (vmin, vmax), title) in enumerate(mats):
        ax = fig.add_subplot(gs[0, 2 + j])
        im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"{M.shape[1]} genes (clustered)", fontsize=11)
        if hline_every:
            for y in np.arange(hline_every, M.shape[0], hline_every) - 0.5:
                ax.axhline(y, color="white" if j != 1 else PALETTE_EDGE, lw=0.8)
        cax = fig.add_subplot(gs[1, 2 + j])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(cbar_label, fontsize=11)
        cb.ax.tick_params(labelsize=9)
        if j == 1:                                    # binary: two labelled levels
            cb.set_ticks([0.25, 0.75], labels=["0  not detected", "1  detected"])
        else:
            cb.ax.xaxis.set_major_locator(plt.MaxNLocator(5, prune="both" if j else None))
    fig.suptitle(f"Fly motor preMN, 24 h APF: {title_suffix}", fontsize=15, y=0.985)
    finalize_figure(fig, fname)


PALETTE_EDGE = "#3775BA"


def main():
    d = load_data()
    C, Y, cells = d["counts"], d["Y_ge1"], d["cells"]
    row, col = display_order(cells, Y)
    Ps = [((lam, k, tag), load_fit("Y_ge1", lam, k)["P"]) for lam, k, tag in FITS]

    # full 1000 x 100 matrices; white rules separate the 5 hemilineage blocks
    o = lambda M: M[np.ix_(row, col)]
    panels(cells.iloc[row].reset_index(drop=True), o(C), o(Y), [(f, o(P)) for f, P in Ps],
           "every entry of the 1000 cells $\\times$ 100 genes matrix", "fig10_side_by_side",
           (17, 8.2), hline_every=200)

    # zoom: 12 cells per hemilineage (first library block) x 40 central genes
    rows_z = np.concatenate([row[h * 200:h * 200 + 12] for h in range(5)])
    cols_z = col[30:70]
    z = lambda M: M[np.ix_(rows_z, cols_z)]
    panels(cells.iloc[rows_z].reset_index(drop=True), z(C), z(Y), [(f, z(P)) for f, P in Ps],
           "zoom on 60 cells $\\times$ 40 genes (12 cells per hemilineage)", "fig11_side_by_side_zoom",
           (17, 8.2), hline_every=12)


if __name__ == "__main__":
    main()
