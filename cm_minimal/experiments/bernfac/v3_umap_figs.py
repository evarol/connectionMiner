"""Worker U: 3x3 UMAP galleries, zooms, drivers heatmap and metric scorecard for the v3 deck."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Rectangle
from figstyle import (PALETTE, FS, FS_LEGEND, FS_SMALL, FS_TICK, FS_TINY, apply_publication_style,
                      finalize_figure)
from umap_compare import QUAL23
from v3style import ROOT, STUDY, V3, FIGS, ORDER, name, colour

apply_publication_style()
CACHE = STUDY / "full/umap/cache"
TAB = V3 / "tables"
cells = pd.read_csv(ROOT / "outputs/connectionminer_ready_inputs/bundles/motor/premn_24h/cells.csv.gz")
cells["birthtime"] = cells["birthtime"].fillna("precursor")
dep = pd.read_csv(TAB / "cell_depth.csv.gz")
assert (dep.cell_id.values == cells.cell_id.values).all()
cells["log10_umi"] = dep.log10_umi.values
U = {r: np.load(CACHE / f"{r}_umap.npy") for r in ORDER}
N = len(cells)
RNG = np.random.default_rng(0)
PERM = RNG.permutation(N)
BG = "#E4E4E4"


def style_ax(ax, r):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.annotate(r, xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                fontweight="bold", color=colour(r), fontsize=FS, va="bottom", ha="left")
    ax.annotate(name(r), xy=(0, 1.0), xycoords="axes fraction", xytext=(27, 4), textcoords="offset points",
                fontsize=FS, va="bottom", ha="left", color=PALETTE["grey_dark"])


def grid_fig():
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 3, left=0.0, right=0.74, bottom=0.0, top=0.95, wspace=0.05, hspace=0.16)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(9)]
    lax = fig.add_axes([0.765, 0.0, 0.235, 0.95]); lax.axis("off")
    return fig, axes, lax


def cat_grid(values, cmap, out, title, order_cats=None, s=0.8, ncol=1, under=None):
    """values: array of category labels; cmap: dict cat->colour. under: categories drawn first."""
    fig, axes, lax = grid_fig()
    v = np.asarray(values).astype(str)
    idx = PERM
    if under is not None:  # draw majority categories first, minorities on top
        first = np.isin(v[idx], under)
        idx = np.concatenate([idx[first], idx[~first]])
    cols = np.array([cmap[x] for x in v[idx]], dtype=object)
    for ax, r in zip(axes, ORDER):
        Z = U[r][idx]
        ax.scatter(Z[:, 0], Z[:, 1], c=list(cols), s=s, lw=0, rasterized=True, antialiased=False)
        style_ax(ax, r)
    cats = order_cats or sorted(cmap)
    counts = pd.Series(v).value_counts()
    h = [Line2D([], [], marker="o", ls="", ms=8, mfc=cmap[k], mec="none") for k in cats]
    lab = [f"{k}  ({counts.get(k, 0):,})" for k in cats]
    lax.legend(h, lab, loc="upper left", ncol=ncol, title=title, fontsize=FS_LEGEND if ncol == 1 else FS_LEGEND - 1,
               title_fontsize=FS, handletextpad=0.2, columnspacing=0.6, labelspacing=0.35, alignment="left")
    lax.text(0, 0.0, f"UMAP of each embedding\n{N:,} cells; axes arbitrary", fontsize=FS_SMALL,
             color=PALETTE["grey_mid"], va="bottom")
    finalize_figure(fig, FIGS / out)


def hl_cmap():
    hl_u = sorted(set(cells.hemilineage), key=lambda s: (len(s), s))
    return dict(zip(hl_u, QUAL23)), hl_u


def fig_hemilineage():
    cm, hl_u = hl_cmap()
    cat_grid(cells.hemilineage, cm, "fig_umap_grid_hemilineage", "hemilineage", hl_u, ncol=2)


def fig_library():
    lib_u = sorted(set(cells.library))
    t20 = plt.get_cmap("tab20")
    cm = dict(zip(lib_u, [t20(i % 20) for i in range(20)] + [PALETTE["grey_dark"]]))
    cat_grid(cells.library, cm, "fig_umap_grid_library", "library (21 lanes)", lib_u, ncol=2)


def fig_sex():
    cm = {"F": "#B64342", "M": "#0F4D92"}
    cat_grid(cells.Sex, cm, "fig_umap_grid_sex", "sex", ["F", "M"], s=0.8)


def fig_birthtime():
    cm = {"secondary": "#C9C9C9", "primary": "#B64342", "earlysecondary": "#0F4D92", "precursor": "#E69F00"}
    cat_grid(cells.birthtime, cm, "fig_umap_grid_birthtime", "birth time",
             ["primary", "earlysecondary", "secondary", "precursor"], s=1.0, under=["secondary"])


def fig_depth():
    fig, axes, lax = grid_fig()
    x = cells.log10_umi.values
    lo, hi = np.percentile(x, [1, 99])
    idx = PERM
    for ax, r in zip(axes, ORDER):
        Z = U[r][idx]
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=x[idx], s=0.8, lw=0, cmap=plt.get_cmap("viridis", 24), vmin=lo, vmax=hi, rasterized=True, antialiased=False)
        style_ax(ax, r)
    cax = fig.add_axes([0.80, 0.30, 0.025, 0.55])
    cb = fig.colorbar(sc, cax=cax, extend="both")
    cb.set_label(r"$\log_{10}$ total UMI per cell", fontsize=FS)
    cb.ax.tick_params(labelsize=FS_TICK)
    lax.text(0, 0.0, f"colour clipped to 1st-99th pct\n{N:,} cells", fontsize=FS_SMALL,
             color=PALETTE["grey_mid"], va="bottom")
    finalize_figure(fig, FIGS / "fig_umap_grid_depth")


# ---------------------------------------------------------------- zooms
def zoom_fig(mask, values, cmap_or_cont, out, title, cats=None, crop=True, s_fg=3.0, cbar_label=None):
    fig, axes, lax = grid_fig()
    sub = np.where(mask)[0]
    sub = sub[np.random.default_rng(1).permutation(len(sub))]
    for ax, r in zip(axes, ORDER):
        Z = U[r]
        ax.scatter(Z[:, 0], Z[:, 1], c=BG, s=0.5, lw=0, rasterized=True, antialiased=False)
        if cats is not None:
            ax.scatter(Z[sub, 0], Z[sub, 1], c=[cmap_or_cont[v] for v in values[sub]], s=s_fg, lw=0, rasterized=True, antialiased=False)
        else:
            vmin, vmax, cm = cmap_or_cont
            sc = ax.scatter(Z[sub, 0], Z[sub, 1], c=values[sub], s=s_fg, lw=0, cmap=cm, vmin=vmin, vmax=vmax, antialiased=False,
                            rasterized=True)
        if crop:
            q = np.percentile(Z[sub], [0.5, 99.5], axis=0)
            pad = 0.08 * (q[1] - q[0])
            ax.set_xlim(q[0, 0] - pad[0], q[1, 0] + pad[0]); ax.set_ylim(q[0, 1] - pad[1], q[1, 1] + pad[1])
        style_ax(ax, r)
    if cats is not None:
        counts = pd.Series(values[mask]).value_counts()
        h = [Line2D([], [], marker="o", ls="", ms=9, mfc=cmap_or_cont[k], mec="none") for k in cats]
        lax.legend(h, [f"{k}  ({counts.get(k, 0):,})" for k in cats], loc="upper left", title=title,
                   title_fontsize=FS, fontsize=FS_LEGEND + 1, alignment="left")
    else:
        cax = fig.add_axes([0.80, 0.35, 0.025, 0.5])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(cbar_label, fontsize=FS); cb.ax.tick_params(labelsize=FS_TICK)
        lax.text(0, 0.97, title, fontsize=FS, va="top")
    lax.text(0, 0.0, f"{mask.sum():,} highlighted cells;\nothers in grey" + ("\npanels cropped to subset" if crop else ""),
             fontsize=FS_SMALL, color=PALETTE["grey_mid"], va="bottom")
    finalize_figure(fig, FIGS / out)


def zooms():
    hl = cells.hemilineage.values
    cm, _ = hl_cmap()
    # (a) method-dependent hemilineage pairs 05B/06B and 13A/19A
    za = ["05B", "06B", "13A", "19A"]
    zcm = {"05B": "#B64342", "06B": "#E69F00", "13A": "#0F4D92", "19A": "#009E73"}
    zoom_fig(np.isin(hl, za), hl, zcm, "fig_umap_zoom_hemilineage_pairs", "hemilineage", cats=za)
    # (b) depth extremes: bottom and top decile of total UMI
    x = cells.log10_umi.values
    q10, q90 = np.percentile(x, [10, 90])
    grp = np.where(x <= q10, "lowest 10% UMI", np.where(x >= q90, "highest 10% UMI", "mid"))
    zoom_fig(grp != "mid", grp, {"lowest 10% UMI": "#0F4D92", "highest 10% UMI": "#B64342"},
             "fig_umap_zoom_depth_extremes", "sequencing depth", cats=["lowest 10% UMI", "highest 10% UMI"],
             crop=False, s_fg=1.6)
    # (c) 14A split by birth time
    bt = cells.birthtime.values
    m = hl == "14A"
    zoom_fig(m, bt, {"primary": "#B64342", "secondary": "#0F4D92", "earlysecondary": "#009E73", "precursor": "#E69F00"},
             "fig_umap_zoom_14A_birthtime", "14A by birth time", cats=["primary", "secondary", "precursor"])


# ---------------------------------------------------------------- drivers heatmap
def drivers():
    d = pd.read_csv(TAB / "umap_drivers_full.csv").set_index("rep").loc[ORDER]
    cols = [("hemilineage_knn_balacc", "hemilineage", 1 / 23), ("library_knn_balacc", "library", 1 / 21),
            ("Sex_knn_balacc", "sex", 1 / 2), ("birthtime_knn_balacc", "birth time", 1 / 4),
            ("birthtime_within_hl_balacc", "birth time\nwithin HL", 0.35),
            ("neurotransmitter_knn_balacc", "neuro-\ntransmitter", 1 / 3),
            ("depth_abs_rho_top5", "log depth", None), ("detect_abs_rho_top5", "detection\nrate", None),
            ("depth_abs_rho_pc1", r"log depth" + "\n(comp. 1)", None)]
    out = d[[c for c, _, _ in cols]].copy()
    out.insert(0, "name", [name(r) for r in ORDER])
    out.to_csv(TAB / "umap_drivers.csv", float_format="%.4f")
    M = out[[c for c, _, _ in cols]].values
    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    nk = 6
    # colour = fraction of the way from chance to 1 for kNN; raw |rho| for correlations
    C = M.copy()
    for j, (_, _, ch) in enumerate(cols):
        if ch is not None:
            C[:, j] = (M[:, j] - ch) / (1 - ch)
    xs = np.array([j + (0.35 if j >= nk else 0) for j in range(len(cols))])
    for j in range(len(cols)):
        cmap = plt.get_cmap("Blues" if j < nk else "Reds")
        for i in range(9):
            ax.add_patch(Rectangle((xs[j] - 0.5, i - 0.5), 1, 1, fc=cmap(0.08 + 0.8 * np.clip(C[i, j], 0, 1)),
                                   ec="white", lw=1.5))
            ax.text(xs[j], i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=FS_TICK,
                    color="white" if C[i, j] > 0.6 else PALETTE["grey_dark"])
    ax.set_xlim(-0.5, xs[-1] + 0.5); ax.set_ylim(8.5, -0.5)
    ax.set_yticks(range(9), [f"{r}  {name(r)}" for r in ORDER])
    for t, r in zip(ax.get_yticklabels(), ORDER):
        t.set_color(colour(r)); t.set_fontweight("bold")
    ax.set_xticks(xs, [lab + (f"\n(chance {ch:.2f})" if ch and ch != 0.35 else ("\n(chance ~0.35)" if ch else "")) for _, lab, ch in cols], fontsize=FS_TICK - 1)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(np.mean(xs[:nk]), -2.15, "what the embedding encodes: kNN balanced accuracy (15-NN, 5-fold)",
            ha="center", fontsize=FS, color="#0F4D92", fontweight="bold")
    ax.text(np.mean(xs[nk:]), -2.15, r"depth in the axes:" + "\n" + r"max $|\rho|$, top-5 comps", ha="center", fontsize=FS,
            color="#B64342", fontweight="bold")
    finalize_figure(fig, FIGS / "fig_umap_drivers")


# ---------------------------------------------------------------- metric scorecard
def scorecard():
    m = pd.read_csv(STUDY / "full/umap/metrics.csv").set_index("rep").loc[ORDER]
    cols = [("hl_knn_balacc", "hemilineage\nkNN bal. acc.", True), ("hl_leiden_best_ARI", "Leiden\nARI (best res.)", True),
            ("hl_silhouette_cos", "silhouette\n(cosine)", True), ("hl_nn_purity", "15-NN\npurity", True),
            ("lib_knn_balacc", "library kNN\nbal. acc. ↓", False), ("lib_mix_rel", "library\nmixing", True),
            ("depth_max_abs_top2", r"depth $|\rho|$" + "\n(top-2) ↓", False)]
    M = m[[c for c, _, _ in cols]].values
    fig, ax = plt.subplots(figsize=(15, 7.2))
    for j, (c, lab, hib) in enumerate(cols):
        v = M[:, j]
        sc = (v - v.min()) / (v.max() - v.min())
        if not hib:
            sc = 1 - sc
        best = np.argmax(v) if hib else np.argmin(v)
        for i in range(9):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=plt.get_cmap("Greens")(0.05 + 0.75 * sc[i]),
                                   ec="white", lw=1.5))
            ax.text(j, i, f"{v[i]:.2f}", ha="center", va="center", fontsize=FS_TICK + 1,
                    fontweight="bold" if i == best else "normal",
                    color="white" if sc[i] > 0.7 else PALETTE["grey_dark"])
        ax.add_patch(Rectangle((j - 0.47, best - 0.47), 0.94, 0.94, fc="none", ec=PALETTE["grey_dark"], lw=2.5))
    ax.set_xlim(-0.5, len(cols) - 0.5); ax.set_ylim(8.5, -0.5)
    ax.set_yticks(range(9), [f"{r}  {name(r)}" for r in ORDER])
    for t, r in zip(ax.get_yticklabels(), ORDER):
        t.set_color(colour(r)); t.set_fontweight("bold")
    ax.set_xticks(range(len(cols)), [lab for _, lab, _ in cols], fontsize=FS_TICK)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(len(cols) - 0.5, 8.95, "shade = min-max scaled within column (darker = better); box = best; ↓ lower is better",
            ha="right", va="top", fontsize=FS_SMALL, color=PALETTE["grey_mid"])
    finalize_figure(fig, FIGS / "fig_metrics_all")


if __name__ == "__main__":
    which = sys.argv[1:] or ["grids", "zooms", "drivers", "scorecard"]
    if "grids" in which:
        fig_hemilineage(); fig_library(); fig_sex(); fig_birthtime(); fig_depth()
    if "zooms" in which:
        zooms()
    if "drivers" in which:
        drivers()
    if "scorecard" in which:
        scorecard()
