"""Step 5: publication figures for the BernFac real-data study.

Reads tables/ and fits/ under outputs/bernfac_20261020/real and writes
figures/<name>.png (dpi 200) + .pdf in the house style of the
scientific-figure-making skill (figstyle.py via common.apply_publication_style).

Usage:  python make_figures.py [name ...]     (default: all figures)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import (DEFAULT_COLORS, HEMI_COLORS, HEMILINEAGES, KS, LAMBDAS, LIB_COLORS,
                    PALETTE, SEX_COLORS, TABLES, apply_publication_style,
                    display_order, finalize_figure, load_data, load_fit)
from figstyle import (CMAP_DENSITY, CMAP_PROB, CMAP_SEQ, CMAP_SEQ_R, CMAP_Y, FS, FS_LEGEND,
                      FS_SMALL, FS_TICK, FS_TINY, k_ramp, lambda_ramp)
from figstyle import panel_label as _panel_label

apply_publication_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

D = load_data()
ROW, COL = display_order(D["cells"], D["Y_ge1"])
CELLS = D["cells"].iloc[ROW].reset_index(drop=True)
LAM_COLORS = dict(zip(LAMBDAS, lambda_ramp(len(LAMBDAS))))
K_COLORS = dict(zip(KS, k_ramp(len(KS))))
GRAY = PALETTE["grey"]
GAUGE_LAB = {"orthogonal R": "orthogonal $R$", "invertible A (UA, VA^-T)": r"invertible $A$ ($UA$, $VA^{-\top}$)"}
# baseline encodings: reds / neutrals, always dashed-type lines
BASE_GENE_RATE = dict(color=PALETTE["red_2"], ls=":", lw=2.2)
BASE_ROW_COL = dict(color=PALETTE["red_strong"], ls="--", lw=2.0)
BASE_DEPTH = dict(color=PALETTE["red_strong"], ls="--", lw=2.0)
BASE_SELF = dict(color=PALETTE["grey_mid"], ls=":", lw=2.2)
BASE_REPS = {"Y_ge1_raw": ("raw $Y$ (100-d)", dict(color=PALETTE["red_strong"], ls=":", lw=2.2)),
             "log1p_counts": ("log1p counts", dict(color=PALETTE["red_2"], ls="--", lw=2.0)),
             "log1p_cp10k": ("log1p CP10k", dict(color=PALETTE["grey_mid"], ls="-.", lw=2.0))}
PCA_STYLE = dict(color=PALETTE["grey_dark"], marker="s", ms=5, lw=1.6, ls="-")
CHANCE = dict(color=PALETTE["neutral"], lw=2.0, ls="-", zorder=0)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def annotation_bar(ax, col, colors, label):
    cats = list(colors)
    idx = CELLS[col].map({c: i for i, c in enumerate(cats)}).to_numpy()[:, None]
    ax.imshow(idx, aspect="auto", cmap=ListedColormap([colors[c] for c in cats]),
              vmin=-0.5, vmax=len(cats) - 0.5, rasterized=True)
    ax.set_xticks([0]); ax.set_xticklabels([label], rotation=90, fontsize=FS_SMALL)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def hemi_ticks(ax):
    """Label hemilineage blocks on the y axis of a cell-ordered heatmap."""
    h = CELLS["hemilineage"].to_numpy()
    centers = [np.mean(np.where(h == x)[0]) for x in HEMILINEAGES]
    ax.set_yticks(centers); ax.set_yticklabels(HEMILINEAGES)
    for b in np.where(h[1:] != h[:-1])[0]:
        ax.axhline(b + 0.5, color="white", lw=1.0)


def heat(ax, M, cmap=CMAP_PROB, vmin=0, vmax=1, title=None, ylabels=False, title_fs=FS_SMALL,
         interpolation="nearest"):
    im = ax.imshow(M[ROW][:, COL], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation=interpolation, rasterized=True)
    if ylabels:
        hemi_ticks(ax)
    else:
        ax.set_yticks([])
        h = CELLS["hemilineage"].to_numpy()
        for b in np.where(h[1:] != h[:-1])[0]:
            ax.axhline(b + 0.5, color="white", lw=0.8)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=title_fs)
    return im


def lam_lines(ax, df, ycol, ecol=None, xcol="k"):
    for lam in LAMBDAS:
        s = df[df["lam"] == lam].sort_values(xcol)
        if s.empty:
            continue
        ax.errorbar(s[xcol], s[ycol], yerr=None if ecol is None else s[ecol], marker="o", ms=5,
                    lw=2.0, color=LAM_COLORS[lam], label=f"$\\lambda={lam:g}$", capsize=2.5)
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS)
    ax.minorticks_off(); ax.set_xlabel("rank $k$")


def grid_heat(ax, df, val, title, fmt="{:.3f}", cmap=CMAP_SEQ, best="max", vmax=None):
    M = df.pivot(index="lam", columns="k", values=val).loc[LAMBDAS, KS].to_numpy()
    lo, hi = np.nanmin(M), (np.nanmax(M) if vmax is None else vmax)
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=lo, vmax=hi)
    rgba = im.cmap(im.norm(M))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            lum = 0.299 * rgba[i, j, 0] + 0.587 * rgba[i, j, 1] + 0.114 * rgba[i, j, 2]
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=FS_TINY,
                    color="white" if lum < 0.5 else "black")
    if best:
        bi = np.unravel_index(np.nanargmax(M) if best == "max" else np.nanargmin(M), M.shape)
        ax.add_patch(plt.Rectangle((bi[1] - 0.5, bi[0] - 0.5), 1, 1, fill=False, ec=PALETTE["highlight"], lw=3))
    ax.set_xticks(range(len(KS))); ax.set_xticklabels(KS)
    ax.set_yticks(range(len(LAMBDAS))); ax.set_yticklabels([f"{l:g}" for l in LAMBDAS])
    ax.set_xlabel("rank $k$"); ax.set_ylabel("$\\lambda$"); ax.set_title(title, fontsize=FS_SMALL + 1)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def panel_label(ax, s, x=-0.16, y=1.04):
    _panel_label(ax, s, x=x, y=y)


def best_cv():
    s = pd.read_csv(TABLES / "cv_summary.csv")
    s = s[(s.dataset == "Y_ge1") & (s.model == "bernfac")]
    b = s.loc[s["nll_mean"].idxmin()]
    return float(b["lam"]), int(b["k"])


def lam_handles(lams=LAMBDAS):
    return [Line2D([], [], color=LAM_COLORS[l], marker="o", ms=5, lw=2.0, label=f"$\\lambda={l:g}$") for l in lams]


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def fig_data():
    C, Y1, Y2 = D["counts"], D["Y_ge1"], D["Y_ge2"]
    fig = plt.figure(figsize=(17, 9.5))
    # columns: 3 annotation bars | spacer | heatmap | spacer | heatmap | spacer | heatmap | spacer | cell marginal
    gs = GridSpec(2, 11, figure=fig, width_ratios=[0.13, 0.13, 0.13, 0.42, 3, 0.42, 3, 0.42, 3, 0.15, 1.1],
                  height_ratios=[1, 5], wspace=0.12, hspace=0.06)
    HC = [4, 6, 8]
    for i, (col, cols, lab) in enumerate([("hemilineage", HEMI_COLORS, "hemilineage"),
                                          ("library", LIB_COLORS, "library"), ("Sex", SEX_COLORS, "sex")]):
        annotation_bar(fig.add_subplot(gs[1, i]), col, cols, lab)
    axs = [fig.add_subplot(gs[1, HC[j]]) for j in range(3)]
    im0 = heat(axs[0], np.log1p(C), cmap=CMAP_SEQ, vmin=0, vmax=np.log1p(C).max())
    heat(axs[1], Y1, cmap=CMAP_Y); heat(axs[2], Y2, cmap=CMAP_Y)
    titles = ["log1p(UMI counts)", r"$Y = \mathbf{1}[\mathrm{count} \geq 1]$ (primary)", r"$Y = \mathbf{1}[\mathrm{count} \geq 2]$ (sensitivity)"]
    for a in axs:
        a.set_xlabel(f"{C.shape[1]} genes (clustered)", fontsize=FS_SMALL + 1)
    cb = fig.colorbar(im0, ax=axs[0], location="bottom", fraction=0.035, pad=0.06, aspect=30)
    cb.set_label("log1p count", fontsize=FS_SMALL)
    cb.ax.tick_params(labelsize=FS_SMALL - 1)
    # gene marginals above each heatmap
    for j, (M, lab) in enumerate([(np.log1p(C), "mean log1p"), (Y1, "detect. rate"), (Y2, "rate (≥2)")]):
        a = fig.add_subplot(gs[0, HC[j]], sharex=axs[j])
        a.bar(np.arange(C.shape[1]), M[:, COL].mean(0), width=1.0,
              color=PALETTE["blue_main"] if j else PALETTE["teal"])
        a.set_ylabel(lab, fontsize=FS_SMALL); a.tick_params(labelbottom=False, bottom=False, labelsize=FS_SMALL - 1)
        a.set_title(titles[j], fontsize=FS)
        a.locator_params(axis="y", nbins=3)
        if j == 0:
            panel_label(a, "a", x=-0.45, y=1.0)
    # cell marginals
    a = fig.add_subplot(gs[1, 10], sharey=axs[0])
    yy = np.arange(len(ROW))
    a.plot(Y1[ROW].mean(1), yy, lw=0.8, color=PALETTE["blue_main"], label="rate ≥1")
    a.plot(Y2[ROW].mean(1), yy, lw=0.8, color=PALETTE["red_strong"], label="rate ≥2")
    a.set_xlabel("cell detection\nrate", fontsize=FS_SMALL + 1)
    a.tick_params(labelleft=False, left=False, labelsize=FS_SMALL - 1)
    a.spines["left"].set_visible(False)
    a.locator_params(axis="x", nbins=3)
    ax2 = fig.add_subplot(gs[0, 10])
    ax2.axis("off")
    ax2.legend(*a.get_legend_handles_labels(), fontsize=FS_SMALL, loc="lower center", handlelength=1.2,
               borderaxespad=0.0)
    for ln in ax2.get_legend().get_lines():
        ln.set_linewidth(2.5)
    handles = [Patch(color=v, label=k) for k, v in HEMI_COLORS.items()] + \
              [Patch(color=v, label=k) for k, v in LIB_COLORS.items()] + \
              [Patch(color=v, label=f"sex {k}") for k, v in SEX_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=11, fontsize=FS_LEGEND,
               handlelength=1.1, columnspacing=1.0, handletextpad=0.4)
    fig.text(0.5, 0.0, "cells (rows) sorted by hemilineage → library → total UMI; "
             f"overall detection {Y1.mean():.3f} (≥1), {Y2.mean():.3f} (≥2)", ha="center", va="top",
             fontsize=FS_SMALL + 1)
    return finalize_figure(fig, "fig1_data_overview")


def fig_gallery():
    lams, ks = [0.1, 1.0, 10.0], [1, 2, 5, 20]
    cv = pd.read_csv(TABLES / "cv_summary.csv") if (TABLES / "cv_summary.csv").exists() else None
    fig = plt.figure(figsize=(14.5, 8.8))
    gs = GridSpec(len(lams), len(ks) + 2, figure=fig, width_ratios=[0.12, 1.3] + [1] * len(ks),
                  wspace=0.08, hspace=0.36)
    annotation_bar(fig.add_subplot(gs[:, 0]), "hemilineage", HEMI_COLORS, "hemilin.")
    ay = fig.add_subplot(gs[:, 1])
    heat(ay, D["Y_ge1"], cmap=CMAP_Y, title="observed $Y$ (count ≥ 1)", ylabels=False, title_fs=FS)
    ay.set_xlabel("genes")
    for i, lam in enumerate(lams):
        for j, k in enumerate(ks):
            a = fig.add_subplot(gs[i, j + 2])
            P = load_fit("Y_ge1", lam, k)["P"]
            t = f"$\\lambda={lam:g}$, $k={k}$"
            if cv is not None:
                r = cv[(cv.dataset == "Y_ge1") & (cv.model == "bernfac") & (cv.lam == lam) & (cv.k == k)]
                if len(r):
                    t += f"\nheld-out NLL {r['nll_mean'].iloc[0]:.3f}"
            im = heat(a, P, title=t, title_fs=FS_SMALL + 1)
    cb = fig.colorbar(im, ax=fig.axes[2:], fraction=0.02, pad=0.015)
    cb.set_label(r"$P = \sigma(UV^\top)$")
    fig.suptitle("BernFac fitted probabilities on Y_ge1 (same cell / gene order as Fig. 1)", y=0.995, fontsize=FS + 1)
    return finalize_figure(fig, "fig2_P_gallery")


def fig_grid_overview():
    g = pd.read_csv(TABLES / "grid_results.csv"); g = g[g.dataset == "Y_ge1"]
    cv = pd.read_csv(TABLES / "cv_summary.csv"); cv = cv[(cv.dataset == "Y_ge1") & (cv.model == "bernfac")]
    dec = pd.read_csv(TABLES / "decoding.csv"); dec = dec[(dec.dataset == "Y_ge1") & (dec.rep == "bernfac")]
    info = pd.read_csv(TABLES / "info_content.csv"); info = info[(info.dataset == "Y_ge1")]
    fig, axs = plt.subplots(2, 4, figsize=(19, 10))
    grid_heat(axs[0, 0], g, "loglik_per_entry", "train log-lik / entry\n(↑, overfits)")
    grid_heat(axs[0, 1], cv, "nll_mean", "held-out NLL / entry\n(↓; colour capped 0.56)", cmap=CMAP_SEQ_R, best="min", vmax=0.56)
    grid_heat(axs[0, 2], cv, "auc_mean", "held-out AUC (↑)")
    grid_heat(axs[0, 3], g, "iterations", "PG-EM iterations (cap 500)", fmt="{:.0f}", cmap="cividis", best=None)
    grid_heat(axs[1, 0], dec, "hemilineage_logreg", "hemilineage bal. acc. (↑)")
    grid_heat(axs[1, 1], dec, "library_logreg", "library bal. acc.\n(batch, ↓ better)", cmap=CMAP_SEQ_R, best="min")
    grid_heat(axs[1, 2], dec, "depth_spearman_meanP", r"$\rho$(mean $P$, total UMI)", cmap=CMAP_SEQ, best=None)
    grid_heat(axs[1, 3], info, "spearman_pos", r"$\rho$($P$, log1p count | $Y$=1) (↑)")
    for a, l in zip(axs.ravel(), "abcdefgh"):
        panel_label(a, l, x=-0.2)
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    return finalize_figure(fig, "fig3_grid_overview")


def fig_cv():
    s = pd.read_csv(TABLES / "cv_summary.csv"); cal = pd.read_csv(TABLES / "cv_calibration.csv")
    b1 = s[s.dataset == "Y_ge1"]; bf_ = b1[b1.model == "bernfac"]
    base = b1[b1.model != "bernfac"].set_index("model")
    fig, axg = plt.subplots(2, 3, figsize=(18, 10.5))
    axs = list(axg.ravel()[:5]); leg_ax = axg.ravel()[5]
    for a, (m, lab) in zip(axs[:3], [("nll", "held-out NLL / entry"), ("auc", "held-out AUC"), ("brier", "held-out Brier")]):
        lam_lines(a, bf_, f"{m}_mean", f"{m}_std")
        a.axhline(base.loc["gene_rate", f"{m}_mean"], label="gene rate", **BASE_GENE_RATE)
        a.axhline(base.loc["row_col", f"{m}_mean"], label="cell+gene logistic", **BASE_ROW_COL)
        a.set_ylabel(lab)
    axs[0].set_ylim(0.47, 0.66)
    axs[0].text(0.03, 0.97, f"off scale: up to {bf_.nll_mean.max():.2f}\n($\\lambda$ ≤ 1, $k$ ≥ 10)", transform=axs[0].transAxes,
                va="top", fontsize=FS_SMALL)
    # dedicated legend panel (shared by a-c)
    h, l = axs[0].get_legend_handles_labels()
    leg_ax.axis("off")
    leg_ax.legend(h, l, loc="center left", fontsize=FS, title="panels a–c\nBernFac (colour = $\\lambda$) / baselines",
                  title_fontsize=FS, ncol=2, columnspacing=1.5, alignment="left")
    lam_b, k_b = best_cv()
    a = axs[3]
    c1 = cal[cal.dataset == "Y_ge1"]
    for (lab, sel, col) in [(f"best $\\lambda={lam_b:g}$, $k={k_b}$", (c1.model == "bernfac") & (c1.lam == lam_b) & (c1.k == k_b), PALETTE["blue_main"]),
                            ("$\\lambda=0.03$, $k=20$ (overfit)", (c1.model == "bernfac") & (c1.lam == 0.03) & (c1.k == 20), PALETTE["red_strong"]),
                            ("$\\lambda=10$, $k=2$ (over-shrunk)", (c1.model == "bernfac") & (c1.lam == 10) & (c1.k == 2), PALETTE["violet"]),
                            ("cell+gene logistic", c1.model == "row_col", PALETTE["grey_mid"])]:
        t = c1[sel].groupby("bin").apply(lambda d: pd.Series(dict(
            p=np.average(d.mean_pred, weights=d["count"]), f=np.average(d.frac_pos, weights=d["count"]))))
        a.plot(t.p, t.f, marker="o", ms=5, lw=2.0, color=col, label=lab)
    a.plot([0, 1], [0, 1], color="k", lw=1.0, ls="--")
    a.set_xlabel("predicted $P$ (held-out)"); a.set_ylabel("observed fraction $Y$=1")
    a.legend(fontsize=FS_LEGEND, loc="upper left", handlelength=1.2); a.set_aspect("equal")
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    # extension: stronger shrinkage / larger k (run_cv_extension.py)
    a = axs[4]
    ext = pd.read_csv(TABLES / "cv_extension_summary.csv")
    allc = pd.concat([bf_[["lam", "k", "nll_mean", "nll_std"]], ext[["lam", "k", "nll_mean", "nll_std"]]])
    allc = allc.drop_duplicates(["lam", "k"])
    for k, col in [(10, K_COLORS[10]), (20, K_COLORS[20]), (40, PALETTE["grey_dark"])]:
        q = allc[allc.k == k].sort_values("lam")
        a.errorbar(q.lam, q.nll_mean, yerr=q.nll_std, marker="o", ms=5, lw=2.0, color=col, label=f"$k={k}$", capsize=2.5)
    a.axhline(np.log(2), color=PALETTE["grey_mid"], ls=":", lw=2.0)
    a.text(0.04, np.log(2) - 0.006, r"$P \equiv 0.5$ ($U=V=0$)", fontsize=FS_SMALL, va="top")
    a.axhline(base.loc["row_col", "nll_mean"], **BASE_ROW_COL)
    a.set_xscale("log"); a.set_xlabel("$\\lambda$"); a.set_ylabel("held-out NLL / entry"); a.set_ylim(0.46, 0.72)
    a.set_title("$\\lambda$ sweep incl. extension ($\\lambda$ ≤ 100, $k$ ≤ 40)", fontsize=FS_SMALL + 1)
    a.legend(fontsize=FS_LEGEND, loc="lower left")
    for ax, l in zip(axs, "abcde"):
        panel_label(ax, l)
    fig.tight_layout(w_pad=2.5, h_pad=2.5)
    return finalize_figure(fig, "fig4_heldout_cv")


def fig_info():
    info = pd.read_csv(TABLES / "info_content.csv")
    ref = info[info.dataset == "reference"].set_index("rep")
    b = info[info.dataset == "Y_ge1"]
    lam_b, k_b = best_cv()
    fig, axs = plt.subplots(1, 4, figsize=(19, 6.2))
    for a, (c, lab) in zip(axs[:2], [("spearman_all", "Spearman $\\rho$($P$, log1p count)\nall entries"),
                                     ("spearman_pos", "Spearman $\\rho$($P$, log1p count)\nwithin $Y$=1 entries")]):
        lam_lines(a, b, c)
        a.axhline(ref.loc["depth_x_abundance", c], label="depth × abundance", **BASE_DEPTH)
        if c == "spearman_all":
            a.axhline(ref.loc["Y_ge1", c], label="$Y$ itself", **BASE_SELF)
        a.set_ylabel(lab)
    h, l = axs[0].get_legend_handles_labels()
    P = load_fit("Y_ge1", lam_b, k_b)["P"]
    L = np.log1p(D["counts"]); pos = D["Y_ge1"] == 1
    a = axs[2]
    a.hexbin(L[pos], P[pos], gridsize=40, bins="log", cmap=CMAP_DENSITY, mincnt=1, rasterized=True)
    q = pd.DataFrame(dict(c=D["counts"][pos], p=P[pos])).assign(c=lambda d: d.c.clip(upper=20)).groupby("c").p.median()
    a.plot(np.log1p(q.index), q.values, color=PALETTE["red_strong"], marker="o", ms=4, lw=2.0, label="median $P$ per count (≤20)")
    a.set_xlabel("log1p count ($Y$=1 entries)"); a.set_ylabel(f"$P$ ($\\lambda={lam_b:g}$, $k={k_b}$)")
    a.legend(fontsize=FS_LEGEND, loc="lower right")
    gc = pd.read_csv(TABLES / "gene_corr_best.csv")
    a = axs[3]
    bins = np.linspace(-0.4, 1, 29)
    a.hist(gc["bernfac_pos"].dropna(), bins=bins, color=PALETTE["blue_main"], alpha=0.8, label="BernFac best")
    a.hist(gc["depth_x_abundance_pos"].dropna(), bins=bins, color=PALETTE["red_2"], alpha=0.65, label="depth × abundance")
    a.axvline(0, color="k", lw=1.0)
    a.set_ylim(0, a.get_ylim()[1] * 1.3)
    a.set_xlabel(r"gene-wise Spearman $\rho$ within $Y$=1"); a.set_ylabel("genes"); a.legend(fontsize=FS_LEGEND, loc="upper left")
    for ax, l_ in zip(axs, "abcd"):
        panel_label(ax, l_, x=-0.22)
    fig.tight_layout(w_pad=2.0, rect=(0, 0.13, 1, 1))
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.28, -0.02), ncol=4, fontsize=FS_LEGEND,
               columnspacing=1.2)
    return finalize_figure(fig, "fig5_information_content")


def fig_batch():
    dec = pd.read_csv(TABLES / "decoding.csv")
    b = dec[(dec.dataset == "Y_ge1") & (dec.rep == "bernfac")]
    base = dec[dec.rep != "bernfac"]
    fig, axs = plt.subplots(2, 3, figsize=(19, 11.5))
    chance = {"hemilineage": 0.2, "library": 0.25, "Sex": 0.5}
    for a, t in zip(axs[0], ["hemilineage", "library", "Sex"]):
        lam_lines(a, b, f"{t}_logreg")
        for rep, (lab, st) in BASE_REPS.items():
            a.axhline(base[base.rep == rep][f"{t}_logreg"].iloc[0], label=lab, **st)
        pca = base[base.rep == "pca_Y_ge1"].sort_values("k")
        a.plot(pca.k, pca[f"{t}_logreg"], label="PCA-$k$ of $Y$", **PCA_STYLE)
        a.axhline(chance[t], label="chance", **CHANCE)
        a.set_ylabel(f"{t.lower()} balanced acc.\n(logreg, 5-fold)")
        a.set_title({"hemilineage": "biology", "library": "batch (4 libraries)", "Sex": "sex (2 F, 2 M libraries)"}[t])
    h, l = axs[0, 0].get_legend_handles_labels()
    a = axs[1, 0]
    for lam in LAMBDAS:
        s = b[b.lam == lam].sort_values("k")
        a.plot(s.sil_library, s.sil_hemilineage, marker="o", ms=5, lw=2.0, color=LAM_COLORS[lam], label=f"$\\lambda={lam:g}$")
        for _, r in s.iterrows():
            if lam in (0.1, 10):
                a.annotate(f"k{int(r.k)}", (r.sil_library, r.sil_hemilineage), fontsize=FS_TINY,
                           xytext=(3, 3), textcoords="offset points")
    for rep, (lab, st) in BASE_REPS.items():
        r = base[base.rep == rep].iloc[0]
        a.scatter(r.sil_library, r.sil_hemilineage, marker="x", s=60, lw=2, color=st["color"])
        off, ha = {"Y_ge1_raw": ((6, -14), "left"), "log1p_counts": ((-12, -26), "right"),
                   "log1p_cp10k": ((6, 4), "left")}[rep]
        a.annotate(lab, (r.sil_library, r.sil_hemilineage), fontsize=FS_TINY, xytext=off, ha=ha,
                   textcoords="offset points", color=st["color"])
    a.axhline(0, color="k", lw=0.8); a.axvline(0, color="k", lw=0.8)
    a.set_xlabel("silhouette by library"); a.set_ylabel("silhouette by hemilineage")
    a = axs[1, 1]
    lam_lines(a, b, "depth_spearman_meanP")
    for rep, (lab, st) in BASE_REPS.items():
        a.axhline(base[base.rep == rep]["depth_spearman_meanP"].iloc[0], label=lab, **st)
    a.set_ylabel("Spearman $\\rho$(cell mean $P$,\ntotal UMI)")
    a = axs[1, 2]
    lam_lines(a, b, "depth_cv_r2")
    pca = base[base.rep == "pca_Y_ge1"].sort_values("k")
    a.plot(pca.k, pca.depth_cv_r2, label="PCA-$k$ of $Y$", **PCA_STYLE)
    a.set_ylabel("CV $R^2$ of log total UMI\nfrom embedding")
    for ax, l_ in zip(axs.ravel(), "abcdef"):
        panel_label(ax, l_, x=-0.2)
    fig.tight_layout(w_pad=2.5, h_pad=2.5, rect=(0, 0.08, 1, 1))
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=FS_LEGEND + 1,
               columnspacing=1.5)
    return finalize_figure(fig, "fig6_batch_vs_biology")


def fig_convergence():
    g = pd.read_csv(TABLES / "grid_results.csv"); g1 = g[g.dataset == "Y_ge1"]
    fig = plt.figure(figsize=(19, 10.5))
    gs = GridSpec(2, 6, figure=fig, hspace=0.5, wspace=0.42, height_ratios=[1, 1.15])
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gb = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :], wspace=0.5, width_ratios=[1, 1, 1.05])
    top = []
    for i, lam in enumerate(LAMBDAS):
        a = fig.add_subplot(gs[0, i], sharey=top[0] if top else None)
        top.append(a)
        for k in KS:
            f = load_fit("Y_ge1", lam, k)
            J = f["obj"]
            rel = np.diff(J) / np.maximum(1.0, np.abs(J[:-1]))   # the solver's stopping statistic
            a.semilogy(np.arange(1, len(J)), np.maximum(rel, 1e-14), color=K_COLORS[k], lw=1.6, label=f"$k={k}$")
        a.axhline(1e-7, color="k", ls="--", lw=1.0)
        a.set_title(f"$\\lambda={lam:g}$"); a.set_xlabel("iteration")
        a.set_ylim(1e-9, 1)
        a.locator_params(axis="x", nbins=3)
        if i == 0:
            a.set_ylabel("$\\Delta J_t\\,/\\,|J_{t-1}|$\n(tol $10^{-7}$ dashed)")
        else:
            a.tick_params(labelleft=False)
    h, l = top[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=6, fontsize=FS_LEGEND + 1,
               title="rank $k$", title_fontsize=FS_LEGEND + 1, columnspacing=1.5)
    panel_label(top[0], "a", x=-0.55)
    a = fig.add_subplot(gb[0])
    grid_heat(a, g1, "iterations", "iterations to stop (cap 500)", fmt="{:.0f}", cmap="cividis", best=None)
    panel_label(a, "b")
    a = fig.add_subplot(gb[1])
    g1 = g1.assign(log_spread=np.log10(g1.restart_J_spread_rel.clip(lower=1e-12)))
    grid_heat(a, g1, "log_spread", "log10 restart spread of J (rel.)", fmt="{:.1f}", cmap=CMAP_PROB, best=None)
    panel_label(a, "c")
    a = fig.add_subplot(gb[2])
    for lam in LAMBDAS:
        s = g1[g1.lam == lam]
        for _, r in s.iterrows():
            Js = np.array([float(x) for x in r.restart_J.split(";")])
            a.scatter(np.full(len(Js), r.k) * (1 + 0.04 * (LAMBDAS.index(lam) - 2.5)),
                      (Js.max() - Js) / abs(Js.max()) + 1e-12, s=22, color=LAM_COLORS[lam],
                      edgecolor="k", lw=0.3)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xticks(KS); a.set_xticklabels(KS); a.minorticks_off()
    a.set_xlabel("rank $k$"); a.set_ylabel("$(J_{\\mathrm{best}} - J_{\\mathrm{restart}})\\,/\\,|J_{\\mathrm{best}}|$\n($+10^{-12}$)")
    a.set_title("per-restart deficit (svd + 3 random)", fontsize=FS_SMALL + 1)
    a.legend(handles=[Patch(color=LAM_COLORS[l_], label=f"$\\lambda={l_:g}$") for l_ in LAMBDAS], fontsize=FS_LEGEND,
             ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=1.0)
    panel_label(a, "d", x=-0.25)
    return finalize_figure(fig, "fig7_convergence")


def fig_identifiability():
    t = pd.read_csv(TABLES / "identifiability.csv"); gd = pd.read_csv(TABLES / "gauge_demo.csv")
    from common import FITS
    fig, axs = plt.subplots(1, 4, figsize=(20, 7.2))
    a = axs[0]
    t1 = t[t.seed > 0]
    settings = list(dict.fromkeys(zip(t1.lam, t1.k)))
    metrics = [("U_raw_rel", "$U$ raw"), ("U_procrustes_rel", "$U$ Procrustes"), ("gram_rel", r"Gram $UU^\top$"), ("P_meanabs", r"mean $|\Delta P|$")]
    mcol = [PALETTE["red_strong"], PALETTE["blue_main"], PALETTE["green_3"], PALETTE["teal"]]
    w = 0.2
    for mi, (m, lab) in enumerate(metrics):
        for si, (l, k) in enumerate(settings):
            v = t1[(t1.lam == l) & (t1.k == k)][m]
            a.scatter(np.full(len(v), si + (mi - 1.5) * w), v.clip(lower=1e-12), color=mcol[mi], s=30,
                      edgecolor="k", lw=0.3, label=lab if si == 0 else None)
    a.set_yscale("log"); a.set_xticks(range(len(settings))); a.set_xticklabels([f"$\\lambda={l:g}$\n$k={k}$" for l, k in settings])
    a.set_ylabel("difference vs seed 0\n(rel. Frobenius; $|\\Delta P|$ absolute)")
    a.legend(fontsize=FS_LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0)
    a.set_title("5 random seeds")
    z = np.load(FITS / "identifiability_lam1_k5.npz")
    U0 = z["U"][0]
    fcols = [PALETTE["blue_main"], PALETTE["red_strong"], PALETTE["green_3"], PALETTE["violet"], PALETTE["teal"],
             "#E69F00", "#555555", "#AAAAAA", PALETTE["blue_secondary"], PALETTE["red_2"]]
    for a, U, ttl in [(axs[1], z["U"][1], "$U$ seed 1 vs seed 0 (raw)"), (axs[2], z["U_aligned"][1], "after Procrustes rotation")]:
        for c in range(U.shape[1]):
            a.scatter(U0[:, c], U[:, c], s=4, alpha=0.5, color=fcols[c % len(fcols)], label=f"factor {c+1}", rasterized=True)
        lim = np.abs(U0).max() * 1.1
        a.plot([-lim, lim], [-lim, lim], color="k", lw=1.0, ls="--")
        a.set_xlabel("$U$ seed 0"); a.set_ylabel("$U$ seed 1"); a.set_title(ttl + "\n($\\lambda=1$, $k=5$)")
    axs[1].legend(fontsize=FS_LEGEND, markerscale=3, loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=5,
                  columnspacing=1.0, handletextpad=0.2)
    a = axs[3]
    g = gd[gd["transform"] != "identity"]
    for i, (tr, col) in enumerate([("orthogonal R", PALETTE["blue_main"]), ("invertible A (UA, VA^-T)", PALETTE["red_strong"])]):
        s = g[g["transform"] == tr]
        a.bar(np.arange(len(s)) + (i - 0.5) * 0.38, -s.J_minus_J0.to_numpy() + 1e-9, width=0.38, color=col,
              edgecolor="k", lw=1.0, label=GAUGE_LAB[tr] + f":\nmax $|\\Delta P|$={s.P_maxabs_vs_orig.max():.1e}")
    a.set_yscale("log"); a.set_xticks(range(len(settings))); a.set_xticklabels([f"$\\lambda={l:g}$\n$k={k}$" for l, k in settings])
    a.set_ylabel("$J(\\mathrm{original}) - J(\\mathrm{transformed})$\n($+10^{-9}$)")
    a.legend(fontsize=FS_LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    a.set_title("gauge: $P$ unchanged;\nprior fixes all but $O(k)$")
    for ax, l_ in zip(axs, "abcd"):
        panel_label(ax, l_, x=-0.3, y=1.12)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.3, top=0.86, wspace=0.5)
    return finalize_figure(fig, "fig8_identifiability")


def fig_ge2():
    s = pd.read_csv(TABLES / "cv_summary.csv"); dec = pd.read_csv(TABLES / "decoding.csv")
    info = pd.read_csv(TABLES / "info_content.csv")
    fig, axs = plt.subplots(1, 4, figsize=(19, 6.3))
    lc3 = dict(zip([0.3, 1.0, 3.0], lambda_ramp(3)))
    for dset, ls, mk in [("Y_ge1", "-", "o"), ("Y_ge2", "--", "^")]:
        cvb = s[(s.dataset == dset) & (s.model == "bernfac") & s.lam.isin([0.3, 1, 3]) & s.k.isin([2, 5, 10])]
        d = dec[(dec.dataset == dset) & (dec.rep == "bernfac") & dec.lam.isin([0.3, 1, 3]) & dec.k.isin([2, 5, 10])]
        inf = info[(info.dataset == dset) & info.lam.isin([0.3, 1, 3]) & info.k.isin([2, 5, 10])]
        for a, df, c, lab in [(axs[0], cvb, "auc_mean", "held-out AUC"), (axs[1], d, "hemilineage_logreg", "hemilineage bal. acc."),
                              (axs[2], d, "library_logreg", "library bal. acc."), (axs[3], inf, "spearman_pos", r"$\rho$($P$, log1p count | count ≥ 1)")]:
            for lam in [0.3, 1.0, 3.0]:
                q = df[df.lam == lam].sort_values("k")
                a.plot(q.k, q[c], ls=ls, marker=mk, ms=7, lw=2.2, color=lc3[lam], label=f"{dset} $\\lambda={lam:g}$")
            a.set_xscale("log"); a.set_xticks([2, 5, 10]); a.set_xticklabels([2, 5, 10]); a.minorticks_off()
            a.set_xlabel("rank $k$"); a.set_ylabel(lab)
    h, l = axs[0].get_legend_handles_labels()
    for ax, l_ in zip(axs, "abcd"):
        panel_label(ax, l_, x=-0.22)
    fig.suptitle("Binarization sensitivity: count ≥ 1 (solid) vs count ≥ 2 (dashed);\n"
                 "AUC is on each binarization's own held-out entries", fontsize=FS + 1, y=0.985)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.8, bottom=0.25, wspace=0.32)
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=FS_LEGEND + 1, columnspacing=1.5)
    return finalize_figure(fig, "fig9_binarization_sensitivity")


FIGS = dict(data=fig_data, gallery=fig_gallery, grid=fig_grid_overview, cv=fig_cv, info=fig_info,
            batch=fig_batch, convergence=fig_convergence, ident=fig_identifiability, ge2=fig_ge2)

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        for p in FIGS[name]():
            print(p)
