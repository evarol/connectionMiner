"""Figures for the shared synthetic benchmark (house style: figstyle.py, colours from v3style.py).

    python syn_figures.py      # reads v3/synthetic/{results,summary,calibration}.csv and data/

fig_syn_design       P_true snippet per world (cells sorted by type / latent angle / block)
fig_syn_kl           KL(P_true || P_hat) on test entries, mean +- sd over 3 seeds, best per world marked
fig_syn_embed        embedding recovery (kNN balanced accuracy, k-means ARI, Leiden best ARI) on G3-G5
fig_syn_calibration  reliability curves on G2 and G3 against observed Y and against P_true
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import syn_generators as sg  # noqa: E402
import v3style as vs  # noqa: E402

OUT = vs.V3 / "synthetic"
FIGS = vs.FIGS

EXTRA = {"PO": ("Poisson counts", "#8C6D1F"), "B1": ("cell + gene logistic", "#767676"),
         "B0": ("gene rate", "#BDBDBD"), "oracle": ("oracle P_true", "#272727")}
P_METHODS = ["R3", "R4", "R5", "R6", "PO", "R7", "R8", "R9", "B1", "B0"]
E_METHODS = ["R1", "R2", "R3", "R4", "R5", "R6", "PO", "R7", "R8", "R9"]
SUBTITLE = {"G1": "logit $a_i+b_j+u_i^\\top v_j$, rank 8",
            "G2": "NB, $\\mu_{ij}=s_i e^{b_j+u_i^\\top v_j}$",
            "G3": "10 types + depth, NB",
            "G4": "2-D latent, sin/cos/tanh map",
            "G5": "8 x 10 block probabilities"}


def name(c):
    return f"{c} {vs.name(c)}" if c in vs.METHODS else EXTRA[c][0]


def colour(c):
    return vs.colour(c) if c in vs.METHODS else EXTRA[c][1]


# ----------------------------------------------------------------------------- design
def fig_design():
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.6), gridspec_kw=dict(wspace=0.12))
    for ax, g in zip(axes, sg.DATASETS):
        D = sg.load(OUT / "data" / f"{g}_seed0.npz")
        rows = D["order"][np.linspace(0, sg.N - 1, 250).astype(int)]
        P = D["P"]
        if D.get("col_labels") is not None:                       # G5: genes by true column block
            cols = np.lexsort((-P.mean(0), D["col_labels"]))
        elif D.get("labels") is not None and g == "G3":           # G3: genes by the type where they peak
            tm = np.stack([P[D["labels"] == t].mean(0) for t in range(D["labels"].max() + 1)])
            cols = np.lexsort((-tm.max(0), tm.argmax(0)))
        else:                                                     # G1, G2, G4: leading direction of centred logit
            Pc = np.clip(P, 1e-6, 1 - 1e-6); L = np.log(Pc / (1 - Pc)); L = L - L.mean(0)
            Ul, _, Vl = np.linalg.svd(L, full_matrices=False)
            cols = np.argsort(Vl[0])
            if g in ("G1", "G2"):                                 # cells along the same direction
                rows = np.argsort(Ul[:, 0])[np.linspace(0, sg.N - 1, 250).astype(int)]
        cols = cols[np.linspace(0, sg.M - 1, 120).astype(int)]
        im = ax.imshow(D["P"][np.ix_(rows, cols)], aspect="auto", cmap=fs.CMAP_PROB, vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_title(f"{sg.TITLES[g]}\n{SUBTITLE[g]}", fontsize=fs.FS_SMALL + 1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("120 of 300 genes")
        if g == "G1":
            ax.set_ylabel("250 of 2000 cells (sorted)")
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("$P_{\\mathrm{true}}=\\Pr(X_{ij}\\geq 1)$")
    fs.finalize_figure(fig, FIGS / "fig_syn_design")


# ----------------------------------------------------------------------------- dot plots
def _dotpanel(ax, sub, metric, methods, higher_better, log=False):
    sub = sub.set_index("method")
    ys = np.arange(len(methods))[::-1]
    vals = []
    for y, m in zip(ys, methods):
        if m not in sub.index or not np.isfinite(sub.loc[m, f"{metric}_mean"]):
            vals.append(np.nan); continue
        mu, sd = sub.loc[m, f"{metric}_mean"], sub.loc[m, f"{metric}_std"]
        vals.append(mu)
        ax.errorbar(mu, y, xerr=sd if np.isfinite(sd) else None, fmt="o", color=colour(m), ms=8, capsize=3,
                    elinewidth=1.6, mec="white", mew=0.8, zorder=3)
    v = np.array(vals)
    cand = [i for i, m in enumerate(methods) if m not in ("B0", "B1") and np.isfinite(v[i])]
    if cand:
        bi = cand[int(np.nanargmax(v[cand]) if higher_better else np.nanargmin(v[cand]))]
        ax.plot(v[bi], ys[bi], marker="o", ms=17, mfc="none", mec=fs.PALETTE["grey_dark"], mew=1.8, zorder=4)
    if log:
        ax.set_xscale("log")
        from matplotlib import ticker  # 1-2-5 ticks with plain labels: default minor labels collide on narrow panels
        lo, hi = ax.get_xlim()
        for subs in (((1,),) if hi / lo > 30 else ()) + ((1, 2, 5), (1, 2, 3, 5), (1, 2, 3, 4, 5, 6, 8)):
            ticks = [m * 10.0 ** e for e in range(-6, 1) for m in subs if lo <= m * 10.0 ** e <= hi]
            if len(ticks) >= 2:
                break
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_yticks(ys)
    ax.grid(axis="x", color=fs.PALETTE["neutral"], lw=0.8, alpha=0.6)
    ax.set_ylim(-0.7, len(methods) - 0.3)
    return v


def fig_kl(summ):
    fig, axes = plt.subplots(1, 5, figsize=(19, 6.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, g in zip(axes, sg.DATASETS):
        _dotpanel(ax, summ[summ.dataset == g], "kl", P_METHODS, higher_better=False, log=True)
        ax.set_title(f"{sg.TITLES[g]}", fontsize=fs.FS_TITLE)
        ax.set_xlabel("KL$(P_{\\mathrm{true}}\\,\\|\\,\\hat P)$, nats")
    axes[0].set_yticklabels([name(m) for m in P_METHODS])
    for t, m in zip(axes[0].get_yticklabels(), P_METHODS):
        t.set_color(colour(m) if m not in ("B0", "B1") else fs.PALETTE["grey"])
    fig.text(0.5, -0.04, "Test entries, mean $\\pm$ sd over 3 seeds; ring = lowest KL among the models "
             "(baselines excluded). Log scale.", ha="center", fontsize=fs.FS_SMALL)
    fs.finalize_figure(fig, FIGS / "fig_syn_kl")


def fig_embed(summ):
    mets = [("knn_balacc", "kNN balanced accuracy"), ("kmeans_ari", "k-means ARI ($K$ = true)"),
            ("leiden_best_ari", "Leiden ARI (best resolution)")]
    gs = list(sg.DATASETS[2:])
    truth = {"G3": "true cell type (10)", "G4": "8 regions of the true latent", "G5": "true row block (8)"}
    fig, axes = plt.subplots(len(mets), 3, figsize=(16, 13.5), sharey=True, gridspec_kw=dict(wspace=0.08, hspace=0.42))
    for j, g in enumerate(gs):
        for i, (m, lab) in enumerate(mets):
            ax = axes[i, j]
            _dotpanel(ax, summ[summ.dataset == g], m, E_METHODS, higher_better=True)
            ax.set_xlabel(lab)
            if i == 0:
                ax.set_title(f"{sg.TITLES[g]}\nvs {truth[g]}", fontsize=fs.FS_TITLE)
    for i in range(len(mets)):
        axes[i, 0].set_yticklabels([name(m) for m in E_METHODS])
        for t, m in zip(axes[i, 0].get_yticklabels(), E_METHODS):
            t.set_color(colour(m))
    fig.text(0.5, 0.05, "Embeddings as in the real-data UMAP comparison (R1 log-norm PCA-50, R2 PCA-50 of $Y$, "
             "factor models: SVD of $UV^\\top$, R7 logit mean, R8 $z$, R9 $RB$); mean $\\pm$ sd over 3 seeds; "
             "ring = best.", ha="center", fontsize=fs.FS_SMALL)
    fs.finalize_figure(fig, FIGS / "fig_syn_embed")


# ----------------------------------------------------------------------------- calibration
def fig_calibration(cal):
    methods = ["R3", "R4", "R5", "R6", "PO", "R7", "R8", "R9", "B1"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), gridspec_kw=dict(wspace=0.22, hspace=0.3))
    for i, g in enumerate(["G2", "G3"]):
        for j, (col, lab) in enumerate([("frac_pos", "observed detection rate"),
                                        ("mean_ptrue", "mean $P_{\\mathrm{true}}$ in bin")]):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color=fs.PALETTE["neutral"], lw=1.5, ls="--", zorder=1)
            for m in methods:
                c = cal[(cal.dataset == g) & (cal.method == m)].copy()
                c["w"] = c["count"]
                agg = c.groupby("bin").apply(lambda d: pd.Series(dict(
                    x=np.average(d.mean_pred, weights=np.maximum(d.w, 1e-9)),
                    y=np.average(d[col], weights=np.maximum(d.w, 1e-9)), n=d.w.sum())), include_groups=False)
                agg = agg[agg.n >= 100]
                ax.plot(agg.x, agg.y, "-o", color=colour(m), ms=4.5, lw=1.8, label=name(m), alpha=0.9)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
            ax.set_xlabel("mean predicted $\\hat p$ (10 bins, test)")
            ax.set_ylabel(lab)
            target = "$Y$" if j == 0 else "$P_{\\mathrm{true}}$"
            ax.set_title(f"{sg.TITLES[g]}: $\\hat p$ vs {target}")
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=fs.FS_LEGEND)
    fs.finalize_figure(fig, FIGS / "fig_syn_calibration")


def main():
    fs.apply_publication_style()
    summ = pd.read_csv(OUT / "summary.csv")
    cal = pd.read_csv(OUT / "calibration.csv")
    fig_design()
    fig_kl(summ)
    fig_embed(summ)
    fig_calibration(cal)


if __name__ == "__main__":
    main()
