"""Two v3 figures in the shared palette: held-out test scores for every model, and the
representation metrics before and after regressing log depth out of each embedding.

Reads models/comparison.csv and full/umap/metrics_depth_regressed.csv; refits nothing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle as fs  # noqa: E402
import v3style as st  # noqa: E402

plt = fs.plt
POIS = ("Poisson counts", "#4D4D4D")
# comparison.csv model name -> v3 code
CODE = {"BernFac logit": "R3", "BernFac logit + offsets": "R4", "BernFac cloglog + offsets": "R5",
        "NB count factorization": "R6", "mixture of local factors": "R7", "neural manifold": "R8",
        "latent block model": "R9", "Poisson count factorization": "Pois"}


def label(c):
    return f"{c}  {POIS[0]}" if c == "Pois" else f"{c}  {st.name(c)}"


def colour(c):
    return POIS[1] if c == "Pois" else st.colour(c)


def fig_test_all():
    df = pd.read_csv(st.STUDY / "models/comparison.csv")
    base = df.loc[df.model == "gene rate", "test_nll"].item()
    df = df[df.model.isin(CODE)].assign(code=lambda d: d.model.map(CODE))
    order = ["R3", "R4", "R5", "R6", "Pois", "R7", "R8", "R9"]
    df = df.set_index("code").loc[order[::-1]]
    best = df.test_nll.min()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    y = np.arange(len(df))
    cols = [colour(c) for c in df.index]
    panels = [("test_nll", r"$\Delta$ test NLL vs best ($\times10^{-3}$, lower is better)", 1e3, best),
              ("test_auc", "test AUC (higher is better)", 1, None),
              ("test_brier", r"test Brier ($\times10^{-2}$, lower is better)", 1e2, None)]
    for k, (ax, (col, xl, sc, ref)) in enumerate(zip(axes, panels)):
        v = (df[col] - (ref or 0)) * sc
        ax.barh(y, v, color=cols, height=0.7)
        lo = 0 if ref else v.min() - 0.15 * (v.max() - v.min())
        ax.set_xlim(lo, v.max() + 0.22 * (v.max() - lo))
        for yi, (vi, raw) in enumerate(zip(v, df[col])):
            ax.text(vi + 0.01 * (v.max() - lo), yi, f"{raw:.4f}", va="center", fontsize=fs.FS_LEGEND)
        ax.set_xlabel(xl)
        ax.set_yticks(y, [label(c) for c in df.index] if k == 0 else [])
        fs.panel_label(ax, "abc"[k], x=-0.42 if k == 0 else -0.05)
    for t, c in zip(axes[0].get_yticklabels(), df.index):
        t.set_color(colour(c))
    axes[0].text(0.98, 0.98, f"gene-rate baseline: {base:.4f}\n(+{(base - best) * 1e3:.1f}, off scale)",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=fs.FS_LEGEND, color="#767676")
    fig.suptitle("8.04 M held-out test entries, each model selected on validation; numbers are the raw scores",
                 fontsize=fs.FS)
    fig.tight_layout()
    fs.finalize_figure(fig, st.FIGS / "fig_test_all")


def fig_depth_regressed():
    df = pd.read_csv(st.STUDY / "full/umap/metrics_depth_regressed.csv")
    mets = [("hl_knn_balacc", "hemilineage kNN ↑"), ("hl_leiden_best_ARI", "hemilineage Leiden ARI ↑"),
            ("lib_knn_balacc", "library kNN ↓"), ("depth_max_abs_top2", r"depth $|\rho|$ (top-2) ↓")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.4), sharey=True)
    codes = st.ORDER[::-1]
    y = np.arange(len(codes))
    for k, (ax, (m, t)) in enumerate(zip(axes, mets)):
        a = df[df.variant == "as is"].set_index("rep").loc[codes, m]
        r = df[df.variant == "depth regressed"].set_index("rep").loc[codes, m]
        for yi, c in enumerate(codes):
            ax.plot([a[c], r[c]], [yi, yi], color="#CFCECE", lw=2.5, zorder=1)
        ax.scatter(a, y, s=60, facecolor="white", edgecolor=[st.colour(c) for c in codes], lw=2, zorder=2,
                   label="as fitted")
        ax.scatter(r, y, s=60, color=[st.colour(c) for c in codes], zorder=3, label="log depth regressed out")
        ax.set_title(t, fontsize=fs.FS_TITLE)
        fs.panel_label(ax, "abcd"[k], x=-0.05)
        if k == 0:
            ax.set_yticks(y, [f"{c}  {st.name(c)}" for c in codes])
            for tl, c in zip(ax.get_yticklabels(), codes):
                tl.set_color(st.colour(c))
    fig.suptitle("Representation metrics before (open) and after (filled) regressing log depth out of each embedding",
                 fontsize=fs.FS)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=fs.FS_LEGEND)
    fs.finalize_figure(fig, st.FIGS / "fig_depth_regressed")


if __name__ == "__main__":
    fs.apply_publication_style()
    fig_test_all()
    fig_depth_regressed()
