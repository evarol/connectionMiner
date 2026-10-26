"""Figures for the Poisson / NB count factorization (house style from bernfac/figstyle.py).

  fig_selection    validation Bernoulli NLL vs lambda, one line per k, one panel per family
  fig_calibration  10-bin reliability on validation and test for the selected Poisson and NB fits
  fig_dispersion   distribution of fitted gene dispersions phi_j (selected NB fit)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = HERE.parents[4] / "outputs/bernfac_20261020/models/counts"
FIG = OUT / "figures"
FAM_LABEL = {"poisson": "Poisson", "nb": "Negative binomial"}
FAM_COLOR = {"poisson": fs.PALETTE["teal"], "nb": fs.PALETTE["blue_main"]}


def fig_selection(sel, ref):
    fams = [f for f in ("poisson", "nb") if f in set(sel.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(6.2 * len(fams), 4.6), sharey=True, squeeze=False)
    ks = sorted(sel.k.unique())
    cols = dict(zip(ks, fs.k_ramp(len(ks) + 1)[1:]))
    for ax, fam in zip(axes[0], fams):
        g = sel[sel.family == fam]
        for k in ks:
            h = g[g.k == k].sort_values("lam")
            if len(h):
                ax.plot(h.lam, h.val_nll, "o-", color=cols[k], label=f"$k={k}$")
        b = g.loc[g.val_nll.idxmin()]
        ax.plot([b.lam], [b.val_nll], "o", ms=14, mfc="none", mec=fs.PALETTE["red_strong"], mew=2.2,
                label="selected")
        if ref is not None:
            ax.axhline(ref, color=fs.PALETTE["grey_mid"], ls="--", lw=1.6, label="cloglog BernFac (clean)")
        ax.set_xscale("log"); ax.set_xticks([3, 10, 30, 100]); ax.set_xticklabels(["3", "10", "30", "100"])
        ax.set_xlabel(r"prior precision $\lambda$")
        ax.set_title(FAM_LABEL[fam])
    axes[0, 0].set_ylabel("validation Bernoulli NLL / entry")
    hl = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l, h)
    order = [f"$k={k}$" for k in ks] + ["selected", "cloglog BernFac (clean)"]
    labs = [l for l in order if l in hl]
    axes[0, -1].legend([hl[l] for l in labs], labs, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    for i, ax in enumerate(axes[0]):
        fs.panel_label(ax, "ab"[i])
    fs.finalize_figure(fig, FIG / "fig_selection")


def fig_calibration(cal, best):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        ax.plot([0, 1], [0, 1], color=fs.PALETTE["neutral"], lw=1.6, zorder=0)
        for fam, bj in best.items():
            c = cal[(cal.family == fam) & (cal.k == bj["k"]) & (cal.lam == bj["lam"]) & (cal.split == split)
                    & (cal.depth == "train")]
            ax.plot(c.mean_pred, c.frac_pos, "o-", color=FAM_COLOR[fam],
                    label=f"{FAM_LABEL[fam]} ($k={bj['k']}$, $\\lambda={bj['lam']:g}$)")
        ax.set_xlabel(r"predicted $P(x_{ij}\geq 1)$")
        ax.set_title("validation" if split == "val" else "test")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    axes[0].set_ylabel("observed detection rate")
    axes[1].legend(loc="lower right", fontsize=fs.FS_SMALL)
    for i, ax in enumerate(axes):
        fs.panel_label(ax, "ab"[i])
    fs.finalize_figure(fig, FIG / "fig_calibration")


def fig_dispersion(best):
    if "nb" not in best:
        return
    z = np.load(OUT / "fits/best_nb.npz")
    phi = z["phi"]
    mean_log = z["b"]                                     # gene offset: log expression per UMI
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.hist(np.log10(phi), bins=60, color=fs.PALETTE["blue_secondary"], edgecolor="white", lw=0.4)
    med = np.median(phi)
    ax.axvline(np.log10(med), color=fs.PALETTE["red_strong"], lw=2, label=f"median $\\phi={med:.0f}$")
    ax.set_xlabel(r"$\log_{10}\,\phi_j$"); ax.set_ylabel("genes"); ax.legend(loc="upper left")
    ax.set_title(f"fitted dispersions ({len(phi):,} genes)")
    cap = np.mean(phi >= 0.99e6)
    ax.text(0.03, 0.78, f"{cap:.0%} of genes at the cap\n$\\phi=10^6$ (Poisson-like)", transform=ax.transAxes,
            ha="left", va="top", fontsize=fs.FS_SMALL)
    ax = axes[1]
    ax.hexbin(mean_log / np.log(10), np.log10(phi), gridsize=60, cmap=fs.CMAP_DENSITY, bins="log", mincnt=1)
    ax.set_xlabel(r"gene offset $b_j/\ln 10$"); ax.set_ylabel(r"$\log_{10}\,\phi_j$")
    ax.set_title("dispersion vs gene abundance")
    for i, ax in enumerate(axes):
        fs.panel_label(ax, "ab"[i])
    fs.finalize_figure(fig, FIG / "fig_dispersion")


def main():
    fs.apply_publication_style()
    sel = pd.read_csv(OUT / "tables/selection.csv")
    cal = pd.read_csv(OUT / "tables/calibration.csv")
    best = {f: json.loads((OUT / f"best_{f}.json").read_text()) for f in ("poisson", "nb")
            if (OUT / f"best_{f}.json").exists()}
    ref = None
    rp = OUT.parent / "rescored/best_per_model.csv"           # CSO rescoring of earlier models (train-mask fits)
    if rp.exists():
        r = pd.read_csv(rp)
        ref = float(r.loc[r.model == "cloglog + offsets", "val_nll"].iloc[0])
    fig_selection(sel, ref)
    fig_calibration(cal, best)
    fig_dispersion(best)


if __name__ == "__main__":
    main()
