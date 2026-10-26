"""W1 (v4): figures from v4/tables/test_ci{tag}.csv (written by w1_bootstrap.py).

fig_test_ci          forest plot: Delta test NLL and Delta Brier vs R6 and vs R4, 95% two-way CI
fig_test_ci_deciles  Delta test NLL vs R6 / R4 by cell-depth decile and gene detection-rate decile
Usage: python w1_test_ci_figures.py [--tag suffix]   (models are taken from the table, so re-running
after w1_bootstrap.py --extra ... picks up refits automatically; extras get grey markers)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import figstyle as fs  # noqa: E402
import v3style as vs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

S = vs.STUDY / "v4"
NAME = {c: vs.name(c) for c in vs.PROB_MODELS} | {"PO": "Poisson counts", "GR": "gene-rate baseline"}
COL = {c: vs.colour(c) for c in vs.PROB_MODELS} | {"PO": "#4D4D4D", "GR": "#9E9E9E"}


def label(c):
    if c in ("PO", "GR"):
        return NAME[c]
    return f"{c}  {NAME[c]}" if c in NAME else c


def forest(ax, d, ref, metric, scale, xlim, order, title, unit):
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for yi, c in enumerate(order):
        r = d[(d.model == c) & (d.ref == ref) & (d.metric == metric)].iloc[0]
        v, lo, hi = r.delta * scale, r.ci_lo * scale, r.ci_hi * scale
        col = COL.get(c, "#767676")
        if c == ref:
            ax.plot(0, yi, marker="D", color=col, ms=10, mec="k")
            txt = "reference"
        elif v > xlim[1] or v < xlim[0]:
            x = xlim[1] if v > xlim[1] else xlim[0]
            ax.plot(x, yi, marker=">" if v > xlim[1] else "<", color=col, ms=13, clip_on=False)
            txt = f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]  off scale"
        else:
            ax.plot([lo, hi], [yi, yi], color=col, lw=3.5, solid_capstyle="butt")
            dist = lo > 0 or hi < 0
            ax.plot(v, yi, marker="o", ms=10, color=col if dist else "white", mec=col, mew=2.5)
            txt = f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        if c == "PO" and metric == "nll":
            rc = d[(d.model == c) & (d.ref == ref) & (d.metric == "nll_clip")].iloc[0]
            txt += f"\nclipped p: {rc.delta * scale:+.2f} [{rc.ci_lo * scale:+.2f}, {rc.ci_hi * scale:+.2f}]"
        ax.text(1.03, yi, txt, transform=tr, va="center", ha="left", fontsize=fs.FS_SMALL - 0.5)
    ax.axvline(0, color="k", lw=1.2)
    ax.set_xlim(*xlim)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label(c) for c in order])
    ax.set_xlabel(unit)
    ax.set_title(title, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    d = pd.read_csv(S / f"tables/test_ci{a.tag}.csv")
    fs.apply_publication_style()
    al = d[d.stratum == "all"]
    tot = al[(al.metric == "nll") & (al.ref == al.ref.iloc[0])].sort_values("value_model")
    order = list(tot.model)

    # ---------------- forest
    fig, ax = plt.subplots(2, 2, figsize=(21, max(12.5, 1.25 * len(order) + 1.5)),
                           gridspec_kw=dict(wspace=1.05, hspace=0.42 if len(order) <= 9 else 0.25))
    forest(ax[0, 0], al, "R6", "nll", 1e3, (-0.15, 1.6), order, "Test NLL minus NB counts (R6)",
           "Δ test NLL (10⁻³ nats / entry)")
    forest(ax[0, 1], al, "R4", "nll", 1e3, (-1.4, 0.35), order, "Test NLL minus BernFac + offsets (R4)",
           "Δ test NLL (10⁻³ nats / entry)")
    forest(ax[1, 0], al, "R6", "brier", 1e4, (-0.3, 5.5), order, "Brier score minus R6",
           "Δ Brier (10⁻⁴)")
    forest(ax[1, 1], al, "R4", "brier", 1e4, (-5.4, 1.0), order, "Brier score minus R4",
           "Δ Brier (10⁻⁴)")
    for x, l in zip(ax.ravel(), "abcd"):
        fs.panel_label(x, l, x=-0.62, y=1.02)
    fig.text(0.5, 0.005, "Two-way (cell × gene) bootstrap, 95% CI, "
             f"{int(d.n_entries.max()):,} test entries. Filled = CI excludes 0; open = not distinguishable. "
             "Negative = better than the reference.", ha="center", fontsize=fs.FS_SMALL)
    fs.finalize_figure(fig, S / f"figures/fig_test_ci{a.tag}")

    # ---------------- deciles
    show = [c for c in ["R3", "R4", "R5", "R6", "R7", "R8"] if c in order] + \
           [c for c in order if c not in vs.PROB_MODELS and c not in ("PO", "GR")]
    fig, ax = plt.subplots(2, 2, figsize=(19, 11.5), gridspec_kw=dict(wspace=0.25, hspace=0.38))
    for ci, ref in enumerate(["R6", "R4"]):
        for ri, (strat, xl) in enumerate([("depth", "cell-depth decile (train UMI, 1 = shallowest)"),
                                          ("generate", "gene detection-rate decile (1 = rarest)")]):
            x = ax[ri, ci]
            for k, c in enumerate(show):
                if c == ref:
                    continue
                s = d[(d.stratum == strat) & (d.ref == ref) & (d.metric == "nll") & (d.model == c)].sort_values("level")
                off = (k - len(show) / 2) * 0.06
                col = COL.get(c, "#767676")
                x.fill_between(s.level, s.ci_lo * 1e3, s.ci_hi * 1e3, color=col, alpha=0.15, lw=0)
                x.plot(s.level, s.delta * 1e3, color=col, lw=2.2, label=label(c))
                dist = s.distinguishable.to_numpy()
                x.scatter(s.level[dist] + off, s.delta[dist] * 1e3, color=col, s=45, zorder=3)
                x.scatter(s.level[~dist] + off, s.delta[~dist] * 1e3, facecolor="white", edgecolor=col, s=45,
                          zorder=3, lw=2)
            x.axhline(0, color="k", lw=1.2)
            x.set_xticks(range(1, 11))
            x.set_xlabel(xl)
            x.set_ylabel("Δ test NLL (10⁻³ nats / entry)")
            x.set_title(f"minus {ref} ({NAME[ref]})", loc="left")
    h, lab = ax[0, 0].get_legend_handles_labels()
    h2, lab2 = ax[0, 1].get_legend_handles_labels()
    for hh, ll in zip(h2, lab2):
        if ll not in lab:
            h.append(hh); lab.append(ll)
    srt = sorted(range(len(lab)), key=lambda i: lab[i])
    h, lab = [h[i] for i in srt], [lab[i] for i in srt]
    fig.legend(h, lab, loc="lower center", ncol=len(lab), bbox_to_anchor=(0.5, -0.045), fontsize=fs.FS_LEGEND)
    for x, l in zip(ax.ravel(), "abcd"):
        fs.panel_label(x, l, x=-0.12)
    fig.text(0.5, -0.075, "Bands: two-way bootstrap 95% CI within the decile. Filled marker = CI excludes 0. "
             "Poisson, latent blocks and the gene-rate baseline are off scale (see test_ci.csv).",
             ha="center", fontsize=fs.FS_SMALL)
    fs.finalize_figure(fig, S / f"figures/fig_test_ci_deciles{a.tag}")


if __name__ == "__main__":
    main()
