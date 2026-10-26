"""Full-data held-out CV figure: NLL / AUC / Brier vs lambda for each rank k, with baselines."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figstyle import PALETTE, apply_publication_style, finalize_figure

OUT = Path(__file__).resolve().parents[3] / "outputs/bernfac_20261020/full"


def main():
    allr = pd.read_csv(OUT / "tables/cv_summary_all.csv")
    d = allr[allr.model == "no offsets"]
    do = allr[allr.model == "offsets"]
    dc = allr[(allr.model == "cloglog") & (allr.k > 0)]
    base = json.loads((OUT / "tables/baselines.json").read_text())
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    ks = sorted(d.k.unique())
    ramp = plt.get_cmap("Blues")(np.linspace(0.4, 1.0, len(ks)))
    for ax, metric, lab in zip(axes, ["nll", "auc", "brier"],
                               ["held-out NLL / entry", "held-out AUC", "held-out Brier"]):
        for c, k in zip(ramp, ks):
            s = d[d.k == k].sort_values("lam")
            ax.plot(s.lam, s[metric], "o-", color=c, lw=2, ms=5, label=rf"$k={k}$")
            so = do[do.k == k].sort_values("lam")
            if len(so):
                ax.plot(so.lam, so[metric], "s--", color=c, lw=1.6, ms=5, mfc="white",
                        label=rf"$k={k}$ + offsets")
            sc = dc[dc.k == k].sort_values("lam")
            if len(sc):
                ax.plot(sc.lam, sc[metric], "D:", color=PALETTE["violet"], alpha=0.35 + 0.2 * ks.index(k), lw=1.8, ms=5,
                        label=rf"$k={k}$ cloglog")
        ax.axhline(base["additive_logistic"][metric], color=PALETTE["red_strong"], ls="--", lw=1.6,
                   label="cell + gene logistic")
        ax.axhline(base["gene_rate"][metric], color=PALETTE["red_2"], ls=":", lw=1.6, label="gene rate")
        k0 = allr[(allr.model == "cloglog") & (allr.k == 0)]
        if len(k0):
            ax.axhline(k0[metric].item(), color=PALETTE["violet"], ls="--", lw=1.4, label="cell + gene cloglog")
        best = allr[allr.k > 0].loc[allr[allr.k > 0].nll.idxmin()]
        ax.plot(best.lam, best[metric], marker="*", ms=16, color="#FFD700", mec="k", zorder=5)
        ax.set_xscale("log"); ax.set_xlabel(r"prior precision $\lambda$"); ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", ncol=3, bbox_to_anchor=(0, -0.22))
    axes[0].set_title(rf"best: $k={int(best.k)},\ \lambda={best.lam:g}$ + offsets (NLL {best.nll:.4f})", fontsize=12, pad=14)
    fig.suptitle(f"Full data, {int(best.n_heldout):,} held-out entries (5%)", fontsize=14, y=1.06)
    for i, ax in enumerate(axes):
        ax.text(-0.12, 1.05, "abc"[i], transform=ax.transAxes, fontsize=16, weight="bold")
    finalize_figure(fig, OUT / "figures/fig_full_cv")


if __name__ == "__main__":
    main()
