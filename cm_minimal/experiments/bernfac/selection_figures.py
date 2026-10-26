"""Tables and figures for the BernFac synthetic model-selection study.

Reads  outputs/bernfac_20261020/synthetic/results/grid_metrics.csv (+ fits_*.npz)
Writes results/selection_table.csv, results/selection_summary.csv and
       figures/fig*.png|pdf  (house style: scientific-figure-making skill)

Selection logic: for each dataset and criterion, the selected grid point is
the argmin of the criterion over (lambda, k).  Regret = oracle KL of the
full-data fit at the selected point - min oracle KL over the grid.
For C2 (count split) the deployed model is the full-data refit at the
selected hyperparameters, so its regret uses the same full-data oracle grid.

Run:  python selection_figures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from figstyle import (FS, FS_LEGEND, FS_SMALL, FS_TINY, PALETTE, apply_publication_style,
                      finalize_figure, k_ramp, lambda_ramp, panel_label)

import matplotlib.pyplot as plt  # noqa: E402  (figstyle selects the Agg backend)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from selection_study import KS, LAMBDAS, RESULTS
from synthetic_scenarios import OUT as DATA_DIR, SCENARIOS

FIG = RESULTS.parent / "figures"
CRITERIA = {  # column -> (short label, colour)
    "C1_entry": ("C1 entry-CV", PALETTE["blue_main"]),
    "C2_split": ("C2 count split", PALETTE["teal"]),
    "C3_rep": ("C3 replicate", PALETTE["green_3"]),
    "C4_cellfold": ("C4 cell fold-in", PALETTE["violet"]),
    "C5_bic": ("C5 BIC", PALETTE["red_strong"]),
    "C5_aic": ("C5 AIC", PALETTE["red_2"]),
    "train_nll": ("train NLL", PALETTE["neutral"]),
}
SCEN_LABEL = {"S1_k3": "S1 logistic rank 3", "S1_k8": "S1 logistic rank 8",
              "S2": "S2 Poisson, clusters", "S3": "S3 Poisson + batch"}
# mathtext snippets (Computer Modern math, sans-serif text)
M_PTRUE, M_PBIO, M_PHAT = r"$P_{\mathrm{true}}$", r"$P_{\mathrm{bio}}$", r"$\hat{P}$"
M_LAM = r"$\lambda$"


def apply_style():
    apply_publication_style()


def save(fig, name):
    paths = finalize_figure(fig, FIG / name)
    size = paths[0].stat().st_size / 1e6
    assert size < 1.5, name


def grid(df, col):
    """(len(LAMBDAS), len(KS)) matrix of a column for one dataset."""
    return df.pivot(index="lam", columns="k", values=col).reindex(index=LAMBDAS, columns=KS).to_numpy()


# ----------------------------------------------------------------------------
# Selection tables
# ----------------------------------------------------------------------------
def selection_table(G):
    rows = []
    for name, df in G.groupby("dataset"):
        best = df.loc[df.kl.idxmin()]
        targets = {"kl": best}
        if df.kl_bio.notna().any():
            targets["kl_bio"] = df.loc[df.kl_bio.idxmin()]
        for col in CRITERIA:
            if df[col].isna().all():
                continue
            s = df.loc[df[col].idxmin()]
            r = dict(dataset=name, scenario=df.scenario.iloc[0], seed=df.seed.iloc[0], criterion=col,
                     sel_lam=s.lam, sel_k=int(s.k), oracle_lam=best.lam, oracle_k=int(best.k),
                     kl_sel=s.kl, kl_best=best.kl, regret_kl=s.kl - best.kl,
                     rel_regret=(s.kl - best.kl) / best.kl,
                     rmse_sel=s.rmse, rmse_best=df.rmse.min(), lcor_sel=s.lcor,
                     hit=bool(s.lam == best.lam and s.k == best.k),
                     spearman=spearmanr(df[col], df.kl).correlation)
            if "kl_bio" in targets:
                b = targets["kl_bio"]
                r.update(kl_bio_sel=s.kl_bio, kl_bio_best=b.kl_bio, regret_kl_bio=s.kl_bio - b.kl_bio,
                         bio_oracle_lam=b.lam, bio_oracle_k=int(b.k),
                         dec_library_sel=s.dec_library, dec_cluster_sel=s.dec_cluster)
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(RESULTS / "selection_table.csv", index=False)
    S = (T.groupby(["scenario", "criterion"])
         .agg(hits=("hit", "sum"), n=("hit", "size"), regret_kl_mean=("regret_kl", "mean"),
              regret_kl_max=("regret_kl", "max"), rel_regret_mean=("rel_regret", "mean"),
              kl_best_mean=("kl_best", "mean"), spearman_mean=("spearman", "mean"),
              sel=("sel_k", lambda x: ",".join(map(str, x))))
         .reset_index())
    S.to_csv(RESULTS / "selection_summary.csv", index=False)
    return T, S


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def box(ax, xy, w, h, text, fc, ec=PALETTE["grey"], fs=FS_SMALL + 1, color="black", weight="normal"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            weight=weight, linespacing=1.35)


def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18, lw=1.8,
                                 color=PALETTE["grey"], shrinkA=2, shrinkB=2, **kw))


def fig_design():
    fig, ax = plt.subplots(figsize=(17, 8))
    ax.set_xlim(0, 13.4); ax.set_ylim(-0.1, 6.3); ax.axis("off")
    # truth
    box(ax, (0.05, 4.15), 2.75, 1.95,
        "Ground truth " + M_PTRUE + "\n"
        r"S1: $\sigma(U^{*}V^{*\top})$" "\n"
        r"S2: $1-e^{-\mu}$, Poisson" "\n"
        "S3: + library batch\n"
        "(" + M_PBIO + " = batch removed)", PALETTE["green_1"])
    box(ax, (0.05, 1.85), 2.75, 1.6, "Calibrated to real\n1000×100 preMN $Y$\n(4 libraries × 5\nhemilineages, density 0.49)",
        "white")
    arrow(ax, (1.42, 3.45), (1.42, 4.15))
    # sample
    box(ax, (3.25, 4.45), 2.3, 1.3, r"Sample $Y \sim \mathrm{Bern}(P_{\mathrm{true}})$" "\n"
        r"(counts $X \to \mathbf{1}[X \geq 1]$)", PALETTE["neutral"])
    arrow(ax, (2.8, 5.1), (3.25, 5.1))
    # splits
    splits = [("C1 entry CV\n5 × 10% entries masked", PALETTE["red_1"]),
              ("C2 count split\n" r"$X_A \sim \mathrm{Bin}(X, \frac{1}{2})$, $X_B = X - X_A$", PALETTE["red_1"]),
              ("C3 replicate\n" r"$Y_{\mathrm{rep}} \sim \mathrm{Bern}(P_{\mathrm{true}})$", PALETTE["red_1"]),
              ("C4 cell split: fit 80% cells,\nfold-in $u_i$ on half genes", PALETTE["red_1"]),
              ("C5 BIC / AIC\n" r"$k(n+m)$ params", PALETTE["red_1"])]
    for i, (t, c) in enumerate(splits):
        y = 4.95 - i * 1.07
        box(ax, (6.05, y), 2.85, 0.9, t, c, fs=FS_SMALL)
        arrow(ax, (5.55, 5.1), (6.05, y + 0.45))
    # fit grid
    box(ax, (9.4, 2.55), 1.95, 2.3, "Fit grid\n" r"$\lambda \in \{0.03, \ldots, 10\}$" "\n"
        r"$\times\ k \in \{1, \ldots, 20\}$" "\n(svd + 2 random\nstarts, PG-EM)",
        PALETTE["blue_secondary"], color="white")
    for i in range(5):
        arrow(ax, (8.9, 4.95 - i * 1.07 + 0.45), (9.4, 3.7))
    # outputs
    box(ax, (11.75, 4.0), 1.6, 1.55, "Criterion\nargmin\n" r"$(\lambda, k)$", PALETTE["blue_main"],
        color="white", weight="bold")
    box(ax, (11.75, 1.55), 1.6, 1.7, "Oracle\nKL, RMSE,\nlogit corr\nvs " + M_PTRUE + " / " + M_PBIO,
        PALETTE["green_2"])
    arrow(ax, (11.35, 4.2), (11.75, 4.7))
    arrow(ax, (11.35, 3.1), (11.75, 2.4))
    arrow(ax, (12.55, 3.25), (12.55, 4.0))
    ax.text(12.65, 3.62, "regret", fontsize=FS_SMALL + 1, ha="left", va="center", style="italic")
    ax.plot([2.8, 3.0, 3.0, 12.55], [4.3, 4.3, 0.45, 0.45], "--", color=PALETTE["grey"], lw=1.6)
    arrow(ax, (12.55, 0.45), (12.55, 1.55), linestyle="--")
    ax.text(10.3, 0.58, M_PTRUE, fontsize=FS_SMALL + 1, ha="center", color=PALETTE["grey"])
    ax.text(7.0, 0.0, "truth is used only by the oracle (and to draw " + r"$Y_{\mathrm{rep}}$" + " for C3)",
            fontsize=FS_SMALL + 1, ha="center", style="italic", color=PALETTE["grey"])
    save(fig, "fig1_design_schematic")


def fig_heatmaps(G):
    cols = ["kl", "C1_entry", "C2_split", "C3_rep", "C4_cellfold", "C5_bic"]
    titles = ["Oracle KL", "C1 entry-CV", "C2 count split", "C3 replicate", "C4 cell fold-in", "C5 BIC"]
    fig, axes = plt.subplots(len(SCENARIOS), len(cols), figsize=(19, 12.5))
    for r, sc in enumerate(SCENARIOS):
        df = G[G.dataset == f"{sc}_seed0"]
        okl = grid(df, "kl")
        oi = np.unravel_index(np.nanargmin(okl), okl.shape)
        for c, col in enumerate(cols):
            ax = axes[r, c]
            M = grid(df, col)
            if r == 0:
                ax.set_title(titles[c], fontsize=FS)
            if np.all(np.isnan(M)):
                ax.axis("off"); ax.text(0.5, 0.5, "n/a\n(no counts)", ha="center", va="center",
                                        transform=ax.transAxes, color=PALETTE["grey"], fontsize=FS_SMALL + 1)
                continue
            D = np.log10(M - np.nanmin(M) + 1e-4)  # distance to the column's minimum, log scale
            im = ax.imshow(D, cmap="viridis_r", aspect="auto", vmin=-4, vmax=1)
            si = np.unravel_index(np.nanargmin(M), M.shape)
            ax.plot(oi[1], oi[0], "o", ms=15, mfc="none", mec="white", mew=2.6)
            ax.plot(si[1], si[0], "*", ms=15, color=PALETTE["highlight"], mec="k", mew=0.8)
            ax.set_xticks(range(len(KS)), KS, fontsize=FS_TINY)
            ax.set_yticks(range(len(LAMBDAS)), [f"{l:g}" for l in LAMBDAS], fontsize=FS_TINY)
            ax.tick_params(length=0)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            if c == 0:
                ax.set_ylabel(f"{SCEN_LABEL[sc]}\n" + M_LAM, fontsize=FS_SMALL + 1)
            if r == len(SCENARIOS) - 1:
                ax.set_xlabel("$k$")
    fig.subplots_adjust(top=0.88, wspace=0.32, hspace=0.3)
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.015)
    cb.set_label(r"$\log_{10}$(value $-$ grid minimum)")
    handles = [Line2D([], [], marker="o", ls="none", ms=13, mfc="none", mec="k", mew=2, label="oracle-best"),
               Line2D([], [], marker="*", ls="none", ms=15, color=PALETTE["highlight"], mec="k", mew=0.8,
                      label="criterion argmin")]
    fig.suptitle("Grid landscapes (seed 0). White circle = oracle-best; gold star = criterion argmin",
                 y=0.975, fontsize=FS + 1)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, 0.955), ncol=2, fontsize=FS_LEGEND + 1)
    save(fig, "fig2_grid_heatmaps")


def fig_scatter(G):
    crit = ["C1_entry", "C2_split", "C3_rep", "C4_cellfold"]
    fig, axes = plt.subplots(len(crit), len(SCENARIOS), figsize=(17, 14))
    cmap = plt.get_cmap("viridis")
    for c, sc in enumerate(SCENARIOS):
        df = G[G.dataset == f"{sc}_seed0"]
        x = df.kl - df.kl.min()
        for r, col in enumerate(crit):
            ax = axes[r, c]
            if df[col].isna().all():
                ax.axis("off")
                if c == 0:
                    ax.text(0.5, 0.5, f"{CRITERIA[col][0]}:\nn/a for S1\n(no counts)", ha="center", va="center",
                            transform=ax.transAxes, fontsize=FS, color=PALETTE["grey"])
                continue
            y = df[col] - df[col].min()
            sc_ = ax.scatter(x + 1e-4, y + 1e-4, c=np.log(df.k), cmap=cmap, vmin=np.log(min(KS)), vmax=np.log(max(KS)),
                             s=30 + 14 * np.log10(df.lam / 0.03 + 1), edgecolor="k", lw=0.4)
            lim = [1e-4, max(x.max(), y.max()) * 2]
            ax.plot(lim, lim, "--", color=PALETTE["grey"], lw=1.4)
            ax.set_xscale("log"); ax.set_yscale("log")
            rho = spearmanr(df[col], df.kl).correlation
            ax.text(0.05, 0.9, rf"$\rho$ = {rho:.2f}", transform=ax.transAxes, fontsize=FS_SMALL + 1)
            if r == 0:
                ax.set_title(SCEN_LABEL[sc], fontsize=FS)
            if c == 0:
                ax.set_ylabel(f"{CRITERIA[col][0]}\n(value $-$ min)")
            if r == len(crit) - 1:
                ax.set_xlabel(r"oracle KL $-$ min")
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    cb = fig.colorbar(sc_, ax=axes, shrink=0.4, pad=0.02, ticks=np.log(KS))
    cb.ax.set_yticklabels(KS); cb.set_label(r"$k$ (marker size $\sim \lambda$)")
    fig.suptitle("Criterion vs oracle KL over the 42 grid points (seed 0); dashed = identity", y=0.93, fontsize=FS + 1)
    save(fig, "fig3_criterion_vs_oracle")


def fig_regret(T):
    crit = [c for c in CRITERIA]
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(19, 7))
    for a, sc in enumerate(SCENARIOS):
        ax = axes[a]
        t = T[T.scenario == sc]
        for i, col in enumerate(crit):
            v = t[t.criterion == col].regret_kl.to_numpy()
            if len(v) == 0:
                ax.text(i, 2e-5, "n/a", ha="center", fontsize=FS_SMALL, color=PALETTE["grey"])
                continue
            ax.bar(i, np.mean(v) + 1e-5, color=CRITERIA[col][1], edgecolor="k", lw=1.5, width=0.75)
            ax.plot(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v + 1e-5, "o", color="k", ms=5)
        ax.set_yscale("log"); ax.set_ylim(5e-6, None)
        top = ax.get_ylim()[1]
        ax.set_ylim(5e-6, top * 8)  # headroom for the in-panel note
        ax.set_xticks(range(len(crit)), [CRITERIA[c][0] for c in crit], rotation=50, ha="right",
                      rotation_mode="anchor", fontsize=FS_SMALL + 1)
        ax.set_title(SCEN_LABEL[sc], fontsize=FS)
        ax.text(0.03, 0.97, f"best KL ~ {t.kl_best.mean():.4f}", transform=ax.transAxes, fontsize=FS_SMALL,
                va="top")
        panel_label(ax, "abcd"[a], x=-0.2, y=1.05)
    axes[0].set_ylabel("regret: KL(selected) $-$ KL(best)\n" r"(nats/entry, $+10^{-5}$)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4_regret")


def fig_convergence():
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(19, 6.5))
    kcols = dict(zip([1, 3, 8, 20], k_ramp(4)))
    lss = {0.1: ":", 1.0: "-", 10.0: "--"}
    for a, sc in enumerate(SCENARIOS):
        f = np.load(RESULTS / f"fits_{sc}_seed0.npz")
        ax = axes[a]
        for k, color in kcols.items():
            for lam, ls in lss.items():
                i = np.flatnonzero((f["k"] == k) & (f["lam"] == lam))[0]
                tr = f["trace"][i]
                tr = tr[~np.isnan(tr)]
                gap = tr[-1] - tr[:-1] + 1e-3
                ax.plot(np.arange(1, len(gap)), gap[1:], ls, color=color, lw=2.0)
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_title(SCEN_LABEL[sc], fontsize=FS); ax.set_xlabel("PG-EM iteration")
        panel_label(ax, "abcd"[a], x=-0.2, y=1.05)
    axes[0].set_ylabel(r"$J(\mathrm{final}) - J(\mathrm{iter})$  ($+10^{-3}$)")
    h = [Line2D([], [], color=c, lw=3, label=f"$k={k}$") for k, c in kcols.items()] + \
        [Line2D([], [], color=PALETTE["grey"], ls=ls, lw=2.2, label=rf"$\lambda={lam:g}$") for lam, ls in lss.items()]
    fig.tight_layout(w_pad=2.0, rect=(0, 0.1, 1, 1))
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=7, fontsize=FS_LEGEND + 1,
               columnspacing=1.8, handlelength=2.6)
    save(fig, "fig5_convergence")


def fig_calibration(G, T):
    fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(18, 10.5))
    for a, sc in enumerate(SCENARIOS):
        name = f"{sc}_seed0"
        d = np.load(DATA_DIR / f"{name}.npz")
        f = np.load(RESULTS / f"fits_{name}.npz")
        P = d["P_true"].ravel()
        s = T[(T.dataset == name) & (T.criterion == "C1_entry")].iloc[0]
        i = np.flatnonzero((f["k"] == s.sel_k) & (f["lam"] == s.sel_lam))[0]
        Ph = f["P_hat"][i].astype(float).ravel()
        ax = axes[0, a]
        ax.hexbin(P, Ph, gridsize=60, bins="log", cmap="Blues", mincnt=1, extent=(0, 1, 0, 1), rasterized=True)
        ax.plot([0, 1], [0, 1], "--", color=PALETTE["red_strong"], lw=1.6)
        ax.set_title(f"{SCEN_LABEL[sc]}\nC1 pick: " + rf"$\lambda={s.sel_lam:g}$, $k={s.sel_k}$", fontsize=FS)
        ax.set_xlabel(M_PTRUE); ax.set_aspect("equal")
        ax.text(0.05, 0.95, f"RMSE {s.rmse_sel:.3f}\nlogit r {s.lcor_sel:.3f}", transform=ax.transAxes,
                fontsize=FS_SMALL, va="top")
        # reliability: binned mean of P_true by P_hat bins (and an under-fit/over-fit contrast)
        ax = axes[1, a]
        bins = np.linspace(0, 1, 21)
        for (lam, k, lab, col) in [(s.sel_lam, s.sel_k, "C1 pick", PALETTE["blue_main"]),
                                   (10.0, 20, r"$\lambda=10$, $k=20$", PALETTE["teal"]),
                                   (0.03, 20, r"$\lambda=0.03$, $k=20$", PALETTE["red_strong"])]:
            j = np.flatnonzero((f["k"] == k) & (f["lam"] == lam))[0]
            q = f["P_hat"][j].astype(float).ravel()
            idx = np.digitize(q, bins) - 1
            xs = [q[idx == b].mean() for b in range(20) if np.sum(idx == b) > 50]
            ys = [P[idx == b].mean() for b in range(20) if np.sum(idx == b) > 50]
            ax.plot(xs, ys, "o-", color=col, ms=5, lw=2.0, label=lab)
        ax.plot([0, 1], [0, 1], "--", color=PALETTE["grey"], lw=1.2)
        ax.set_xlabel(M_PHAT + " (binned)"); ax.set_aspect("equal")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if a == 0:
            ax.legend(fontsize=FS_LEGEND, loc="upper left")
        panel_label(axes[0, a], "abcd"[a], x=-0.25, y=1.12)
        panel_label(axes[1, a], "efgh"[a], x=-0.25, y=1.04)
    axes[0, 0].set_ylabel(M_PHAT); axes[1, 0].set_ylabel("mean " + M_PTRUE + " in bin")
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    save(fig, "fig6_calibration")


def fig_batch(G, ref):
    df = G[G.scenario == "S3"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 7.2))
    m = df.groupby(["lam", "k"])[["kl", "kl_bio", "dec_library", "dec_cluster"]].mean().reset_index()
    for ax, col, title in [(axes[0], "kl", "KL to " + M_PTRUE + " (with batch)"),
                           (axes[1], "kl_bio", "KL to " + M_PBIO + " (batch removed)")]:
        M = grid(m, col)
        im = ax.imshow(np.log10(M), cmap="viridis_r", aspect="auto")
        i = np.unravel_index(np.argmin(M), M.shape)
        ax.plot(i[1], i[0], "o", ms=15, mfc="none", mec="white", mew=2.6)
        ax.set_xticks(range(len(KS)), KS); ax.set_yticks(range(len(LAMBDAS)), [f"{l:g}" for l in LAMBDAS])
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_xlabel("$k$"); ax.set_ylabel(M_LAM); ax.set_title(title + "\n(mean of 3 seeds)", fontsize=FS)
        cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cb.set_label(r"$\log_{10}$ KL")
    r = ref[ref.dataset.str.startswith("S3")].mean(numeric_only=True)
    refs = [("P_true", M_PTRUE, dict(color=PALETTE["red_strong"], ls="--", lw=2.0)),
            ("P_bio", M_PBIO, dict(color=PALETTE["green_3"], ls="--", lw=3.0)),
            ("Y", "raw $Y$", dict(color=PALETTE["grey"], ls="-.", lw=2.0))]
    for ax, col, chance, refkey, title in [(axes[2], "dec_library", 0.25, "library", "Library decoding"),
                                           (axes[3], "dec_cluster", 0.2, "cluster", "Hemilineage decoding")]:
        for lam, color in zip(LAMBDAS, lambda_ramp(len(LAMBDAS))):
            mm = m[m.lam == lam]
            ax.plot(mm.k, mm[col], "o-", color=color, ms=5, lw=2.0, label=rf"$\lambda={lam:g}$")
        ax.axhline(chance, color=PALETTE["neutral"], ls=":", lw=2.2, label="chance")
        for key, lab, st in refs:
            ax.axhline(r[f"{key}_dec_{refkey}"], label=lab, **st)
        ax.set_xscale("log"); ax.set_xticks(KS, KS); ax.minorticks_off(); ax.set_xlabel("$k$"); ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=FS)
    axes[2].set_ylabel("5-fold CV accuracy\n(rows of logit " + M_PHAT + ")")
    for ax, l_ in zip(axes, "abcd"):
        panel_label(ax, l_, x=-0.25, y=1.1)
    h, l = axes[2].get_legend_handles_labels()
    fig.tight_layout(w_pad=2.5, rect=(0, 0.13, 1, 1))
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=FS_LEGEND + 1,
               columnspacing=1.8)
    save(fig, "fig7_batch_S3")


def fig_count_split(G, T):
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    for sc, color in [("S2", PALETTE["blue_main"]), ("S3", PALETTE["red_strong"])]:
        df = G[G.scenario == sc]
        m = df.groupby(["lam", "k"])[["kl", "C2_kl_half", "C2_kl_mapped", "C2_kl_unmapped", "C2_split"]].mean().reset_index()
        axes[0].scatter(m.C2_kl_half, m.C2_split, color=color, s=36, edgecolor="k", lw=0.3, label=sc)
        axes[1].scatter(m.kl, m.C2_kl_mapped, color=color, s=36, edgecolor="k", lw=0.3,
                        label=f"{sc} mapped " + r"$1-(1-p)^2$")
        axes[1].scatter(m.kl, m.C2_kl_unmapped, color=color, s=40, marker="x", lw=1.5, alpha=0.7,
                        label=f"{sc} un-mapped " + r"$p_{\mathrm{half}}$")
        # best k per lambda: full-data oracle vs count-split criterion
        for lam_lab, key, ls, mk, dx in [("full oracle", "kl", "-", "o", 1.0), ("C2 criterion", "C2_split", "--", "x", 1.12)]:
            bk = m.loc[m.groupby("lam")[key].idxmin()]
            off = 0.15 if sc == "S3" else -0.15
            axes[2].plot(bk.lam * dx, bk.k + off, ls, marker=mk, ms=9, mew=2, lw=2.0, color=color, label=f"{sc} {lam_lab}")
    axes[0].set_xlabel(r"$\mathrm{KL}(p_{\mathrm{half}}\,\|\,\hat{p}_A)$  [half-depth oracle]")
    axes[0].set_ylabel(r"C2: NLL of $Y_B$ under $\hat{p}_A$")
    axes[0].set_xscale("log"); axes[0].legend(fontsize=FS_LEGEND)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    lim = [G.kl.min() * 0.8, G.C2_kl_unmapped.max() * 1.2]
    axes[1].plot(lim, lim, "--", color=PALETTE["grey"], lw=1.4)
    axes[1].set_xlabel("KL of full-data fit to " + M_PTRUE); axes[1].set_ylabel("KL of half-data fit to " + M_PTRUE)
    axes[1].legend(fontsize=FS_LEGEND, loc="upper left")
    axes[2].set_xscale("log"); axes[2].set_xlabel(M_LAM); axes[2].set_ylabel(r"best $k$ at this $\lambda$")
    axes[2].legend(fontsize=FS_LEGEND)
    for ax, l_ in zip(axes, "abc"):
        panel_label(ax, l_, x=-0.18, y=1.04)
    fig.suptitle("Count splitting: half-depth shift (mean of 3 seeds per grid point)", y=0.985, fontsize=FS + 1)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.13, top=0.85, wspace=0.3)
    save(fig, "fig8_count_split")


def main():
    apply_style()
    G = pd.read_csv(RESULTS / "grid_metrics.csv")
    ref = pd.read_csv(RESULTS / "reference_decoding.csv")
    T, S = selection_table(G)
    pd.set_option("display.width", 200)
    print(S.to_string(index=False))
    import sys
    if "--skip-done" not in sys.argv:
        fig_design()
        fig_heatmaps(G)
        fig_scatter(G)
        fig_regret(T)
    fig_convergence()
    fig_calibration(G, T)
    fig_batch(G, ref)
    fig_count_split(G, T)


if __name__ == "__main__":
    main()
