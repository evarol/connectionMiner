"""Tables and figures for the v4 synthetic benchmark (house style figstyle.py, colours v3style.py).

    python syn_figures_v4.py          # after `synthetic_benchmark.py collect`

Tables (outputs/bernfac_20261020/v4/tables):
  syn_v4_selection.csv        method x world x seed: selected (validation) and oracle (test KL) config, regret,
                              grid-edge flags, iterations / convergence of the selected fit
  syn_v4_summary.csv          method x world: mean / sd over seeds of KL, RMSE, test NLL (selected and oracle)
  syn_v4_vs_v3.csv            v3 vs v4 selected KL per method x world
  syn_v4_embed.csv            embedding metrics (G3-G5), selected + oracle configs, R1/R2 full and train-only
  syn_v4_depthstd.csv         task 5 depth-standardization ground-truth check
  syn_v4_count_variants.csv   R6 / Poisson on G1 with artificial vs genuine counts
  syn_v4_r8_budget.csv        R8 iterations / termination for every config
  syn_v4_grid_all_configs.csv every config: val NLL, test NLL, test KL, RMSE, AUC, iterations, converged
Figures (outputs/bernfac_20261020/v4/figures): fig_syn_kl_v4, fig_syn_select_v4, fig_syn_embed_v4, fig_syn_depthstd
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import syn_generators as sg  # noqa: E402
import syn_methods as sm  # noqa: E402
import v3style as vs  # noqa: E402

V4 = vs.STUDY / "v4"
OUT, TABLES, FIGS = V4 / "synthetic", V4 / "tables", V4 / "figures"
WORLDS = list(sg.DATASETS)
VARIANTS = ["G1zP", "G1zNB"]
P_METHODS = ["R3", "R4", "R5", "R6", "PO", "R7", "R8", "R9", "B1", "B0"]
MODELS = ["R3", "R4", "R5", "R6", "PO", "R7", "R8", "R9"]
EXTRA = {"PO": ("Poisson counts", "#8C6D1F"), "B1": ("cell + gene logistic", "#767676"),
         "B0": ("gene rate", "#BDBDBD"), "oracle": ("oracle P_true", "#272727")}
BF_AXES = {"k": list(sm.BF_K), "lam": list(sm.BF_LAM)}
AXES = {"R3": BF_AXES, "R4": BF_AXES, "R5": BF_AXES, "R6": BF_AXES, "PO": BF_AXES,
        "R7": {"K": [1, 2, 4], "d": [4, 8, 16]},
        "R8": {"d": list(sm.R8_D), "lamV": list(sm.R8_LAMV)},
        "R9": {"Kr": list(sm.R9_GRID)}}
PRETTY = {"lam": "λ", "lamV": "λ_V"}
WSHORT = {"G1": "G1 low-rank logit", "G2": "G2 NB counts", "G3": "G3 cell types", "G4": "G4 manifold",
          "G5": "G5 blocks"}


def name(c):
    return f"{c} {vs.name(c)}" if c in vs.METHODS else EXTRA[c][0]


def colour(c):
    return vs.colour(c) if c in vs.METHODS else EXTRA[c][1]


def parse(cfg):
    return {k: float(v) for k, v in re.findall(r"(\w+)=([\d.eE+-]+)", str(cfg))}


def cfg_label(m, cfg):
    p = parse(cfg)
    if m == "R9":
        return f"K={int(p['Kr'])}"
    return ", ".join(f"{PRETTY.get(k, k)}={v:g}" for k, v in p.items())


def edge_axes(m, cfg):
    if m not in AXES:
        return []
    p = parse(cfg)
    out = []
    for ax, vals in AXES[m].items():
        if p.get(ax) == min(vals):
            out.append(f"{ax}=min")
        elif p.get(ax) == max(vals):
            out.append(f"{ax}=max")
    return out


def tobool(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


# ----------------------------------------------------------------------------- tables
def load():
    grid = pd.read_csv(OUT / "grid_all_configs.csv")
    grid["selected"], grid["oracle"] = tobool(grid["selected"]), tobool(grid["oracle"])
    emb = pd.read_csv(OUT / "embeddings.csv")
    emb["selected"], emb["oracle"] = tobool(emb["selected"]), tobool(emb["oracle"])
    dep = pd.read_csv(OUT / "depthstd.csv")
    dep["selected"], dep["oracle"] = tobool(dep["selected"]), tobool(dep["oracle"])
    return grid, emb, dep


def tables(grid, emb, dep):
    TABLES.mkdir(parents=True, exist_ok=True)
    grid.to_csv(TABLES / "syn_v4_grid_all_configs.csv", index=False)
    rows = []
    for (g, s, m), sub in grid.groupby(["dataset", "seed", "method"]):
        if m == "oracle":
            continue
        se = sub[sub.selected].iloc[0]
        orc = sub[sub.oracle].iloc[0]
        e = edge_axes(m, se.config)
        rows.append(dict(method=m, world=g, seed=s, grid_size=len(sub), n_failed=int(sub.val_nll.isna().sum()),
                         selected_config=se.config, at_grid_edge=bool(e), edge_axes=";".join(e),
                         oracle_config_by_test_kl=orc.config, oracle_at_edge=bool(edge_axes(m, orc.config)),
                         val_nll_selected=se.val_nll, test_nll_selected=se.test_nll, kl_selected=se.kl,
                         rmse_selected=se.rmse, kl_oracle=orc.kl, test_nll_oracle=orc.test_nll,
                         regret_kl=se.kl - orc.kl, iterations_selected=se.iterations,
                         converged_selected=se.converged, n_configs_converged=int(tobool(sub.converged).sum()),
                         phi_spearman=se.get("phi_spearman", np.nan),
                         phi_median_log_ratio=se.get("phi_median_log_ratio", np.nan)))
    sel = pd.DataFrame(rows)
    sel.to_csv(TABLES / "syn_v4_selection.csv", index=False)
    agg = sel.groupby(["world", "method"]).agg(
        kl_sel_mean=("kl_selected", "mean"), kl_sel_sd=("kl_selected", "std"),
        kl_oracle_mean=("kl_oracle", "mean"), kl_oracle_sd=("kl_oracle", "std"),
        regret_mean=("regret_kl", "mean"), rmse_sel_mean=("rmse_selected", "mean"),
        test_nll_sel_mean=("test_nll_selected", "mean"), test_nll_sel_sd=("test_nll_selected", "std"),
        n_seeds=("seed", "count"), n_edge=("at_grid_edge", "sum"),
        configs=("selected_config", lambda x: "; ".join(x)),
        oracle_configs=("oracle_config_by_test_kl", lambda x: "; ".join(x))).reset_index()
    orc = grid[grid.method == "oracle"].groupby("dataset").test_nll.mean()
    agg["test_nll_truth"] = agg.world.map(orc)
    agg.to_csv(TABLES / "syn_v4_summary.csv", index=False)
    # v3 comparison
    v3 = pd.read_csv(vs.V3 / "synthetic" / "summary.csv")
    v3 = v3[["dataset", "method", "kl_mean", "kl_std", "configs"]].rename(
        columns={"dataset": "world", "kl_mean": "kl_v3_mean", "kl_std": "kl_v3_sd", "configs": "configs_v3"})
    cmp_ = agg[agg.world.isin(WORLDS)].merge(v3, on=["world", "method"], how="left")
    cmp_ = cmp_[["world", "method", "kl_v3_mean", "kl_v3_sd", "configs_v3", "kl_sel_mean", "kl_sel_sd", "configs",
                 "kl_oracle_mean", "oracle_configs", "n_edge"]]
    cmp_.to_csv(TABLES / "syn_v4_vs_v3.csv", index=False)
    emb.to_csv(TABLES / "syn_v4_embed.csv", index=False)
    dep.to_csv(TABLES / "syn_v4_depthstd.csv", index=False)
    cv = sel[sel.method.isin(["R6", "PO", "R4"]) & sel.world.isin(["G1"] + VARIANTS)]
    cv.groupby(["world", "method"]).agg(kl_sel_mean=("kl_selected", "mean"), kl_sel_sd=("kl_selected", "std"),
                                         kl_oracle_mean=("kl_oracle", "mean"),
                                         test_nll_mean=("test_nll_selected", "mean"),
                                         configs=("selected_config", lambda x: "; ".join(x))).reset_index() \
        .to_csv(TABLES / "syn_v4_count_variants.csv", index=False)
    r8 = grid[grid.method == "R8"][["dataset", "seed", "config", "iterations", "termination", "converged", "seconds",
                                     "val_nll", "kl", "selected", "oracle"]]
    r8.to_csv(TABLES / "syn_v4_r8_budget.csv", index=False)
    return sel, agg


# ----------------------------------------------------------------------------- figure helpers
def _logticks(ax):
    from matplotlib import ticker
    lo, hi = ax.get_xlim()
    for subs in (((1,),) if hi / lo > 30 else ()) + ((1, 2, 5), (1, 2, 3, 5), (1, 2, 3, 4, 5, 6, 8)):
        ticks = [m * 10.0 ** e for e in range(-6, 1) for m in subs if lo <= m * 10.0 ** e <= hi]
        if len(ticks) >= 2:
            break
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def _colour_ticklabels(ax, codes):
    for t, m in zip(ax.get_yticklabels(), codes):
        t.set_color(colour(m) if m not in ("B0", "B1") else fs.PALETTE["grey"])


# ----------------------------------------------------------------------------- KL figure
def fig_kl(sel):
    fig = plt.figure(figsize=(22, 7.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[5, 1.05], wspace=0.17, left=0.115, right=0.99, top=0.86,
                             bottom=0.2)
    gl = outer[0, 0].subgridspec(1, 5, wspace=0.10)
    axes = [fig.add_subplot(gl[0, i]) for i in range(5)]
    ys = np.arange(len(P_METHODS))[::-1]
    for ax, g in zip(axes, WORLDS):
        sub = sel[sel.world == g]
        best = None
        for y, m in zip(ys, P_METHODS):
            d = sub[sub.method == m]
            if d.empty:
                continue
            mu, sd = d.kl_selected.mean(), d.kl_selected.std()
            ax.errorbar(mu, y + 0.12, xerr=sd, fmt="o", color=colour(m), ms=8, capsize=3, elinewidth=1.6,
                        mec="white", mew=0.8, zorder=3)
            if m not in ("B0", "B1"):
                mo, so = d.kl_oracle.mean(), d.kl_oracle.std()
                ax.errorbar(mo, y - 0.2, xerr=so, fmt="D", color=colour(m), ms=6.5, mfc="white", mew=1.6,
                            capsize=2, elinewidth=1.0, zorder=3)
                if best is None or mu < best[0]:
                    best = (mu, y + 0.12)
                ne = int(d.at_grid_edge.sum())
                if ne:
                    ax.text(1.0, y, f"edge {ne}/{len(d)}", transform=ax.get_yaxis_transform(), ha="right",
                            va="center", fontsize=fs.FS_TINY - 0.5, color="#E66100")
        if best:
            ax.plot(*best, marker="o", ms=17, mfc="none", mec=fs.PALETTE["grey_dark"], mew=1.8, zorder=4)
        ax.set_xscale("log")
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo * 0.8, hi * 4.0)      # room for the edge tags at the right
        _logticks(ax)
        ax.set_yticks(ys)
        ax.set_yticklabels([name(m) for m in P_METHODS] if g == "G1" else [])
        if g == "G1":
            _colour_ticklabels(ax, P_METHODS)
        ax.set_ylim(-0.7, len(P_METHODS) - 0.3)
        ax.grid(axis="x", color=fs.PALETTE["neutral"], lw=0.8, alpha=0.6)
        ax.set_title(WSHORT[g], fontsize=fs.FS_TITLE)
        ax.set_xlabel("KL$(P_{\\mathrm{true}}\\,\\|\\,\\hat P)$, nats")
    # count-variant panel
    ax = fig.add_subplot(outer[0, 1])
    wl = ["G1", "G1zP", "G1zNB"]
    wlab = {"G1": "as\nbuilt", "G1zP": "Poisson\ngiven Y", "G1zNB": "NB\ngiven Y"}
    for xi, w in enumerate(wl):
        for m, dx in (("R6", -0.18), ("PO", 0.0), ("R4", 0.18)):
            d = sel[(sel.world == w) & (sel.method == m)]
            if w != "G1" and m == "R4":
                d = sel[(sel.world == "G1") & (sel.method == "R4")]   # binary data identical to G1
            if d.empty:
                continue
            ax.errorbar(xi + dx, d.kl_selected.mean(), yerr=d.kl_selected.std(), fmt="o", color=colour(m), ms=8,
                        capsize=3, mec="white", mew=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([wlab[w] for w in wl], fontsize=fs.FS_TINY)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("KL$(P_{\\mathrm{true}}\\,\\|\\,\\hat P)$, nats")
    ax.set_title("G1 counts: same $P_{\\mathrm{true}}$ and $Y$", fontsize=fs.FS_TITLE)
    ax.grid(axis="y", color=fs.PALETTE["neutral"], lw=0.8, alpha=0.6)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=colour(m), ms=8, label=name(m))
                       for m in ("R6", "PO", "R4")], loc="upper right", fontsize=fs.FS_TINY, ncol=1, handletextpad=0.3)
    leg = [Line2D([], [], marker="o", ls="", color="#555", ms=8, label="validation-selected config"),
           Line2D([], [], marker="D", ls="", color="#555", mfc="white", mew=1.6, ms=6.5,
                  label="oracle config (lowest test KL; not selectable)"),
           Line2D([], [], marker="o", ls="", mfc="none", mec=fs.PALETTE["grey_dark"], mew=1.8, ms=15,
                  label="lowest selected KL in the world")]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.45, 0.995), ncol=3, fontsize=fs.FS_LEGEND)
    fig.text(0.45, 0.01, "Test entries, mean $\\pm$ sd over 3 seeds, log scale.  Orange 'edge n/3': the selected "
             "config sits at the end of a grid axis in n seeds.  R8: fixed 250 outer iterations for every config.\n"
             "Right panel: G1 as built uses X = Y(1 + Poisson(1)); the two controls redraw only the positive counts "
             "from a zero-truncated Poisson or NB with Pr(X $\\geq$ 1) = $P_{\\mathrm{true}}$ (R4 sees identical binary data).",
             ha="center", fontsize=fs.FS_SMALL)
    fs.finalize_figure(fig, FIGS / "fig_syn_kl_v4")


# ----------------------------------------------------------------------------- selection figure
def fig_select(grid, sel):
    g0 = grid[grid.method.isin(MODELS) & grid.dataset.isin(WORLDS)]
    mean = g0.groupby(["method", "dataset", "config"], sort=False)["val_nll"].mean().reset_index()
    mean["delta"] = 1e3 * (mean["val_nll"] - mean.groupby(["method", "dataset"])["val_nll"].transform("min"))
    kl = g0.groupby(["method", "dataset", "config"], sort=False)["kl"].mean()
    fig = plt.figure(figsize=(20, 15.5))
    gsp = fig.add_gridspec(2, 4, hspace=0.30, wspace=0.95, left=0.075, right=0.925, top=0.88, bottom=0.05)
    norm = LogNorm(vmin=0.1, vmax=1000)
    cmap = plt.get_cmap(fs.CMAP_SEQ_R).copy()
    for idx, m in enumerate(MODELS):
        r, c = divmod(idx, 4)
        order = {"R3": None}
        cfgs = list(dict.fromkeys(grid.loc[grid.method == m, "config"]))
        if m in AXES:   # sort configs in grid order
            ax_names = list(AXES[m])
            cfgs.sort(key=lambda s: tuple(parse(s).get(a, 0) for a in ax_names))
        del order
        if not cfgs:
            continue
        M = np.full((len(cfgs), len(WORLDS)), np.nan)
        for j, g in enumerate(WORLDS):
            mm = mean[(mean.method == m) & (mean.dataset == g)].set_index("config")["delta"]
            M[:, j] = [mm.get(k, np.nan) for k in cfgs]
        sub = gsp[r, c].subgridspec(12, 1)
        ax = fig.add_subplot(sub[:len(cfgs), 0])
        ax.imshow(np.clip(M, 0.1, None), cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=fs.FS_TINY - 1, color="#999")
                    continue
                txt = "0" if v < 0.05 else (f"{v:.1f}" if v < 10 else f"{v:.0f}")
                ax.text(j, i, txt, ha="center", va="center", fontsize=fs.FS_TINY,
                        color="white" if norm(max(v, 0.1)) > 0.55 else "#272727")
        heads = []
        for j, g in enumerate(WORLDS):
            s = sel[(sel.method == m) & (sel.world == g)]
            if s.empty:
                heads.append(g); continue
            picks = s.selected_config.value_counts()
            for cfg, n in picks.items():
                i = cfgs.index(cfg)
                edge = bool(edge_axes(m, cfg))
                ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False,
                                       ec="#E66100" if edge else "black", lw=3.0 if n == picks.max() else 1.6,
                                       ls="-" if n == picks.max() else (0, (2, 1.5))))
            # oracle (lowest mean test KL over seeds) marked with a star in the cell corner
            kk = kl.loc[(m, g)] if (m, g) in set(kl.index.droplevel(2)) else None
            if kk is not None:
                io = cfgs.index(kk.idxmin())
                ax.plot(j + 0.33, io - 0.30, marker="*", ms=11, color="#D62728", mec="white", mew=0.6, zorder=5)
            heads.append(f"{g}\n{picks.max()}/{len(s)}")
        ax.set_xticks(range(len(WORLDS))); ax.set_xticklabels(heads, fontsize=fs.FS_TINY + 0.5)
        ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=2)
        ax.set_yticks(range(len(cfgs))); ax.set_yticklabels([cfg_label(m, k) for k in cfgs], fontsize=fs.FS_TINY + 0.5)
        ax.tick_params(axis="y", length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(f"{m} {name(m).split(' ', 1)[1] if m in vs.METHODS else EXTRA[m][0]}", color=colour(m),
                     fontweight="bold", fontsize=fs.FS_TITLE - 1, pad=36, loc="left", x=-0.02)
    cax = fig.add_axes([0.945, 0.35, 0.011, 0.35])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("validation ΔNLL vs best config (×10⁻³ nats)", fontsize=fs.FS_SMALL)
    cb.ax.tick_params(labelsize=fs.FS_TINY)
    fig.text(0.01, 0.995, "v4 synthetic tuning: validation NLL per config (mean of 3 seeds); header n/3 = seeds that "
             "chose the boxed config", fontsize=fs.FS_SMALL + 1, va="top")
    fig.text(0.01, 0.973, "Box: validation-selected config (thick = most common, dashed = other seeds); black = "
             "interior, orange = at the end of a grid axis.   Red star: oracle config (lowest mean test KL).",
             fontsize=fs.FS_SMALL, va="top", color=fs.PALETTE["grey"])
    fig.text(0.01, 0.952, "True rank: 8 in G1 and G2.  R8: d in {2, 5, 10} x λ_V in {10, 100, 1000}, fixed 250 "
             "outer iterations.  R9 ties K_c = K_r.", fontsize=fs.FS_SMALL, va="top", color=fs.PALETTE["grey"])
    fs.finalize_figure(fig, FIGS / "fig_syn_select_v4")


# ----------------------------------------------------------------------------- embedding figure
def fig_embed(emb):
    mets = [("knn_balacc", "kNN balanced accuracy"), ("kmeans_ari", "k-means ARI ($K$ = true)"),
            ("leiden_best_ari", "Leiden ARI (best resolution)")]
    gs3 = ["G3", "G4", "G5"]
    truth = {"G3": "true cell type (10)", "G4": "8 regions of the true latent", "G5": "true row block (8)"}
    rows = [("R1", "full"), ("R1", "train"), ("R2", "full"), ("R2", "train")] + [(m, "model") for m in MODELS]
    lab = {("R1", "full"): "R1 processed counts, all entries", ("R1", "train"): "R1 processed counts, train only",
           ("R2", "full"): "R2 binary PCA, all entries", ("R2", "train"): "R2 binary PCA, train only"}
    fig, axes = plt.subplots(3, 3, figsize=(17, 15), sharey=True, gridspec_kw=dict(wspace=0.08, hspace=0.42))
    ys = np.arange(len(rows))[::-1]
    for j, g in enumerate(gs3):
        for i, (met, mlab) in enumerate(mets):
            ax = axes[i, j]
            best = None
            for y, (m, rep) in zip(ys, rows):
                d = emb[(emb.dataset == g) & (emb.method == m) & (emb.rep == rep)]
                ds = d[d.selected]
                if ds.empty:
                    continue
                mu, sd = ds[met].mean(), ds[met].std()
                filled = rep != "full"
                ax.errorbar(mu, y, xerr=sd, fmt="o", color=colour(m), ms=8, capsize=3, elinewidth=1.6,
                            mfc=colour(m) if filled else "white", mec=colour(m) if not filled else "white",
                            mew=1.6 if not filled else 0.8, zorder=3)
                if rep != "full" and (best is None or mu > best[0]):
                    best = (mu, y)
                do = d[d.oracle]
                if rep == "model" and len(do):
                    ax.plot(do[met].mean(), y - 0.28, marker="D", ms=5, mfc="white", mec=colour(m), mew=1.3,
                            ls="", zorder=3)
            if best:
                ax.plot(*best, marker="o", ms=17, mfc="none", mec=fs.PALETTE["grey_dark"], mew=1.8, zorder=4)
            ax.set_yticks(ys)
            ax.set_ylim(-0.7, len(rows) - 0.3)
            ax.grid(axis="x", color=fs.PALETTE["neutral"], lw=0.8, alpha=0.6)
            ax.set_xlabel(mlab)
            if i == 0:
                ax.set_title(f"{WSHORT[g]}\nvs {truth[g]}", fontsize=fs.FS_TITLE)
    names = [lab.get(r, name(r[0])) for r in rows]
    for i in range(3):
        axes[i, 0].set_yticklabels(names)
        _colour_ticklabels(axes[i, 0], [r[0] for r in rows])
    leg = [Line2D([], [], marker="o", ls="", color="#555", ms=8, label="train entries only (models: selected config)"),
           Line2D([], [], marker="o", ls="", mfc="white", mec="#555", mew=1.6, ms=8, label="R1/R2 on all entries (v3 protocol)"),
           Line2D([], [], marker="D", ls="", mfc="white", mec="#555", mew=1.3, ms=5, label="oracle config (lowest test KL)"),
           Line2D([], [], marker="o", ls="", mfc="none", mec=fs.PALETTE["grey_dark"], mew=1.8, ms=15,
                  label="best among train-only")]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, fontsize=fs.FS_LEGEND)
    fig.text(0.5, 0.045, "Mean $\\pm$ sd over 3 seeds.  Train only: held-out entries (30%) zero-filled before "
             "normalization (R1) or mean-imputed after centring (R2); factor models are fitted on train entries.",
             ha="center", fontsize=fs.FS_SMALL)
    fs.finalize_figure(fig, FIGS / "fig_syn_embed_v4")


# ----------------------------------------------------------------------------- depth standardization figure
def fig_depth(dep):
    d = dep[dep.selected & dep.dataset.isin(["G1", "G2", "G3", "G4"])]
    ms = ["R4", "R5", "R6", "PO"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), gridspec_kw=dict(wspace=0.28, hspace=0.55))
    for j, g in enumerate(["G1", "G2", "G3", "G4"]):
        ax = axes[0, j]
        sub = d[d.dataset == g]
        x = np.arange(len(ms))
        for xi, m in enumerate(ms):
            s = sub[sub.method == m]
            if s.empty:
                continue
            for dx, col, mk, fc in ((-0.22, "kl_std", "o", colour(m)), (0.0, "kl_std_nostd", "o", "white"),
                                    (0.22, "kl_obs", "s", colour(m))):
                ax.errorbar(xi + dx, s[col].mean(), yerr=s[col].std(), fmt=mk, color=colour(m), mfc=fc,
                            mec=colour(m), mew=1.6, ms=8, capsize=3)
        gap = sub.kl_truth_gap.mean()
        ax.axhline(gap, color=fs.PALETTE["grey"], ls="--", lw=1.4)
        ax.set_xticks(x); ax.set_xticklabels(ms)
        for t, m in zip(ax.get_xticklabels(), ms):
            t.set_color(colour(m))
        ax.set_xlim(-0.5, len(ms) - 0.5)
        ax.set_ylim(0, None)
        ax.set_title(f"{WSHORT[g]}", fontsize=fs.FS_TITLE)
        if j == 0:
            ax.set_ylabel("mean KL over all entries, nats")
        ax.grid(axis="y", color=fs.PALETTE["neutral"], lw=0.8, alpha=0.6)
        ax2 = axes[1, j]
        for xi, m in enumerate(ms):
            s = sub[sub.method == m]
            if s.empty:
                continue
            ax2.errorbar(xi - 0.15, s.rho_resid_obs.mean(), yerr=s.rho_resid_obs.std(), fmt="o", color=colour(m),
                         mfc="white", mew=1.6, ms=8, capsize=3)
            ax2.errorbar(xi + 0.15, s.rho_resid_std.mean(), yerr=s.rho_resid_std.std(), fmt="o", color=colour(m),
                         ms=8, capsize=3)
        rt = sub.rho_resid_true.mean()
        ax2.axhline(rt, color=fs.PALETTE["grey"], ls="--", lw=1.4)
        ax2.axhline(0, color=fs.PALETTE["neutral"], lw=1)
        ax2.set_xticks(x); ax2.set_xticklabels(ms)
        for t, m in zip(ax2.get_xticklabels(), ms):
            t.set_color(colour(m))
        ax2.set_xlim(-0.5, len(ms) - 0.5)
        ax2.set_ylim(-0.6, 1.05)
        if j == 0:
            ax2.set_ylabel("Spearman ρ(true cell offset,\nrow mean of logit p)")
        ax2.set_title(f"{g}: depth left in each cell's profile", fontsize=fs.FS_SMALL + 1)
        ax2.grid(axis="y", color=fs.PALETTE["neutral"], lw=0.8, alpha=0.6)
    leg1 = [Line2D([], [], marker="o", ls="", color="#555", ms=8, label="standardized $\\tilde{p}$ vs true standardized $P$"),
            Line2D([], [], marker="o", ls="", mfc="white", mec="#555", mew=1.6, ms=8,
                   label="observed $\\hat{p}$ vs true standardized $P$"),
            Line2D([], [], marker="s", ls="", color="#555", ms=8, label="observed $\\hat{p}$ vs true observed $P$"),
            Line2D([], [], color=fs.PALETTE["grey"], ls="--", label="KL(standardized truth ‖ observed truth)")]
    fig.legend(handles=leg1, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=fs.FS_LEGEND)
    leg2 = [Line2D([], [], marker="o", ls="", mfc="white", mec="#555", mew=1.6, ms=8, label="observed $\\hat{p}$"),
            Line2D([], [], marker="o", ls="", color="#555", ms=8, label="standardized $\\tilde{p}$ (offset set to median)"),
            Line2D([], [], color=fs.PALETTE["grey"], ls="--", label="true standardized $P$")]
    fig.legend(handles=leg2, loc="upper center", bbox_to_anchor=(0.5, 0.505), ncol=3, fontsize=fs.FS_LEGEND)
    fig.text(0.5, 0.0, "Validation-selected config per method, mean $\\pm$ sd over 3 seeds.  True standardized $P$: "
             "generator with every cell's offset (G1 $a_i$, G2/G3 log depth, G4 $d_i$) set to its median.",
             ha="center", fontsize=fs.FS_SMALL)
    fs.finalize_figure(fig, FIGS / "fig_syn_depthstd")


def main():
    fs.apply_publication_style()
    grid, emb, dep = load()
    sel, agg = tables(grid, emb, dep)
    fig_kl(sel)
    fig_select(grid, sel)
    fig_embed(emb)
    fig_depth(dep)
    pd.set_option("display.width", 250)
    print(agg.pivot(index="method", columns="world", values="kl_sel_mean").round(4))


if __name__ == "__main__":
    main()
