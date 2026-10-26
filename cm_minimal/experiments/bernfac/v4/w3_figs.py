"""W3 step 4: tables and figures from the w3 caches.

Tables  (v4/tables): rep_metrics_v4.csv (long), depth_axes_v4.csv, confounds_v4.csv
Figures (v4/figures): fig_rep_fair, fig_depth_axes_v4, fig_confounds,
                      fig_umap_grid_hemilineage_v4, fig_umap_grid_depth_v4
"""
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from w3_common import CACHE, FIGS, ORDER, TABS, colour, load_cells, name
from figstyle import (PALETTE, FS, FS_LEGEND, FS_SMALL, FS_TICK, FS_TINY, apply_publication_style,
                      finalize_figure, panel_label)
from umap_compare import QUAL23

apply_publication_style()
REPS = [r for r in ORDER if (CACHE / f"w3_metrics_{r}.csv").exists()]
GREY = PALETTE["grey_mid"]


# ------------------------------------------------------------------ tables
def build_tables():
    m = pd.concat([pd.read_csv(CACHE / f"w3_metrics_{r}.csv") for r in REPS], ignore_index=True)
    m.insert(1, "name", m.rep.map(name))
    m["d_requested"] = m["d_requested"].astype(str)
    m.to_csv(TABS / "rep_metrics_v4.csv", index=False)
    d = pd.concat([pd.read_csv(CACHE / f"w3_depth_{r}.csv") for r in REPS], ignore_index=True)
    d.insert(1, "name", d.rep.map(name))
    d.to_csv(TABS / "depth_axes_v4.csv", index=False)
    cf = json.loads((CACHE / "w3_confounds_raw.json").read_text())
    rows = []
    for metric in ("lib_knn_within_hl_balacc", "lib_knn_within_hl_chance", "sex_knn_within_mixed_lib_balacc",
                   "sex_knn_balacc_all", "lib_knn_balacc_all", "birthtime_knn_balacc"):
        s = m[(m.metric == metric) & (m.d_requested == "20")]
        for _, r in s.iterrows():
            rows.append(dict(section="embedding (d=20)", rep=r.rep, item=f"{metric} [{r.scaling}]", value=r.value))
    for r in cf["rox_detection"]:
        rows.append(dict(section="roX detection", rep="data", item=f"{r['gene']} detected, {r['sex']}, {r['libraries']} libraries (n={r['n']})",
                         value=r["detected_frac"]))
    for k in ("mixed", "single-sex", "all"):
        for kk, v in cf[f"rox_call_{k}"].items():
            rows.append(dict(section="sex from roX", rep="data", item=f"{kk}, {k} libraries", value=v))
    for k, v in cf["eta2_log_umi"].items():
        rows.append(dict(section="depth variance explained (eta2 of log10 UMI)", rep="data", item=k, value=v))
    for k, v in cf["primary_frac_by_depth"].items():
        rows.append(dict(section="primary-neuron fraction", rep="data", item=k, value=v))
    for k, v in cf["library_by_hemilineage_chi2"].items():
        rows.append(dict(section="library x hemilineage composition", rep="data", item=k, value=v))
    for g, v in cf["rox_in_R1_hvgs"].items():
        rows.append(dict(section="roX in R1 HVGs", rep="data", item=g, value=float(v)))
    pd.DataFrame(rows).to_csv(TABS / "confounds_v4.csv", index=False)
    return m, d, cf


def get(m, metric, dreq, scaling):
    s = m[(m.metric == metric) & (m.d_requested == str(dreq)) & (m.scaling == scaling)].set_index("rep")
    return s


def rep_ticks(ax, reps, flags=None):
    ax.set_yticks(range(len(reps)))
    labs = [f"{r} {name(r)}" + (flags.get(r, "") if flags else "") for r in reps]
    ax.set_yticklabels(labs)
    for t, r in zip(ax.get_yticklabels(), reps):
        t.set_color(colour(r)); t.set_fontweight("bold")
    ax.set_ylim(len(reps) - 0.5, -0.5)


# ------------------------------------------------------------------ fig_rep_fair
def fig_rep_fair(m):
    reps = REPS
    cols = [("hl_probe_balacc", "linear probe\nbalanced accuracy"), ("hl_knn_balacc", "15-NN\nbalanced accuracy"),
            ("hl_leiden23_ari", "Leiden ARI\n(23 clusters, 5 seeds)"), ("hl_leiden23_ami", "Leiden AMI\n(23 clusters, 5 seeds)")]
    fig, axes = plt.subplots(2, 4, figsize=(18, 12.5))
    for i, dreq in enumerate((20, 40)):
        for j, (met, lab) in enumerate(cols):
            ax = axes[i, j]
            for k, r in enumerate(reps):
                ax.axhspan(k - 0.5, k + 0.5, color="#F2F2F2" if k % 2 else "white", zorder=0, lw=0)
            if met == "hl_probe_balacc":
                s = get(m, met, dreq, "invariant")
                for k, r in enumerate(reps):
                    if r not in s.index:
                        continue
                    fl = isinstance(s.loc[r, "flag"], str) and "only" in s.loc[r, "flag"]
                    ax.plot(s.loc[r, "value"], k, "D", ms=9, color=colour(r), mec="black", mew=0.8,
                            alpha=0.45 if fl else 1, zorder=3)
            else:
                u, w = get(m, met, dreq, "unwhitened"), get(m, met, dreq, "whitened")
                for k, r in enumerate(reps):
                    if r not in u.index:
                        continue
                    fl = isinstance(u.loc[r, "flag"], str) and "only" in u.loc[r, "flag"]
                    a = 0.45 if fl else 1
                    xu, xw = u.loc[r, "value"], w.loc[r, "value"]
                    ax.plot([xu, xw], [k, k], color=GREY, lw=1.5, zorder=2, alpha=a)
                    ax.errorbar(xu, k - 0.0, xerr=u.loc[r, "sd"] if np.isfinite(u.loc[r, "sd"]) else None,
                                fmt="o", ms=9, mfc="white", mec=colour(r), mew=2, ecolor=colour(r), capsize=3,
                                alpha=a, zorder=3)
                    ax.errorbar(xw, k, xerr=w.loc[r, "sd"] if np.isfinite(w.loc[r, "sd"]) else None,
                                fmt="o", ms=9, color=colour(r), mec="black", mew=0.6, ecolor="black", capsize=3,
                                alpha=a, zorder=4)
            if i == 0:
                ax.set_title(lab, fontsize=FS)
            ax.grid(axis="x", color="#DDDDDD", lw=0.8, zorder=1)
            ax.tick_params(axis="y", length=0)
            flags = {r: " †" for r in reps if (dreq == 40 and r in ("R8", "R11"))}
            flags.update({"R9": " ‡"})
            rep_ticks(ax, reps, flags)
            if j > 0:
                ax.set_yticklabels([])
        axes[i, 0].annotate(f"d = {dreq}", xy=(-0.95, 0.5), xycoords="axes fraction", fontsize=FS + 3,
                            fontweight="bold", rotation=90, va="center", ha="center")
    # shared x ranges per column; R9 (near one-hot) is off-scale and printed at the left edge
    k9 = reps.index("R9") if "R9" in reps else None
    for j, (met, _) in enumerate(cols):
        sel = m[(m.metric == met) & (m.d_requested.isin(["20", "40"])) & (m.rep != "R9")]
        lo = sel.value.min() - (sel.sd.fillna(0).max() + 0.02)
        hi = min(1.0, sel.value.max() + sel.sd.fillna(0).max() + 0.015)
        for i, dreq in enumerate((20, 40)):
            axes[i, j].set_xlim(lo, hi)
            if k9 is not None:
                v9 = m[(m.metric == met) & (m.d_requested == str(dreq)) & (m.rep == "R9")].value
                axes[i, j].text(lo, k9, f"  ◄ {v9.min():.2f}" + ("" if v9.max() - v9.min() < 0.005 else f"-{v9.max():.2f}"),
                                va="center", ha="left", fontsize=FS_SMALL, color=colour("R9"), fontweight="bold")
    h = [Line2D([], [], marker="D", ls="", ms=9, color="#777777", mec="black", label="probe (scale-invariant)"),
         Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec="#444444", mew=2, label="unwhitened (principal coordinates, S-scaled)"),
         Line2D([], [], marker="o", ls="", ms=9, color="#444444", label="whitened (unit variance per PC)"),
         Line2D([], [], color=GREY, lw=1.5, label="change from whitening")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.55, 0.012), fontsize=FS_LEGEND + 1)
    fig.text(0.99, 0.008, "Every embedding centred and rotated to principal axes, first d axes kept. Error bars: SD over 5 Leiden seeds.\n"
             "† only 20 dims exist, shown at 20 (faded).  ‡ R9 is nearly one-hot (cluster memberships), not a comparable metric space; off scale, value printed.",
             ha="right", va="top", fontsize=FS_SMALL, color=GREY)
    fig.tight_layout(rect=(0.02, 0.04, 1, 1))
    finalize_figure(fig, FIGS / "fig_rep_fair")


# ------------------------------------------------------------------ fig_depth_axes_v4
def fig_depth(d, cf):
    d = d.set_index("rep").loc[REPS]
    cols = [("depth_rho_pc1", "PC1\n|ρ|"), ("depth_rho_max_pc12", "max PC1-2\n|ρ|"),
            ("depth_rho_max_pc15", "max PC1-5\n|ρ|"), ("depth_r2_top10", "R² from\ntop-10 PCs"),
            ("depth_rho_max_pc15_within_stratum", "max PC1-5\n|ρ|"),
            ("depth_r2_top10_within_stratum", "R² from\ntop-10 PCs")]
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1.25, 0.62], wspace=0.25, hspace=0.45)
    ax = fig.add_subplot(gs[:, 0])
    M = d[[c for c, _ in cols]].values
    cmap = plt.get_cmap("Reds")
    for i in range(len(REPS)):
        for j in range(len(cols)):
            xj = j + (0.3 if j >= 4 else 0)
            ax.add_patch(Rectangle((xj - 0.5, i - 0.5), 1, 1, fc=cmap(0.05 + 0.8 * M[i, j]), ec="white", lw=1.5))
            ax.text(xj, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=FS_TICK,
                    color="white" if M[i, j] > 0.6 else PALETTE["grey_dark"])
    xs = [j + (0.3 if j >= 4 else 0) for j in range(len(cols))]
    ax.set_xlim(-0.5, xs[-1] + 0.5)
    ax.set_xticks(xs, [l for _, l in cols], fontsize=FS_TICK)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    rep_ticks(ax, REPS)
    nr = d["null_rho_max_pc15_within_stratum"].mean(); n2 = d["null_r2_top10_within_stratum"].mean()
    ax.text(np.mean(xs[4:]), len(REPS) - 0.2, f"permutation null: |ρ| {nr:.2f}, R² {n2:.3f}", ha="center",
            va="top", fontsize=FS_SMALL, color=GREY)
    ax.text(np.mean(xs[4:]), -1.75, "within HL × birth time", ha="center", va="bottom", fontsize=FS, fontweight="bold", color=PALETTE["red_strong"])
    ax.text(np.mean(xs[:4]), -1.75, "all cells", ha="center", va="bottom", fontsize=FS, fontweight="bold", color=PALETTE["red_strong"])
    ax.text(np.mean(xs[:4]), len(REPS) - 0.2, "centred-SVD axes; |ρ| = Spearman with log UMI", ha="center", va="top",
            fontsize=FS_SMALL, color=GREY)
    panel_label(ax, "a", x=-0.36, y=1.08)
    # b: pooled vs within-stratum R2
    bx = fig.add_subplot(gs[0, 1])
    for k, r in enumerate(REPS):
        x0, x1 = d.loc[r, "depth_r2_top10"], d.loc[r, "depth_r2_top10_within_stratum"]
        bx.plot([x0, x1], [k, k], color=GREY, lw=1.5, zorder=1)
        bx.plot(x0, k, "o", ms=8, mfc="white", mec=colour(r), mew=2, zorder=2)
        bx.plot(x1, k, "o", ms=8, color=colour(r), mec="black", mew=0.5, zorder=3)
    bx.set_yticks(range(len(REPS)), REPS)
    for t, r in zip(bx.get_yticklabels(), REPS):
        t.set_color(colour(r)); t.set_fontweight("bold")
    bx.set_ylim(len(REPS) - 0.5, -0.5); bx.set_xlim(0, 1.02); bx.tick_params(axis="y", length=0, labelsize=FS_TICK - 1)
    bx.grid(axis="x", color="#DDDDDD", lw=0.8)
    bx.set_xlabel("R² of log UMI from top-10 PCs")
    bx.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="#444444", mew=2, label="all cells"),
                       Line2D([], [], marker="o", ls="", color="#444444", label="within HL × birth time")],
              loc="lower left", fontsize=FS_LEGEND - 1, bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)
    panel_label(bx, "b", x=-0.2, y=1.1)
    # c: what explains depth in the data
    cx = fig.add_subplot(gs[1, 1])
    e = cf["eta2_log_umi"]
    items = [("stratum", "HL × birth time"), ("birthtime", "birth time"), ("hemilineage", "hemilineage"),
             ("library", "library"), ("Sex", "sex")]
    v = [e[k] for k, _ in items]
    cx.barh(range(len(items)), v, color=PALETTE["blue_main"], height=0.65)
    for k, x in enumerate(v):
        cx.text(x + 0.005, k, f"{x:.2f}", va="center", fontsize=FS_SMALL)
    cx.set_yticks(range(len(items)), [l for _, l in items]); cx.invert_yaxis()
    cx.set_xlim(0, 0.42)
    cx.set_xlabel("share of log UMI variance (η²)")
    panel_label(cx, "c", x=-0.3, y=1.02)
    finalize_figure(fig, FIGS / "fig_depth_axes_v4")


# ------------------------------------------------------------------ fig_confounds
def fig_confounds(m, cf, cells):
    reps = REPS
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 0.9, 1.15], wspace=0.5, hspace=0.6)
    for j, (met, lab, chance) in enumerate([
            ("lib_knn_within_hl_balacc", "library 15-NN bal. accuracy\nwithin hemilineage (d = 20)", None),
            ("sex_knn_within_mixed_lib_balacc", "sex 15-NN bal. accuracy within\n14 mixed-sex libraries (d = 20)", 0.5)]):
        ax = fig.add_subplot(gs[:, j])
        u, w = get(m, met, 20, "unwhitened"), get(m, met, 20, "whitened")
        for k, r in enumerate(reps):
            ax.axhspan(k - 0.5, k + 0.5, color="#F2F2F2" if k % 2 else "white", zorder=0, lw=0)
            ax.plot([u.loc[r, "value"], w.loc[r, "value"]], [k, k], color=GREY, lw=1.5, zorder=2)
            ax.plot(u.loc[r, "value"], k, "o", ms=10, mfc="white", mec=colour(r), mew=2, zorder=3)
            ax.plot(w.loc[r, "value"], k, "o", ms=10, color=colour(r), mec="black", mew=0.6, zorder=4)
        if chance is None:
            chance = get(m, "lib_knn_within_hl_chance", 20, "unwhitened")["value"].mean()
            ctext = f"chance {chance:.3f}"
        else:
            ctext = "chance 0.5"
        ax.axvline(chance, color=PALETTE["red_strong"], ls="--", lw=1.5, zorder=1)
        ax.text(chance, len(reps) - 0.55, " " + ctext, color=PALETTE["red_strong"], fontsize=FS_SMALL, ha="left", va="bottom")
        ax.set_title(lab, fontsize=FS, pad=12)
        ax.grid(axis="x", color="#DDDDDD", lw=0.8, zorder=1)
        ax.tick_params(axis="y", length=0)
        rep_ticks(ax, reps)
        if j == 1:
            ax.set_yticklabels([r for r in reps])
            for t, r in zip(ax.get_yticklabels(), reps):
                t.set_color(colour(r)); t.set_fontweight("bold")
        panel_label(ax, "ab"[j], x=-0.5 if j == 0 else -0.2, y=1.0)
    h = [Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec="#444444", mew=2, label="unwhitened"),
         Line2D([], [], marker="o", ls="", ms=9, color="#444444", label="whitened")]
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.33, -0.02), fontsize=FS_LEGEND + 1)
    # c: roX detection by sex
    cx = fig.add_subplot(gs[0, 2])
    det = pd.DataFrame(cf["rox_detection"])
    groups = [("roX1", "mixed"), ("roX1", "single-sex"), ("roX2", "mixed"), ("roX2", "single-sex")]
    x = np.arange(len(groups))
    for o, (s, col) in enumerate([("F", PALETTE["red_strong"]), ("M", PALETTE["blue_main"])]):
        vals = [det[(det.gene == g) & (det.libraries == k) & (det.sex == s)].detected_frac.iloc[0] for g, k in groups]
        cx.bar(x + (o - 0.5) * 0.38, vals, 0.38, color=col, label={"F": "female", "M": "male"}[s])
    cx.set_xticks(x, [f"{g}\n{'mixed' if k == 'mixed' else 'single'}" for g, k in groups], fontsize=FS_TICK - 1)
    cx.set_ylabel("cells with ≥ 1 UMI")
    cx.set_ylim(0, 1.32)
    cx.legend(loc="upper left", ncol=2, fontsize=FS_LEGEND)
    auc = cf["rox_call_mixed"]["auc"]; ba = cf["rox_call_mixed"]["best_balacc"]
    cx.set_title(f"roX detection by annotated sex\n(roX predicts label, AUC {auc:.3f})", fontsize=FS)
    panel_label(cx, "c", x=-0.3, y=1.02)
    # d: depth by birth time
    dx = fig.add_subplot(gs[1, 2])
    order = ["primary", "earlysecondary", "secondary", "precursor"]
    data = [cells.loc[cells.birthtime == b, "log10_umi"].values for b in order]
    bp = dx.boxplot(data, vert=True, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=2))
    for p_, c_ in zip(bp["boxes"], [PALETTE["red_strong"], PALETTE["blue_main"], "#C9C9C9", "#E69F00"]):
        p_.set_facecolor(c_)
    dx.set_xticks(range(1, 5), ["primary", "early\nsec.", "second-\nary", "precur-\nsor"], fontsize=FS_TICK - 1)
    dx.set_ylabel("log10 UMI per cell")
    pf = cf["primary_frac_by_depth"]
    dx.set_title(f"depth by birth time (η² {cf['eta2_log_umi']['birthtime']:.2f})", fontsize=FS)
    dx.text(0.98, 0.97, f"top depth decile:\n{100 * pf['top_decile']:.0f}% primary\n(vs {100 * pf['overall']:.0f}% overall)",
            transform=dx.transAxes, ha="right", va="top", fontsize=FS_SMALL)
    panel_label(dx, "d", x=-0.3, y=1.02)
    finalize_figure(fig, FIGS / "fig_confounds")


# ------------------------------------------------------------------ UMAP grids
def umap_grid(cells, kind):
    reps = [r for r in ORDER if (CACHE / f"w3_{r}_umap.npy").exists()]
    U = {r: np.load(CACHE / f"w3_{r}_umap.npy") for r in reps}
    perm = np.random.default_rng(0).permutation(len(cells))
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 4, left=0.0, right=1.0, bottom=0.0, top=0.95, wspace=0.05, hspace=0.17)
    axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(12)]
    if kind == "hemilineage":
        hl = cells.hemilineage.astype(str).values
        hl_u = sorted(set(hl), key=lambda s: (len(s), s))
        cm = dict(zip(hl_u, QUAL23))
        cols = np.array([cm[h] for h in hl[perm]], dtype=object)
    else:
        x = cells.log10_umi.values
        lo, hi = np.percentile(x, [1, 99])
    for ax, r in zip(axes, reps):
        Z = U[r][perm]
        if kind == "hemilineage":
            ax.scatter(Z[:, 0], Z[:, 1], c=list(cols), s=0.6, lw=0, rasterized=True, antialiased=False)
        else:
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=x[perm], s=0.6, lw=0, cmap=plt.get_cmap("viridis", 24),
                            vmin=lo, vmax=hi, rasterized=True, antialiased=False)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.annotate(r, xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
                    fontweight="bold", color=colour(r), fontsize=FS, va="bottom", ha="left")
        ax.annotate(name(r), xy=(0, 1.0), xycoords="axes fraction", xytext=(46 if len(r) > 2 else 36, 3), textcoords="offset points",
                    fontsize=FS, va="bottom", ha="left", color=PALETTE["grey_dark"])
    for ax in axes[len(reps):]:
        ax.axis("off")
    lax = axes[-1]
    if kind == "hemilineage":
        counts = pd.Series(hl).value_counts()
        h = [Line2D([], [], marker="o", ls="", ms=7, mfc=cm[k], mec="none") for k in hl_u]
        lax.legend(h, [f"{k} ({counts[k]:,})" for k in hl_u], loc="center left", ncol=3, title="hemilineage",
                   fontsize=FS_TINY, title_fontsize=FS_LEGEND, handletextpad=0.1, columnspacing=0.5,
                   labelspacing=0.25, borderaxespad=0, bbox_to_anchor=(0.02, 0.5))
        out = "fig_umap_grid_hemilineage_v4"
    else:
        cax = lax.inset_axes([0.15, 0.45, 0.7, 0.08])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
        cb.set_label(r"$\log_{10}$ total UMI per cell" + "\n(clipped to 1st-99th pct)", fontsize=FS_LEGEND)
        cb.ax.tick_params(labelsize=FS_TICK)
        out = "fig_umap_grid_depth_v4"
    finalize_figure(fig, FIGS / out)


if __name__ == "__main__":
    which = sys.argv[1:] or ["tables", "fair", "depth", "confounds", "umap"]
    m, d, cf = build_tables()
    cells = load_cells()
    if "fair" in which:
        fig_rep_fair(m)
    if "depth" in which:
        fig_depth(d, cf)
    if "confounds" in which:
        fig_confounds(m, cf, cells)
    if "umap" in which:
        umap_grid(cells, "hemilineage"); umap_grid(cells, "depth")
