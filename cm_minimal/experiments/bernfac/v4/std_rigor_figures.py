"""Figures for W5 (depth-standardization rigor), from v4/tables written by std_rigor_tables.py.

fig_std_rigor               residual depth in p~ vs references (R^2, within-stratum |rho|) and retention
fig_std_depthgenes_heldout  depth gap of genes chosen on one half of the cells, evaluated on the other
fig_std_reference           reference depth c_bar at the 10th / 50th / 90th percentile of c_i
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "v3"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import figstyle as fs
import v3style as vs
import predictors as pr

fs.apply_publication_style()
V4 = pr.STUDY / "v4"
TABS, FIGS = V4 / "tables", V4 / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
T1 = pd.read_csv(TABS / "std_rigor.csv").set_index("source")
T2 = pd.read_csv(TABS / "std_depthgenes_heldout.csv")
T3 = pd.read_csv(TABS / "std_reference.csv")
T4 = pd.read_csv(TABS / "std_retention.csv")
MODELS = ["R4", "R5", "R6"]
GREY = fs.PALETTE["grey_mid"]
REFCOL = {"Y": "#7F7F7F", "PC": vs.colour("R1"), "R1": vs.colour("R1")}
MNAME = {"R4": "R4 logit + offsets", "R5": "R5 cloglog", "R6": "R6 NB counts"}


def light(hexcol, f=0.55):
    c = np.array(matplotlib_to_rgb(hexcol))
    return tuple(c + (1 - c) * f)


def matplotlib_to_rgb(h):
    from matplotlib.colors import to_rgb
    return to_rgb(h)


# =========================================================================== fig_std_rigor
fig, axes = plt.subplots(1, 3, figsize=(20, 7.2), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1.0], wspace=0.55))

# ---- A: CV R^2 of log UMI from 10 PCs
ax = axes[0]
rows = [("Y", "binary Y (data)", REFCOL["Y"]), ("PC", "processed counts", REFCOL["PC"]),
        ("R1", "R1 HVG PCA", REFCOL["R1"])]
for c in MODELS:
    rows += [(c, f"{c} p̂", vs.colour(c)), (f"{c}~p50", f"{c} p̃", vs.colour(c))]
y = np.arange(len(rows))[::-1].astype(float)
y[3:] -= 0.5; y[:3] += 0.0
h = 0.38
for yi, (src, lab, col) in zip(y, rows):
    tilde = "~" in src
    tot, part = T1.loc[src, "r2_total_p"], T1.loc[src, "r2_partial_p"]
    ax.barh(yi + h / 2, tot, h, color=light(col, 0.55) if tilde else col, edgecolor=col, lw=1.2)
    ax.barh(yi - h / 2, part, h, color="white", edgecolor=col, hatch="///", lw=1.2)
    ax.text(tot + 0.015, yi + h / 2, f"{tot:.2f}", va="center", fontsize=fs.FS_TINY)
    ax.text(part + 0.015, yi - h / 2, f"{part:.2f}", va="center", fontsize=fs.FS_TINY, color=fs.PALETTE["grey"])
perm = T1.loc[[r[0] for r in rows], "r2_total_p_perm"].mean()
ax.axvline(perm, color="k", ls="--", lw=1.4)
ax.text(perm - 0.035, y.max() + 0.75, f"depth shuffled\nwithin strata ({perm:.2f})", fontsize=fs.FS_SMALL - 1,
        va="bottom", ha="right")
ax.set_yticks(y); ax.set_yticklabels([r[1] for r in rows])
ax.set_xlim(0, 1.12); ax.set_ylim(y.min() - 0.8, y.max() + 1.9)
ax.set_xlabel("CV R² of log UMI from 10 PCs (p scale)")
ax.legend(handles=[Patch(fc=GREY, ec=GREY, label="total"),
                   Patch(fc="white", ec=GREY, hatch="///", label="beyond hemilineage × birth time")],
          loc="upper right", fontsize=fs.FS_LEGEND - 1, bbox_to_anchor=(1.04, 1.01))
ax.set_title("Depth recoverable from the embedding", loc="left")
fs.panel_label(ax, "a", x=-0.42)

# ---- B: median |rho| within hemilineage x birth-time strata
ax = axes[1]
rows = [("Y", "binary Y (data)", REFCOL["Y"]), ("PC", "processed counts", REFCOL["PC"])]
for c in MODELS:
    rows += [(c, f"{c} p̂", vs.colour(c)), (f"{c}~p50", f"{c} p̃", vs.colour(c))]
y = np.arange(len(rows))[::-1].astype(float); y[2:] -= 0.5
for yi, (src, lab, col) in zip(y, rows):
    v = T1.loc[src, "med_abs_rho_hlbt"]
    ax.barh(yi, v, 0.62, color=light(col, 0.55) if "~" in src else col, edgecolor=col, lw=1.2)
    ax.text(v * 1.08, yi + 0.18, f"{v:.3f}", va="center", fontsize=fs.FS_TINY)
    ax.plot(T1.loc[src, "med_abs_rho_hlbt_perm"], yi, "|", color="k", ms=16, mew=2.2)
    d = T1.loc[src, "med_abs_rho_hlbt_bernoulli_draw"] if "med_abs_rho_hlbt_bernoulli_draw" in T1 else np.nan
    if np.isfinite(d):
        ax.plot(d, yi, "D", color="white", mec="k", ms=8, mew=1.6, zorder=5)
ax.set_xscale("log"); ax.set_xlim(0.006, 2.2)
ax.set_yticks(y); ax.set_yticklabels([r[1] for r in rows])
ax.set_ylim(y.min() - 0.7, y.max() + 1.6)
ax.set_xlabel("median |Spearman ρ| with log UMI\nwithin hemilineage × birth time")
ax.legend(handles=[Line2D([], [], marker="|", ls="", color="k", ms=14, mew=2.2, label="depth shuffled (noise floor)"),
                   Line2D([], [], marker="D", ls="", color="white", mec="k", ms=8, mew=1.6,
                          label="binary draw from p (data scale)")],
          loc="upper right", fontsize=fs.FS_LEGEND - 1, bbox_to_anchor=(1.02, 1.02))
ax.set_title("Per-gene depth correlation", loc="left")
fs.panel_label(ax, "b", x=-0.42)

# ---- C: retention p~ / p^
ax = axes[2]
r = T4[T4.ref == "p50"].set_index("model")
mets = [("markers115_ratio_sum_between_hl_var", "between-hemilineage\nvariance, markers"),
        ("markers115_corr_hl_profiles_all", "hemilineage mean\nprofiles (r), markers"),
        ("random3000_ratio_sum_between_hl_var", "between-hemilineage\nvariance, random genes"),
        ("random3000_ratio_sum_birthtime_within_hl_var", "birth-time variance\nwithin HL, random genes")]
w = 0.26
for k, c in enumerate(MODELS):
    vals = [r.loc[c, m] for m, _ in mets]
    yy = np.arange(len(mets))[::-1] + (1 - k) * w
    ax.barh(yy, vals, w * 0.92, color=vs.colour(c), label=MNAME[c])
    for v, yv in zip(vals, yy):
        ax.text(v + 0.02, yv, f"{v:.2f}", va="center", fontsize=fs.FS_TINY)
ax.axvline(1, color="k", lw=1.2, ls=":")
ax.set_yticks(np.arange(len(mets))[::-1]); ax.set_yticklabels([l for _, l in mets])
ax.set_xlim(0, 1.32); ax.set_ylim(-0.6, len(mets) - 0.4)
ax.set_xlabel("p̃ relative to p̂ (ratio, or r)")
ax.legend(loc="upper center", fontsize=fs.FS_LEGEND - 1, bbox_to_anchor=(0.45, -0.17), ncol=3, columnspacing=0.8,
          handlelength=1.2)
ax.set_title("What p̃ keeps (median reference)", loc="left")
fs.panel_label(ax, "c", x=-0.62)
fs.finalize_figure(fig, FIGS / "fig_std_rigor")

# =========================================================================== fig_std_depthgenes_heldout
V3GAP = {"Y": 0.4155, "R4": 0.4102, "R4~p50": 0.2400, "R5": 0.4014, "R5~p50": 0.1620, "R6": 0.3999,
         "R6~p50": 0.1402}
srcs = [("Y", "data Y", REFCOL["Y"]), ("PC", "log-norm\ncounts", REFCOL["PC"])]
for c in MODELS:
    srcs += [(c, f"{c} p̂", vs.colour(c)), (f"{c}~p50", f"{c} p̃", vs.colour(c))]
fig, axes = plt.subplots(1, 2, figsize=(19, 6.6), gridspec_kw=dict(wspace=0.22, width_ratios=[1.05, 1]))
x = np.arange(len(srcs)).astype(float); x[1:] += 0.25; x[2:] += 0.4
ax = axes[0]
for xi, (s, lab, col) in zip(x, srcs):
    fc = light(col, 0.55) if "~" in s else col
    for dx, half, hatch in ((-0.2, "selection_half", None), (0.2, "heldout_half", "///")):
        d = T2[(T2.source == s) & (T2.ntop == "top 8") & (T2.half == half)].iloc[0]
        ax.bar(xi + dx, d.gap_mean, 0.38, yerr=d.gap_sd, color=fc if hatch is None else "white", edgecolor=col,
               hatch=hatch, lw=1.3, error_kw=dict(lw=1.3, capsize=3))
        if half == "heldout_half":
            ax.text(xi + dx, d.gap_mean + d.gap_sd + 0.012, f"{d.gap_mean:.2f}", ha="center", fontsize=fs.FS_TINY)
    if s in V3GAP:
        ax.plot(xi, V3GAP[s], "_", color="k", ms=30, mew=2.5, zorder=6)
ax.set_xticks(x); ax.set_xticklabels([s[1] for s in srcs])
ax.set_ylabel("depth gap: mean p (top − bottom depth quartile)")
ax.set_ylim(0, 0.53)
ax.legend(handles=[Patch(fc=GREY, ec=GREY, label="genes chosen and scored on the same half"),
                   Patch(fc="white", ec=GREY, hatch="///", label="chosen on one half, scored on the other"),
                   Line2D([], [], marker="_", ls="", color="k", ms=22, mew=2.5, label="v3 value (all cells, in-sample)")],
          loc="upper right", fontsize=fs.FS_LEGEND - 1, ncol=1, bbox_to_anchor=(1.0, 1.04))
ax.set_title("Top-8 depth-sensitive genes, 20 random half splits", loc="left")
fs.panel_label(ax, "a", x=-0.1)

ax = axes[1]
sets = [("top 8", "gap", "top 8, quartiles within HL"), ("top 8", "gap_hlbt", "top 8, within HL × birth time"),
        ("top 50", "gap", "top 50, within HL"), ("300 random broad genes", "gap", "300 random broad genes")]
w = 0.2
setcol = ["#272727", "#767676", "#AFAFAF", "#E0E0E0"]
for k, (nt, col_, lab) in enumerate(sets):
    vals, errs = [], []
    for s, _, _ in srcs:
        d = T2[(T2.source == s) & (T2.ntop == nt) & (T2.half == "heldout_half")].iloc[0]
        vals.append(d[f"{col_}_mean"]); errs.append(d[f"{col_}_sd"])
    ax.bar(x + (k - 1.5) * w, vals, w * 0.92, yerr=errs, color=setcol[k], edgecolor="k", lw=0.6, label=lab,
           error_kw=dict(lw=1, capsize=2))
ax.set_xticks(x); ax.set_xticklabels([s[1] for s in srcs])
for t, (s, _, col) in zip(ax.get_xticklabels(), srcs):
    t.set_color(col if s not in ("Y",) else "k")
ax.set_ylabel("held-out depth gap")
ax.set_ylim(0, 0.53)
ax.legend(loc="upper right", fontsize=fs.FS_LEGEND - 1, ncol=2, bbox_to_anchor=(1.0, 1.04), columnspacing=1.0)
ax.set_title("Held-out gap by gene set and stratification", loc="left")
fs.panel_label(ax, "b", x=-0.1)
fs.finalize_figure(fig, FIGS / "fig_std_depthgenes_heldout")

# =========================================================================== fig_std_reference
pan = [("mean_p_random_genes", "mean p̃ (3,000 random genes)", "changes"),
       ("heldout_gap_top8", "held-out depth gap, top-8 genes", "changes"),
       ("markers115_ratio_sum_between_hl_var", "between-HL variance p̃ / p̂, markers", "changes"),
       ("med_abs_rho_hlbt", "median |ρ| with depth within strata", "stable"),
       ("markers115_corr_hl_profiles_all", "r of HL mean profiles, p̃ vs p̂, markers", "stable"),
       ("r2_total_logit", "CV R² of log UMI, 10 PCs of logit p̃", "stable")]
fig, axes = plt.subplots(2, 3, figsize=(19, 10.5), gridspec_kw=dict(hspace=0.85, wspace=0.34))
xs = np.array([10, 50, 90])
for k, (ax, (m, lab, kind)) in enumerate(zip(axes.ravel(), pan)):
    for c in MODELS:
        d = T3[T3.model == c].set_index("ref").loc[["p10", "p50", "p90"]]
        ax.plot(xs, d[m].values, "-o", color=vs.colour(c), lw=2.4, ms=8, label=MNAME[c])
    if m == "mean_p_random_genes":
        ph = T3.groupby("model").mean_p_random_genes_phat.first().mean()
        ax.axhline(ph, color="k", ls="--", lw=1.3)
        ax.text(12, ph + 0.002, f"p̂ ({ph:.3f})", fontsize=fs.FS_SMALL, va="bottom")
    if m == "markers115_ratio_sum_between_hl_var":
        ax.axhline(1, color="k", ls=":", lw=1.2)
    if kind == "stable":
        lo, hi = ax.get_ylim(); mid = (lo + hi) / 2; span = max(hi - lo, 0.04 if m != "markers115_corr_hl_profiles_all" else 0.02)
        ax.set_ylim(mid - span * 0.75, mid + span * 0.75)
    if m == "med_abs_rho_hlbt":
        ax.set_ylim(0, 0.32)
    if m == "r2_total_logit":
        ax.set_ylim(0.3, 1.0)
    if m == "markers115_corr_hl_profiles_all":
        ax.set_ylim(0.97, 1.002)
    ax.set_xticks(xs); ax.set_xticklabels(["10th", "50th", "90th"])
    ax.set_xlabel("reference c̄: percentile of c_i")
    ax.set_title(lab, loc="left", fontsize=fs.FS_TITLE - 1)
    fs.panel_label(ax, "abcdef"[k], x=-0.2, y=1.06)
axes[0, 0].legend(loc="upper left", fontsize=fs.FS_LEGEND - 1)
for row, txt in ((0, "Changes with the reference (absolute level)"), (1, "Does not change (rank and structure)")):
    p0 = axes[row, 0].get_position()
    fig.text(0.02, p0.y1 + 0.075, txt, fontsize=fs.FS + 2, fontweight="bold", ha="left", va="bottom")
fs.finalize_figure(fig, FIGS / "fig_std_reference")
