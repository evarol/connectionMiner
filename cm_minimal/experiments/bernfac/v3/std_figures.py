"""Depth-standardized probabilities p~ (R4, R5, R6) next to the observed p^ of every model.

p^ uses each cell's own offset c_i (a_i for R4/R5, o_i + a_i for R6) and is what was scored on
held-out entries. p~ = predictors.standardized(code): the same model with c_i replaced by
c_bar = median_i c_i, i.e. each cell's state (u_i) at a typical sequencing depth. p~ is a display /
analysis quantity, not a predictor of observed detection, and is not scored.

Same cells, genes and ordering as heat_figures.py (v3/cache/heat_selection.json, heat_blocks.npz).
Whole-matrix numbers come from std_quant.py (v3/cache/std_quant.npz, std_embed.json).

Outputs (v3/figures): fig_heat_std_overview, fig_heat_std_depth, fig_heat_std_diff, fig_std_quant
         (v3/tables): depth_std_metrics.csv
"""
import json, os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import rankdata
import predictors as pr
import figstyle as fs
import v3style as vs

fs.apply_publication_style()
CACHE, FIGS, TABS = pr.CACHE, vs.FIGS, vs.V3 / "tables"
sel = json.loads((CACHE / "heat_selection.json").read_text())
HL_COL = dict(zip(sel["hemilineages"], ["#1F5FA8", "#C0392B", "#2E8B57", "#7B4FA0", "#8C5A2B"]))
DOT = dict(marker="o", s=5, c="#2A7FFF", edgecolors="white", linewidths=0.35, zorder=5)
STD = list(pr.STD_MODELS)
PO_COL = "#7F7F7F"


def colour(c):
    c = c.rstrip("~")
    return PO_COL if c == "PO" else vs.colour(c)


def short(c):
    return "Poisson counts" if c == "PO" else vs.name(c)


# --------------------------------------------------------------------------- snippet blocks
D = dict(np.load(CACHE / "heat_blocks.npz", allow_pickle=True))
f = CACHE / "heat_std_blocks.npz"
if not f.exists():
    out = {}
    for c in STD:
        p = pr.standardized(c)
        out[f"P_{c}~"] = p.block(D["rows"], D["cols"])[0]
        out[f"c_{c}"] = p.c_obs[D["rows"]]
        out[f"cbar_{c}"] = p.c_ref
        del p
    np.savez(f, **out)
D.update(dict(np.load(f)))

gg = sel["gene_groups"]
names = sel["gene_names"]
gene_group = np.concatenate([[k] * len(v) for k, v in gg.items()])
gene_name = np.array([names[str(j)] for j in D["cols"]])
row_dep = np.concatenate([[g["depth"]] * len(g["rows"]) for g in sel["cell_groups"]])


def group_of(k):
    return "marker" if k.startswith("marker") else k


def gene_idx(kind):
    """Same gene subsets as heat_figures.gene_idx."""
    if kind == "overview":
        idx = []
        for k, v in gg.items():
            nk = {"depth": 6, "rare": 3, "housekeeping": 3}.get(k, 3)
            take = list(range(len(v)))[:nk] if k != "housekeeping" else [0, 2, 3]
            start = int(np.flatnonzero(gene_group == k)[0])
            idx += [start + t for t in take]
        return np.array(idx)
    if kind == "depth":
        return np.flatnonzero(gene_group == "depth")
    raise KeyError(kind)


def col_bounds():
    return np.cumsum([len(g["rows"]) for g in sel["cell_groups"]])


def draw(ax, A, gi, cmap="magma", vmin=0, vmax=1, norm=None, dots=False, ylabels=False, xlabels=False,
         title=None, title_colour="black", gene_fs=9.5, hl_y=-0.075):
    """genes x cells heatmap with the heat_figures.draw separators and labels."""
    A = A[:, gi].T
    im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=None if norm else vmin, vmax=None if norm else vmax,
                   norm=norm, interpolation="nearest")
    cb = col_bounds()
    for i, x in enumerate(cb[:-1]):
        ax.axvline(x - 0.5, color="white", lw=2.2 if i % 2 == 1 else 0.6)
    gsel = gene_group[gi]
    for r in np.flatnonzero(gsel[1:] != gsel[:-1]):
        ax.axhline(r + 0.5, color="white", lw=1.6 if group_of(gsel[r]) != group_of(gsel[r + 1]) else 0.6)
    if dots:
        yy, xx = np.nonzero(D["H"][:, gi].T)
        ax.scatter(xx, yy, **DOT)
    ax.set_yticks(range(len(gi)))
    if ylabels:
        ax.set_yticklabels(gene_name[gi], fontsize=gene_fs)
        for t, k in zip(ax.get_yticklabels(), gsel):
            if k.startswith("marker"):
                t.set_color(HL_COL[k.split()[1]]); t.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    centres = np.r_[0, cb[:-1]] + np.diff(np.r_[0, cb]) / 2 - 0.5
    ax.set_xticks(centres)
    if xlabels:
        ax.set_xticklabels(["lo", "hi"] * len(sel["hemilineages"]), fontsize=9)
        for h_i, h in enumerate(sel["hemilineages"]):
            ax.text(((cb[2 * h_i - 1] if h_i else 0) + cb[2 * h_i + 1]) / 2 - 0.5, hl_y, h, color=HL_COL[h],
                    fontsize=10.5, fontweight="bold", ha="center", va="top", transform=ax.get_xaxis_transform())
    else:
        ax.set_xticklabels([])
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=12.5, color=title_colour, pad=4)
    return im


def P_of(key):
    if key == "Y":
        return D["Y"].astype(float)
    return D[f"P_{key}"]


def pair_header(fig, ax1, ax2, code, y_pad=0.035):
    """Shared header + bracket over an observed / standardized pair of panels."""
    b1, b2 = ax1.get_position(), ax2.get_position()
    y = b1.y1 + y_pad
    fig.add_artist(Line2D([b1.x0, b1.x0, b2.x1, b2.x1], [y - 0.008, y, y, y - 0.008],
                          transform=fig.transFigure, color=colour(code), lw=2))
    fig.text((b1.x0 + b2.x1) / 2, y + 0.006, f"{code}  {vs.name(code)}", ha="center", va="bottom",
             fontsize=13.5, color=colour(code), fontweight="bold")


def panel_title(key):
    """(title, colour) for a single panel; pair members get 'observed' / 'depth-standardized'."""
    if key == "Y":
        return "binary $Y$ (data)", "black"
    if key.endswith("~"):
        return "depth-standardized $\\tilde p$", colour(key)
    if key in STD:
        return "observed $\\hat p$", colour(key)
    return f"{key}  {vs.name(key)}", colour(key)


def heat_rows(layout, gi, figsize, dots_all=False, gene_fs=9.5, right=None, hspace=0.42):
    """layout: list of rows, each a list of panel keys (None = empty slot). Pairs (R4, R4~) are drawn
    with a small gap; other neighbours with a larger gap. Returns fig, axes dict, gridspec, cbar slot."""
    ncol = max(len(r) for r in layout)
    # width ratios: panel, gap, panel, ...; small gap inside a pair
    fig = plt.figure(figsize=figsize)
    wr, colmap = [], []
    for k in range(ncol):
        if k:
            inside = any(r[k - 1] is not None and r[k] is not None and r[k] == r[k - 1] + "~" for r in layout
                         if k < len(r))
            wr.append(0.025 if inside else 0.09)
        colmap.append(len(wr)); wr.append(1.0)
    wr += [0.06, 0.05]                                   # gap, colourbar
    if right:
        wr += [0.75, right]
    gs = fig.add_gridspec(len(layout), len(wr), width_ratios=wr, wspace=0.0, hspace=hspace)
    axes = {}
    for r, row in enumerate(layout):
        for k, key in enumerate(row):
            if key is None:
                continue
            ax = fig.add_subplot(gs[r, colmap[k]])
            t, tc = panel_title(key)
            im = draw(ax, P_of(key), gi, dots=(key == "Y") or dots_all, ylabels=(k == 0), xlabels=True,
                      title=t, title_colour=tc, gene_fs=gene_fs)
            axes[key] = ax
    cax = fig.add_subplot(gs[0, len(wr) - (3 if right else 1)])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$\\Pr(y=1)$ / $Y$", fontsize=13); cb.ax.tick_params(labelsize=11)
    return fig, axes, gs, colmap


def dot_legend(fig, x, y):
    h = [Line2D([], [], marker="o", markersize=5, markerfacecolor=DOT["c"], markeredgecolor="white", linestyle="none")]
    fig.legend(h, ["held-out entry (not used in fitting; scored on these)"], loc="lower center",
               bbox_to_anchor=(x, y), fontsize=11.5, frameon=False, handletextpad=0.3)


def heat_overview():
    gi = gene_idx("overview")
    layout = [["Y", "R3", "R4", "R4~", "R5", "R5~"],
              ["R6", "R6~", "R7", "R8", "R9", None]]
    fig, axes, gs, colmap = heat_rows(layout, gi, figsize=(20, 11.2), hspace=0.42)
    fig.canvas.draw()
    for c in STD:
        pair_header(fig, axes[c], axes[c + "~"], c, y_pad=0.03)
    ax = fig.add_subplot(gs[1, colmap[5]]); ax.axis("off")
    cb = {c: float(D[f"cbar_{c}"]) for c in STD}
    ax.text(0.04, 1.0,
            "How to read\n\n"
            "Same cells and genes as the\noverview figure: 5 hemilineages,\n12 low- then 12 high-depth\ncells each.\n\n"
            "observed $\\hat p$: each cell's own\noffset $c_i$ (what was scored).\n\n"
            "depth-standardized $\\tilde p$:\n$c_i$ replaced by the median $\\bar c$\nfor every cell; gene level $b_j$\n"
            "and cell state $u_i^\\top v_j$ kept.\n\n"
            "$c_i = a_i$ (R4, R5),\n$c_i = o_i + a_i$ (R6, $o_i$ = log UMI).",
            transform=ax.transAxes, va="top", ha="left", fontsize=11, color="#333333", linespacing=1.15)
    fig.suptitle("Observed vs depth-standardized probabilities: the same cells at a typical sequencing depth",
                 fontsize=15, y=0.995)
    dot_legend(fig, 0.45, 0.0)
    fs.finalize_figure(fig, FIGS / "fig_heat_std_overview")


def heat_depth():
    gi = gene_idx("depth")
    layout = [["Y", "R4", "R4~", "R5", "R5~"],
              ["R3", "R6", "R6~", "R7", "R8"]]
    fig, axes, gs, colmap = heat_rows(layout, gi, figsize=(22, 9.6), dots_all=True, gene_fs=10.5,
                                      right=1.35, hspace=0.62)
    fig.canvas.draw()
    for c in STD:
        pair_header(fig, axes[c], axes[c + "~"], c, y_pad=0.045)
    ax = fig.add_subplot(gs[:, -1])
    dg = gene_group == "depth"
    lo, hi = row_dep == "low", row_dep == "high"
    emp = [D["ref"][lo][:, dg].mean(), D["ref"][hi][:, dg].mean()]
    ax.plot([0, 1], emp, color="black", lw=3.4, marker="s", ms=10, label="empirical rate (train)", zorder=6)
    for c in vs.PROB_MODELS:
        P = D[f"P_{c}"][:, dg]
        ax.plot([0, 1], [P[lo].mean(), P[hi].mean()], color=colour(c), lw=2, marker="o", ms=6, label=f"{c} $\\hat p$")
    for c in STD:
        P = D[f"P_{c}~"][:, dg]
        ax.plot([0, 1], [P[lo].mean(), P[hi].mean()], color=colour(c), lw=2.6, ls="--", marker="o", ms=8,
                mfc="white", mew=2, label=f"{c} $\\tilde p$", zorder=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["low depth", "high depth"])
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylabel("mean probability over the 8 genes")
    ax.set_title("depth gap", fontsize=14)
    ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2, handlelength=2.2,
              columnspacing=0.8)
    fig.suptitle("Depth-sensitive genes: standardizing the cell offset shrinks the jump between low- and "
                 "high-depth cells (dashed); observed $\\hat p$ keeps it", fontsize=15, y=1.0)
    dot_legend(fig, 0.4, -0.06)
    fs.finalize_figure(fig, FIGS / "fig_heat_std_depth")


def heat_diff():
    gi = gene_idx("overview")
    fig = plt.figure(figsize=(19, 9.2))
    gs = fig.add_gridspec(2, 5, height_ratios=[0.07, 1], width_ratios=[1, 1, 1, 0.05, 0.05],
                          wspace=0.07, hspace=0.04)
    norm = TwoSlopeNorm(0, -0.5, 0.5)
    cnorm = TwoSlopeNorm(0, -1.5, 1.5)
    for k, c in enumerate(STD):
        axc = fig.add_subplot(gs[0, k])
        off = (D[f"c_{c}"] - D[f"cbar_{c}"])[None]
        imc = axc.imshow(off, aspect="auto", cmap="PuOr_r", norm=cnorm, interpolation="nearest")
        for i, x in enumerate(col_bounds()[:-1]):
            axc.axvline(x - 0.5, color="white", lw=2.2 if i % 2 == 1 else 0.6)
        axc.set_xticks([]); axc.set_yticks([0])
        axc.set_yticklabels(["$c_i-\\bar c$"] if k == 0 else [], fontsize=11)
        axc.tick_params(length=0)
        for s in axc.spines.values():
            s.set_visible(False)
        Dd = D[f"P_{c}~"] - D[f"P_{c}"]
        axc.set_title(f"{c}  {vs.name(c)}:  $\\tilde p - \\hat p$\nmean |shift| {np.abs(Dd[:, gi]).mean():.3f}",
                      fontsize=13, color=colour(c), pad=6)
        ax = fig.add_subplot(gs[1, k])
        imd = draw(ax, Dd, gi, cmap="RdBu_r", norm=norm, ylabels=(k == 0), xlabels=True, gene_fs=10.5,
                   hl_y=-0.04)
    cax = fig.add_subplot(gs[1, 4]); cb = fig.colorbar(imd, cax=cax)
    cb.set_label("$\\tilde p - \\hat p$  (red: standardized higher)", fontsize=12.5); cb.ax.tick_params(labelsize=11)
    cax = fig.add_subplot(gs[0, 4]); cb = fig.colorbar(imc, cax=cax, orientation="vertical")
    cb.set_ticks([-1, 1]); cb.ax.tick_params(labelsize=10)
    fig.text(0.925, 0.93, "top strip: cell offset $c_i-\\bar c$\norange = deeper than median,\npurple = shallower",
             fontsize=10.5, color="#444444", ha="left", va="bottom")
    fig.suptitle("Standardization moves whole cells: low-depth cells up, high-depth cells down. Genes that are off "
                 "stay off, so the hemilineage pattern is kept", fontsize=14.5, y=1.02)
    fs.finalize_figure(fig, FIGS / "fig_heat_std_diff")


# --------------------------------------------------------------------------- whole-matrix metrics
def quant_table():
    Z = np.load(CACHE / "std_quant.npz")
    sources = list(Z["sources"])
    is_rand, genes = Z["is_rand"], Z["genes"]
    HLS = list(Z["hls"])
    m = len(np.load(CACHE / "gene_train_rate.npy"))
    rate_all = np.load(CACHE / "gene_train_rate.npy")
    dec = np.minimum((rankdata(rate_all, method="average")[genes] - 1) * 10 // m, 9).astype(int)
    mk = [(j, k.split()[1]) for k, v in gg.items() if k.startswith("marker") for j in v]
    pos = {int(g): i for i, g in enumerate(genes)}
    mi = np.array([pos[j] for j, _ in mk]); mh = [HLS.index(h) for _, h in mk]
    di = np.array([pos[j] for j in gg["depth"]])
    recs = []
    for s in sources:
        rho, eta = Z[f"rho__{s}"], Z[f"eta2__{s}"]
        hm, lo, hi = Z[f"hlmean__{s}"], Z[f"lo__{s}"], Z[f"hi__{s}"]
        # marker contrast: mean p in the marker's hemilineage minus the mean over the other 22
        contrast = np.array([hm[h, j] - np.delete(hm[:, j], h).mean() for h, j in zip(mh, mi)])
        kind = "data" if s == "Y" else ("p_tilde" if s.endswith("~") else "p_hat")
        r = dict(source=s, code=s.rstrip("~"), kind=kind,
                 name="binary Y (data)" if s == "Y" else short(s.rstrip("~")) + (" (depth-standardized)" if kind == "p_tilde" else ""),
                 median_abs_rho_depth=np.nanmedian(np.abs(rho[is_rand])),
                 mean_rho_depth=np.nanmean(rho[is_rand]),
                 frac_genes_abs_rho_gt_0p3=np.nanmean(np.abs(rho[is_rand]) > 0.3),
                 median_abs_rho_depth_8depthgenes=np.nanmedian(np.abs(rho[di])),
                 median_eta2_hl_all=np.nanmedian(eta[is_rand]),
                 median_eta2_hl_20markers=np.nanmedian(eta[mi]),
                 median_eta2_hl_top10pct_rate=np.nanmedian(eta[is_rand & (dec == 9)]),
                 mean_marker_contrast_20markers=float(np.mean(contrast)),
                 depth_gap_hi_minus_lo_all=float(np.mean(hi[is_rand] - lo[is_rand])),
                 depth_gap_hi_minus_lo_8depthgenes=float(np.mean(hi[di] - lo[di])))
        for d in range(10):
            r[f"median_abs_rho_decile{d + 1}"] = np.nanmedian(np.abs(rho[is_rand & (dec == d)]))
        recs.append(r)
    df = pd.DataFrame(recs)
    df.insert(3, "sample", "all 16,460 cells x 3,000 random genes (seed 0); markers/depth genes added separately")
    df.to_csv(TABS / "depth_std_metrics.csv", index=False)
    return df


def fig_quant(df):
    df = df.set_index("source")
    order = ["R3", "R4", "R5", "R6", "PO", "R7", "R8", "R9"]
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.8), gridspec_kw=dict(width_ratios=[1, 1.25, 1], wspace=0.42))
    # A: median |rho| with depth
    ax = axs[0]
    ypos = np.arange(len(order))[::-1]
    for y, c in zip(ypos, order):
        v = df.loc[c, "median_abs_rho_depth"]
        ax.scatter(v, y, s=110, color=colour(c), zorder=4)
        if c in STD:
            w = df.loc[c + "~", "median_abs_rho_depth"]
            ax.annotate("", xy=(w, y), xytext=(v, y), arrowprops=dict(arrowstyle="->", color=colour(c), lw=2))
            ax.scatter(w, y, s=110, facecolor="white", edgecolor=colour(c), lw=2.4, zorder=5)
            ax.text(w, y + 0.28, f"{w:.2f}", ha="center", va="bottom", fontsize=11, color=colour(c))
        ax.text(v, y + 0.28, f"{v:.2f}", ha="center", va="bottom", fontsize=11, color=colour(c))
    yv = df.loc["Y", "median_abs_rho_depth"]
    ax.axvline(yv, color="black", ls=":", lw=2)
    ax.text(yv + 0.01, ypos[-1] - 0.75, f"binary $Y$ {yv:.2f}", fontsize=11, va="bottom")
    ax.set_yticks(ypos); ax.set_yticklabels([f"{c} {short(c)}" for c in order])
    for t, c in zip(ax.get_yticklabels(), order):
        t.set_color(colour(c))
    ax.set_ylim(-1, len(order) - 0.3)
    ax.set_xlim(0, max(0.8, df.median_abs_rho_depth.max() + 0.08))
    ax.set_xlabel("median |Spearman $\\rho$| with log UMI\n(per gene, within hemilineage)")
    ax.set_title("A  depth dependence", loc="left", fontweight="bold")
    # B: by detection-rate decile
    ax = axs[1]
    x = np.arange(1, 11)
    cols = [f"median_abs_rho_decile{d}" for d in x]
    for c in order:
        ax.plot(x, df.loc[c, cols].values.astype(float), color=colour(c), lw=2, marker="o", ms=5, label=f"{c} $\\hat p$")
    for c in STD:
        ax.plot(x, df.loc[c + "~", cols].values.astype(float), color=colour(c), lw=2.6, ls="--", marker="o", ms=7,
                mfc="white", mew=2, label=f"{c} $\\tilde p$")
    ax.plot(x, df.loc["Y", cols].values.astype(float), color="black", lw=2.4, ls=":", label="binary $Y$")
    ax.set_xticks(x)
    ax.set_xlabel("gene detection-rate decile (1 = rarest)")
    ax.set_ylabel("median |$\\rho$| with log UMI")
    ax.set_title("B  by gene detection rate", loc="left", fontweight="bold")
    ax.legend(fontsize=10.5, ncol=3, loc="upper left", handlelength=1.8, columnspacing=0.8)
    ax.set_ylim(0, 1.15 * np.nanmax(df[cols].values.astype(float)) + 0.12)
    # C: hemilineage variance share
    ax = axs[2]
    for y, c in zip(ypos, order):
        for metric, mk_, dy in (("median_eta2_hl_all", "o", 0.17), ("median_eta2_hl_20markers", "D", -0.17)):
            v = df.loc[c, metric]
            ax.scatter(v, y + dy, s=95, marker=mk_, color=colour(c), zorder=4)
            if c in STD:
                w = df.loc[c + "~", metric]
                ax.scatter(w, y + dy, s=95, marker=mk_, facecolor="white", edgecolor=colour(c), lw=2.2, zorder=5)
                ax.plot([v, w], [y + dy] * 2, color=colour(c), lw=1.5, zorder=3)
    for metric, mk_, dy in (("median_eta2_hl_all", "o", 0.17), ("median_eta2_hl_20markers", "D", -0.17)):
        ax.axvline(df.loc["Y", metric], color="black", ls=":" if mk_ == "o" else "-.", lw=1.6)
    ax.set_yticks(ypos); ax.set_yticklabels([c for c in order])
    for t, c in zip(ax.get_yticklabels(), order):
        t.set_color(colour(c))
    ax.set_ylim(-1, len(order) - 0.3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("median share of variance between hemilineages")
    ax.set_title("C  cell-type structure kept", loc="left", fontweight="bold")
    h = [Line2D([], [], marker="o", color="#555555", ls="none", ms=9, label="all genes"),
         Line2D([], [], marker="D", color="#555555", ls="none", ms=8, label="20 marker genes"),
         Line2D([], [], marker="o", color="#555555", mfc="white", mew=2, ls="none", ms=9, label="open: $\\tilde p$"),
         Line2D([], [], color="black", ls="-.", lw=1.6, label="binary $Y$,\nmarkers")]
    ax.legend(handles=h, fontsize=10.5, loc="center", bbox_to_anchor=(0.4, 0.43), handletextpad=0.3)
    fig.suptitle("Whole-matrix sample (all 16,460 cells x 3,000 random genes): filled = observed $\\hat p$, "
                 "open = depth-standardized $\\tilde p$", fontsize=14.5, y=1.02)
    fs.finalize_figure(fig, FIGS / "fig_std_quant")


if __name__ == "__main__":
    which = sys.argv[1:] or ["heat", "quant"]          # or any of: overview depth diff quant
    if "heat" in which:
        which += ["overview", "depth", "diff"]
    for k, fn in (("overview", heat_overview), ("depth", heat_depth), ("diff", heat_diff)):
        if k in which:
            fn()
    if "quant" in which:
        df = quant_table()
        print(df.drop(columns=["sample"]).round(3).to_string())
        fig_quant(df)
