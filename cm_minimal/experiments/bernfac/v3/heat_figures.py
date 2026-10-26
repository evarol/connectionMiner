"""Heatmap snippets (cells x genes) for every probability model, plus snippet metrics.

Selection from heat_select.py (v3/cache/heat_selection.json). Orientation: genes are rows, cells
are columns (5 hemilineages x {low, high} depth x 12 cells). The reference is the empirical
detection rate on TRAIN entries within hemilineage x depth stratum (all cells of that hemilineage
in the bottom or top total-UMI quartile), so it carries the depth effect but no cell-level detail.

Outputs (v3/figures): fig_heat_overview, fig_heat_markers, fig_heat_depth, fig_heat_diff
         (v3/tables): heat_metrics.csv
"""
import json, os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
import predictors as pr
import figstyle as fs
import v3style as vs

fs.apply_publication_style()
CACHE, FIGS, TABS = pr.CACHE, vs.FIGS, vs.V3 / "tables"
TABS.mkdir(parents=True, exist_ok=True)
sel = json.loads((CACHE / "heat_selection.json").read_text())
MODELS = vs.PROB_MODELS                                   # R3..R9
HL_COL = dict(zip(sel["hemilineages"], ["#1F5FA8", "#C0392B", "#2E8B57", "#7B4FA0", "#8C5A2B"]))
DOT = dict(marker="o", s=5, c="#2A7FFF", edgecolors="white", linewidths=0.35, zorder=5)


# --------------------------------------------------------------------------- compute snippet blocks
def compute():
    f = CACHE / "heat_blocks.npz"
    if f.exists():
        return dict(np.load(f, allow_pickle=True))
    rows = np.concatenate([g["rows"] for g in sel["cell_groups"]])
    ggroups = sel["gene_groups"]
    cols = np.concatenate([ggroups[k] for k in ggroups])
    keep = np.load(pr.DATA / "genes_kept.npy")
    X = sp.load_npz(pr.ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h/G_prime.npz").tocsc()[:, keep]
    Xs = X[:, cols].tocsr()
    Y = np.load(pr.DATA / "Y.npy", mmap_mode="r")
    H = np.load(pr.DATA / "heldout.npy", mmap_mode="r")
    Te = np.load(pr.DATA / "test.npy", mmap_mode="r")
    cells = pd.read_csv(pr.ROOT / "outputs/connectionminer_ready_inputs/bundles/motor/premn_24h/cells.csv.gz")
    hl = cells.hemilineage.values
    dq = np.load(CACHE / "depth_quartile.npy")
    out = dict(rows=rows, cols=cols, counts=Xs[rows].toarray(), Y=np.asarray(Y[rows][:, cols]),
               H=np.asarray(H[rows][:, cols]), T=np.asarray(Te[rows][:, cols]))
    # reference rates (train entries) per row: hemilineage x depth stratum, and hemilineage only
    ref, ref_hl = np.zeros(out["Y"].shape), np.zeros(out["Y"].shape)
    pos = 0
    for g in sel["cell_groups"]:
        d = 1 if g["depth"] == "low" else 2
        for arr, mask in ((ref, (hl == g["hemilineage"]) & (dq == d)), (ref_hl, hl == g["hemilineage"])):
            rr = np.flatnonzero(mask)
            yy = np.asarray(Y[rr][:, cols], float); tt = ~np.asarray(H[rr][:, cols])
            arr[pos:pos + len(g["rows"])] = (yy * tt).sum(0) / tt.sum(0)
        pos += len(g["rows"])
    out["ref"], out["ref_hl"] = ref, ref_hl
    for c in MODELS + ["PO"]:
        p = pr.load(c)
        P, Q = p.block(rows, cols)
        out[f"P_{c}"], out[f"Q_{c}"] = P, Q
        if c == "R6":
            out["mu_R6"] = p.mu_block(rows, cols)
        del p
    np.savez(f, **out)
    return out


D = compute()
names = sel["gene_names"]
gg = sel["gene_groups"]
gene_group = np.concatenate([[k] * len(v) for k, v in gg.items()])
gene_name = np.array([names[str(j)] for j in D["cols"]])
row_hl = np.concatenate([[g["hemilineage"]] * len(g["rows"]) for g in sel["cell_groups"]])
row_dep = np.concatenate([[g["depth"]] * len(g["rows"]) for g in sel["cell_groups"]])
depth = np.load(CACHE / "cell_depth.npy")[D["rows"]]


def group_of(k):
    return "marker" if k.startswith("marker") else k


# --------------------------------------------------------------------------- metrics table
def metrics():
    recs = []
    Yv, Hv, ref = D["Y"].astype(float), D["H"], D["ref"]
    groups = {"all": np.ones(len(gene_group), bool)}
    for k in ("marker", "depth", "rare", "housekeeping"):
        groups[k] = np.array([group_of(g) == k for g in gene_group])
    lo, hi = row_dep == "low", row_dep == "high"
    srcs = {c: (D[f"P_{c}"], D[f"Q_{c}"]) for c in MODELS + ["PO"]}
    srcs["reference (HL x depth, train)"] = (ref, 1 - ref)
    srcs["reference (HL only, train)"] = (D["ref_hl"], 1 - D["ref_hl"])
    for c, (P, Q) in srcs.items():
        lp = np.log(np.maximum(np.where(Yv > 0, P, Q), 1e-300))
        for gname, gm in groups.items():
            Hm = Hv[:, gm]
            Pg, Rg = P[:, gm], ref[:, gm]
            recs.append(dict(
                code=c, name=vs.name(c) if c in vs.METHODS else ("Poisson counts" if c == "PO" else c),
                genes=gname, n_genes=int(gm.sum()), n_heldout=int(Hm.sum()),
                heldout_nll=float(-lp[:, gm][Hm].mean()), heldout_nll_test_only=float(-lp[:, gm][D["T"][:, gm]].mean()),
                corr_ref_hl_depth=float(np.corrcoef(Pg.ravel(), Rg.ravel())[0, 1]),
                corr_ref_hl_only=float(np.corrcoef(Pg.ravel(), D["ref_hl"][:, gm].ravel())[0, 1]),
                mean_abs_diff_ref=float(np.abs(Pg - Rg).mean()),
                mean_P_low=float(Pg[lo].mean()), mean_P_high=float(Pg[hi].mean()),
                depth_gap_P=float(Pg[hi].mean() - Pg[lo].mean()),
                depth_gap_ref=float(Rg[hi].mean() - Rg[lo].mean()),
                within_group_sd=float(np.mean([Pg[(row_hl == h) & (row_dep == d)].std(0).mean()
                                               for h in sel["hemilineages"] for d in ("low", "high")]))))
    df = pd.DataFrame(recs)
    df.to_csv(TABS / "heat_metrics.csv", index=False)
    return df


M = metrics()
print(M[M.genes.isin(["all", "depth", "marker"])][["code", "genes", "heldout_nll", "corr_ref_hl_depth",
                                                    "mean_abs_diff_ref", "depth_gap_P", "depth_gap_ref",
                                                    "within_group_sd"]].round(3).to_string())


# --------------------------------------------------------------------------- plotting helpers
def gene_idx(kind):
    if kind == "overview":
        idx = []
        for k, v in gg.items():
            n = {"depth": 6, "rare": 3, "housekeeping": 3}.get(k, 3)
            take = list(range(len(v)))[:n] if k != "housekeeping" else [0, 2, 3]
            start = int(np.flatnonzero(gene_group == k)[0])
            idx += [start + t for t in take]
        return np.array(idx)
    if kind == "markers":
        return np.flatnonzero([g.startswith("marker") for g in gene_group])
    if kind == "depth":
        return np.flatnonzero(gene_group == "depth")
    raise KeyError(kind)


def col_bounds():
    """Column boundaries between (hemilineage, depth) groups."""
    sizes = [len(g["rows"]) for g in sel["cell_groups"]]
    return np.cumsum(sizes)


def gene_label_colour(k):
    return HL_COL[k.split()[1]] if k.startswith("marker") else "#222222"


def draw(ax, A, gi, cmap="magma", vmin=0, vmax=1, norm=None, dots=False, ylabels=False, xlabels=False,
         title=None, title_colour="black"):
    A = A[:, gi].T                                       # genes x cells
    im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=None if norm else vmin, vmax=None if norm else vmax,
                   norm=norm, interpolation="nearest")
    cb = col_bounds()
    for i, x in enumerate(cb[:-1]):
        ax.axvline(x - 0.5, color="white", lw=2.2 if i % 2 == 1 else 0.6)
    gsel = gene_group[gi]
    for r in np.flatnonzero(gsel[1:] != gsel[:-1]):
        if group_of(gsel[r]) != group_of(gsel[r + 1]) or True:
            ax.axhline(r + 0.5, color="white", lw=1.6 if group_of(gsel[r]) != group_of(gsel[r + 1]) else 0.6)
    if dots:
        Hm = D["H"][:, gi].T
        yy, xx = np.nonzero(Hm)
        ax.scatter(xx, yy, **DOT)
    ax.set_yticks(range(len(gi)))
    if ylabels:
        ax.set_yticklabels(gene_name[gi], fontsize=9.5)
        for t, k in zip(ax.get_yticklabels(), gsel):
            t.set_color(gene_label_colour(k))
            if k.startswith("marker"):
                t.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    centres = np.r_[0, cb[:-1]] + np.diff(np.r_[0, cb]) / 2 - 0.5
    ax.set_xticks(centres)
    if xlabels:
        ax.set_xticklabels(["lo", "hi"] * len(sel["hemilineages"]), fontsize=9)
        for h_i, h in enumerate(sel["hemilineages"]):
            ax.text(((cb[2 * h_i - 1] if h_i else 0) + cb[2 * h_i + 1]) / 2 - 0.5, -0.07 * 27 / len(gi) - 0.04, h, color=HL_COL[h],
                    fontsize=11, fontweight="bold", ha="center", va="top", transform=ax.get_xaxis_transform())
    else:
        ax.set_xticklabels([])
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, color=title_colour, pad=5)
    return im


def method_title(c):
    return f"{c}  {vs.name(c)}"


def add_hl_header(fig, ax_left, ax_right, y):
    pass


PANELS = [("counts", "log$_{1+}$ counts"), ("Y", "binary $Y$ (data)")] + [(c, method_title(c)) for c in MODELS]


def panel_data(key):
    if key == "counts":
        return np.log1p(D["counts"])
    if key == "Y":
        return D["Y"].astype(float)
    return D[f"P_{key}"]


def heat_grid(gi, panels, fname, ncol=5, figsize=(19, 9.5), suptitle=None, dots_all=False, label_every=False,
              extra_right=None, hspace=0.32):
    nrow = int(np.ceil(len(panels) / ncol))
    fig = plt.figure(figsize=figsize)
    wr = [1] * ncol + ([0.05] if extra_right is None else [0.05, 0.42, extra_right[0]])
    gs = fig.add_gridspec(nrow, len(wr), width_ratios=wr, wspace=0.06, hspace=hspace)
    vmax_c = float(np.log1p(D["counts"][:, gi]).max())
    ims = {}
    for k, (key, title) in enumerate(panels):
        r, c = divmod(k, ncol)
        ax = fig.add_subplot(gs[r, c])
        tc = vs.colour(key) if key in vs.METHODS else "black"
        if key == "counts":
            im = draw(ax, panel_data(key), gi, cmap="viridis", vmin=0, vmax=vmax_c, dots=False,
                      ylabels=(c == 0), xlabels=(r == nrow - 1) or label_every, title=title)
            ims["counts"] = im
        elif key == "mu_R6":
            im = draw(ax, np.maximum(D["mu_R6"], 1e-3), gi, cmap="viridis",
                      norm=LogNorm(1e-2, max(1.0, float(D["mu_R6"][:, gi].max()))), dots=dots_all,
                      ylabels=(c == 0), xlabels=(r == nrow - 1) or label_every,
                      title="R6  NB expected count $\\hat\\mu$", title_colour=vs.colour("R6"))
            ims["mu"] = im
        else:
            im = draw(ax, panel_data(key), gi, dots=(key == "Y") or (dots_all and key != "counts"),
                      ylabels=(c == 0), xlabels=(r == nrow - 1) or label_every, title=title, title_colour=tc)
            ims["p"] = im
    # colourbars
    cax = fig.add_subplot(gs[0, ncol])
    cb = fig.colorbar(ims["p"], cax=cax)
    cb.set_label("$\\Pr(y=1)$ / $Y$", fontsize=13)
    cb.ax.tick_params(labelsize=11)
    if nrow > 1 and ("counts" in ims or "mu" in ims):
        cax2 = fig.add_subplot(gs[1, ncol])
        key = "mu" if "mu" in ims else "counts"
        cb2 = fig.colorbar(ims[key], cax=cax2)
        cb2.set_label("$\\hat\\mu$ (NB)" if key == "mu" else "log$_{1+}$ count", fontsize=13)
        cb2.ax.tick_params(labelsize=11)
    return fig, gs


def legend_note(fig, x=0.5, y=0.005, extra=""):
    h = [Line2D([], [], **{k: v for k, v in DOT.items() if k not in ("s", "c", "edgecolors", "linewidths", "zorder")},
                markersize=5, markerfacecolor=DOT["c"], markeredgecolor="white", linestyle="none")]
    fig.legend(h, ["held-out entry (not used in fitting; scored on these)"], loc="lower center",
               bbox_to_anchor=(x, y), fontsize=11.5, frameon=False, handletextpad=0.3)
    if extra:
        fig.text(0.5, y - 0.03, extra, ha="center", va="top", fontsize=11.5, color="#444444")


# --------------------------------------------------------------------------- 1. overview
gi = gene_idx("overview")
fig, gs = heat_grid(gi, PANELS, "overview", ncol=5, figsize=(19, 9.6), hspace=0.2)
ax = fig.add_subplot(gs[1, 4]); ax.axis("off")
ax.text(0.0, 0.97, "How to read\n\n"
        "Columns: cells. In each\nhemilineage 12 low-depth,\nthen 12 high-depth cells\n(bottom / top UMI quartile).\n\n"
        "Rows: 3 markers per\nhemilineage (name colour),\nthen 6 depth-sensitive,\n3 rare, 3 ubiquitous genes.\n\n"
        "Model panels: fitted\n$\\Pr(y_{ij}=1)$ for every entry,\nincl. held-out ones (blue\ndots on the data panel).",
        transform=ax.transAxes, va="top", ha="left", fontsize=11.5, color="#333333", linespacing=1.2)
fig.suptitle("Cells $\\times$ genes: 5 hemilineages $\\times$ {low, high} depth $\\times$ 12 cells; "
             "rows = markers (coloured by hemilineage), depth-sensitive, rare, ubiquitous genes",
             fontsize=14.5, y=0.985)
legend_note(fig, y=0.0)
fs.finalize_figure(fig, FIGS / "fig_heat_overview")

# --------------------------------------------------------------------------- 2. markers
gi = gene_idx("markers")
fig, gs = heat_grid(gi, [p for p in PANELS if p[0] != "counts"], "markers", ncol=4, figsize=(19, 10.4), hspace=0.2)
fig.suptitle("Marker genes: top-4 detection-enriched genes per hemilineage "
             "(03B, 14A, 12B well separated; 05B/06B most confused by 15-NN)", fontsize=14.5, y=0.985)
legend_note(fig, y=0.0)
fs.finalize_figure(fig, FIGS / "fig_heat_markers")

# --------------------------------------------------------------------------- 3. depth
gi = gene_idx("depth")
panels = [("counts", "log$_{1+}$ counts"), ("Y", "binary $Y$ (data)"), ("R3", method_title("R3")),
          ("R4", method_title("R4")), ("R5", method_title("R5")), ("R6", method_title("R6")),
          ("mu_R6", ""), ("R7", method_title("R7")), ("R8", method_title("R8")), ("R9", method_title("R9"))]
fig, gs = heat_grid(gi, panels, "depth", ncol=5, figsize=(21, 8.6), dots_all=True, extra_right=(1.25,), hspace=0.42)
# right: mean P over depth genes, low vs high depth, vs empirical stratum rate
ax = fig.add_subplot(gs[:, -1])
dg = gene_group == "depth"
lo, hi = row_dep == "low", row_dep == "high"
emp = [D["ref"][lo][:, dg].mean(), D["ref"][hi][:, dg].mean()]
ax.plot([0, 1], emp, color="black", lw=3.2, marker="s", ms=9, label="empirical (train)", zorder=4)
for k, c in enumerate(MODELS):
    P = D[f"P_{c}"][:, dg]
    v = [P[lo].mean(), P[hi].mean()]
    ax.plot([0, 1], v, color=vs.colour(c), lw=2, marker="o", ms=6, alpha=0.95, label=c)
ax.set_xticks([0, 1]); ax.set_xticklabels(["low depth", "high depth"])
ax.set_xlim(-0.25, 1.25)
ax.set_ylabel("mean $\\hat p$ over the 8 genes")
ax.set_title("depth gap", fontsize=13)
ax.legend(fontsize=10.5, loc="upper left", ncol=1, handlelength=1.2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.suptitle("Depth-sensitive genes (detection rises most from the bottom to the top UMI quartile within hemilineage)",
             fontsize=14.5, y=1.0)
legend_note(fig, x=0.42, y=-0.075)
fs.finalize_figure(fig, FIGS / "fig_heat_depth")

# --------------------------------------------------------------------------- 4. differences vs reference
gi = gene_idx("overview")
fig = plt.figure(figsize=(19, 10.2))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.06, hspace=0.32)
ax = fig.add_subplot(gs[0, 0])
imr = draw(ax, D["ref"], gi, ylabels=True, title="reference: train rate in\nhemilineage $\\times$ depth stratum")
norm = TwoSlopeNorm(0, -0.6, 0.6)
for k, c in enumerate(MODELS):
    r, cc = divmod(k + 1, 4)
    ax = fig.add_subplot(gs[r, cc])
    s = M[(M.code == c) & (M.genes == "all")].iloc[0]
    imd = draw(ax, D[f"P_{c}"] - D["ref"], gi, cmap="RdBu_r", norm=norm, ylabels=(cc == 0), xlabels=(r == 1),
               title=f"{c} {vs.name(c)} $-$ reference\nmean |diff| {s.mean_abs_diff_ref:.3f},  "
                     f"cell-level sd {s.within_group_sd:.3f}", title_colour=vs.colour(c))
cax = fig.add_subplot(gs[0, 4]); cb = fig.colorbar(imr, cax=cax); cb.set_label("rate", fontsize=13)
cax = fig.add_subplot(gs[1, 4]); cb = fig.colorbar(imd, cax=cax); cb.set_label("$\\hat p - $ reference", fontsize=13)
fig.suptitle("Difference from the empirical rate within hemilineage $\\times$ depth stratum "
             "(red = model above, blue = below).  Cell-level sd: spread of $\\hat p$ across cells within a block",
             fontsize=14.5, y=0.985)
fs.finalize_figure(fig, FIGS / "fig_heat_diff")
