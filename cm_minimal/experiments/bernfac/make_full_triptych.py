"""Triptych for the FULL fly preMN 24h data: log counts | binary Y | BernFac P.

fig_full_triptych       : all 16,460 cells x 19,543 genes. Rendered with area-averaging
                          (each screen pixel = mean of the entries it covers), so dense
                          blocks read as detection / probability density.
fig_full_triptych_zoom  : 10 cells per hemilineage (23) x 3 top marker genes per hemilineage,
                          small enough that every entry is its own square.

Cells: hemilineage -> library -> depth. Genes: grouped by the hemilineage in which they are
most often detected, then by overall detection rate (full view); markers chosen by
detection-rate enrichment (zoom).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.colors import ListedColormap

from bernfac import sigmoid
from figstyle import apply_publication_style, finalize_figure

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/bernfac_20261020/full"
COUNTS = ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h/G_prime.npz"
CELLS = ROOT / "outputs/connectionminer_ready_inputs/bundles/motor/premn_24h/cells.csv.gz"
GENES = ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h/genes.csv"


def palette(n):
    cols = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    return cols[:n]


def load():
    cells = pd.read_csv(CELLS)
    keep = np.load(OUT / "data/genes_kept.npy")
    X = sp.load_npz(COUNTS).tocsr()[:, keep]
    Y = np.load(OUT / "data/Y.npy")
    genes = pd.read_csv(GENES).iloc[keep, 0].to_numpy()
    z = np.load(OUT / "fits/best_full.npz")
    return cells, X, Y, genes, z


def orders(cells, Y):
    hl = sorted(cells.hemilineage.unique())
    cells = cells.assign(_h=cells.hemilineage.map({h: i for i, h in enumerate(hl)}),
                         _d=np.log1p(cells.get("total_umi", pd.Series(np.zeros(len(cells))))))
    row = cells.sort_values(["_h", "library"]).index.to_numpy()
    H = np.zeros((len(cells), len(hl)), np.float32)
    H[np.arange(len(cells)), cells._h.to_numpy()] = 1
    rate = (H.T @ Y.astype(np.float32)) / H.sum(0)[:, None]       # hemilineage x gene detection rate
    enrich = rate - (rate.sum(0, keepdims=True) - rate) / (len(hl) - 1)   # vs mean of other hemilineages
    top = enrich.argmax(0)
    col = np.lexsort((-enrich.max(0), top))           # grouped by most-enriched hemilineage, strongest first
    return hl, row, col, rate


def annotate(ax, labels, names, colors, title):
    idx = pd.Series(labels).map({n: i for i, n in enumerate(names)}).to_numpy()[:, None]
    ax.imshow(idx, aspect="auto", interpolation="nearest", cmap=ListedColormap(colors),
              vmin=-0.5, vmax=len(names) - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, rotation=90, fontsize=10, pad=4)
    for s in ax.spines.values():
        s.set_visible(False)


def triptych(cells_o, mats, fname, suptitle, interp, xlabel, block_rows=None, gene_ticks=None):
    apply_publication_style()
    fig = plt.figure(figsize=(17, 8.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.03, 1, 1, 1], height_ratios=[1, 0.035],
                          wspace=0.12, hspace=0.14 if gene_ticks is None else 0.32)
    hl = sorted(cells_o.hemilineage.unique())
    annotate(fig.add_subplot(gs[0, 0]), cells_o.hemilineage.to_numpy(), hl, palette(len(hl)), "hemi.")
    specs = [("viridis", r"$\log(1+\mathrm{UMI\ count})$", "log-transformed counts"),
             (ListedColormap(["white", "black"]), r"$y_{ij}=\mathbf{1}[\mathrm{count}\geq 1]$", "binary matrix $Y$"),
             ("magma", r"$\hat p_{ij}=\sigma(\hat{\mathbf{u}}_i^{\top}\hat{\mathbf{v}}_j)$", None)]
    for j, (M, (cmap, clab, title)) in enumerate(zip(mats[:3], specs)):
        ax = fig.add_subplot(gs[0, 1 + j])
        vmax = np.percentile(M[M > 0], 99) if j == 0 else 1
        im = ax.imshow(M, aspect="auto", interpolation=interp, interpolation_stage="data",
                       cmap=cmap if j != 1 or interp == "nearest" else "Greys", vmin=0, vmax=vmax)
        ax.set_title(title or mats[3], fontsize=13, pad=8)
        ax.set_yticks([])
        if gene_ticks is None:
            ax.set_xlabel(xlabel, fontsize=11)
        if gene_ticks is not None:
            ax.set_xticks(np.arange(len(gene_ticks)), gene_ticks, rotation=90, fontsize=5)
        else:
            ax.set_xticks([])
        if block_rows is not None:
            for y in block_rows:
                ax.axhline(y - 0.5, color="white" if j != 1 else "#3775BA", lw=0.6)
        cax = fig.add_subplot(gs[1, 1 + j])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max" if j == 0 else "neither")
        if j == 1 and interp == "nearest":
            cb.set_ticks([0.25, 0.75], labels=["0  not detected", "1  detected"])
        elif j == 1:
            cb.set_label("detection rate (area-averaged)", fontsize=10)
        cb.set_label(clab if j != 1 or interp == "nearest" else "fraction detected (area-averaged)", fontsize=11)
    fig.suptitle(suptitle, fontsize=15, y=0.99)
    finalize_figure(fig, OUT / "figures" / fname)


def main():
    cells, X, Y, genes, z = load()
    U, V, lam, k = z["U"], z["V"], float(z["lam"]), int(z["k"])
    hl, row, col, rate = orders(cells, Y)
    cells_o = cells.iloc[row].reset_index(drop=True)
    breaks = np.flatnonzero(np.diff(pd.Categorical(cells_o.hemilineage, hl).codes)) + 1
    ptitle = rf"$\hat P$, $\lambda={lam:g},\ k={k}$ (CV-best)"

    # full view (float32; P computed in one go: 16460 x 19543 x 4 B = 1.3 GB)
    Lc = np.log1p(X[row][:, col].toarray().astype(np.float32))
    Yo = Y[np.ix_(row, col)].astype(np.float32)
    P = sigmoid((U[row].astype(np.float32) @ V[col].astype(np.float32).T)).astype(np.float32)
    triptych(cells_o, [Lc, Yo, P, ptitle], "fig_full_triptych",
             f"Fly motor preMN, 24 h APF: all {Y.shape[0]:,} cells $\\times$ {Y.shape[1]:,} detected genes",
             "antialiased", f"all {Y.shape[1]:,} genes (by enriched hemilineage)", block_rows=breaks)
    del Lc, Yo, P

    # markers: ALL cells x 40 most enriched genes per hemilineage (block diagonal)
    other_m = (rate.sum(0, keepdims=True) - rate) / (len(hl) - 1)
    enr = rate - other_m
    cols_m = []
    for i in range(len(hl)):
        cols_m += [g for g in np.argsort(-enr[i]) if g not in cols_m][:40]
    cols_m = np.array(cols_m)
    Lm = np.log1p(X[row][:, cols_m].toarray().astype(np.float32))
    Ym = Y[np.ix_(row, cols_m)].astype(np.float32)
    Pm = sigmoid(U[row].astype(np.float32) @ V[cols_m].astype(np.float32).T).astype(np.float32)
    triptych(cells_o, [Lm, Ym, Pm, ptitle], "fig_full_triptych_markers",
             f"All {Y.shape[0]:,} cells $\\times$ top-40 enriched genes per hemilineage ({len(cols_m)} genes)",
             "antialiased", f"{len(cols_m)} marker genes (40 per hemilineage)", block_rows=breaks)
    del Lm, Ym, Pm

    # zoom: 10 cells per hemilineage x 3 marker genes per hemilineage
    rng = np.random.default_rng(0)
    rows_z = np.concatenate([rng.choice(np.flatnonzero(cells.hemilineage.to_numpy() == h), 10, replace=False)
                             for h in hl])
    other = (rate.sum(0, keepdims=True) - rate) / (len(hl) - 1)
    enrich = rate - other
    cols_z = []
    for i in range(len(hl)):                      # top-3 most enriched genes per hemilineage, no repeats
        cols_z += [g for g in np.argsort(-enrich[i]) if g not in cols_z][:3]
    cols_z = np.array(cols_z)
    cz = cells.iloc[rows_z].reset_index(drop=True)
    Lz = np.log1p(X[rows_z][:, cols_z].toarray())
    Yz = Y[np.ix_(rows_z, cols_z)].astype(float)
    Pz = sigmoid(U[rows_z] @ V[cols_z].T)
    triptych(cz, [Lz, Yz, Pz, ptitle], "fig_full_triptych_zoom",
             "Zoom: 10 cells per hemilineage $\\times$ 3 marker genes per hemilineage (every entry visible)",
             "nearest", "marker genes", block_rows=np.arange(10, len(rows_z), 10),
             gene_ticks=genes[cols_z])


if __name__ == "__main__":
    main()
