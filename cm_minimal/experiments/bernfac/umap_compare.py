"""Representation comparison on the FULL motor preMN 24h APF dataset.

R1  processed counts : normalize_total(1e4) -> log1p -> 2000 HVGs (seurat_v3 on raw
                       counts) -> scale(max_value=10) -> PCA 50
R2  binary Y         : Y = 1[count >= 1] on the kept genes -> PCA 50 of column-centred Y
                       (implicit centring via a LinearOperator + scipy svds, Y stays sparse)
R3  BernFac P        : Psi = U V^T from fits/best_full.npz; rotation-invariant embedding
                       E = A S from the thin SVD Psi = A S B^T, computed from QR(U), QR(V) and
                       the SVD of the k x k matrix R_U R_V^T (n x m is never formed)

Each embedding -> UMAP (n_neighbors=15, min_dist=0.3, random_state=0) and a metrics table.

Usage
    OMP_NUM_THREADS=4 python umap_compare.py                 # R1, R2
    OMP_NUM_THREADS=4 python umap_compare.py --with-bernfac  # + R3 (needs fits/best_full.npz)
    python umap_compare.py --with-bernfac --wait-hours 6     # poll every 10 min for the fit
Cached embeddings / UMAPs in umap/cache are reused unless --recompute.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import LinearOperator, svds  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

COUNTS = ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h/G_prime.npz"
GENES = ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h/genes.csv"
CELLS = ROOT / "outputs/connectionminer_ready_inputs/bundles/motor/premn_24h/cells.csv.gz"
FULL = ROOT / "outputs/bernfac_20261020/full"
KEPT = FULL / "data/genes_kept.npy"
FIT = FULL / "fits/best_full.npz"
FIT_OFF = FULL / "fits/best_full_offsets.npz"
FIT_CLL = FULL / "fits/best_full_cloglog.npz"
MODELS = FULL.parent / "models"
# Embeddings saved by the model-comparison workers (cells x dims); name -> (label, path)
EXTRA = {"R6": ("R6 NB counts", MODELS / "counts/embedding_nb.npy"),
         "R7": ("R7 mixture of local factors", MODELS / "mfa/embedding.npy"),
         "R8": ("R8 neural manifold (z)", MODELS / "neural/embedding.npy"),
         "R9": ("R9 latent blocks (R B)", MODELS / "lbm/embedding.npy")}    # cloglog link log(-log(1-p)) = a1' + 1b' + UV'; R5 embeds UV' only   # offset model Psi = a1' + 1b' + UV'; R4 embeds UV' only
OUT = FULL / "umap"
CACHE = OUT / "cache"
FIGS = OUT / "figures"

SEED = 0
NAMES = {"R1": "R1 processed counts", "R2": "R2 binary Y", "R3": "R3 BernFac P", "R4": "R4 BernFac + offsets", "R5": "R5 BernFac cloglog",
         "R6": "R6 NB counts", "R7": "R7 mixture of local factors", "R8": "R8 neural manifold", "R9": "R9 latent blocks"}


# ----------------------------------------------------------------------------- data
def load_data():
    X = sp.load_npz(COUNTS).tocsr()
    cells = pd.read_csv(CELLS)
    assert X.shape[0] == len(cells), (X.shape, len(cells))
    if KEPT.exists():
        kept = np.load(KEPT)
        print(f"loaded genes_kept.npy ({kept.size} genes)")
    else:
        kept = np.flatnonzero(np.asarray((X > 0).sum(0)).ravel() >= 1)
        KEPT.parent.mkdir(parents=True, exist_ok=True)
        if not KEPT.exists():
            np.save(KEPT, kept)
        print(f"wrote genes_kept.npy ({kept.size} genes)")
    X = X[:, kept].tocsr()
    X.sort_indices()
    return X, cells, kept


# ----------------------------------------------------------------------------- embeddings
def embed_r1(X):
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(X.astype(np.float32))
    a.layers["counts"] = a.X.copy()
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=2000, layer="counts")
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a = a[:, a.var["highly_variable"].values].copy()
    sc.pp.scale(a, max_value=10)
    sc.pp.pca(a, n_comps=50, svd_solver="arpack", random_state=SEED)
    return a.obsm["X_pca"].astype(np.float64), {"hvg_flavor": "seurat_v3 (raw counts)",
                                                 "var_ratio_sum": float(a.uns["pca"]["variance_ratio"].sum())}


def embed_r2(X, k=50):
    Y = (X >= 1).astype(np.float64).tocsr()
    n, m = Y.shape
    mu = np.asarray(Y.mean(0)).ravel()
    YT = Y.T.tocsr()

    def mv(v):
        v = np.asarray(v).reshape(m, -1)
        return Y @ v - (mu @ v)[None, :]

    def rmv(u):
        u = np.asarray(u).reshape(n, -1)
        return YT @ u - np.outer(mu, u.sum(0))

    op = LinearOperator((n, m), matvec=mv, rmatvec=rmv, matmat=mv, rmatmat=rmv, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    u, s, vt = svds(op, k=k, v0=rng.standard_normal(min(n, m)), solver="arpack")
    o = np.argsort(s)[::-1]
    u, s = u[:, o], s[o]
    # sign convention: largest-|loading| cell positive per component
    sgn = np.sign(u[np.abs(u).argmax(0), np.arange(k)])
    E = u * s * sgn
    tot = float(((Y.multiply(Y)).sum() - n * (mu ** 2).sum()))
    return E, {"centred_frob_sq": tot, "var_ratio_sum": float((s ** 2).sum() / tot)}


def embed_r3(fit_path=None):
    fit_path = Path(fit_path or FIT)
    z = np.load(fit_path)
    U, V = np.asarray(z["U"], float), np.asarray(z["V"], float)
    Qu, Ru = np.linalg.qr(U)
    Qv, Rv = np.linalg.qr(V)
    W, s, Zt = np.linalg.svd(Ru @ Rv.T)
    E = Qu @ (W * s)
    sgn = np.sign(E[np.abs(E).argmax(0), np.arange(E.shape[1])])
    E = E * sgn
    return E, {"k": int(U.shape[1]), "n": int(U.shape[0]), "m": int(V.shape[0]),
               "singular_values_top5": s[:5].round(3).tolist(), "fit_path": str(fit_path), "fit_mtime": time.ctime(fit_path.stat().st_mtime)}


def run_umap(E):
    import umap
    return umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=SEED).fit_transform(E).astype(np.float32)


# ----------------------------------------------------------------------------- metrics
def knn_graph(E, k=15):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(E)
    d, idx = nn.kneighbors(E)
    return d[:, 1:], idx[:, 1:]


def cv_balacc(E, y, model):
    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pred = cross_val_predict(model, E, y, cv=cv, n_jobs=1)
    return balanced_accuracy_score(y, pred), accuracy_score(y, pred)


def leiden_scores(E, labels, resolutions=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0)):
    import anndata as ad
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    a = ad.AnnData(np.zeros((E.shape[0], 1), np.float32))
    a.obsm["E"] = E.astype(np.float32)
    sc.pp.neighbors(a, n_neighbors=15, use_rep="E", random_state=SEED)
    rows = []
    for r in resolutions:
        sc.tl.leiden(a, resolution=r, key_added="cl", random_state=SEED, flavor="igraph",
                     n_iterations=2, directed=False)
        cl = a.obs["cl"].values
        rows.append({"res": r, "n_clusters": int(len(set(cl))),
                     "ARI": adjusted_rand_score(labels, cl), "NMI": normalized_mutual_info_score(labels, cl)})
    return pd.DataFrame(rows)


def neighbour_entropy(idx, lab):
    codes, uniq = pd.factorize(lab)
    L = len(uniq)
    nb = codes[idx]
    k = idx.shape[1]
    cnt = np.zeros((idx.shape[0], L))
    np.add.at(cnt, (np.repeat(np.arange(idx.shape[0]), k), nb.ravel()), 1)
    p = cnt / k
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(p > 0, p * np.log(p), 0), 1)
    return H / np.log(min(k, L))


def metrics_for(E, cells, logumi, name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    hl = cells["hemilineage"].astype(str).values
    lib = cells["library"].astype(str).values
    sex = cells["Sex"].astype(str).values
    knn = KNeighborsClassifier(n_neighbors=15)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(E), 5000, replace=False)
    out = {}
    out["hl_knn_balacc"], out["hl_knn_acc"] = cv_balacc(E, hl, knn)
    out["hl_logreg_balacc"], out["hl_logreg_acc"] = cv_balacc(E, hl, lr)
    out["hl_silhouette_euc"] = silhouette_score(E[sub], hl[sub], metric="euclidean")
    out["hl_silhouette_cos"] = silhouette_score(E[sub], hl[sub], metric="cosine")
    ld = leiden_scores(E, hl)
    ld.insert(0, "rep", name)
    best = ld.loc[ld["ARI"].idxmax()]
    out["hl_leiden_best_ARI"], out["hl_leiden_best_ARI_res"] = best["ARI"], best["res"]
    out["hl_leiden_best_NMI"] = ld["NMI"].max()
    r1 = ld.loc[ld["res"] == 1.0].iloc[0]
    out["hl_leiden_r1_ARI"], out["hl_leiden_r1_NMI"], out["hl_leiden_r1_nclust"] = r1["ARI"], r1["NMI"], r1["n_clusters"]
    out["lib_knn_balacc"], out["lib_knn_acc"] = cv_balacc(E, lib, knn)
    out["sex_knn_balacc"], out["sex_knn_acc"] = cv_balacc(E, sex, knn)
    out["lib_silhouette_euc"] = silhouette_score(E[sub], lib[sub], metric="euclidean")
    _, idx = knn_graph(E, 15)
    H = neighbour_entropy(idx, lib)
    out["lib_mix_entropy"] = H.mean()
    # reference: same statistic for random neighbours (ideal mixing given library sizes)
    ridx = rng.integers(0, len(E), size=idx.shape)
    out["lib_mix_entropy_random_ref"] = neighbour_entropy(ridx, lib).mean()
    out["lib_mix_rel"] = out["lib_mix_entropy"] / out["lib_mix_entropy_random_ref"]
    out["hl_nn_purity"] = (hl[idx] == hl[:, None]).mean()
    out["depth_spearman_c1"] = spearmanr(E[:, 0], logumi)[0]
    out["depth_spearman_c2"] = spearmanr(E[:, 1], logumi)[0]
    out["depth_max_abs_top2"] = max(abs(out["depth_spearman_c1"]), abs(out["depth_spearman_c2"]))
    out["depth_max_abs_all"] = max(abs(spearmanr(E[:, j], logumi)[0]) for j in range(E.shape[1]))
    out["dim"] = E.shape[1]
    print(f"[{name}] metrics in {time.time() - t0:.0f}s", flush=True)
    return {"rep": name, **{k: float(v) for k, v in out.items()}}, ld


# ----------------------------------------------------------------------------- figures
QUAL23 = ["#222222", "#F3C300", "#875692", "#F38400", "#A1CAF1", "#BE0032", "#C2B280", "#848482",
          "#008856", "#E68FAC", "#0067A5", "#F99379", "#604E97", "#F6A600", "#B3446C", "#DCD300",
          "#882D17", "#8DB600", "#654522", "#E25822", "#2B3D26", "#00A2D3", "#FF4A46"]


def make_figures(reps, umaps, cells, logumi, metrics):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from figstyle import PALETTE, FS_LEGEND, apply_publication_style, finalize_figure, panel_label
    apply_publication_style()
    hl = cells["hemilineage"].astype(str).values
    lib = cells["library"].astype(str).values
    hl_u = sorted(set(hl), key=lambda s: (len(s), s))
    lib_u = sorted(set(lib))
    hl_c = dict(zip(hl_u, QUAL23))
    lib_cmap = plt.get_cmap("tab20")
    lib_cols = [lib_cmap(i % 20) for i in range(20)] + [PALETTE["grey_dark"]]
    lib_c = dict(zip(lib_u, lib_cols))
    order = np.random.default_rng(SEED).permutation(len(hl))  # avoid over-plotting by class

    nr = len(reps)
    fig, axes = plt.subplots(nr, 3, figsize=(17.5, 5.2 * nr), squeeze=False)
    letters = iter("abcdefghijkl")
    for i, r in enumerate(reps):
        Z = umaps[r][order]
        for j, (title, cols) in enumerate([
                ("hemilineage", np.array([hl_c[h] for h in hl[order]])),
                ("library", np.array([lib_c[x] for x in lib[order]], dtype=object)),
                (r"$\log_{10}$ total UMI", None)]):
            ax = axes[i, j]
            if cols is None:
                sc_ = ax.scatter(Z[:, 0], Z[:, 1], c=logumi[order], s=0.6, cmap="viridis", rasterized=True, lw=0)
                cb = fig.colorbar(sc_, ax=ax, fraction=0.04, pad=0.01)
                cb.ax.tick_params(labelsize=FS_LEGEND)
            else:
                ax.scatter(Z[:, 0], Z[:, 1], c=list(cols), s=0.6, rasterized=True, lw=0)
            ax.set_xticks([]), ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            if i == 0:
                ax.set_title(f"colour: {title}")
            if j == 0:
                ax.set_ylabel(NAMES[r], fontweight="bold")
                panel_label(ax, next(letters), x=-0.08)
            ax.text(0.01, 0.01, "UMAP", transform=ax.transAxes, fontsize=9, color=PALETTE["grey_mid"])
    h_hl = [Line2D([], [], marker="o", ls="", ms=6, color=hl_c[h], label=h) for h in hl_u]
    h_lib = [Line2D([], [], marker="o", ls="", ms=6, color=lib_c[x], label=x) for x in lib_u]
    fig.legend(handles=h_hl, title="hemilineage", loc="upper center", bbox_to_anchor=(0.3, 0.0),
               ncol=12, fontsize=10, handletextpad=0.1, columnspacing=0.6)
    fig.legend(handles=h_lib, title="library", loc="upper center", bbox_to_anchor=(0.72, 0.0),
               ncol=7, fontsize=9, handletextpad=0.1, columnspacing=0.5)
    fig.tight_layout()
    finalize_figure(fig, FIGS / "umap_grid")

    # ---- metrics bars
    m = metrics.set_index("rep").loc[reps]
    panels = [("hemilineage (higher = better)",
               [("hl_knn_balacc", "kNN bal. acc."), ("hl_logreg_balacc", "LogReg bal. acc."),
                ("hl_leiden_best_ARI", "Leiden best ARI"), ("hl_leiden_best_NMI", "Leiden best NMI"),
                ("hl_silhouette_euc", "silhouette")]),
              ("batch / library (lower = less batch)",
               [("lib_knn_balacc", "library kNN bal. acc."), ("sex_knn_balacc", "sex kNN bal. acc."),
                ("lib_silhouette_euc", "library silhouette")]),
              ("mixing & depth",
               [("lib_mix_rel", "library NN entropy\n/ random (↑ mixing)"),
                ("depth_max_abs_top2", r"max $|\rho_s|$ top-2 comp." "\nvs log UMI (↓)")])]
    rep_cols = {"R1": PALETTE["red_strong"], "R2": PALETTE["neutral"], "R3": PALETTE["blue_main"],
                "R4": PALETTE["teal"], "R5": PALETTE["violet"],
                "R6": "#E69F00", "R7": "#009E73", "R8": "#CC79A7", "R9": "#56B4E9"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [5, 3, 2]})
    for ax, (title, items), let in zip(axes, panels, "abc"):
        x = np.arange(len(items))
        w = 0.8 / len(reps)
        for i, r in enumerate(reps):
            vals = [m.loc[r, k] for k, _ in items]
            b = ax.bar(x + (i - (len(reps) - 1) / 2) * w, vals, w, color=rep_cols[r], label=NAMES[r],
                       edgecolor="black", lw=0.6)
            for bb, v in zip(b, vals):
                ax.text(bb.get_x() + bb.get_width() / 2, max(v, 0) + 0.01, f"{v:.2f}", ha="center",
                        va="bottom", fontsize=9, rotation=90)
        ax.set_xticks(x, [l for _, l in items], rotation=30, ha="right", fontsize=11)
        ax.set_title(title)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(min(0, ax.get_ylim()[0]), max(1.0, ax.get_ylim()[1]) * 1.12)
        panel_label(ax, let, x=-0.12)
    bb = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
    axes[1].axhline(1 / 21, color=PALETTE["grey_dark"], ls="--", lw=1, zorder=5)
    axes[1].text(2.45, 0.15, "library chance 1/21 (dashed)", fontsize=9, ha="right", zorder=6, bbox=bb)
    axes[1].plot([0.55, 1.45], [0.5, 0.5], color=PALETTE["grey_dark"], ls=":", lw=1.5, zorder=5)
    axes[1].text(1.0, 0.52, "sex chance 0.5", fontsize=9, ha="center", zorder=6, bbox=bb)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(reps), fontsize=13, bbox_to_anchor=(0.5, 1.0))
    finalize_figure(fig, FIGS / "metrics_bars")


# ----------------------------------------------------------------------------- main
def get_embedding(name, X, recompute):
    f = CACHE / f"{name}_embedding.npz"
    if f.exists() and not recompute:
        z = np.load(f, allow_pickle=True)
        return z["E"], json.loads(str(z["info"]))
    t0 = time.time()
    if name == "R3":
        E, info = embed_r3()
    elif name == "R4":
        E, info = embed_r3(FIT_OFF)
    elif name == "R5":
        E, info = embed_r3(FIT_CLL)
    elif name in EXTRA:
        E = np.load(EXTRA[name][1]).astype(np.float64)
        info = {"source": str(EXTRA[name][1]), "dim": int(E.shape[1])}
    else:
        E, info = {"R1": embed_r1, "R2": embed_r2}[name](X)
    info["seconds"] = round(time.time() - t0, 1)
    np.savez_compressed(f, E=E.astype(np.float32), info=json.dumps(info))
    print(f"[{name}] embedding {E.shape} in {info['seconds']}s  {info}", flush=True)
    return E.astype(np.float32), info


def main():
    global FIT
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-bernfac", action="store_true", help="also compute R3 from fits/best_full.npz")
    ap.add_argument("--wait-hours", type=float, default=0.0, help="poll every 10 min for the fit up to this long")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--fit", default=None, help=f"BernFac npz with U, V (default {FIT})")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--with-offsets", action="store_true", help="also compute R4 from fits/best_full_offsets.npz")
    ap.add_argument("--with-cloglog", action="store_true", help="also compute R5 from fits/best_full_cloglog.npz")
    ap.add_argument("--extra", nargs="*", default=[], help="extra worker embeddings to include, e.g. R6 R7 R8 R9")
    args = ap.parse_args()
    if args.fit:
        FIT = Path(args.fit).resolve()
    for d in (CACHE, FIGS):
        d.mkdir(parents=True, exist_ok=True)

    X, cells, kept = load_data()
    logumi = np.log10(np.asarray(X.sum(1)).ravel())
    reps = ["R1", "R2"]
    if args.with_bernfac:
        t_end = time.time() + 3600 * args.wait_hours
        while not FIT.exists() and time.time() < t_end:
            print(f"waiting for {FIT} ...", flush=True)
            time.sleep(600)
        if FIT.exists():
            while time.time() - FIT.stat().st_mtime < 120:  # let a just-written fit finish
                time.sleep(30)
            reps.append("R3")
        else:
            print(f"WARNING: {FIT} not found; skipping R3")

    if args.with_offsets and FIT_OFF.exists():
        reps.append("R4")
    if args.with_cloglog and FIT_CLL.exists():
        reps.append("R5")
    reps += [r for r in args.extra if EXTRA[r][1].exists()]

    mfile, lfile = OUT / "metrics.csv", OUT / "leiden_by_resolution.csv"
    if args.figures_only:
        metrics = pd.read_csv(mfile)
        reps = [r for r in reps if r in set(metrics["rep"])]
        umaps = {r: np.load(CACHE / f"{r}_umap.npy") for r in reps}
        make_figures(reps, umaps, cells, logumi, metrics)
        return

    rows, leids, umaps, infos = [], [], {}, {}
    for r in reps:
        E, infos[r] = get_embedding(r, X, args.recompute)
        uf = CACHE / f"{r}_umap.npy"
        if uf.exists() and not args.recompute:
            umaps[r] = np.load(uf)
        else:
            t0 = time.time()
            umaps[r] = run_umap(E)
            np.save(uf, umaps[r])
            print(f"[{r}] UMAP in {time.time() - t0:.0f}s", flush=True)
        row, ld = metrics_for(E, cells, logumi, r)
        rows.append(row), leids.append(ld)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(mfile, index=False)
    pd.concat(leids).to_csv(lfile, index=False)
    (OUT / "run_info.json").write_text(json.dumps({"n_cells": int(X.shape[0]), "n_genes_kept": int(kept.size),
                                                   "reps": infos}, indent=2))
    pd.DataFrame(np.column_stack([cells["cell_id"].values] + [umaps[r] for r in reps]),
                 columns=["cell_id"] + [f"{r}_umap{i}" for r in reps for i in (1, 2)]
                 ).to_csv(OUT / "umap_coords.csv.gz", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(metrics.set_index("rep").T.round(3))
    make_figures(reps, umaps, cells, logumi, metrics)


if __name__ == "__main__":
    main()
