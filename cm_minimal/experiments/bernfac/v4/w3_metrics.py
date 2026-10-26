"""W3 step 2: one evaluation protocol applied identically to every representation.

Input: principal coordinates P (centred SVD, columns sorted by singular value) from w3_build.py.
Variants: d in {20, 40, native} (if a representation has fewer than d columns, all are used and the
row is flagged) x scaling in {unwhitened (P = U S), whitened (U, unit variance per PC)}.

Per variant (long-format rows in v4/cache/w3_metrics_<code>.csv)
  hl_probe_balacc     multinomial logistic regression on standardised coordinates, 5-fold CV
                      (standardisation makes it identical for whitened and unwhitened; stored as 'invariant')
  hl_knn_balacc       15-NN, 5-fold stratified CV (seed 0); also k = 5 and 30 for the d = 20 variants
  hl_leiden_ari/ami   Leiden on the 15-NN UMAP-weighted graph (scanpy defaults, igraph flavour), resolution
                      bisected per seed until exactly 23 clusters (closest count kept if 23 is skipped);
                      5 seeds, mean and sd
  hl_silhouette       Euclidean silhouette on a fixed 5,000-cell subsample
  lib_knn_within_hl   library 15-NN balanced accuracy computed inside each hemilineage (libraries with
                      >= 10 cells in that hemilineage), cell-weighted mean over hemilineages; chance stored
  sex_knn_within_lib  sex 15-NN balanced accuracy inside each mixed-sex library (>= 50 cells of each sex)
  hl_knn_depthresid_bt hemilineage kNN after regressing log UMI out of every coordinate WITHIN birth-time class
Depth (v4/cache/w3_depth_<code>.csv; on the centred-SVD axes, so scaling does not matter)
  |rho_s| of log10 UMI with PC1, max over PC1-2, max over PC1-5; R^2 of log UMI from top-10 PCs (OLS);
  the same within hemilineage x birth-time strata (>= 30 cells): pooled within-stratum R^2 (log UMI and PCs
  demeaned per stratum) and cell-weighted mean over strata of max_{PC1-5} |rho_s|, each with a
  within-stratum permutation null.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score, balanced_accuracy_score,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from w3_common import CACHE, ORDER, load_cells, load_counts

SEEDS = range(5)
TARGET = 23


def knn_cv(E, y, k=15, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(KNeighborsClassifier(k), E, y, cv=cv)
    return balanced_accuracy_score(y, p)


def probe_cv(E, y):
    Z = StandardScaler().fit_transform(E)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), Z, y, cv=cv)
    return balanced_accuracy_score(y, p)


def leiden_fixed_k(E, labels, target=TARGET, seeds=SEEDS):
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(np.zeros((E.shape[0], 1), np.float32))
    a.obsm["E"] = E.astype(np.float32)
    sc.pp.neighbors(a, n_neighbors=15, use_rep="E", random_state=0)
    out = []
    for seed in seeds:
        def run(res):
            sc.tl.leiden(a, resolution=res, key_added="cl", random_state=seed, flavor="igraph",
                         n_iterations=2, directed=False)
            cl = a.obs["cl"].values
            return len(set(cl)), np.asarray(cl)
        lo, hi = np.log(0.005), np.log(5.0)
        best = None
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            n, cl = run(np.exp(mid))
            if best is None or abs(n - target) < abs(best[0] - target):
                best = (n, cl, np.exp(mid))
            if n == target:
                break
            if n < target:
                lo = mid
            else:
                hi = mid
        n, cl, res = best
        out.append(dict(seed=seed, n_clusters=n, resolution=res, ari=adjusted_rand_score(labels, cl),
                        ami=adjusted_mutual_info_score(labels, cl)))
    return pd.DataFrame(out)


def lib_within_hl(E, hl, lib):
    accs, w, ch = [], [], []
    for h in np.unique(hl):
        m = hl == h
        vc = pd.Series(lib[m]).value_counts()
        keep = vc.index[vc >= 10]
        mm = m & np.isin(lib, keep)
        if len(keep) < 2 or mm.sum() < 100:
            continue
        accs.append(knn_cv(E[mm], lib[mm]))
        w.append(mm.sum()), ch.append(1 / len(keep))
    return np.average(accs, weights=w), np.average(ch, weights=w), len(accs)


def sex_within_lib(E, lib, sex):
    accs = []
    for L in np.unique(lib):
        m = lib == L
        vc = pd.Series(sex[m]).value_counts()
        if len(vc) < 2 or vc.min() < 50:
            continue
        accs.append(knn_cv(E[m], sex[m]))
    return float(np.mean(accs)), len(accs)


def resid_within(E, x, groups):
    out = np.empty_like(E)
    for g in np.unique(groups):
        m = groups == g
        D = np.column_stack([np.ones(m.sum()), x[m]])
        out[m] = E[m] - D @ np.linalg.lstsq(D, E[m], rcond=None)[0]
    return out


def demean(A, groups):
    A = np.array(A, dtype=float, copy=True)
    for g in np.unique(groups):
        m = groups == g
        A[m] -= A[m].mean(0)
    return A


def r2_ols(y, X):
    D = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(D, y, rcond=None)[0]
    return 1 - ((y - D @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def depth_metrics(P, cells, rng):
    x = cells["log10_umi"].values
    st = cells["stratum"].values
    rho = [abs(spearmanr(P[:, j], x).statistic) for j in range(min(5, P.shape[1]))]
    out = dict(depth_rho_pc1=rho[0], depth_rho_max_pc12=max(rho[:2]), depth_rho_max_pc15=max(rho),
               depth_r2_top10=r2_ols(x, P[:, :10]))
    vc = pd.Series(st).value_counts()
    ok = np.isin(st, vc.index[vc >= 30])
    Pk, xk, sk = P[ok, :10], x[ok], st[ok]

    def within(xv):
        xd = demean(xv[:, None], sk)[:, 0]
        Pd = demean(Pk, sk)
        b = np.linalg.lstsq(Pd, xd, rcond=None)[0]
        r2 = 1 - ((xd - Pd @ b) ** 2).sum() / (xd ** 2).sum()
        vals, ws = [], []
        for g in np.unique(sk):
            m = sk == g
            vals.append(max(abs(spearmanr(Pk[m, j], xv[m]).statistic) for j in range(min(5, Pk.shape[1]))))
            ws.append(m.sum())
        return r2, float(np.average(vals, weights=ws))

    out["depth_r2_top10_within_stratum"], out["depth_rho_max_pc15_within_stratum"] = within(xk)
    xp = xk.copy()
    for g in np.unique(sk):
        m = np.flatnonzero(sk == g)
        xp[m] = xk[rng.permutation(m)]
    out["null_r2_top10_within_stratum"], out["null_rho_max_pc15_within_stratum"] = within(xp)
    out["n_strata"] = int(len(np.unique(sk)))
    out["frac_cells_in_strata"] = float(ok.mean())
    return out


def evaluate(code, leiden=True):
    cells = load_cells()
    hl = cells["hemilineage"].astype(str).values
    lib = cells["library"].astype(str).values
    sex = cells["Sex"].astype(str).values
    bt = cells["birthtime"].astype(str).values
    x = cells["log10_umi"].values
    z = np.load(CACHE / f"w3_{code}_pc.npz", allow_pickle=True)
    P = z["P"].astype(np.float64)
    native = P.shape[1]
    sub = np.random.default_rng(0).choice(len(P), 5000, replace=False)
    rows = []

    def add(dreq, dused, scaling, metric, value, sd=np.nan, n=1, flag=""):
        rows.append(dict(rep=code, d_requested=dreq, d_used=dused, scaling=scaling, metric=metric,
                         value=float(value), sd=float(sd), n=n, flag=flag))

    done = {}
    for dreq in (20, 40, "native"):
        du = native if dreq == "native" else min(dreq, native)
        flag = f"only {native} dims available" if (dreq != "native" and dreq > native) else ""
        key = du
        if key in done:   # identical to an earlier variant: copy rows with the new label
            for r in [r for r in rows if r["d_requested"] == done[key]]:
                rows.append({**r, "d_requested": dreq, "flag": flag or f"same as d={done[key]}"})
            continue
        done[key] = dreq
        Pd = P[:, :du]
        t0 = time.time()
        add(dreq, du, "invariant", "hl_probe_balacc", probe_cv(Pd, hl), flag=flag)
        for scaling, E in (("unwhitened", Pd), ("whitened", Pd / Pd.std(0))):
            add(dreq, du, scaling, "hl_knn_balacc", knn_cv(E, hl), flag=flag)
            add(dreq, du, scaling, "hl_silhouette", silhouette_score(E[sub], hl[sub]), flag=flag)
            a, ch, nh = lib_within_hl(E, hl, lib)
            add(dreq, du, scaling, "lib_knn_within_hl_balacc", a, n=nh, flag=flag)
            add(dreq, du, scaling, "lib_knn_within_hl_chance", ch, n=nh, flag=flag)
            s_, nl = sex_within_lib(E, lib, sex)
            add(dreq, du, scaling, "sex_knn_within_mixed_lib_balacc", s_, n=nl, flag=flag)
            if dreq == 20:
                for k in (5, 30):
                    add(dreq, du, scaling, f"hl_knn{k}_balacc", knn_cv(E, hl, k=k), flag=flag)
                add(dreq, du, scaling, "hl_knn_balacc_depthresid_within_bt", knn_cv(resid_within(E, x, bt), hl), flag=flag)
                add(dreq, du, scaling, "sex_knn_balacc_all", knn_cv(E, sex), flag=flag)
                add(dreq, du, scaling, "lib_knn_balacc_all", knn_cv(E, lib), flag=flag)
                add(dreq, du, scaling, "birthtime_knn_balacc", knn_cv(E, bt), flag=flag)
            if leiden:
                ld = leiden_fixed_k(E, hl)
                fl = flag + ("; 23 not hit for some seeds" if (ld.n_clusters != TARGET).any() else "")
                add(dreq, du, scaling, "hl_leiden23_ari", ld.ari.mean(), ld.ari.std(ddof=1), len(ld), fl)
                add(dreq, du, scaling, "hl_leiden23_ami", ld.ami.mean(), ld.ami.std(ddof=1), len(ld), fl)
                add(dreq, du, scaling, "hl_leiden23_nclusters", ld.n_clusters.mean(), ld.n_clusters.std(ddof=1), len(ld), fl)
                add(dreq, du, scaling, "hl_leiden23_resolution", ld.resolution.mean(), ld.resolution.std(ddof=1), len(ld), fl)
        print(code, dreq, f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(CACHE / f"w3_metrics_{code}.csv", index=False)
    dm = depth_metrics(P, cells, np.random.default_rng(0))
    pd.DataFrame([{"rep": code, "native_dim": native, **dm}]).to_csv(CACHE / f"w3_depth_{code}.csv", index=False)
    print(code, json.dumps({k: round(v, 3) for k, v in dm.items()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="*", default=ORDER)
    ap.add_argument("--no-leiden", action="store_true")
    a = ap.parse_args()
    for c in a.codes:
        evaluate(c, leiden=not a.no_leiden)
