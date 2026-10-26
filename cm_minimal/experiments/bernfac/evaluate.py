"""Step 4b/4c: information content and batch-vs-biology evaluations of the
full-data grid fits (requires run_grid.py outputs).

All comparisons are rotation-invariant: we use P = sigmoid(U V') or the cell
embedding E = A S from the SVD of logit(P) = U V' (common.cell_embedding).

(b) Information content -- does P, fit only to binary Y, rank expression level?
    Spearman / Pearson of P vs log1p(counts): overall, within Y_ge1 = 1 entries,
    and gene-wise (per gene across cells; median over genes). Reference
    predictor "depth x abundance": outer product of cell detection rate and
    gene detection rate (rank-1, no latent biology), and Y itself.

(c) Batch vs biology -- decode hemilineage, library, sex from each cell
    representation with stratified 5-fold CV:
      logreg : standardised features, L2 logistic regression (C = 1)
      knn    : 15-nearest-neighbour vote, Euclidean on the raw representation
    metric = balanced accuracy (chance 0.20 / 0.25 / 0.50). Also silhouette
    (Euclidean) by hemilineage and by library, and sequencing depth:
    Spearman(cell mean P, total_umi) and 5-fold CV R^2 of log(total_umi) from
    the representation. Comparators: raw Y (100-d), log1p counts, log1p CP10k
    (counts / total_umi * 1e4), and PCA-k of Y and of log1p counts.

Outputs: tables/info_content.csv, tables/gene_corr_<tag>.csv (best model),
         tables/decoding.csv
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (KS, KS_GE2, LAMBDAS, LAMBDAS_GE2, TABLES, cell_embedding,
                    load_data, load_fit)

TARGETS = ["hemilineage", "library", "Sex"]


# ----------------------------------------------------------------------------
# (b) information content
# ----------------------------------------------------------------------------
def info_content(P, L, Y1):
    pos = Y1 == 1
    row = dict(
        spearman_all=spearmanr(P.ravel(), L.ravel())[0],
        pearson_all=pearsonr(P.ravel(), L.ravel())[0],
        spearman_pos=spearmanr(P[pos], L[pos])[0],
        pearson_pos=pearsonr(P[pos], L[pos])[0],
    )
    g_all, g_pos = gene_corr(P, L, Y1)
    row["gene_spearman_all_median"] = float(np.nanmedian(g_all))
    row["gene_spearman_pos_median"] = float(np.nanmedian(g_pos))
    row["gene_spearman_pos_frac_gt0"] = float(np.nanmean(g_pos > 0))
    return row


def gene_corr(P, L, Y1):
    m = P.shape[1]
    g_all, g_pos = np.full(m, np.nan), np.full(m, np.nan)
    for j in range(m):
        if np.ptp(P[:, j]) > 0:
            g_all[j] = spearmanr(P[:, j], L[:, j])[0]
        s = Y1[:, j] == 1
        if s.sum() > 10 and np.ptp(P[s, j]) > 0 and np.ptp(L[s, j]) > 0:
            g_pos[j] = spearmanr(P[s, j], L[s, j])[0]
    return g_all, g_pos


# ----------------------------------------------------------------------------
# (c) decoding
# ----------------------------------------------------------------------------
def decode(X, cells, seed=0):
    out = {}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for t in TARGETS:
        y = cells[t].astype(str).to_numpy()
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
        out[f"{t}_logreg"] = balanced_accuracy_score(y, cross_val_predict(lr, X, y, cv=cv))
        knn = KNeighborsClassifier(15)
        out[f"{t}_knn"] = balanced_accuracy_score(y, cross_val_predict(knn, X, y, cv=cv))
    out["sil_hemilineage"] = silhouette_score(X, cells["hemilineage"])
    out["sil_library"] = silhouette_score(X, cells["library"])
    logd = np.log(cells["total_umi"].to_numpy(float))
    pred = cross_val_predict(LinearRegression(), X, logd, cv=KFold(5, shuffle=True, random_state=seed))
    out["depth_cv_r2"] = r2_score(logd, pred)
    return out


def fit_job(args):
    dataset, lam, k = args
    d = load_data()
    f = load_fit(dataset, lam, k)
    P = f["P"]
    L = np.log1p(d["counts"])
    row = dict(dataset=dataset, rep="bernfac", lam=lam, k=k)
    info = dict(row, **info_content(P, L, d["Y_ge1"]))
    E = cell_embedding(f["U"], f["V"])
    dec = dict(row, **decode(E, d["cells"]))
    dec["depth_spearman_meanP"] = spearmanr(P.mean(1), d["cells"]["total_umi"])[0]
    dec["depth_spearman_maxfactor"] = max(abs(spearmanr(E[:, c], d["cells"]["total_umi"])[0])
                                          for c in range(E.shape[1]))
    print(dataset, lam, k, "hemi", round(dec["hemilineage_logreg"], 3),
          "lib", round(dec["library_logreg"], 3), flush=True)
    return info, dec


def baseline_job(args):
    name, k = args
    d = load_data()
    cells, Y, C = d["cells"], d["Y_ge1"], d["counts"]
    L = np.log1p(C)
    cp10k = np.log1p(C / cells["total_umi"].to_numpy(float)[:, None] * 1e4)
    X = {"Y_ge1_raw": Y, "Y_ge2_raw": d["Y_ge2"], "log1p_counts": L, "log1p_cp10k": cp10k,
         "pca_Y_ge1": Y, "pca_log1p_counts": L, "pca_log1p_cp10k": cp10k}[name]
    if name.startswith("pca"):
        X = PCA(k, random_state=0).fit_transform(X - X.mean(0))
    dec = dict(dataset="Y_ge1", rep=name, lam=np.nan, k=k, **decode(X, cells))
    dec["depth_spearman_meanP"] = spearmanr(X.mean(1), cells["total_umi"])[0] if not name.startswith("pca") else np.nan
    return dec


def main():
    d = load_data()
    jobs = [("Y_ge1", l, k) for l in LAMBDAS for k in KS] + \
           [("Y_ge2", l, k) for l in LAMBDAS_GE2 for k in KS_GE2]
    bjobs = [(n, None) for n in ["Y_ge1_raw", "Y_ge2_raw", "log1p_counts", "log1p_cp10k"]]
    bjobs += [(n, k) for n in ["pca_Y_ge1", "pca_log1p_counts", "pca_log1p_cp10k"] for k in KS]
    with Pool(11) as pool:
        res = pool.map(fit_job, jobs, chunksize=1)
        bres = pool.map(baseline_job, bjobs, chunksize=1)
    info = pd.DataFrame([r[0] for r in res])
    # reference predictors for (b)
    Y1, L = d["Y_ge1"], np.log1p(d["counts"])
    refs = {"depth_x_abundance": np.outer(Y1.mean(1), Y1.mean(0)), "Y_ge1": Y1,
            "Y_ge2": d["Y_ge2"]}
    for name, P in refs.items():
        # (Y_ge1 is constant on its own positives, so its *_pos entries are NaN)
        with np.errstate(all="ignore"):
            r = info_content(P, L, Y1)
        info = pd.concat([info, pd.DataFrame([dict(dataset="reference", rep=name, lam=np.nan, k=np.nan, **r)])])
    info["rep"] = info["rep"].fillna("bernfac")
    # gene-wise correlations for the CV-best model vs the depth x abundance reference
    cv = pd.read_csv(TABLES / "cv_summary.csv")
    cv = cv[(cv.dataset == "Y_ge1") & (cv.model == "bernfac")]
    best = cv.loc[cv["nll_mean"].idxmin()]
    P = load_fit("Y_ge1", float(best["lam"]), int(best["k"]))["P"]
    ga, gp = gene_corr(P, L, Y1)
    ra, rp = gene_corr(refs["depth_x_abundance"], L, Y1)
    pd.DataFrame(dict(gene=d["genes"]["gene"], detect_rate=Y1.mean(0), bernfac_all=ga, bernfac_pos=gp,
                      depth_x_abundance_all=ra, depth_x_abundance_pos=rp,
                      best_lam=float(best["lam"]), best_k=int(best["k"]))
                 ).to_csv(TABLES / "gene_corr_best.csv", index=False)
    info.to_csv(TABLES / "info_content.csv", index=False)
    dec = pd.DataFrame([r[1] for r in res] + bres)
    dec.to_csv(TABLES / "decoding.csv", index=False)




# ----------------------------------------------------------------------------
# (b, follow-up) is the within-Y=1 ranking just sequencing depth?
# ----------------------------------------------------------------------------
def depth_partial():
    """Gene-wise Spearman of P vs log1p(count) within Y_ge1 = 1 cells, after
    rank-residualising both on log(total_umi) (partial Spearman). Run with
    `python evaluate.py partial` after main(); writes tables/info_depth_partial.csv."""
    from scipy.stats import rankdata
    d = load_data()
    Y1, L = d["Y_ge1"], np.log1p(d["counts"])
    logd = np.log(d["cells"]["total_umi"].to_numpy(float))

    def resid(x, z):
        x, z = rankdata(x), rankdata(z)
        b = np.polyfit(z, x, 1)
        return x - np.polyval(b, z)

    rows = []
    fits = [("Y_ge1", l, k) for l in LAMBDAS for k in KS]
    refs = {"depth_x_abundance": np.outer(Y1.mean(1), Y1.mean(0))}
    for name, P in [(f, load_fit(*f)["P"]) for f in fits] + list(refs.items()):
        raw, part = [], []
        for j in range(Y1.shape[1]):
            s = Y1[:, j] == 1
            if s.sum() > 10 and np.ptp(P[s, j]) > 0 and np.ptp(L[s, j]) > 0:
                raw.append(spearmanr(P[s, j], L[s, j])[0])
                part.append(np.corrcoef(resid(P[s, j], logd[s]), resid(L[s, j], logd[s]))[0, 1])
        tag = dict(rep="bernfac", lam=name[1], k=name[2]) if isinstance(name, tuple) else dict(rep=name, lam=np.nan, k=np.nan)
        rows.append(dict(tag, gene_spearman_pos_median=np.median(raw),
                         gene_partial_spearman_pos_median=np.median(part),
                         gene_partial_frac_gt0=float(np.mean(np.array(part) > 0))))
    pd.DataFrame(rows).to_csv(TABLES / "info_depth_partial.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    import sys
    depth_partial() if sys.argv[1:] == ["partial"] else main()
