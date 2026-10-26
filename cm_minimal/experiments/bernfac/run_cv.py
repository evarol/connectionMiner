"""Step 4a: held-out entry cross-validation for BernFac on the real subset.

Protocol: a random permutation of all n*m entries (seed 2026) is cut into ten
10% blocks; folds 0..4 each hold out one block (so the five held-out sets are
disjoint and together cover 50% of entries). The fit sees the other 90% via
the solver's mask argument (masked entries never enter the likelihood); the
same svd + 3 random restart rule as the grid fits picks the best J.

Held-out metrics per entry: Bernoulli NLL (nats), ROC AUC, Brier score, and a
10-bin reliability (calibration) table.

Baselines on the same folds:
  gene_rate    P_ij = training detection rate of gene j
  row_col      additive logistic model logit P_ij = a_i + b_j (no interaction),
               fit by L2-penalised logistic regression (C = 10) on one-hot
               cell + gene indicators. This is the "no latent structure beyond
               cell depth and gene abundance" reference.

Outputs: tables/cv_folds.csv (one row per fit x fold), tables/cv_summary.csv
(mean +- sd over folds), tables/cv_calibration.csv.
"""
from __future__ import annotations

import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import KS, KS_GE2, LAMBDAS, LAMBDAS_GE2, TABLES, load_data
from run_grid import fit_best

N_FOLDS = 5
HOLDOUT_FRAC = 0.10
CV_SEED = 2026
N_BINS = 10
EPS = 1e-12


def fold_mask(shape, fold):
    """Boolean mask, True = observed (training) entry, for the given fold."""
    n_tot = shape[0] * shape[1]
    perm = np.random.default_rng(CV_SEED).permutation(n_tot)
    block = int(round(HOLDOUT_FRAC * n_tot))
    held = perm[fold * block:(fold + 1) * block]
    mask = np.ones(n_tot, dtype=bool)
    mask[held] = False
    return mask.reshape(shape)


def heldout_metrics(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return dict(nll=float(nll), auc=float(roc_auc_score(y, p)),
                brier=float(np.mean((p - y) ** 2)))


def calibration_rows(y, p, **tags):
    edges = np.linspace(0, 1, N_BINS + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, N_BINS - 1)
    rows = []
    for b in range(N_BINS):
        sel = idx == b
        if sel.any():
            rows.append(dict(tags, bin=b, mean_pred=float(p[sel].mean()),
                             frac_pos=float(y[sel].mean()), count=int(sel.sum())))
    return rows


def baseline_predictions(Y, mask):
    n, m = Y.shape
    train_rate = (Y * mask).sum(0) / mask.sum(0)
    out = {"gene_rate": np.broadcast_to(train_rate, Y.shape)}
    ii, jj = np.nonzero(mask)
    def design(i, j):
        r = np.arange(len(i))
        return sp.hstack([sp.csr_matrix((np.ones(len(i)), (r, i)), shape=(len(i), n)),
                          sp.csr_matrix((np.ones(len(i)), (r, j)), shape=(len(i), m))]).tocsr()
    lr = LogisticRegression(C=10.0, max_iter=2000).fit(design(ii, jj), Y[ii, jj])
    ai, aj = np.indices(Y.shape)
    out["row_col"] = lr.predict_proba(design(ai.ravel(), aj.ravel()))[:, 1].reshape(Y.shape)
    return out


def job(args):
    dataset, lam, k, fold = args
    Y = load_data()[dataset]
    mask = fold_mask(Y.shape, fold)
    t0 = time.time()
    if lam is None:      # baselines
        preds = baseline_predictions(Y, mask)
    else:
        (_, _, r), _ = fit_best(Y, k, lam, mask=mask)
        preds = {"bernfac": r.P}
    held = ~mask
    rows, cal = [], []
    for model, P in preds.items():
        y, p = Y[held], P[held]
        tags = dict(dataset=dataset, model=model, lam=lam, k=k, fold=fold)
        rows.append(dict(tags, **heldout_metrics(y, p),
                         train_nll=heldout_metrics(Y[mask], P[mask])["nll"]))
        cal += calibration_rows(y, p, **tags)
    print(f"{dataset} lam={lam} k={k} fold={fold} nll={rows[0]['nll']:.4f} "
          f"auc={rows[0]['auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return rows, cal


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    jobs = []
    for f in range(N_FOLDS):
        jobs += [("Y_ge1", l, k, f) for l in LAMBDAS for k in KS]
        jobs += [("Y_ge2", l, k, f) for l in LAMBDAS_GE2 for k in KS_GE2]
        jobs += [("Y_ge1", None, None, f), ("Y_ge2", None, None, f)]
    jobs.sort(key=lambda j: 0 if j[1] is None else -j[2] / j[1] ** 0.5)
    with Pool(int(os.environ.get("NPROC", 11))) as pool:
        res = pool.map(job, jobs, chunksize=1)
    folds = pd.DataFrame([r for rs, _ in res for r in rs])
    cal = pd.DataFrame([c for _, cs in res for c in cs])
    folds.to_csv(TABLES / "cv_folds.csv", index=False)
    cal.to_csv(TABLES / "cv_calibration.csv", index=False)
    g = folds.fillna({"lam": -1, "k": -1}).groupby(["dataset", "model", "lam", "k"])
    summ = g[["nll", "auc", "brier", "train_nll"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(TABLES / "cv_summary.csv", index=False)


if __name__ == "__main__":
    main()
