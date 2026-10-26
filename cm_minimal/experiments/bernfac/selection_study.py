"""BernFac model selection on synthetic data with known P_true.

For every dataset (synthetic_scenarios.py) and every grid point
lambda in LAMBDAS x k in KS (lambda_u = lambda_v = lambda) we fit BernFac
(init 'svd' + 2 'random' restarts, best final MAP objective, max_iter 400) and
compute

Oracle metrics of the full-data fit P_hat (need the truth):
    kl      mean_ij KL(Bern(p_ij) || Bern(p_hat_ij))  = cross-entropy - H(p)
            (= expected held-out NLL with infinite replicates, minus a constant)
    rmse    sqrt(mean (p_hat - p)^2)
    lcor    Pearson correlation of logit(p_hat) and logit(p) over entries
    (S3: the same three against P_bio, plus library / cluster decoding
     accuracy from the rows of logit(P_hat))

Data-driven criteria (mean NLL per evaluated entry, lower is better):
    C1  entry-wise CV: 5 folds, each holds out a disjoint random 10% of entries
    C2  count splitting (count scenarios): fit on Y_A = 1[X_A>=1], score Y_B.
        Both halves have p_half = 1-exp(-mu/2), so the score is internally
        consistent; it selects for the *half-depth* problem.  We also record
        the oracle KL of the mapped prediction 1-(1-p_hat_A)^2 (exact under
        Poisson) against P_true.
    C3  replicate split: score the full-data fit on an independent replicate
        Y_rep ~ Bern(P_true) (unbiased for cross-entropy; simulation-only)
    C4  cell split + fold-in: 5 row folds; fit on 80% of cells; for held-out
        cells fit u_i with V frozen on a random half of their genes and score
        the other half (both directions)
    C5  BIC / AIC with k(n+m) parameters (reference only)

Note on speed: bernfac.update_rows builds the (n,k,k) Gram tensor with a
3-operand einsum, which is ~13x slower than the equivalent GEMM.  We swap in
an algebraically identical implementation at import (checked to 1e-12 below);
bernfac.py itself is untouched.

Run:  python selection_study.py [--jobs 12]
Writes outputs/bernfac_20261020/synthetic/results/{grid_metrics.csv, fits_<dataset>.npz}
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

import bernfac as bf
from synthetic_scenarios import OUT as DATA_DIR, SCENARIOS, SEEDS

RESULTS = DATA_DIR.parent / "results"
LAMBDAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
KS = [1, 2, 3, 5, 8, 12, 20]
MAX_ITER = 400
EPS = 1e-6


# ----------------------------------------------------------------------------
# Faster, identical row update (see module docstring)
# ----------------------------------------------------------------------------
_einsum_update_rows = bf.update_rows


def update_rows_gemm(K, W, F, lam):
    m, k = F.shape
    A = (W @ (F[:, :, None] * F[:, None, :]).reshape(m, k * k)).reshape(-1, k, k) + lam * np.eye(k)
    return np.linalg.solve(A, (K @ F)[..., None])[..., 0]


def _check_update_rows():
    rng = np.random.default_rng(0)
    K, W, F = rng.random((30, 7)) - 0.5, rng.random((30, 7)), rng.standard_normal((7, 4))
    assert np.allclose(_einsum_update_rows(K, W, F, 0.3), update_rows_gemm(K, W, F, 0.3), atol=1e-12)


_check_update_rows()
bf.update_rows = update_rows_gemm


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def fit_best(Y, k, lam, mask=None, seed=0):
    """init 'svd' (1 start) + 'random' (2 starts); keep the largest final J."""
    r_svd = bf.fit(Y, k, lam, lam, mask=mask, max_iter=MAX_ITER, init="svd", seed=seed)
    r_rnd = bf.fit(Y, k, lam, lam, mask=mask, max_iter=MAX_ITER, init="random",
                   seed=seed + 1, n_restarts=2)
    best = r_svd if r_svd.obj[-1] >= r_rnd.obj[-1] else r_rnd
    best.restart_objs = [r_svd.obj[-1]] + list(r_rnd.restart_objs)
    best.init = "svd" if best is r_svd else "random"
    return best


def nll_logits(Y, psi, mask):
    """Mean Bernoulli NLL over mask, from logits."""
    return float(np.mean((bf.softplus(psi) - Y * psi)[mask]))


def oracle(P_hat, P):
    ph = np.clip(P_hat, EPS, 1 - EPS)
    p = np.clip(P, EPS, 1 - EPS)
    kl = np.mean(p * np.log(p / ph) + (1 - p) * np.log((1 - p) / (1 - ph)))
    rmse = np.sqrt(np.mean((P_hat - P) ** 2))
    lcor = np.corrcoef(np.log(ph / (1 - ph)).ravel(), np.log(p / (1 - p)).ravel())[0, 1]
    return float(kl), float(rmse), float(lcor)


def fold_in_rows(Y, mask, V, lam, max_iter=300, tol=1e-8):
    """MAP u_i for new rows with V frozen: PG-EM on the convex ridge-logistic
    problem, reusing bernfac.pg_mean / bernfac.update_rows."""
    K = np.where(mask, Y - 0.5, 0.0)
    U = np.zeros((Y.shape[0], V.shape[1]))
    for _ in range(max_iter):
        W = np.where(mask, bf.pg_mean(U @ V.T), 0.0)
        U_new = bf.update_rows(K, W, V, lam)
        done = np.max(np.abs(U_new - U)) < tol
        U = U_new
        if done:
            break
    return U


def decode_accuracy(X, labels, seed=0):
    """5-fold CV accuracy of multinomial logistic regression on rows of X."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clf, X, labels, cv=cv)))


def logit(P):
    P = np.clip(P, EPS, 1 - EPS)
    return np.log(P / (1 - P))


# ----------------------------------------------------------------------------
# One grid point
# ----------------------------------------------------------------------------
def run_point(name, lam, k):
    t0 = time.time()
    d = dict(np.load(DATA_DIR / f"{name}.npz"))
    Y, P = d["Y"].astype(float), d["P_true"]
    n, m = Y.shape
    full = np.ones_like(Y, dtype=bool)
    row = dict(dataset=name, scenario=name.rsplit("_seed", 1)[0], seed=int(name.rsplit("seed", 1)[1]),
               lam=lam, k=k)

    # full-data fit + oracle
    r = fit_best(Y, k, lam)
    psi = r.U @ r.V.T
    P_hat = bf.sigmoid(psi)
    row["kl"], row["rmse"], row["lcor"] = oracle(P_hat, P)
    row["train_nll"] = nll_logits(Y, psi, full)
    row["iters"], row["converged"], row["init"] = r.iterations, r.converged, r.init
    row["restart_spread"] = float(np.max(r.restart_objs) - np.min(r.restart_objs))
    ll = r.loglik[-1]
    row["C5_bic"] = float(-2 * ll + np.log(n * m) * k * (n + m)) / (n * m)
    row["C5_aic"] = float(-2 * ll + 2 * k * (n + m)) / (n * m)

    # C3 replicate
    row["C3_rep"] = nll_logits(d["Y_rep"].astype(float), psi, full)

    # C1 entry-wise CV (5 disjoint 10% folds)
    c1 = []
    for f in range(5):
        test = d["entry_fold"] == f
        rf = fit_best(Y, k, lam, mask=~test, seed=10 + f)
        c1.append(nll_logits(Y, rf.U @ rf.V.T, test))
    row["C1_entry"] = float(np.mean(c1))

    # C2 count splitting
    if "Y_A" in d:
        YA, YB = d["Y_A"].astype(float), d["Y_B"].astype(float)
        ra = fit_best(YA, k, lam, seed=20)
        psi_a = ra.U @ ra.V.T
        row["C2_split"] = nll_logits(YB, psi_a, full)
        pa = bf.sigmoid(psi_a)
        row["C2_kl_half"] = oracle(pa, d["P_half"])[0]          # its own (half-depth) oracle
        row["C2_kl_mapped"] = oracle(1 - (1 - pa) ** 2, P)[0]    # mapped to full depth
        row["C2_kl_unmapped"] = oracle(pa, P)[0]                 # half model used as-is

    # C4 cell split + fold-in
    c4 = []
    for f in range(5):
        te = d["row_fold"] == f
        rf = fit_best(Y[~te], k, lam, seed=30 + f)
        Yt, half = Y[te], d["gene_half"][te]
        for fit_mask in (half, ~half):
            Ut = fold_in_rows(Yt, fit_mask, rf.V, lam)
            c4.append(nll_logits(Yt, Ut @ rf.V.T, ~fit_mask))
    row["C4_cellfold"] = float(np.mean(c4))

    # decoding (all scenarios; batch targets for S3)
    L = logit(P_hat)
    row["dec_library"] = decode_accuracy(L, d["library"])
    row["dec_cluster"] = decode_accuracy(L, d["cluster"])
    if "P_bio" in d:
        row["kl_bio"], row["rmse_bio"], row["lcor_bio"] = oracle(P_hat, d["P_bio"])
    row["seconds"] = time.time() - t0
    trace = np.full(MAX_ITER + 1, np.nan)
    trace[: len(r.obj)] = r.obj
    return row, P_hat.astype(np.float32), trace


def reference_rows(name):
    """Decoding accuracy from the truth / raw data (reference lines)."""
    d = dict(np.load(DATA_DIR / f"{name}.npz"))
    out = dict(dataset=name)
    for key in ("P_true", "P_bio"):
        if key in d:
            out[f"{key}_dec_library"] = decode_accuracy(logit(d[key]), d["library"])
            out[f"{key}_dec_cluster"] = decode_accuracy(logit(d[key]), d["cluster"])
    out["Y_dec_library"] = decode_accuracy(d["Y"].astype(float), d["library"])
    out["Y_dec_cluster"] = decode_accuracy(d["Y"].astype(float), d["cluster"])
    return out


def run_point_cached(name, lam, k):
    """run_point with a per-task checkpoint (results/points/*.pkl) so the
    grid can be resumed after an interruption; logs one line per task."""
    import pickle
    path = RESULTS / "points" / f"{name}_lam{lam:g}_k{k}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    out = run_point(name, lam, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(out))
    tmp.rename(path)
    with open(RESULTS / "progress.log", "a") as fh:
        fh.write(f"{name} lam={lam:g} k={k} {out[0]['seconds']:.0f}s\n")
    return out


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    names = args.datasets or [f"{s}_seed{sd}" for s in SCENARIOS for sd in SEEDS]
    tasks = [(nm, lam, k) for k in sorted(KS, reverse=True) for nm in names for lam in LAMBDAS]
    # call through the *imported* module so loky workers import it by name and
    # therefore also apply the update_rows swap above (functions defined in
    # __main__ are shipped by value and would run the slow einsum version)
    import selection_study as ss
    t0 = time.time()
    out = Parallel(n_jobs=args.jobs, verbose=0)(delayed(ss.run_point_cached)(*t) for t in tasks)
    print(f"grid done in {time.time() - t0:.0f}s", flush=True)
    rows = pd.DataFrame([o[0] for o in out]).sort_values(["dataset", "lam", "k"])
    rows.to_csv(RESULTS / "grid_metrics.csv", index=False)
    for nm in names:
        sel = [(t, o) for t, o in zip(tasks, out) if t[0] == nm]
        sel.sort(key=lambda x: (x[0][1], x[0][2]))
        np.savez_compressed(RESULTS / f"fits_{nm}.npz",
                            lam=np.array([t[1] for t, _ in sel]), k=np.array([t[2] for t, _ in sel]),
                            P_hat=np.stack([o[1] for _, o in sel]) if nm.endswith("seed0")
                            else np.zeros(0), trace=np.stack([o[2] for _, o in sel]))
    ref = Parallel(n_jobs=args.jobs)(delayed(ss.reference_rows)(nm) for nm in names)
    pd.DataFrame(ref).to_csv(RESULTS / "reference_decoding.csv", index=False)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
