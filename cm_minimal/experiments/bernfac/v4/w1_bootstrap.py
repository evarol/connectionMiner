"""W1 (v4): two-way (cell x gene) bootstrap of PAIRED per-entry test differences.

For every model a and each reference r in --refs, Delta = mean_test[L_a] - mean_test[L_r] over the
8,044,655 test entries, where L is the per-entry
    nll       Bernoulli NLL (exact, as scored in v3)
    nll_clip  NLL with p clipped to [1e-6, 1 - 1e-6] (tail sensitivity, Poisson)
    brier     (p - y)^2
Negative Delta = a better than the reference.

Bootstrap: Poisson(1) pigeonhole weights on cells and on genes (Owen 2007), entry weight
w_ij = w_i * w_j, ratio estimator sum(w L)/sum(w); the SAME weights are used for every model in a
replicate, so the differences are paired. Reported overall, per cell-depth decile (deciles over
cells of train UMI, as in v3) and per gene detection-rate decile (deciles over genes of train
detection rate, as in v3). AUC is reported as a point estimate only (not bootstrapped).

Per-entry inputs: v3/cache/test_nll_<code>.npy and test_p_<code>.npy (row-major order of the test
mask), plus the gene-rate baseline GR (train detection rate clipped to [1e-6, 1-1e-6], as in
rescore_val_test.py). Models refit later (W2) can be added without editing this file:

    --extra NAME=counts:/path/fit.npz       (countfac npz: U V a b o phi; NB if phi non-empty)
    --extra NAME=logit_off:/path/fit.npz    (U V a b)
    --extra NAME=logit:/path/fit.npz        (U V)
    --extra NAME=cloglog:/path/fit.npz      (U V a b)
    --extra NAME=neural:/path/best.pt
Per-entry NLL / p for an extra are computed once with v3/predictors and cached in v4/cache.

Usage (defaults reproduce the v4 W1 tables):
    python w1_bootstrap.py [--models R3,R4,R5,R6,PO,R7,R8,R9,GR] [--refs R6,R4] [--B 400]
                           [--extra NAME=kind:path ...] [--tag suffix]
Outputs: v4/tables/test_ci{tag}.csv (long format), v4/tables/test_ci_reps{tag}.npz (replicates).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v3"))
sys.path.insert(0, str(HERE.parent))
import predictors as pr  # noqa: E402

STUDY = pr.STUDY
C3 = STUDY / "v3/cache"
V4 = STUDY / "v4"
C4 = V4 / "cache"
TAB = V4 / "tables"
METRICS = ("nll", "nll_clip", "brier")
EPS = 1e-6


def test_index():
    T = np.load(pr.DATA / "test.npy", mmap_mode="r")
    I, J = [], []
    for s in range(0, T.shape[0], 2048):
        r, c = np.nonzero(np.asarray(T[s:s + 2048]))
        I.append((r + s).astype(np.int32)); J.append(c.astype(np.int32))
    return np.concatenate(I), np.concatenate(J), T.shape


def extra_predictor(kind, path):
    z = None if kind == "neural" else np.load(path)
    if kind == "counts":
        return pr.Counts(z["U"], z["V"], z["a"], z["b"], z["o"], z["phi"])
    if kind == "logit_off":
        return pr.Logit(z["U"], z["V"], z["a"], z["b"])
    if kind == "logit":
        return pr.Logit(z["U"], z["V"])
    if kind == "cloglog":
        return pr.Cloglog(z["U"], z["V"], z["a"], z["b"])
    if kind == "neural":
        return pr.Neural(path)
    raise KeyError(kind)


def per_entry(code, I, J, y, extras):
    """(nll, p) float32 vectors over the test entries for one model code."""
    if code == "GR":
        g = np.clip(np.load(C3 / "gene_train_rate.npy"), EPS, 1 - EPS)
        p = g[J]
        nll = -np.where(y > 0, np.log(p), np.log1p(-p))
        return nll, p
    if code in extras:
        f_nll, f_p = C4 / f"test_nll_{code}.npy", C4 / f"test_p_{code}.npy"
        if not f_nll.exists():
            kind, path = extras[code]
            pred = extra_predictor(kind, path)
            nll = np.empty(len(I), np.float32); p = np.empty(len(I), np.float32)
            for s in range(0, len(I), 1_000_000):
                lP, lQ = pred.logpairs(I[s:s + 1_000_000], J[s:s + 1_000_000])
                yy = y[s:s + 1_000_000]
                nll[s:s + 1_000_000] = -np.where(yy > 0, lP, lQ)
                p[s:s + 1_000_000] = np.exp(lP)
            np.save(f_nll, nll); np.save(f_p, p)
        return np.load(f_nll).astype(np.float64), np.load(f_p).astype(np.float64)
    return (np.load(C3 / f"test_nll_{code}.npy").astype(np.float64),
            np.load(C3 / f"test_p_{code}.npy").astype(np.float64))


def auc(p, y):
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    # average ties
    ps = p[order]
    _, first, cnt = np.unique(ps, return_index=True, return_counts=True)
    avg = first + (cnt + 1) / 2.0
    ranks[order] = np.repeat(avg, cnt)
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def decile(v):
    q = np.quantile(v, np.linspace(0, 1, 11))
    return np.clip(np.searchsorted(q, v, side="right") - 1, 0, 9), q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="R3,R4,R5,R6,PO,R7,R8,R9,GR")
    ap.add_argument("--refs", default="R6,R4")
    ap.add_argument("--B", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    extras = {}
    for e in a.extra:
        name, spec = e.split("=", 1)
        kind, path = spec.split(":", 1)
        extras[name] = (kind, path)
    codes = a.models.split(",") + [c for c in extras if c not in a.models.split(",")]
    refs = a.refs.split(",")
    C4.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    I, J, (n, m) = test_index()
    y = np.load(C3 / "test_y.npy").astype(np.float64)
    N = len(I)
    print("test entries", N, flush=True)
    cd, cq = decile(np.load(C3 / "cell_depth.npy"))
    gd, gq = decile(np.load(C3 / "gene_train_rate.npy"))
    grp = (cd[I] * 10 + gd[J]).astype(np.int64)          # 100 joint groups -> marginalise later

    # columns: (code, metric)
    cols, M, aucs = [], [], {}
    for c in codes:
        nll, p = per_entry(c, I, J, y, extras)
        pc = np.clip(p, EPS, 1 - EPS)
        clip = -np.where(y > 0, np.log(pc), np.log1p(-pc))
        bri = (p - y) ** 2
        for met, v in zip(METRICS, (nll, clip, bri)):
            cols.append((c, met)); M.append(v)
        aucs[c] = auc(p, y)
        print(c, "nll %.6f clip %.6f brier %.6f auc %.6f" % (nll.mean(), clip.mean(), bri.mean(), aucs[c]),
              flush=True)
        del nll, p, pc, clip, bri
    M = np.stack(M, 1)                                   # N x ncol float64
    ncol = M.shape[1]

    def group_sums(w):
        S = np.empty((100, ncol))
        for k in range(ncol):
            S[:, k] = np.bincount(grp, w * M[:, k], minlength=100)
        W = np.bincount(grp, w, minlength=100)
        return S.reshape(10, 10, ncol), W.reshape(10, 10)

    def means(S, W):
        """dict stratum -> per-column weighted mean: overall, depth d1..d10, generate g1..g10."""
        out = {("all", 0): S.sum((0, 1)) / W.sum()}
        for k in range(10):
            out[("depth", k + 1)] = S[k].sum(0) / W[k].sum()
            out[("generate", k + 1)] = S[:, k].sum(0) / W[:, k].sum()
        return out

    S0, W0 = group_sums(np.ones(N))
    point = means(S0, W0)
    strata = list(point)
    rng = np.random.default_rng(a.seed)
    reps = np.empty((a.B, len(strata), ncol))
    for b in range(a.B):
        wc = rng.poisson(1.0, n).astype(np.float64)
        wg = rng.poisson(1.0, m).astype(np.float64)
        Sb, Wb = group_sums(wc[I] * wg[J])
        mb = means(Sb, Wb)
        reps[b] = np.stack([mb[s] for s in strata])
        if b % 25 == 0:
            print(f"rep {b}  {time.time() - t0:.0f}s", flush=True)

    ci = {cm: i for i, cm in enumerate(cols)}
    rows = []
    for si, (kind, lev) in enumerate(strata):
        nent = int(W0.sum() if kind == "all" else (W0[lev - 1].sum() if kind == "depth" else W0[:, lev - 1].sum()))
        for ref in refs:
            for c in codes:
                for met in METRICS:
                    ka, kr = ci[(c, met)], ci[(ref, met)]
                    est = point[(kind, lev)][ka] - point[(kind, lev)][kr]
                    rd = reps[:, si, ka] - reps[:, si, kr]
                    lo, hi = np.quantile(rd, [0.025, 0.975])
                    rows.append(dict(stratum=kind, level=lev, n_entries=nent, metric=met, model=c, ref=ref,
                                     value_model=point[(kind, lev)][ka], value_ref=point[(kind, lev)][kr],
                                     delta=est, se=rd.std(ddof=1), ci_lo=lo, ci_hi=hi,
                                     distinguishable=bool(c != ref and (lo > 0 or hi < 0)),
                                     auc_model=aucs[c] if kind == "all" else np.nan,
                                     auc_ref=aucs[ref] if kind == "all" else np.nan))
    d = pd.DataFrame(rows)
    d.to_csv(TAB / f"test_ci{a.tag}.csv", index=False)
    np.savez_compressed(TAB / f"test_ci_reps{a.tag}.npz", reps=reps.astype(np.float32),
                        strata=np.array([f"{k}:{l}" for k, l in strata]),
                        cols=np.array([f"{c}:{mt}" for c, mt in cols]), depth_edges=cq, generate_edges=gq,
                        B=a.B, seed=a.seed)
    print(d[(d.stratum == "all") & (d.metric == "nll")].to_string(), flush=True)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
