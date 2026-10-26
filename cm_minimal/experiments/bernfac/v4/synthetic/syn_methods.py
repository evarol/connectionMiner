"""v4 synthetic benchmark: method wrappers (copied from v3/syn_methods.py and revised, review W4).

Each `run_<code>(D)` fits a grid on the TRAIN entries (split == 0) and returns ALL candidates:

    dict(cands=[dict(config, P (in memory only), emb, val_nll, seconds, iterations, converged, extra)],
         fit_seconds)

The driver (synthetic_benchmark.py) scores every candidate on validation AND test (so the oracle
config by test KL can be shown next to the validation-selected one) and never writes P to disk.

Changes from v3
  * Factor/count grids contain the true rank 8 (G1, G2): k in {4, 8, 16} x lambda in {3, 10, 30, 100}
    for R3, R4, R5, R6 and Poisson; iteration caps raised to 500 (BernFac) and 400 (counts).
  * R7: K in {1, 2, 4} x d in {4, 8, 16}, max_iter 300 (v3: K in {1, 4} x d in {5, 10}, 150).
  * R8: one config per call (run_R8_one), FIXED outer-iteration budget (max_outer = R8_MAX_OUTER, tol
    1e-6) identical for every config and seed; the 240 s wall-clock cap is gone. A 2 h safety
    stop exists only to protect the run and is reported if it ever fires.
  * R9: K_r = K_c in {5, 10, 20, 40, 80}.
  * R1/R2 embeddings in two versions: 'full' (all entries, as v3) and 'train' (held-out entries
    masked, see embed_R1 / embed_R2).
  * Cell offsets and gene terms are returned for R4/R5/R6 so the driver can build the
    depth-standardized p~ (v3/predictors.standardized definition) for task 5.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BF = HERE.parents[1]   # cm_minimal/experiments/bernfac
for p in (BF, BF / "models/counts", BF / "models/mfa", BF / "models/lbm", BF / "models/neural"):
    sys.path.insert(0, str(p))

EPS = 1e-12

BF_K = (4, 8, 16)
BF_LAM = (3.0, 10.0, 30.0, 100.0)
BF_GRID = [(k, lam) for k in BF_K for lam in BF_LAM]
R7_GRID = [(K, d) for K in (1, 2, 4) for d in (4, 8, 16)]
R8_D = (2, 5, 10)
R8_LAMV = (10.0, 100.0, 1000.0)
R8_GRID = [(d, lv) for d in R8_D for lv in R8_LAMV]
R8_MAX_OUTER = 250
R8_SAFETY_SECONDS = 7200.0
R9_GRID = (5, 10, 20, 40, 80)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def bern_nll(P, Y, mask):
    p = np.clip(P[mask], EPS, 1 - EPS)
    y = Y[mask]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def svd_embedding(U, V):
    """Rotation-invariant cell embedding from U V' (countfac.embedding), zero directions dropped (as v3)."""
    import countfac
    E, s = countfac.embedding(np.asarray(U, float), np.asarray(V, float))
    keep = s > 1e-8 * max(s[0], 1e-300)
    return E[:, keep]


def _fail(config, t1, exc):
    return dict(config=config, P=None, val_nll=np.inf, seconds=time.time() - t1, error=repr(exc))


# ----------------------------------------------------------------------------- BernFac family
def _bernfac(D, kind, max_iter=500, tol=1e-6):
    import bernfac_large as bl
    Y, H, val = D["Y"], D["split"] != 0, D["split"] == 1
    t0 = time.time()
    cands = []
    for k, lam in BF_GRID:
        t1 = time.time()
        cfg = f"k={k} lam={lam:g}"
        try:
            if kind == "R3":
                U, V, tr = bl.fit(Y, k, lam, heldout=H, max_iter=max_iter, tol=tol, verbose=False)
                eta0 = U @ V.T
                P = sig(eta0)
                std = None
            elif kind == "R4":
                U, V, a, b, tr = bl.fit_offsets(Y, k, lam, heldout=H, max_iter=max_iter, tol=tol, verbose=False)
                eta0 = U @ V.T + b[None, :]
                P = sig(eta0 + a[:, None])
                std = dict(c=a, P_std=sig(eta0 + np.median(a)))
            else:
                U, V, a, b, tr = bl.fit_cloglog(Y, k, lam, heldout=H, max_iter=max_iter, tol=tol, verbose=False)
                eta0 = U @ V.T + b[None, :]
                P = -np.expm1(-np.exp(np.clip(eta0 + a[:, None], bl.ETA_MIN, bl.ETA_MAX)))
                std = dict(c=a, P_std=-np.expm1(-np.exp(np.clip(eta0 + np.median(a), bl.ETA_MIN, bl.ETA_MAX))))
        except Exception as exc:  # a monotonicity failure is recorded, not hidden
            cands.append(_fail(cfg, t1, exc))
            continue
        cands.append(dict(config=cfg, P=P, val_nll=bern_nll(P, Y, val), seconds=time.time() - t1,
                          iterations=tr["iterations"], converged=bool(tr["converged"]),
                          emb=svd_embedding(U, V), std=std))
    return dict(cands=cands, fit_seconds=time.time() - t0)


def run_R3(D):
    return _bernfac(D, "R3")


def run_R4(D):
    return _bernfac(D, "R4")


def run_R5(D):
    return _bernfac(D, "R5")


# ----------------------------------------------------------------------------- count models
def _counts(D, family, max_iter=400, tol=1e-6):
    import countfac as cf
    X, Y, H, val = D["X"], D["Y"], D["split"] != 0, D["split"] == 1
    Dc = cf.CountData(X, heldout=H, depth="train")
    t0 = time.time()
    cands = []
    for k, lam in BF_GRID:
        t1 = time.time()
        cfg = f"k={k} lam={lam:g}"
        try:
            U, V, a, b, phi, tr = cf.fit(Dc, k, lam, family=family, max_iter=max_iter, tol=tol, verbose=False)
        except Exception as exc:
            cands.append(_fail(cfg, t1, exc))
            continue
        c = a + Dc.o                                   # cell offset o_i + a_i (predictors.cell_offset for R6)
        eta0 = U @ V.T + b[None, :]
        ph = None if phi is None else phi[None, :]
        P, _ = cf.prob_detect(np.exp(np.clip(eta0 + c[:, None], cf.ETA_MIN, cf.ETA_MAX)), ph)
        Ps, _ = cf.prob_detect(np.exp(np.clip(eta0 + np.median(c), cf.ETA_MIN, cf.ETA_MAX)), ph)
        cands.append(dict(config=cfg, P=P, val_nll=bern_nll(P, Y, val), seconds=time.time() - t1,
                          iterations=tr["iterations"], converged=bool(tr["converged"]),
                          emb=svd_embedding(U, V), std=dict(c=c, P_std=Ps),
                          phi=None if phi is None else np.asarray(phi, float)))
    return dict(cands=cands, fit_seconds=time.time() - t0)


def run_R6(D):
    return _counts(D, "nb")


def run_PO(D):
    return _counts(D, "poisson")


# ----------------------------------------------------------------------------- R7 local-factor mixture
def run_R7(D, max_iter=300, tol=1e-6):
    import mfa
    Y, H, val = D["Y"], D["split"] != 0, D["split"] == 1
    t0 = time.time()
    cands = []
    for K, d in R7_GRID:
        t1 = time.time()
        cfg = f"K={K} d={d}"
        opts = mfa.MFAOptions(residual_mode="none", init="kmeans", px=True, max_iter=max_iter, tol=tol, seed=1,
                              lambda_mu=1.0, lambda_B=1.0, mono_tol=1e-7)
        try:
            model, state, info = mfa.fit(Y, K, d, excluded=H, opts=opts)
            P = mfa.predict_rows(model, state, np.arange(Y.shape[0]))
            emb = mfa.logit_mean_embedding(model, state, n_components=50)[0].astype(np.float64)
        except Exception as exc:
            cands.append(_fail(cfg, t1, exc))
            continue
        cands.append(dict(config=cfg, P=P, val_nll=bern_nll(P, Y, val), seconds=time.time() - t1,
                          iterations=info["iterations"], converged=bool(info["converged"]), emb=emb))
    return dict(cands=cands, fit_seconds=time.time() - t0)


# ----------------------------------------------------------------------------- R8 neural manifold
def run_R8_one(D, d, lv, max_outer=R8_MAX_OUTER, tol=1e-6):
    """One R8 config with a fixed outer-iteration budget (same for every config / seed / world)."""
    import torch
    from bernoulli_neural import NeuralConfig, fit_bernoulli_neural_manifold, predict_bernoulli_neural_manifold
    from sklearn.utils.extmath import randomized_svd
    torch.set_num_threads(1)
    Y, train, val = D["Y"], D["split"] == 0, D["split"] == 1
    Us, _, _ = randomized_svd(np.where(train, 4.0 * (Y - 0.5), 0.0), 10, n_iter=4, random_state=0)
    Z0 = Us * np.sqrt(Us.shape[0])
    t1 = time.time()
    cfg_name = f"d={d} lamV={lv:g}"
    cfg = NeuralConfig(latent_dim=d, hidden_widths=(64, 64), head_mode="affine", optimizer_mode="pg_gem",
                       dtype="float64", lambda_output=lv, max_outer=max_outer, tol=tol, n_restarts=1, seed=1,
                       inner_steps=10, head_steps=5, max_seconds=R8_SAFETY_SECONDS)
    try:
        model, hist = fit_bernoulli_neural_manifold(Y.astype(float), d, cfg, observed_mask=train, init_Z=Z0[:, :d])
    except Exception as exc:
        return dict(cands=[_fail(cfg_name, t1, exc)], fit_seconds=time.time() - t1)
    P = np.asarray(predict_bernoulli_neural_manifold(model), np.float64)
    c = dict(config=cfg_name, P=P, val_nll=bern_nll(P, Y, val), seconds=time.time() - t1,
             iterations=int(hist["iteration"][-1]), converged=hist["termination"] == "tol",
             termination=hist["termination"], emb=model.Z.detach().numpy().astype(np.float64))
    return dict(cands=[c], fit_seconds=time.time() - t1)


# ----------------------------------------------------------------------------- R9 latent blocks
def run_R9(D, max_iter=300, tol=1e-7):
    import lbm
    Y, train, val = D["Y"], D["split"] == 0, D["split"] == 1
    data = lbm.BinaryData(Y.astype(float), mask=train)
    t0 = time.time()
    cands = []
    for K in R9_GRID:
        t1 = time.time()
        cfg = f"Kr={K} Kc={K}"
        try:
            mdl = lbm.fit_lbm(data, K, K, lam=1.0, max_iter=max_iter, tol=tol, n_restarts=3, init="mixed", seed=1,
                              mono_tol=1e-6)
        except Exception as exc:
            cands.append(_fail(cfg, t1, exc))
            continue
        P = lbm.predict_matrix(mdl)
        cands.append(dict(config=cfg, P=P, val_nll=bern_nll(P, Y, val), seconds=time.time() - t1,
                          iterations=int(mdl.info["n_iter"]), converged=mdl.info["status"] == "converged",
                          emb=np.asarray(mdl.R @ mdl.B)))
    return dict(cands=cands, fit_seconds=time.time() - t0)


# ----------------------------------------------------------------------------- baselines and representations
def run_B0(D):
    """Gene detection rate (train entries, +0.5/+1 smoothing): p_ij = r_j."""
    Y, train = D["Y"], D["split"] == 0
    t0 = time.time()
    r = (np.where(train, Y, 0).sum(0) + 0.5) / (train.sum(0) + 1.0)
    P = np.broadcast_to(r[None, :], Y.shape).copy()
    return dict(cands=[dict(config="-", P=P, val_nll=bern_nll(P, Y, D["split"] == 1), seconds=time.time() - t0)],
                fit_seconds=time.time() - t0)


def run_B1(D):
    """Additive cell + gene logistic model sigma(a_i + b_j) (alternating Newton on train entries)."""
    import bernfac_large as bl
    Y, H = D["Y"], D["split"] != 0
    t0 = time.time()
    a, b = bl.additive_init(Y, H, iters=40)
    P = sig(a[:, None] + b[None, :])
    return dict(cands=[dict(config="-", P=P, val_nll=bern_nll(P, Y, D["split"] == 1), seconds=time.time() - t0)],
                fit_seconds=time.time() - t0)


def embed_R1(D, n_comps=50, masked=False):
    """normalize_total -> log1p -> scale -> PCA 50 on all 300 genes.

    masked=True ('train'): every held-out entry (split != 0, 30% of entries) is set to 0 BEFORE
    normalize_total, so library sizes, log values, gene scaling and the PCA all use train entries only.
    Because the mask is uniform at random, zero-filling scales every library by about 0.7 and adds
    false zeros at random; no other imputation is done.
    """
    import anndata as ad
    import scanpy as sc
    X = D["X"].astype(np.float32)
    if masked:
        X = np.where(D["split"] == 0, X, 0.0).astype(np.float32)
    a = ad.AnnData(X)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.scale(a, max_value=10)
    sc.pp.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=0)
    return a.obsm["X_pca"].astype(np.float64)


def embed_R2(D, n_comps=50, masked=False):
    """PCA 50 of column-centred Y.

    masked=True ('train'): columns are centred with the TRAIN-entry detection rate of each gene and every
    held-out entry is set to 0 after centring (i.e. mean-imputed), then the SVD is taken.
    """
    Y = D["Y"].astype(np.float64)
    if masked:
        tr = D["split"] == 0
        mu = (Y * tr).sum(0) / np.maximum(tr.sum(0), 1)
        Yc = np.where(tr, Y - mu, 0.0)
    else:
        Yc = Y - Y.mean(0)
    U, s, _ = np.linalg.svd(Yc, full_matrices=False)
    return U[:, :n_comps] * s[:n_comps]


RUNNERS = dict(R3=run_R3, R4=run_R4, R5=run_R5, R6=run_R6, PO=run_PO, R7=run_R7, R9=run_R9, B0=run_B0, B1=run_B1)
