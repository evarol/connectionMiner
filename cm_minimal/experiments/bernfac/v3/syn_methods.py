"""Thin wrappers that fit every v3 method on one synthetic dataset with the shared masks.

Each `run_<code>(D, budget)` fits a small grid on the TRAIN entries, picks the configuration
with the lowest VALIDATION Bernoulli NLL and returns

    dict(P=P_hat (n x m, in memory only), emb=cell embedding or None, config=str,
         grid=[(config, val_nll, seconds, iterations, converged)], fit_seconds)

The model code is imported unchanged from the study packages (no reimplementation):
    R3/R4/R5   bernfac_large.fit / fit_offsets / fit_cloglog
    R6, PO     models/counts/countfac.fit (family 'nb' / 'poisson')
    R7         models/mfa/mfa.fit (residual_mode 'none', k-means init, PX step)
    R8         models/neural/bernoulli_neural (affine head, pg_gem)
    R9         models/lbm/lbm.fit_lbm
    B0, B1     gene-rate and cell+gene additive logistic baselines (bernfac_large.additive_init)
Embeddings follow umap_compare.py: SVD of U V' for the factor models (offsets excluded),
the MFA logit-mean embedding, the neural Z, the LBM R B; R1 = normalize_total -> log1p ->
scale -> PCA 50 (all 300 genes, no HVG step), R2 = PCA 50 of column-centred Y.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BF = HERE.parent
for p in (BF, BF / "models/counts", BF / "models/mfa", BF / "models/lbm", BF / "models/neural"):
    sys.path.insert(0, str(p))

EPS = 1e-12


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def bern_nll(P, Y, mask):
    p = np.clip(P[mask], EPS, 1 - EPS)
    y = Y[mask]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def svd_embedding(U, V):
    """Rotation-invariant cell embedding from U V' (countfac.embedding), zero directions dropped."""
    import countfac
    E, s = countfac.embedding(np.asarray(U, float), np.asarray(V, float))
    keep = s > 1e-8 * max(s[0], 1e-300)
    return E[:, keep]


def _select(cands):
    """cands: list of dicts with val_nll; returns the best one (lowest validation NLL)."""
    ok = [c for c in cands if np.isfinite(c["val_nll"])]
    return min(ok, key=lambda c: c["val_nll"])


def _grid_out(cands, best, t0):
    grid = [(c["config"], c["val_nll"], c["seconds"], c.get("iterations"), c.get("converged")) for c in cands]
    return dict(P=best["P"], emb=best.get("emb"), config=best["config"], grid=grid, fit_seconds=time.time() - t0)


# ----------------------------------------------------------------------------- BernFac family
BF_GRID = [(k, lam) for k in (5, 10, 20) for lam in (30.0, 10.0, 1.0)]   # lambda 30 added: lambda 1 overfits at n = 2000


def _bernfac(D, kind, max_iter=300, tol=1e-6):
    import bernfac_large as bl
    Y, H, val = D["Y"], D["split"] != 0, D["split"] == 1
    t0 = time.time()
    cands = []
    for k, lam in BF_GRID:
        t1 = time.time()
        try:
            if kind == "R3":
                U, V, tr = bl.fit(Y, k, lam, heldout=H, max_iter=max_iter, tol=tol, verbose=False)
                P = sig(U @ V.T)
            elif kind == "R4":
                U, V, a, b, tr = bl.fit_offsets(Y, k, lam, heldout=H, max_iter=max_iter, tol=tol, verbose=False)
                P = sig(U @ V.T + a[:, None] + b[None, :])
            else:
                U, V, a, b, tr = bl.fit_cloglog(Y, k, lam, heldout=H, max_iter=max_iter, tol=tol, verbose=False)
                eta = np.clip(U @ V.T + a[:, None] + b[None, :], bl.ETA_MIN, bl.ETA_MAX)
                P = -np.expm1(-np.exp(eta))
        except Exception as exc:  # a monotonicity failure is recorded, not hidden
            cands.append(dict(config=f"k={k} lam={lam:g}", val_nll=np.inf, seconds=time.time() - t1,
                              error=repr(exc)))
            continue
        cands.append(dict(config=f"k={k} lam={lam:g}", P=P, val_nll=bern_nll(P, Y, val), seconds=time.time() - t1,
                          iterations=tr["iterations"], converged=bool(tr["converged"]), UV=(U, V)))
    best = _select(cands)
    best["emb"] = svd_embedding(*best["UV"])
    return _grid_out(cands, best, t0)


def run_R3(D):
    return _bernfac(D, "R3")


def run_R4(D):
    return _bernfac(D, "R4")


def run_R5(D):
    return _bernfac(D, "R5")


# ----------------------------------------------------------------------------- count models
def _counts(D, family, max_iter=200, tol=1e-6):
    import countfac as cf
    X, Y, H, val = D["X"], D["Y"], D["split"] != 0, D["split"] == 1
    Dc = cf.CountData(X, heldout=H, depth="train")
    t0 = time.time()
    cands = []
    for k, lam in BF_GRID:
        t1 = time.time()
        try:
            U, V, a, b, phi, tr = cf.fit(Dc, k, lam, family=family, max_iter=max_iter, tol=tol, verbose=False)
        except Exception as exc:
            cands.append(dict(config=f"k={k} lam={lam:g}", val_nll=np.inf, seconds=time.time() - t1,
                              error=repr(exc)))
            continue
        eta = np.clip(U @ V.T + a[:, None] + b[None, :] + Dc.o[:, None], cf.ETA_MIN, cf.ETA_MAX)
        P, _ = cf.prob_detect(np.exp(eta), None if phi is None else phi[None, :])
        cands.append(dict(config=f"k={k} lam={lam:g}", P=P, val_nll=bern_nll(P, Y, val), seconds=time.time() - t1,
                          iterations=tr["iterations"], converged=tr["converged"], UV=(U, V)))
    best = _select(cands)
    best["emb"] = svd_embedding(*best["UV"])
    return _grid_out(cands, best, t0)


def run_R6(D):
    return _counts(D, "nb")


def run_PO(D):
    return _counts(D, "poisson")


# ----------------------------------------------------------------------------- R7 local-factor mixture
def run_R7(D, max_iter=150, tol=1e-6):
    import mfa
    Y, H, val = D["Y"], D["split"] != 0, D["split"] == 1
    t0 = time.time()
    cands = []
    for K in (1, 4):
        for d in (5, 10):
            t1 = time.time()
            opts = mfa.MFAOptions(residual_mode="none", init="kmeans", px=True, max_iter=max_iter, tol=tol, seed=1,
                                  lambda_mu=1.0, lambda_B=1.0, mono_tol=1e-7)
            try:
                model, state, info = mfa.fit(Y, K, d, excluded=H, opts=opts)
            except Exception as exc:
                cands.append(dict(config=f"K={K} d={d}", val_nll=np.inf, seconds=time.time() - t1, error=repr(exc)))
                continue
            P = mfa.predict_rows(model, state, np.arange(Y.shape[0]))
            cands.append(dict(config=f"K={K} d={d}", P=P, val_nll=bern_nll(P, Y, val), seconds=time.time() - t1,
                              iterations=info["iterations"], converged=bool(info["converged"]),
                              ms=(model, state)))
    best = _select(cands)
    best["emb"] = mfa.logit_mean_embedding(*best["ms"], n_components=50)[0].astype(np.float64)
    return _grid_out(cands, best, t0)


# ----------------------------------------------------------------------------- R8 neural manifold
def run_R8(D, max_outer=300, tol=1e-6, max_seconds=240):
    import torch
    from bernoulli_neural import NeuralConfig, fit_bernoulli_neural_manifold, predict_bernoulli_neural_manifold
    from sklearn.utils.extmath import randomized_svd
    torch.set_num_threads(1)
    Y, train, val = D["Y"], D["split"] == 0, D["split"] == 1
    # training-only SVD start for Z, unit variance per coordinate (as models/neural/run_full.py)
    Us, _, _ = randomized_svd(np.where(train, 4.0 * (Y - 0.5), 0.0), 10, n_iter=4, random_state=0)
    Z0 = Us * np.sqrt(Us.shape[0])
    t0 = time.time()
    cands = []
    for d in (2, 5, 10):
        for lv in (1.0, 10.0, 100.0):   # 100 added: at m = 300 genes lambda_V = 1 overfits (pilot on G3)
            t1 = time.time()
            cfg = NeuralConfig(latent_dim=d, hidden_widths=(64, 64), head_mode="affine", optimizer_mode="pg_gem",
                               dtype="float64", lambda_output=lv, max_outer=max_outer, tol=tol, n_restarts=1, seed=1,
                               inner_steps=10, head_steps=5, max_seconds=max_seconds)
            try:
                model, hist = fit_bernoulli_neural_manifold(Y.astype(float), d, cfg, observed_mask=train,
                                                            init_Z=Z0[:, :d])
            except Exception as exc:
                cands.append(dict(config=f"d={d} lamV={lv:g}", val_nll=np.inf, seconds=time.time() - t1,
                                  error=repr(exc)))
                continue
            P = np.asarray(predict_bernoulli_neural_manifold(model), np.float64)
            cands.append(dict(config=f"d={d} lamV={lv:g}", P=P, val_nll=bern_nll(P, Y, val),
                              seconds=time.time() - t1, iterations=int(hist["iteration"][-1]),
                              converged=hist["termination"] == "tol",
                              termination=hist["termination"], Z=model.Z.detach().numpy().astype(np.float64)))
    best = _select(cands)
    best["emb"] = best["Z"]
    return _grid_out(cands, best, t0)


# ----------------------------------------------------------------------------- R9 latent blocks
def run_R9(D, max_iter=300, tol=1e-7):
    import lbm
    Y, train, val = D["Y"], D["split"] == 0, D["split"] == 1
    data = lbm.BinaryData(Y.astype(float), mask=train)
    t0 = time.time()
    cands = []
    for Kr, Kc in ((5, 5), (10, 10), (20, 20), (40, 40)):   # (40, 40) added: (20, 20) won at the edge in the pilot
        t1 = time.time()
        try:
            mdl = lbm.fit_lbm(data, Kr, Kc, lam=1.0, max_iter=max_iter, tol=tol, n_restarts=3, init="mixed", seed=1,
                              mono_tol=1e-6)
        except Exception as exc:
            cands.append(dict(config=f"Kr={Kr} Kc={Kc}", val_nll=np.inf, seconds=time.time() - t1, error=repr(exc)))
            continue
        P = lbm.predict_matrix(mdl)
        cands.append(dict(config=f"Kr={Kr} Kc={Kc}", P=P, val_nll=bern_nll(P, Y, val), seconds=time.time() - t1,
                          iterations=int(mdl.info["n_iter"]), converged=mdl.info["status"] == "converged",
                          emb=mdl.R @ mdl.B))
    return _grid_out(cands, _select(cands), t0)


# ----------------------------------------------------------------------------- baselines and representations
def run_B0(D):
    """Gene detection rate (train entries, +0.5/+1 smoothing): p_ij = r_j."""
    Y, train = D["Y"], D["split"] == 0
    t0 = time.time()
    r = (np.where(train, Y, 0).sum(0) + 0.5) / (train.sum(0) + 1.0)
    P = np.broadcast_to(r[None, :], Y.shape).copy()
    return dict(P=P, emb=None, config="-", grid=[], fit_seconds=time.time() - t0)


def run_B1(D):
    """Additive cell + gene logistic model sigma(a_i + b_j) (alternating Newton on train entries)."""
    import bernfac_large as bl
    Y, H = D["Y"], D["split"] != 0
    t0 = time.time()
    a, b = bl.additive_init(Y, H, iters=40)
    return dict(P=sig(a[:, None] + b[None, :]), emb=None, config="-", grid=[], fit_seconds=time.time() - t0)


def embed_R1(D, n_comps=50):
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(D["X"].astype(np.float32))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.scale(a, max_value=10)
    sc.pp.pca(a, n_comps=n_comps, svd_solver="arpack", random_state=0)
    return a.obsm["X_pca"].astype(np.float64)


def embed_R2(D, n_comps=50):
    Y = D["Y"].astype(np.float64)
    Yc = Y - Y.mean(0)
    U, s, _ = np.linalg.svd(Yc, full_matrices=False)
    return U[:, :n_comps] * s[:n_comps]


RUNNERS = dict(R3=run_R3, R4=run_R4, R5=run_R5, R6=run_R6, PO=run_PO, R7=run_R7, R8=run_R8, R9=run_R9,
               B0=run_B0, B1=run_B1)
