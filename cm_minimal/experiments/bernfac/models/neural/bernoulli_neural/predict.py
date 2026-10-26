"""Plug-in prediction, new-row inference with a frozen decoder, and scores (handoff 4.10, 6.2).

prediction_mode = 'map_plugin': p_ij = sigmoid(g_theta_hat(z_hat_i)_j). These are conditional
plug-in probabilities at the joint-MAP estimate, NOT posterior-integrated predictions.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np
import torch

from .data import MaskedBinary
from .objectives import F64, lik_terms
from .pg import pg_mean

PREDICTION_MODE = "map_plugin"


@contextlib.contextmanager
def frozen_decoder(model):
    saved = [(p, p.requires_grad) for p in model.decoder.parameters()]
    try:
        for p, _ in saved:
            p.requires_grad_(False)
        yield
    finally:
        for p, rg in saved:
            p.requires_grad_(rg)


@torch.no_grad()
def predict_logits(model, rows=None, cols=None, Z=None, batch_size=4096):
    Zt = model.Z if Z is None else torch.as_tensor(Z, dtype=model.Z.dtype)
    ridx = torch.arange(Zt.shape[0]) if rows is None else torch.as_tensor(rows, dtype=torch.long)
    cidx = None if cols is None else torch.as_tensor(cols, dtype=torch.long)
    out = []
    for a in range(0, len(ridx), batch_size):
        out.append(model.decoder(Zt[ridx[a:a + batch_size]], cidx))
    return torch.cat(out).numpy()


def predict_bernoulli_neural_manifold(model, rows=None, cols=None, Z=None, batch_size=4096):
    """Plug-in probabilities for a row x column block (prediction_mode='map_plugin')."""
    psi = predict_logits(model, rows, cols, Z, batch_size).astype(np.float64)
    return 1.0 / (1.0 + np.exp(-psi))


@torch.no_grad()
def predict_pairs(model, i, j, Z=None, batch_size=4096, logits=False):
    """Plug-in probabilities at index pairs (i, j), computed by row batches (no n x m array)."""
    i, j = np.asarray(i), np.asarray(j)
    order = np.argsort(i, kind="stable")
    out = np.empty(len(i))
    Zt = model.Z if Z is None else torch.as_tensor(Z, dtype=model.Z.dtype)
    rows_u, start = np.unique(i[order], return_index=True)
    bounds = list(start) + [len(i)]
    for a in range(0, len(rows_u), batch_size):
        ru = rows_u[a:a + batch_size]
        psi = model.decoder(Zt[torch.as_tensor(ru)]).to(F64).numpy()
        for k, r in enumerate(ru):
            sl = order[bounds[a + k]:bounds[a + k + 1]]
            out[sl] = psi[k, j[sl]]
    return out if logits else 1.0 / (1.0 + np.exp(-out))


@torch.no_grad()
def prior_predictive_mc(model, cols=None, n_samples=2000, seed=0, batch_size=500):
    """Monte Carlo prior predictive E_{z ~ N(0, I/lambda_z)}[sigmoid(g(z)_j)] with MC standard
    errors (explicitly labelled; differs from the mode plug-in sigmoid(g(0)))."""
    g = torch.Generator().manual_seed(seed)
    d = model.config.latent_dim
    cidx = None if cols is None else torch.as_tensor(cols, dtype=torch.long)
    s1 = s2 = 0.0
    for a in range(0, n_samples, batch_size):
        k = min(batch_size, n_samples - a)
        z = torch.randn(k, d, generator=g, dtype=F64).to(model.Z.dtype) / math.sqrt(model.config.lambda_z)
        p = torch.sigmoid(model.decoder(z, cidx).to(F64))
        s1 = s1 + p.sum(0)
        s2 = s2 + (p * p).sum(0)
    mean = s1 / n_samples
    var = (s2 / n_samples - mean * mean).clamp_min(0) * n_samples / max(n_samples - 1, 1)
    return dict(mean=mean.numpy(), mc_se=(var / n_samples).sqrt().numpy(), n_samples=n_samples, seed=seed,
                mode="prior_predictive_mc")


# ---------------------------------------------------------------------------------------------
# new-row inference (decoder frozen; only the context cells of each new row enter)
# ---------------------------------------------------------------------------------------------
def _row_objective(model, Z, data, kind, W, lz, idx=None):
    r = torch.arange(Z.shape[0]) if idx is None else idx
    psi = model.decoder(Z[r])
    y, mk, kappa = data.rows_idx(r, Z.dtype)
    lik = lik_terms(kind, psi, y, mk, kappa, None if W is None else W[r]).sum(1, dtype=F64)
    return lik + 0.5 * lz * (Z[r].to(F64) ** 2).sum(1)


def infer_neural_rows(model, Y_new, observed_mask=None, method="pg_gem", n_starts=5, seed=0, max_iter=300,
                      inner_steps=5, tol=1e-9, max_trials=30, armijo=1e-4, backtrack=0.5):
    """MAP row coordinates for new rows with ALL global decoder parameters frozen.

        min_z sum_{j in O_new} [softplus(g(z)_j) - y_j g(z)_j] + lambda_z/2 |z|^2

    Rows are independent, so each row has its own Armijo step. method='pg_gem' uses the frozen
    PG surrogate per outer iteration (with a per-row original-objective check and rollback);
    'direct_bce' uses the original objective. Starts: z = 0 plus (n_starts - 1) draws from the
    prior; the start with the lowest final objective is kept per row. Rows with no observed
    context return z = 0 (the prior MAP); their predictions are mode plug-ins, not prior means.
    """
    data = MaskedBinary(Y_new, observed_mask) if not isinstance(Y_new, MaskedBinary) else Y_new
    dt = model.Z.dtype
    lz = model.config.lambda_z
    n, d = data.n, model.config.latent_dim
    g = torch.Generator().manual_seed(seed)
    best_Z = torch.zeros(n, d, dtype=dt)
    best_f = torch.full((n,), math.inf, dtype=F64)
    best_start = torch.zeros(n, dtype=torch.long)
    has_obs = data.mask.any(1)
    iters_used = []
    with frozen_decoder(model):
        for st in range(n_starts):
            if st == 0:
                Z = torch.zeros(n, d, dtype=dt)
            else:
                Z = (torch.randn(n, d, generator=g, dtype=F64) / math.sqrt(lz)).to(dt)
            Z[~has_obs] = 0
            step = torch.ones(n, dtype=F64)
            with torch.no_grad():
                F_cur = _row_objective(model, Z, data, "F", None, lz)
            it = 0
            for it in range(1, max_iter + 1):
                F_prev = F_cur.clone()
                Z_prev = Z.clone()
                if method == "pg_gem":
                    with torch.no_grad():
                        W = torch.where(data.mask, pg_mean(model.decoder(Z)), torch.zeros((), dtype=dt))
                    kind = "S"
                elif method == "direct_bce":
                    W, kind = None, "F"
                else:
                    raise ValueError(method)
                for _ in range(inner_steps):
                    Zv = Z.clone().requires_grad_(True)
                    with torch.enable_grad():
                        f = _row_objective(model, Zv, data, kind, W, lz)
                        gz, = torch.autograd.grad(f.sum(), Zv)
                    gz[~has_obs] = 0
                    g2 = gz.to(F64).pow(2).sum(1)
                    f0 = f.detach()
                    todo = torch.nonzero(g2 > 1e-30).squeeze(1)
                    stp = step.clone()
                    acc = torch.zeros(n, dtype=torch.bool)
                    Z0 = Z.clone()
                    for _t in range(max_trials):
                        if todo.numel() == 0:
                            break
                        with torch.no_grad():
                            Z[todo] = Z0[todo] - stp[todo, None].to(dt) * gz[todo]
                            ft = _row_objective(model, Z, data, kind, W, lz, todo)
                        ok = torch.isfinite(ft) & (ft <= f0[todo] - armijo * stp[todo] * g2[todo])
                        acc[todo[ok]] = True
                        todo = todo[~ok]
                        stp[todo] *= backtrack
                    Z[todo] = Z0[todo]
                    step = torch.where(acc, stp / backtrack, stp)
                    if not bool(acc.any()):
                        break
                with torch.no_grad():
                    F_cur = _row_objective(model, Z, data, "F", None, lz)
                bad = F_cur > F_prev + 1e-10 * F_prev.abs().clamp_min(1.0)
                if bool(bad.any()):                      # per-row rollback (rows are independent)
                    Z[bad] = Z_prev[bad]
                    F_cur[bad] = F_prev[bad]
                rel = ((F_prev - F_cur) / F_prev.abs().clamp_min(1.0)).max()
                if float(rel) < tol:
                    break
            iters_used.append(it)
            better = F_cur < best_f
            best_Z[better] = Z[better]
            best_f[better] = F_cur[better]
            best_start[better] = st
    return best_Z.numpy(), dict(objective=best_f.numpy(), best_start=best_start.numpy(), iterations=iters_used,
                                rows_without_context=int((~has_obs).sum()), method=method)


# ---------------------------------------------------------------------------------------------
# held-out scores (same convention as bernfac rescore_val_test.scores_from_logit)
# ---------------------------------------------------------------------------------------------
def scores_from_logit_fn(Y, M, logit_rows, block=2048, auc_sample=2_000_000, seed=0, n_bins=10):
    """Bernoulli NLL / Brier / AUC (2 M subsample, rng seed 0) / 10-bin calibration on mask M."""
    from sklearn.metrics import roc_auc_score
    nll = brier = 0.0
    cnt = 0
    ys, ps = [], []
    for s in range(0, Y.shape[0], block):
        Mb = M[s:s + block]
        if not Mb.any():
            continue
        eta = logit_rows(slice(s, s + block))[Mb].astype(np.float64)
        y = Y[s:s + block][Mb].astype(np.float64)
        ll = y * eta - np.logaddexp(0, eta)
        p = 1.0 / (1.0 + np.exp(-eta))
        nll -= ll.sum()
        brier += np.sum((p - y) ** 2)
        cnt += y.size
        ys.append(y.astype(np.int8))
        ps.append(p.astype(np.float32))
    y, p = np.concatenate(ys), np.concatenate(ps)
    edges = np.linspace(0, 1, n_bins + 1)
    bi = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    cal = [dict(bin=b, mean_pred=float(p[bi == b].mean()), frac_pos=float(y[bi == b].mean()),
                count=int((bi == b).sum())) for b in range(n_bins) if (bi == b).any()]
    if len(np.unique(y)) < 2:
        auc = float("nan")
    else:
        idx = np.random.default_rng(seed).choice(y.size, min(auc_sample, y.size), replace=False)
        auc = float(roc_auc_score(y[idx], p[idx]))
    return dict(nll=nll / cnt, brier=brier / cnt, auc=auc, n=cnt, calibration=cal)


def heldout_scores(model, Y, M, Z=None, block=2048, **kw):
    Zt = model.Z if Z is None else torch.as_tensor(Z, dtype=model.Z.dtype)

    @torch.no_grad()
    def fn(sl):
        return model.decoder(Zt[sl]).numpy()

    return scores_from_logit_fn(Y, M, fn, block=block, **kw)
