"""BernFac: Bayesian logistic matrix factorization of a binary matrix via
Polya-Gamma EM (PG-EM).

Model
-----
    Y_ij | U, V  ~  Bernoulli(sigmoid(u_i' v_j))         (i in rows, j in cols)
    u_i ~ N(0, I / lambda_u),   v_j ~ N(0, I / lambda_v)

We maximise the *original* MAP objective

    J(U, V) = sum_obs [ y_ij psi_ij - softplus(psi_ij) ]
              - lambda_u/2 ||U||_F^2 - lambda_v/2 ||V||_F^2 ,   psi = U V'

PG augmentation (Polson, Scott & Windle 2013, JASA 108:1339) writes each
Bernoulli factor as a Gaussian-in-psi mixture over omega ~ PG(1, 0):

    e^{y psi} / (1 + e^psi) = 1/2 e^{kappa psi} E_omega[ e^{-omega psi^2 / 2} ],
    kappa = y - 1/2.

Given psi the posterior of omega is PG(1, psi) with mean
    w(psi) = tanh(psi/2) / (2 psi)    (= 1/4 at psi = 0).

So one EM iteration is:
    E-step : W = w(U V')                      (frozen for the whole M-step)
    M-step : u_i = [lambda_u I + sum_j w_ij v_j v_j']^{-1} sum_j kappa_ij v_j
             v_j = [lambda_v I + sum_i w_ij u_i u_i']^{-1} sum_i kappa_ij u_i
             (the V update uses the *new* U but the *same* W)

Each block update exactly maximises the EM surrogate Q, so J never decreases.
The problem is nonconvex: we get a local optimum, and factors are identifiable
only up to a common rotation (U R, V R). Compare P = sigmoid(U V'), not U, V.

Missing entries: pass mask (True = observed) or NaN in Y. Observed zeros are
data; masked cells do not enter the likelihood.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# Numerically stable scalar functions
# ----------------------------------------------------------------------------
def sigmoid(x):
    """Sign-stable logistic function."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softplus(x):
    """log(1 + e^x) without overflow."""
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def pg_mean(psi):
    """E[omega | psi] for omega ~ PG(1, psi): tanh(psi/2)/(2 psi), 1/4 at 0.

    Even in psi, positive, max 1/4, ~1/(2|psi|) for large |psi|.
    """
    psi = np.asarray(psi, dtype=float)
    a = np.abs(psi)
    out = np.empty_like(a)
    small = a < 1e-4
    s2 = a[small] ** 2
    out[small] = 0.25 - s2 / 48.0 + s2 * s2 / 480.0
    big = ~small
    out[big] = np.tanh(a[big] / 2.0) / (2.0 * a[big])
    return out


# ----------------------------------------------------------------------------
# Objective, gradient, prediction
# ----------------------------------------------------------------------------
def _prep(Y, mask):
    Y = np.asarray(Y, dtype=float)
    if mask is None:
        mask = ~np.isnan(Y)
    mask = np.asarray(mask, dtype=bool)
    Yc = np.where(mask, np.nan_to_num(Y, nan=0.0), 0.0)  # zero at missing
    obs = Yc[mask]
    if not np.all((obs == 0) | (obs == 1)):
        raise ValueError("observed entries of Y must be 0 or 1")
    return Yc, mask


def loglik(Y, U, V, mask=None):
    """Original (unaugmented) Bernoulli log-likelihood over observed cells."""
    Yc, mask = _prep(Y, mask)
    psi = U @ V.T
    return float(np.sum((Yc * psi - softplus(psi))[mask]))


def objective(Y, U, V, lambda_u, lambda_v, mask=None, grad=False):
    """MAP objective J (larger is better), loglik, and optional gradients.

    grad_U = [M*(Y-P)] V - lambda_u U,  grad_V = [M*(Y-P)]' U - lambda_v V
    """
    Yc, mask = _prep(Y, mask)
    psi = U @ V.T
    ll = float(np.sum((Yc * psi - softplus(psi))[mask]))
    J = ll - 0.5 * lambda_u * np.sum(U * U) - 0.5 * lambda_v * np.sum(V * V)
    if not grad:
        return J, ll
    R = np.where(mask, Yc - sigmoid(psi), 0.0)
    return J, ll, R @ V - lambda_u * U, R.T @ U - lambda_v * V


def predict(U, V):
    """Latent Bernoulli probabilities P = sigmoid(U V')."""
    return sigmoid(U @ V.T)


def surrogate_Q(Y, U, V, W, lambda_u, lambda_v, mask=None):
    """EM surrogate Q(U,V; W) (drops constants). Used only in tests."""
    Yc, mask = _prep(Y, mask)
    psi = U @ V.T
    K = np.where(mask, Yc - 0.5, 0.0)
    Wm = np.where(mask, W, 0.0)
    return float(np.sum(K * psi) - 0.5 * np.sum(Wm * psi * psi)
                 - 0.5 * lambda_u * np.sum(U * U) - 0.5 * lambda_v * np.sum(V * V))


# ----------------------------------------------------------------------------
# PG-EM block updates
# ----------------------------------------------------------------------------
def update_rows(K, W, F, lam):
    """Solve [lam I + sum_j W_ij f_j f_j'] x_i = sum_j K_ij f_j for every row i.

    K, W: (n, m) with zeros at missing cells.  F: (m, k) partner factors.
    Returns X: (n, k).  All n k-by-k systems are solved in one batched call.
    """
    m, k = F.shape
    FF = (F[:, :, None] * F[:, None, :]).reshape(m, k * k)    # rows: vec(f_j f_j')
    A = (W @ FF).reshape(-1, k, k) + lam * np.eye(k)          # (n, k, k), one GEMM
    b = K @ F                                                  # (n, k)
    return np.linalg.solve(A, b[..., None])[..., 0]


def init_factors(Y, mask, k, init, scale, rng, U0=None, V0=None):
    n, m = Y.shape
    if init == "random":
        return scale * rng.standard_normal((n, k)), scale * rng.standard_normal((m, k))
    if init == "svd":
        # first-order logit proxy: logit(p) ~ 4 (p - 1/2); zero-filled where missing
        Z = np.where(mask, 4.0 * (Y - 0.5), 0.0)
        Us, s, Vt = np.linalg.svd(Z, full_matrices=False)
        r = np.sqrt(s[:k])
        U, V = Us[:, :k] * r, Vt[:k].T * r
        # tiny jitter so restarts differ and rank-deficient columns are nonzero
        return U + 1e-3 * rng.standard_normal(U.shape), V + 1e-3 * rng.standard_normal(V.shape)
    if init == "provided":
        U0, V0 = np.array(U0, float), np.array(V0, float)
        assert U0.shape == (n, k) and V0.shape == (m, k)
        assert np.all(np.isfinite(U0)) and np.all(np.isfinite(V0))
        return U0.copy(), V0.copy()
    raise ValueError(f"unknown init {init!r}")


@dataclass
class FitResult:
    U: np.ndarray
    V: np.ndarray
    obj: list = field(default_factory=list)       # MAP objective J per iteration (incl. init)
    loglik: list = field(default_factory=list)    # unpenalised log-likelihood per iteration
    rel_improve: list = field(default_factory=list)
    converged: bool = False
    iterations: int = 0
    n_observed: int = 0
    lambda_u: float = 1.0
    lambda_v: float = 1.0
    k: int = 1
    init: str = "random"
    seed: int = 0
    best_restart: int = 0
    grad_norm: float = np.nan
    restart_objs: list = field(default_factory=list)

    @property
    def P(self):
        return predict(self.U, self.V)


def fit(Y, k, lambda_u=1.0, lambda_v=1.0, mask=None, max_iter=200, tol=1e-7,
        init="random", init_scale=0.1, seed=1, n_restarts=1, U0=None, V0=None,
        verbose=False):
    """Fit BernFac by PG-EM. Returns the restart with the largest final J."""
    Yc, mask = _prep(Y, mask)
    if not (lambda_u > 0 and lambda_v > 0):
        raise ValueError("lambda_u and lambda_v must be positive precisions")
    k = int(k)
    K = np.where(mask, Yc - 0.5, 0.0)          # kappa = y - 1/2 (zero if missing)
    rng = np.random.default_rng(seed)
    best = None
    for r in range(n_restarts):
        U, V = init_factors(Yc, mask, k, init, init_scale, rng, U0, V0)
        J, ll = objective(Yc, U, V, lambda_u, lambda_v, mask)
        res = FitResult(U, V, [J], [ll], [], lambda_u=lambda_u, lambda_v=lambda_v,
                        k=k, init=init, seed=seed, n_observed=int(mask.sum()))
        for it in range(1, max_iter + 1):
            # E-step: PG means from the OLD snapshot, frozen for both blocks
            W = np.where(mask, pg_mean(U @ V.T), 0.0)
            # M-step: exact block maximisers of Q
            U = update_rows(K, W, V, lambda_u)
            V = update_rows(K.T, W.T, U, lambda_v)
            J_new, ll = objective(Yc, U, V, lambda_u, lambda_v, mask)
            rel = (J_new - J) / max(1.0, abs(J))
            if rel < -1e-9:
                raise RuntimeError(f"PG-EM decreased J by {J - J_new:.3e} (bug)")
            res.obj.append(J_new); res.loglik.append(ll); res.rel_improve.append(rel)
            J = J_new
            if verbose and (it % 25 == 0 or it == 1):
                print(f"  restart {r} iter {it:4d}  J={J:.4f}  ll={ll:.4f}  rel={rel:.2e}")
            if abs(rel) < tol:
                res.converged = True
                break
        res.U, res.V, res.iterations = U, V, it
        _, _, gU, gV = objective(Yc, U, V, lambda_u, lambda_v, mask, grad=True)
        res.grad_norm = float(np.sqrt(np.sum(gU ** 2) + np.sum(gV ** 2)))
        res.best_restart = r
        if best is None or res.obj[-1] > best.obj[-1]:
            prev = [] if best is None else best.restart_objs
            best = res
            best.restart_objs = prev
        best.restart_objs.append(res.obj[-1])
    return best
