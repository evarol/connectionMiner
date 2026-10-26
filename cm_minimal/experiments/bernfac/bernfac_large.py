"""Memory-bounded PG-EM for BernFac on large binary matrices (e.g. 16k x 20k).

Same algorithm and objective as bernfac.py, implemented in row/column blocks:

  E-step  W = pg_mean(U_old V_old')     -- never stored whole; each block is
                                           regenerated from the SAME snapshot
  M-step A  rows:  [lam I + sum_j w_ij v_j v_j'] u_i = sum_j kappa_ij v_j
  M-step B  cols:  [lam I + sum_i w_ij u_i u_i'] v_j = sum_i kappa_ij u_i
                   (new U, weights from the old snapshot -- handoff section 10)

Y is a dense uint8 matrix (322 M entries ~ 0.3 GB). An optional boolean
`heldout` matrix marks entries excluded from the likelihood (cross-validation).
Large products run in float32; objectives are accumulated in float64.
"""
from __future__ import annotations

import time

import numpy as np

from bernfac import pg_mean, softplus


def _gram_rows(F):
    """(m, k*k) matrix whose row j is vec(f_j f_j')."""
    m, k = F.shape
    return (F[:, :, None] * F[:, None, :]).reshape(m, k * k)


def _solve(A_flat, b, lam, k):
    A = A_flat.reshape(-1, k, k).astype(np.float64) + lam * np.eye(k)
    return np.linalg.solve(A, b.astype(np.float64)[..., None])[..., 0]


def _block_terms(Yb, Hb, Psi):
    """kappa and PG weights for one block, zeroed at held-out entries."""
    K = Yb.astype(np.float32) - 0.5
    W = pg_mean(Psi).astype(np.float32)
    if Hb is not None:
        K[Hb] = 0.0
        W[Hb] = 0.0
    return K, W


def objective(Y, U, V, lam, heldout=None, block=2048):
    """MAP objective J and train log-likelihood over observed (non-held-out) entries."""
    ll = 0.0
    V32 = V.astype(np.float32)
    for s in range(0, Y.shape[0], block):
        Psi = U[s:s + block].astype(np.float32) @ V32.T
        term = Y[s:s + block] * Psi - softplus(Psi)
        if heldout is not None:
            term[heldout[s:s + block]] = 0.0
        ll += float(term.sum(dtype=np.float64))
    pen = 0.5 * lam * (np.sum(U * U) + np.sum(V * V))
    return ll - pen, ll


def heldout_scores(Y, U, V, heldout, block=2048, auc_sample=2_000_000, seed=0):
    """NLL/entry, Brier and AUC on held-out entries."""
    V32 = V.astype(np.float32)
    nll, brier, cnt = 0.0, 0.0, 0
    ys, ps = [], []
    for s in range(0, Y.shape[0], block):
        Hb = heldout[s:s + block]
        if not Hb.any():
            continue
        psi = (U[s:s + block].astype(np.float32) @ V32.T)[Hb].astype(np.float64)
        y = Y[s:s + block][Hb].astype(np.float64)
        p = 1.0 / (1.0 + np.exp(-psi))
        nll += float(np.sum(softplus(psi) - y * psi))
        brier += float(np.sum((p - y) ** 2))
        cnt += y.size
        ys.append(y.astype(np.int8)); ps.append(p.astype(np.float32))
    y, p = np.concatenate(ys), np.concatenate(ps)
    rng = np.random.default_rng(seed)
    if y.size > auc_sample:
        idx = rng.choice(y.size, auc_sample, replace=False)
        y, p = y[idx], p[idx]
    from sklearn.metrics import roc_auc_score
    return dict(nll=nll / cnt, brier=brier / cnt, auc=float(roc_auc_score(y, p)), n_heldout=cnt)


def pgem_iteration(Y, U, V, lam, heldout=None, block=2048):
    """One PG-EM iteration (E-step + both M-step blocks) from snapshot (U, V)."""
    n, m = Y.shape
    k = U.shape[1]
    U_old32, V_old32 = U.astype(np.float32), V.astype(np.float32)

    # M-step A: rows, weights from (U_old, V_old)
    GV = _gram_rows(V_old32)
    U_new = np.empty_like(U)
    for s in range(0, n, block):
        Hb = None if heldout is None else heldout[s:s + block]
        K, W = _block_terms(Y[s:s + block], Hb, U_old32[s:s + block] @ V_old32.T)
        U_new[s:s + block] = _solve(W @ GV, K @ V_old32, lam, k)

    # M-step B: columns, SAME weights (regenerated from U_old, V_old), new U
    U_new32 = U_new.astype(np.float32)
    GU = _gram_rows(U_new32)
    V_new = np.empty_like(V)
    for s in range(0, m, block):
        Hb = None if heldout is None else heldout[:, s:s + block]
        K, W = _block_terms(Y[:, s:s + block], Hb, U_old32 @ V_old32[s:s + block].T)
        V_new[s:s + block] = _solve(W.T @ GU, K.T @ U_new32, lam, k)
    return U_new, V_new


def svd_init(Y, k, heldout=None, seed=0):
    """Truncated-SVD warm start of the logit proxy 4(Y - 1/2) (held-out set to 0)."""
    from sklearn.utils.extmath import randomized_svd
    Z = 4.0 * (Y.astype(np.float32) - 0.5)
    if heldout is not None:
        Z[heldout] = 0.0
    Us, s, Vt = randomized_svd(Z, k, n_iter=4, random_state=seed)
    r = np.sqrt(s)
    return (Us * r).astype(np.float64), (Vt.T * r).astype(np.float64)


def fit(Y, k, lam, heldout=None, U0=None, V0=None, max_iter=300, tol=1e-6, block=2048,
        seed=0, verbose=True, log=None):
    """PG-EM fit; returns (U, V, trace). Warm-start with U0/V0 (e.g. along a lambda path)."""
    if U0 is None:
        U, V = svd_init(Y, k, heldout, seed)
    else:
        U, V = U0.copy(), V0.copy()
    J, ll = objective(Y, U, V, lam, heldout, block)
    trace = dict(obj=[J], loglik=[ll], seconds=[0.0])
    t0 = time.time()
    for it in range(1, max_iter + 1):
        U, V = pgem_iteration(Y, U, V, lam, heldout, block)
        J_new, ll = objective(Y, U, V, lam, heldout, block)
        rel = (J_new - J) / max(1.0, abs(J))
        trace["obj"].append(J_new); trace["loglik"].append(ll); trace["seconds"].append(time.time() - t0)
        if rel < -1e-7:
            raise RuntimeError(f"PG-EM decreased J (rel {rel:.2e})")
        J = J_new
        if verbose and (it <= 2 or it % 10 == 0):
            msg = f"  k={k} lam={lam:g} it={it:3d} J={J:.6e} rel={rel:.2e} t={time.time() - t0:.0f}s"
            print(msg, flush=True)
            if log:
                print(msg, file=log, flush=True)
        if abs(rel) < tol:
            break
    trace["iterations"] = it
    trace["converged"] = abs(rel) < tol
    return U, V, trace


# ----------------------------------------------------------------------------
# Offset model:  Psi = a 1' + 1 b' + U V'
#   a: per-cell offset (sequencing depth / capture), b: per-gene offset (abundance)
#   u_i, v_j ~ N(0, I/lam);  a_i, b_j ~ N(0, 1/lam0) with lam0 small (near-flat)
# PG-EM row step solves for x_i = [u_i; a_i] with features f_j = [v_j; 1] and the
# known offset b_j moved to the right-hand side:
#   [diag(lam,...,lam, lam0) + sum_j w_ij f_j f_j'] x_i = sum_j (kappa_ij - w_ij b_j) f_j
# The column step is the same with x_j = [v_j; b_j], f_i = [u_i; 1], offset a_i.
# Weights are still frozen at the old snapshot for both blocks.
# ----------------------------------------------------------------------------
def _psi(U, V, a, b, rows=slice(None), cols=slice(None)):
    return (U[rows].astype(np.float32) @ V[cols].astype(np.float32).T
            + a[rows, None].astype(np.float32) + b[None, cols].astype(np.float32))


def _solve_aug(A_flat, rhs, lam, lam0, k1):
    reg = np.diag(np.r_[np.full(k1 - 1, lam), lam0])
    A = A_flat.reshape(-1, k1, k1).astype(np.float64) + reg
    return np.linalg.solve(A, rhs.astype(np.float64)[..., None])[..., 0]


def objective_offsets(Y, U, V, a, b, lam, lam0, heldout=None, block=2048):
    ll = 0.0
    for s in range(0, Y.shape[0], block):
        Psi = _psi(U, V, a, b, slice(s, s + block))
        term = Y[s:s + block] * Psi - softplus(Psi)
        if heldout is not None:
            term[heldout[s:s + block]] = 0.0
        ll += float(term.sum(dtype=np.float64))
    pen = 0.5 * lam * (np.sum(U * U) + np.sum(V * V)) + 0.5 * lam0 * (np.sum(a * a) + np.sum(b * b))
    return ll - pen, ll


def pgem_iteration_offsets(Y, U, V, a, b, lam, lam0, heldout=None, block=2048):
    n, m = Y.shape
    k1 = U.shape[1] + 1
    U0, V0, a0, b0 = U.astype(np.float32), V.astype(np.float32), a.astype(np.float32), b.astype(np.float32)

    # rows: x_i = [u_i, a_i], features [v_j, 1], offset b_j
    F = np.c_[V0, np.ones(m, np.float32)]
    GF = _gram_rows(F)
    X = np.empty((n, k1))
    for s in range(0, n, block):
        Hb = None if heldout is None else heldout[s:s + block]
        K, W = _block_terms(Y[s:s + block], Hb, U0[s:s + block] @ V0.T + a0[s:s + block, None] + b0[None, :])
        X[s:s + block] = _solve_aug(W @ GF, (K - W * b0[None, :]) @ F, lam, lam0, k1)
    U_new, a_new = X[:, :-1], X[:, -1]

    # columns: x_j = [v_j, b_j], features [u_i_new, 1], offset a_i_new; weights from old snapshot
    G = np.c_[U_new.astype(np.float32), np.ones(n, np.float32)]
    GG = _gram_rows(G)
    an = a_new.astype(np.float32)
    Xc = np.empty((m, k1))
    for s in range(0, m, block):
        Hb = None if heldout is None else heldout[:, s:s + block]
        K, W = _block_terms(Y[:, s:s + block], Hb, U0 @ V0[s:s + block].T + a0[:, None] + b0[None, s:s + block])
        Xc[s:s + block] = _solve_aug(W.T @ GG, (K - W * an[:, None]).T @ G, lam, lam0, k1)
    return U_new, Xc[:, :-1], a_new, Xc[:, -1]


def additive_init(Y, heldout=None, iters=15, block=2048):
    """Cell/gene offsets of the additive logistic model (alternating Newton), as a warm start."""
    T = np.ones(Y.shape, bool) if heldout is None else ~heldout
    rate = np.clip((Y * T).sum(0) / T.sum(0), 1e-4, 1 - 1e-4)
    a, b = np.zeros(Y.shape[0]), np.log(rate / (1 - rate))
    for _ in range(iters):
        for axis in (1, 0):
            num = np.zeros(Y.shape[0] if axis == 1 else Y.shape[1]); den = np.zeros_like(num)
            for s in range(0, Y.shape[0], block):
                p = 1 / (1 + np.exp(-(a[s:s + block, None] + b[None, :])))
                t = T[s:s + block]; r = (Y[s:s + block] - p) * t; w = p * (1 - p) * t
                if axis == 1:
                    num[s:s + block] = r.sum(1); den[s:s + block] = w.sum(1)
                else:
                    num += r.sum(0); den += w.sum(0)
            if axis == 1:
                a += num / np.maximum(den, 1e-9)
            else:
                b += num / np.maximum(den, 1e-9)
    return a, b


def fit_offsets(Y, k, lam, lam0=1e-3, heldout=None, init=None, max_iter=300, tol=1e-6, block=2048,
                seed=0, verbose=True, log=None):
    """PG-EM for Psi = a1' + 1b' + UV'. init = (U, V, a, b) to warm-start."""
    if init is None:
        a, b = additive_init(Y, heldout, block=block)
        rng = np.random.default_rng(seed)
        U, V = 0.1 * rng.standard_normal((Y.shape[0], k)), 0.1 * rng.standard_normal((Y.shape[1], k))
    else:
        U, V, a, b = (x.copy() for x in init)
    J, ll = objective_offsets(Y, U, V, a, b, lam, lam0, heldout, block)
    tr = dict(obj=[J], loglik=[ll])
    t0 = time.time()
    for it in range(1, max_iter + 1):
        U, V, a, b = pgem_iteration_offsets(Y, U, V, a, b, lam, lam0, heldout, block)
        J_new, ll = objective_offsets(Y, U, V, a, b, lam, lam0, heldout, block)
        rel = (J_new - J) / max(1.0, abs(J))
        tr["obj"].append(J_new); tr["loglik"].append(ll)
        if rel < -1e-7:
            raise RuntimeError(f"PG-EM decreased J (rel {rel:.2e})")
        J = J_new
        if verbose and (it <= 2 or it % 10 == 0):
            msg = f"  [offsets] k={k} lam={lam:g} it={it:3d} J={J:.6e} rel={rel:.2e} t={time.time() - t0:.0f}s"
            print(msg, flush=True)
            if log:
                print(msg, file=log, flush=True)
        if abs(rel) < tol:
            break
    tr["iterations"], tr["converged"] = it, abs(rel) < tol
    return U, V, a, b, tr


def heldout_scores_offsets(Y, U, V, a, b, heldout, block=2048):
    """Held-out scores for the offset model (reuses heldout_scores by folding offsets into factors)."""
    Ua = np.c_[U, a, np.ones_like(a)]
    Vb = np.c_[V, np.ones_like(b), b]
    return heldout_scores(Y, Ua, Vb, heldout, block)


# ----------------------------------------------------------------------------
# Complementary log-log link:  log(-log(1 - p_ij)) = eta_ij = a_i + b_j + u_i' v_j
#   equivalently p_ij = 1 - exp(-mu_ij), mu_ij = exp(eta_ij): y_ij = 1[z_ij >= 1] with
#   z_ij ~ Poisson(mu_ij), so cell depth multiplies the rate of every gene.
# Polya-Gamma augmentation is specific to the logistic link, so blocks are updated by
# damped Fisher scoring instead. The Bernoulli-cloglog log-likelihood is concave in eta,
# hence concave in each row block (u_i, a_i) given V, b. Row objectives are separable,
# so a per-row backtracking line search makes every block step -- and J -- monotone.
#   score    g = y mu/(e^mu - 1) - (1 - y) mu
#   Fisher   w = mu^2 / (e^mu - 1)
# ----------------------------------------------------------------------------
ETA_MIN, ETA_MAX = -30.0, 4.4      # mu <= 81 keeps expm1(mu) finite in float32; p(4.4) = 1 - 7e-36


def cll_terms(eta, y):
    """Log-likelihood, score and Fisher weight of Bernoulli(1 - exp(-exp(eta)))."""
    eta = np.clip(np.asarray(eta, np.float32), ETA_MIN, ETA_MAX)
    mu = np.exp(eta)
    em1 = np.expm1(mu)                                    # e^mu - 1 (>0)
    ll = np.where(y > 0, np.log(-np.expm1(-mu)), -mu)
    g = np.where(y > 0, mu / em1, 0.0) - np.where(y > 0, 0.0, mu)
    w = mu * mu / em1
    return ll, g, w


def _row_ll(Yb, Hb, eta):
    ll, _, _ = cll_terms(eta, Yb)
    if Hb is not None:
        ll = np.where(Hb, 0.0, ll)
    return ll.sum(1, dtype=np.float64)


def _scoring_block(Yb, Hb, X, F, off_b, reg, max_halve=6):
    """One damped Fisher-scoring step for rows X (n_b, k1) with features F (m, k1) and
    per-column offset off_b (m,). Objective per row: sum_j ll(x_i'f_j + off_j) - x' reg x / 2."""
    X32, F32, o32 = X.astype(np.float32), F.astype(np.float32), off_b.astype(np.float32)
    eta = X32 @ F32.T + o32[None, :]
    _, g, w = cll_terms(eta, Yb)
    if Hb is not None:
        g = np.where(Hb, 0.0, g); w = np.where(Hb, 0.0, w)
    k1 = F.shape[1]
    A = (w @ _gram_rows(F32)).astype(np.float64).reshape(-1, k1, k1) + np.diag(reg)
    grad = (g @ F32).astype(np.float64) - X * reg
    step = np.linalg.solve(A, grad[..., None])[..., 0]
    f0 = _row_ll(Yb, Hb, eta) - 0.5 * np.sum(X * X * reg, 1)
    t = np.ones(len(X))
    Xn = X.copy()
    todo = np.ones(len(X), bool)
    for _ in range(max_halve + 1):
        Xc = X[todo] + t[todo, None] * step[todo]
        fc = _row_ll(Yb[todo], None if Hb is None else Hb[todo], Xc.astype(np.float32) @ F32.T + o32[None, :]) \
            - 0.5 * np.sum(Xc * Xc * reg, 1)
        ok = fc >= f0[todo]
        idx = np.flatnonzero(todo)
        Xn[idx[ok]] = Xc[ok]
        todo[idx[ok]] = False
        t[todo] *= 0.5
        if not todo.any():
            break
    return Xn                                            # rows never improved keep X


def objective_cloglog(Y, U, V, a, b, lam, lam0, heldout=None, block=2048):
    ll = 0.0
    for s in range(0, Y.shape[0], block):
        eta = (U[s:s + block].astype(np.float32) @ V.astype(np.float32).T
               + a[s:s + block, None].astype(np.float32) + b[None, :].astype(np.float32))
        Hb = None if heldout is None else heldout[s:s + block]
        ll += float(_row_ll(Y[s:s + block], Hb, eta).sum())
    pen = 0.5 * lam * (np.sum(U * U) + np.sum(V * V)) + 0.5 * lam0 * (np.sum(a * a) + np.sum(b * b))
    return ll - pen, ll


def cloglog_iteration(Y, U, V, a, b, lam, lam0, heldout=None, block=2048):
    n, m = Y.shape
    k = U.shape[1]
    reg = np.r_[np.full(k, lam), lam0]
    F = np.c_[V, np.ones(m)]
    X = np.c_[U, a]
    for s in range(0, n, block):
        Hb = None if heldout is None else heldout[s:s + block]
        X[s:s + block] = _scoring_block(Y[s:s + block], Hb, X[s:s + block], F, b, reg)
    U, a = X[:, :k], X[:, k]
    G = np.c_[U, np.ones(n)]
    Xc = np.c_[V, b]
    for s in range(0, m, block):
        Hb = None if heldout is None else heldout[:, s:s + block].T
        Xc[s:s + block] = _scoring_block(np.ascontiguousarray(Y[:, s:s + block].T), Hb, Xc[s:s + block], G, a, reg)
    return U, Xc[:, :k], a, Xc[:, k]


def cloglog_init(Y, k, heldout=None, seed=0):
    """Offsets from the Poisson-threshold identity: a_i + b_j = log(-log(1 - p))."""
    T = np.ones(Y.shape, bool) if heldout is None else ~heldout
    rg = np.clip((Y * T).sum(0) / T.sum(0), 1e-4, 1 - 1e-4)
    rc = np.clip((Y * T).sum(1) / T.sum(1), 1e-4, 1 - 1e-4)
    ro = np.clip((Y * T).sum() / T.sum(), 1e-4, 1 - 1e-4)
    b = np.log(-np.log1p(-rg))
    a = np.log(-np.log1p(-rc)) - np.log(-np.log1p(-ro))
    rng = np.random.default_rng(seed)
    return 0.05 * rng.standard_normal((Y.shape[0], k)), 0.05 * rng.standard_normal((Y.shape[1], k)), a, b


def fit_cloglog(Y, k, lam, lam0=1e-3, heldout=None, init=None, max_iter=300, tol=1e-6, block=2048,
                seed=0, verbose=True, log=None):
    U, V, a, b = cloglog_init(Y, k, heldout, seed) if init is None else (x.copy() for x in init)
    J, ll = objective_cloglog(Y, U, V, a, b, lam, lam0, heldout, block)
    tr = dict(obj=[J], loglik=[ll])
    t0 = time.time()
    for it in range(1, max_iter + 1):
        U, V, a, b = cloglog_iteration(Y, U, V, a, b, lam, lam0, heldout, block)
        J_new, ll = objective_cloglog(Y, U, V, a, b, lam, lam0, heldout, block)
        rel = (J_new - J) / max(1.0, abs(J))
        tr["obj"].append(J_new); tr["loglik"].append(ll)
        if rel < -1e-9:
            raise RuntimeError(f"cloglog scoring decreased J (rel {rel:.2e})")
        J = J_new
        if verbose and (it <= 2 or it % 10 == 0):
            msg = f"  [cloglog] k={k} lam={lam:g} it={it:3d} J={J:.6e} rel={rel:.2e} t={time.time() - t0:.0f}s"
            print(msg, flush=True)
            if log:
                print(msg, file=log, flush=True)
        if abs(rel) < tol:
            break
    tr["iterations"], tr["converged"] = it, abs(rel) < tol
    return U, V, a, b, tr


def heldout_scores_cloglog(Y, U, V, a, b, heldout, block=2048, auc_sample=2_000_000, seed=0):
    from sklearn.metrics import roc_auc_score
    nll = brier = 0.0; cnt = 0; ys, ps = [], []
    for s in range(0, Y.shape[0], block):
        Hb = heldout[s:s + block]
        if not Hb.any():
            continue
        eta = (U[s:s + block] @ V.T + a[s:s + block, None] + b[None, :])[Hb]
        y = Y[s:s + block][Hb].astype(np.float64)
        ll, _, _ = cll_terms(eta, y)
        p = -np.expm1(-np.exp(np.clip(eta, ETA_MIN, ETA_MAX)))
        nll -= float(ll.sum()); brier += float(np.sum((p - y) ** 2)); cnt += y.size
        ys.append(y.astype(np.int8)); ps.append(p.astype(np.float32))
    y, p = np.concatenate(ys), np.concatenate(ps)
    if y.size > auc_sample:
        idx = np.random.default_rng(seed).choice(y.size, auc_sample, replace=False); y, p = y[idx], p[idx]
    return dict(nll=nll / cnt, brier=brier / cnt, auc=float(roc_auc_score(y, p)), n_heldout=cnt)
