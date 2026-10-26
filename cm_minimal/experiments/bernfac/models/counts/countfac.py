"""Low-rank Poisson and negative-binomial factorization of a count matrix.

Model (both families)
    log mu_ij = o_i + a_i + b_j + u_i'v_j
    o_i = log s_i, a fixed depth offset (s_i = total UMI of cell i)
    u_i, v_j ~ N(0, I/lam);  a_i, b_j ~ N(0, 1/lam0), lam0 = 1e-3

    Poisson   x_ij ~ Poisson(mu_ij)
    NB2       x_ij ~ NB(mean mu_ij, var mu_ij + mu_ij^2/phi_j), gene-wise dispersion phi_j

Only train entries enter the likelihood. The held-out mask is applied by passing
X with held-out counts set to 0 and a 0/1 float mask T (T = 0 on held-out entries).

Fitting is MAP by alternating block Fisher scoring, as in bernfac_large.fit_cloglog.
With eta = log mu and L = log1p(mu/phi), the eta-dependent part of the
per-entry log-likelihood is
    Poisson   h = x eta - mu
    NB2       h = x (eta - L) - phi L
with score dh/deta and Fisher weight
    Poisson   g = x - mu                 w = mu
    NB2       g = (x - mu)/(1 + mu/phi)  w = mu/(1 + mu/phi)
The rest of the log-likelihood depends on phi only and is nonzero only where x > 0:
    Poisson   c = -lgamma(x+1)
    NB2       c = lgamma(x+phi) - lgamma(phi) - x log(phi) - lgamma(x+1)
Both h are concave in eta (log link), so each row problem [u_i; a_i] given (V, b)
and each column problem [v_j; b_j] given (U, a) is concave. Rows are independent,
so a per-row backtracking line search makes each block step, and the penalized
objective J, monotone. For NB, phi_j is updated after each factor sweep by a few
safeguarded Newton steps on log(phi_j) with mu fixed; a step is kept only if it
raises gene j's log-likelihood, so J stays monotone.

The weighted Gram products sum_j w_ij f_j f_j' run in float32; every log-likelihood
value (line searches, objective, dispersion updates) is computed in float64. No n x m matrix is
ever saved.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.special import digamma, gammaln, polygamma

ETA_MIN, ETA_MAX = -40.0, 15.0
LOGPHI_MIN, LOGPHI_MAX = np.log(1e-3), np.log(1e6)


def _gram_rows(F):
    """(m, k*k) matrix whose row j is vec(f_j f_j')."""
    m, k = F.shape
    return (F[:, :, None] * F[:, None, :]).reshape(m, k * k)


# ----------------------------------------------------------------------------- per-entry terms
def entry_terms(eta, x, t, phi=None, need_grad=True):
    """h, g, w for one dense block (dtype follows eta); phi broadcastable or None.
    t is the 0/1 train mask; x is already 0 where t == 0."""
    eta = np.clip(eta, ETA_MIN, ETA_MAX)
    mu = np.exp(eta)
    if phi is None:
        h = x * eta - t * mu
        if not need_grad:
            return h, None, None
        return h, x - t * mu, t * mu
    L = np.log1p(mu / phi)
    h = x * (eta - L) - t * phi * L
    if not need_grad:
        return h, None, None
    r = 1.0 / (1.0 + mu / phi)
    return h, (x - t * mu) * r, t * mu * r


def _eta(X, F, off_c, off_r=None):
    """Linear predictor in float64. All log-likelihood values (line searches, objective, phi
    updates) are evaluated in float64 so that monotonicity checks are not swamped by float32
    rounding; only the big weighted Gram products run in float32."""
    e = np.asarray(X, np.float64) @ np.asarray(F, np.float64).T + np.asarray(off_c, np.float64)[None, :]
    if off_r is not None:
        e += np.asarray(off_r, np.float64)[:, None]
    return e


def scoring_block(Xb, Tb, X, F, off_c, reg, off_r=None, phi=None, max_halve=8):
    """One damped Fisher-scoring step for the rows of X (n_b, k1) with features F (m, k1).
    eta_ij = X_i'F_j + off_c_j (+ off_r_i). Row objective f_i = sum_j h_ij - x_i' diag(reg) x_i / 2.
    Returns (new X, f at the returned X)."""
    F32 = F.astype(np.float32)
    eta = _eta(X, F, off_c, off_r)
    h, g, w = entry_terms(eta, Xb, Tb, phi)
    del eta
    k1 = F.shape[1]
    A = (w.astype(np.float32) @ _gram_rows(F32)).astype(np.float64).reshape(-1, k1, k1) + np.diag(reg)
    grad = (g.astype(np.float32) @ F32).astype(np.float64) - X * reg
    del g, w
    step = np.linalg.solve(A, grad[..., None])[..., 0]
    f0 = h.sum(1, dtype=np.float64) - 0.5 * np.sum(X * X * reg, 1)
    del h
    Xn, fn = X.copy(), f0.copy()
    todo = np.ones(len(X), bool)
    t = 1.0
    for _ in range(max_halve + 1):
        idx = np.flatnonzero(todo)
        Xc = X[idx] + t * step[idx]
        ph = None if phi is None else (phi if np.ndim(phi) == 0 or phi.shape[0] == 1 else phi[idx])
        e = _eta(Xc, F, off_c, None if off_r is None else off_r[idx])
        hc, _, _ = entry_terms(e, Xb[idx], Tb[idx], ph, need_grad=False)
        fc = hc.sum(1, dtype=np.float64) - 0.5 * np.sum(Xc * Xc * reg, 1)
        ok = fc >= f0[idx]
        Xn[idx[ok]] = Xc[ok]
        fn[idx[ok]] = fc[ok]
        todo[idx[ok]] = False
        t *= 0.5
        if not todo.any():
            break
    return Xn, fn                                        # rows never improved keep X


# ----------------------------------------------------------------------------- data container
class CountData:
    """Dense float32 train counts X (held-out = 0), float32 train mask T, depth offset o, and
    the train nonzeros compressed to unique (gene, count) pairs with multiplicities, which is
    all the phi-only terms need."""

    def __init__(self, X, heldout=None, depth="train", s_total=None):
        X = np.asarray(X, np.float32)
        self.n, self.m = X.shape
        self.T = np.ones(X.shape, np.float32) if heldout is None else (~heldout).astype(np.float32)
        self.X = X * self.T
        if depth == "train":
            s = self.X.sum(1, dtype=np.float64)
        elif depth == "total":
            s = X.sum(1, dtype=np.float64) if s_total is None else np.asarray(s_total, np.float64)
        elif depth == "none":
            s = np.ones(self.n)
        else:
            raise ValueError(depth)
        del X
        self.o = np.log(np.maximum(s, 1.0))
        self.n_train = float(self.T.sum(dtype=np.float64))
        jj, ii = np.nonzero(self.X.T)                     # genes-major order
        key = jj.astype(np.int64) * (1 << 32) + self.X.T[jj, ii].astype(np.int64)
        del jj, ii
        key, cnt = np.unique(key, return_counts=True)
        self.nz_col = (key >> 32).astype(np.int64)
        self.nz_x = (key & ((1 << 32) - 1)).astype(np.float64)
        self.nz_w = cnt.astype(np.float64)
        self.lgx1 = float(np.sum(self.nz_w * gammaln(self.nz_x + 1.0)))


def const_terms(D, phi):
    """sum over train entries of the phi-only part c (Poisson: phi None)."""
    if phi is None:
        return -D.lgx1
    p = phi[D.nz_col].astype(np.float64)
    x = D.nz_x
    return float(np.sum(D.nz_w * (gammaln(x + p) - gammaln(p) - x * np.log(p)))) - D.lgx1


def penalty(U, V, a, b, lam, lam0):
    return 0.5 * lam * (np.sum(U * U) + np.sum(V * V)) + 0.5 * lam0 * (np.sum(a * a) + np.sum(b * b))


def objective(D, U, V, a, b, lam, lam0, phi=None, block=1024):
    """(J, train log-likelihood). J = loglik - penalty."""
    F = np.c_[V, np.ones(D.m)]
    ph = None if phi is None else phi[None, :]
    ll = 0.0
    for s in range(0, D.n, block):
        eta = _eta(np.c_[U[s:s + block], a[s:s + block]], F, b, D.o[s:s + block])
        h, _, _ = entry_terms(eta, D.X[s:s + block], D.T[s:s + block], ph, need_grad=False)
        ll += float(h.sum(dtype=np.float64))
    ll += const_terms(D, phi)
    return ll - penalty(U, V, a, b, lam, lam0), ll


def factor_iteration(D, U, V, a, b, lam, lam0, phi=None, block=1024):
    """Row sweep (u_i, a_i) then column sweep (v_j, b_j). Returns new factors and the
    eta-dependent log-likelihood sum at the new factors (from the column sweep)."""
    n, m, k = D.n, D.m, U.shape[1]
    reg = np.r_[np.full(k, lam), lam0]
    F = np.c_[V, np.ones(m)]
    X = np.c_[U, a]
    ph_r = None if phi is None else phi[None, :]
    for s in range(0, n, block):
        X[s:s + block], _ = scoring_block(D.X[s:s + block], D.T[s:s + block], X[s:s + block], F, b, reg,
                                          off_r=D.o[s:s + block], phi=ph_r)
    U, a = X[:, :k].copy(), X[:, k].copy()
    G = np.c_[U, np.ones(n)]
    Xc = np.c_[V, b]
    hsum = 0.0
    for s in range(0, m, block):
        ph_c = None if phi is None else phi[s:s + block][:, None]
        Xb = np.ascontiguousarray(D.X[:, s:s + block].T)
        Tb = np.ascontiguousarray(D.T[:, s:s + block].T)
        Xc[s:s + block], f = scoring_block(Xb, Tb, Xc[s:s + block], G, a + D.o, reg, phi=ph_c)
        hsum += float(f.sum()) + 0.5 * float(np.sum(Xc[s:s + block] ** 2 * reg))
    return U, Xc[:, :k].copy(), a, Xc[:, k].copy(), hsum


# ----------------------------------------------------------------------------- dispersion
def _gene_ll_dense(mu, x, phi):
    """Per-gene sum over train entries of the phi-dependent part of h: -x L - phi L, L = log1p(mu/phi).
    mu and x are 0 at held-out entries, so those entries contribute 0."""
    L = np.log1p(mu / phi[:, None])
    return -((x + phi[:, None]) * L).sum(1)


def _gene_ll_sparse(rows, xs, ws, phi, nb):
    p = phi[rows]
    return np.bincount(rows, ws * (gammaln(xs + p) - gammaln(p) - xs * np.log(p)), minlength=nb)


def _phi_derivs(mu, x, phi, rows, xs, ws, nb):
    """First and second derivative in phi_j of gene j's train log-likelihood (mu fixed;
    mu = x = 0 at held-out entries)."""
    pr = phi[rows]
    d1s = np.bincount(rows, ws * (digamma(xs + pr) - digamma(pr)), minlength=nb)
    d2s = np.bincount(rows, ws * (polygamma(1, xs + pr) - polygamma(1, pr)), minlength=nb)
    q = phi[:, None] + mu
    L = np.log1p(mu / phi[:, None])
    r = (mu - x) / q
    d1 = d1s - L.sum(1) + r.sum(1)
    d2 = d2s + (mu / (phi[:, None] * q)).sum(1) - (r / q).sum(1)
    return d1, d2


def update_phi(D, U, V, a, b, phi, block=1024, newton_steps=2, max_halve=5):
    """Profile update of phi_j with mu fixed: safeguarded Newton on theta = log phi_j.
    A proposed phi_j is kept only if it raises gene j's log-likelihood.
    Returns (new phi, total log-likelihood gain >= 0)."""
    phi = phi.astype(np.float64).copy()
    G = np.c_[U, np.ones(D.n)]
    off = a + D.o
    starts = np.searchsorted(D.nz_col, np.arange(0, D.m + 1))
    gain = 0.0
    for s in range(0, D.m, block):
        e = min(s + block, D.m)
        nb = e - s
        Xb = np.ascontiguousarray(D.X[:, s:e].T).astype(np.float64)
        mu = np.exp(np.clip(np.c_[V[s:e], b[s:e]] @ G.T + off[None, :], ETA_MIN, ETA_MAX))
        mu *= D.T[:, s:e].T
        lo, hi = starts[s], starts[e]
        rows, xs, ws = D.nz_col[lo:hi] - s, D.nz_x[lo:hi], D.nz_w[lo:hi]

        def ll(p, sel):
            """gene log-likelihoods (phi part) for the genes in index array sel."""
            inv = np.full(nb, -1); inv[sel] = np.arange(len(sel))
            pm = inv[rows] >= 0
            return (_gene_ll_dense(mu[sel], Xb[sel], p[sel])
                    + _gene_ll_sparse(inv[rows[pm]], xs[pm], ws[pm], p[sel], len(sel)))

        ph = phi[s:e].copy()
        allg = np.arange(nb)
        f0 = ll(ph, allg)
        f_start = f0.sum()
        for _ in range(newton_steps):
            d1, d2 = _phi_derivs(mu, Xb, ph, rows, xs, ws, nb)
            gth = ph * d1                                   # d/dtheta
            hth = ph * ph * d2 + ph * d1                    # d2/dtheta2
            stp = np.where(hth < 0, -gth / np.minimum(hth, -1e-12), np.sign(gth))
            stp = np.clip(stp, -2.0, 2.0)
            th = np.log(ph)
            todo = np.flatnonzero(np.abs(stp) > 1e-8)
            tt = 1.0
            for _h in range(max_halve + 1):
                if todo.size == 0:
                    break
                cand = ph.copy()
                cand[todo] = np.exp(np.clip(th[todo] + tt * stp[todo], LOGPHI_MIN, LOGPHI_MAX))
                fc = ll(cand, todo)
                ok = fc > f0[todo]
                ph[todo[ok]] = cand[todo[ok]]; f0[todo[ok]] = fc[ok]
                todo = todo[~ok]
                tt *= 0.5
        phi[s:e] = ph
        gain += float(f0.sum() - f_start)
    return phi, gain


def phi_init(D, value=10.0):
    return np.full(D.m, value)


# ----------------------------------------------------------------------------- fit
def init_offsets(D, k, seed=0, scale=0.05):
    """a = 0, b_j = log(sum_i x_ij / sum_i t_ij s_i) (Poisson MLE of b with a = 0), small random U, V."""
    num = D.X.sum(0, dtype=np.float64) + 0.5
    den = (D.T * np.exp(D.o)[:, None].astype(np.float32)).sum(0, dtype=np.float64)
    b = np.log(num / den)
    rng = np.random.default_rng(seed)
    return (scale * rng.standard_normal((D.n, k)), scale * rng.standard_normal((D.m, k)), np.zeros(D.n), b)


def fit(D, k, lam, family="poisson", lam0=1e-3, init=None, phi=None, max_iter=250, tol=1e-6, block=1024,
        seed=0, verbose=True, log=None, phi_steps=4):
    """MAP fit. family 'poisson' or 'nb'. init = (U, V, a, b); phi = starting dispersions (NB).
    Returns U, V, a, b, phi (None for Poisson), trace."""
    U, V, a, b = init_offsets(D, k, seed) if init is None else (np.array(x, np.float64) for x in init)
    nb = family == "nb"
    if nb:
        phi = phi_init(D) if phi is None else np.asarray(phi, np.float64).copy()
    else:
        phi = None
    J, ll = objective(D, U, V, a, b, lam, lam0, phi, block)
    tr = dict(obj=[J], loglik=[ll], seconds=[0.0])
    t0 = time.time()
    rel = np.inf
    for it in range(1, max_iter + 1):
        J_prev = J
        U, V, a, b, hsum = factor_iteration(D, U, V, a, b, lam, lam0, phi, block)
        J_f = hsum + const_terms(D, phi) - penalty(U, V, a, b, lam, lam0)
        if (J_f - J_prev) / max(1.0, abs(J_prev)) < -1e-9:
            raise RuntimeError(f"factor sweep decreased J ({J_prev:.8e} -> {J_f:.8e})")
        if nb:
            phi, gain = update_phi(D, U, V, a, b, phi, newton_steps=phi_steps)
            if gain < 0:
                raise RuntimeError(f"phi update decreased J (gain {gain:.3e})")
            J = J_f + gain
            ll = J + penalty(U, V, a, b, lam, lam0)
        else:
            J, ll = J_f, J_f + penalty(U, V, a, b, lam, lam0)
        rel = (J - J_prev) / max(1.0, abs(J_prev))
        tr["obj"].append(J); tr["loglik"].append(ll); tr["seconds"].append(time.time() - t0)
        if verbose and (it <= 2 or it % 5 == 0):
            extra = "" if phi is None else f" phi_med={np.median(phi):.3g}"
            msg = f"  [{family}] k={k} lam={lam:g} it={it:3d} J={J:.8e} rel={rel:.2e}{extra} t={time.time() - t0:.0f}s"
            print(msg, flush=True)
            if log:
                print(msg, file=log, flush=True)
        if abs(rel) < tol:
            break
    tr["iterations"], tr["converged"] = it, bool(abs(rel) < tol)
    return U, V, a, b, phi, tr


# ----------------------------------------------------------------------------- scoring
def prob_detect(mu, phi=None):
    """P(x >= 1) and log P(x = 0)."""
    log_p0 = -mu if phi is None else -phi * np.log1p(mu / phi)
    return -np.expm1(log_p0), log_p0


def count_loglik(x, mu, phi=None):
    if phi is None:
        return x * np.log(mu) - mu - gammaln(x + 1)
    return (gammaln(x + phi) - gammaln(phi) - gammaln(x + 1) + phi * np.log(phi / (phi + mu))
            + x * np.log(mu / (phi + mu)))


def heldout_scores(Y, Xcounts, o, U, V, a, b, phi, masks, block=1024, auc_sample=2_000_000, seed=0,
                   n_bins=10):
    """Bernoulli NLL / Brier / AUC (2M subsample) / 10-bin calibration on each mask (dict name -> bool n x m),
    plus held-out count log-likelihood per entry. Matches bernfac_large.heldout_scores_cloglog."""
    from sklearn.metrics import roc_auc_score
    acc = {k: dict(nll=0.0, brier=0.0, cll=0.0, cnt=0, ys=[], ps=[]) for k in masks}
    for s in range(0, Y.shape[0], block):
        eta_full = U[s:s + block] @ V.T + a[s:s + block, None] + b[None, :] + o[s:s + block, None]
        for name, M in masks.items():
            Hb = M[s:s + block]
            if not Hb.any():
                continue
            eta = np.clip(eta_full[Hb], ETA_MIN, ETA_MAX)
            mu = np.exp(eta)
            ph = None if phi is None else np.broadcast_to(phi[None, :], Hb.shape)[Hb]
            y = Y[s:s + block][Hb].astype(np.float64)
            xb = Xcounts[s:s + block]
            xb = xb.toarray() if hasattr(xb, "toarray") else xb
            x = xb[Hb].astype(np.float64)
            p, log_p0 = prob_detect(mu, ph)
            log_p1 = np.log(np.maximum(p, 1e-300))
            r = acc[name]
            r["nll"] -= float(np.sum(y * log_p1 + (1 - y) * log_p0))
            r["brier"] += float(np.sum((p - y) ** 2))
            r["cll"] += float(np.sum(count_loglik(x, mu, ph)))
            r["cnt"] += y.size
            r["ys"].append(y.astype(np.int8)); r["ps"].append(p.astype(np.float32))
    out, cal = {}, []
    edges = np.linspace(0, 1, n_bins + 1)
    for name, r in acc.items():
        y, p = np.concatenate(r["ys"]), np.concatenate(r["ps"])
        idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        for bi in range(n_bins):
            sel = idx == bi
            if sel.any():
                cal.append(dict(split=name, bin=bi, mean_pred=float(p[sel].mean()), frac_pos=float(y[sel].mean()),
                                count=int(sel.sum())))
        if y.size > auc_sample:
            ii = np.random.default_rng(seed).choice(y.size, auc_sample, replace=False); y, p = y[ii], p[ii]
        out[name] = dict(nll=r["nll"] / r["cnt"], brier=r["brier"] / r["cnt"], auc=float(roc_auc_score(y, p)),
                         count_ll=r["cll"] / r["cnt"], n=r["cnt"])
    return out, cal


def embedding(U, V):
    """E = A S from the thin SVD U V' = A S B' (via QR of U and V; n x m never formed)."""
    Qu, Ru = np.linalg.qr(U)
    Qv, Rv = np.linalg.qr(V)
    W, s, _ = np.linalg.svd(Ru @ Rv.T)
    E = Qu @ (W * s)
    sgn = np.sign(E[np.abs(E).argmax(0), np.arange(E.shape[1])])
    return E * sgn, s
