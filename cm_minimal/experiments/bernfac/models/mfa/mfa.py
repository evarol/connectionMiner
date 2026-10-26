"""Bernoulli mixture of local Gaussian factor models (Model 1 of the handoff).

Generative model, for row i (a cell) and column j (a gene):

    c_i ~ Categorical(pi)
    z_i | c_i = l ~ N(0, I_d)
    psi_i | z_i, c_i = l ~ N(mu_l + B_l z_i, t_l I)      (t_l = 0 in residual_mode='none')
    y_ij | psi_ij ~ Bernoulli(sigmoid(psi_ij))

Inference is structured variational EM:
  * Every row/component pair (i, l) has its own Gaussian posterior q(z_i, psi_i | c_i = l)
    with moments m_il, S_il, and its own Polya-Gamma variational parameters xi_ilj, one per
    observed cell. The logistic likelihood is replaced by the Jaakkola-Jordan / PG quadratic
    lower bound, so q is exactly Gaussian. xi is set to sqrt(E_q[psi^2]) (posterior mean AND
    variance), never |E psi|.
  * Responsibilities r_il = softmax_l(log pi_l + l_il^bound), where l_il^bound is the
    analytically integrated quadratic bound (handoff eq. in section 3.5, including
    logdet A, log a and c(xi)).
  * Global parameters (pi, mu_l, B_l and optionally t_l) are MAP point estimates with
    Gaussian / Dirichlet / inverse-gamma priors.

The monitored objective is a VARIATIONAL LOWER BOUND
    B(Theta, xi) = sum_i logsumexp_l(log pi_l + l_il^bound) + log p(Theta),
not the exact marginal likelihood. Constants that do not depend on fitted parameters
(Gaussian/Dirichlet normalisers) are dropped, so bounds are only comparable within one
configuration.

Schedule (one "outer iteration" = one pass over the data)
---------------------------------------------------------
With stored xi, a pass computes q* = argmax_q L(q, xi, Theta) for every (i, l), the bound
B(Theta, xi), the responsibilities, and the frozen M-step sufficient statistics. Then:
  isotropic mode: the xi update (xi^2 = E_q[psi^2]) and the M-step are both taken from the
      same q. Given q, the lower bound L(q, xi, Theta) splits into a xi-part (the PG bound
      terms only involve psi and xi) plus a Theta-part (log N(psi; mu + B z, t) and priors),
      so the two updates are a valid simultaneous block-coordinate step.
  none mode: psi = mu + B z, so xi and Theta interact. The M-step uses the frozen xi; the xi
      update is then sqrt(E_q[(mu_new + B_new z)^2]), computed at the start of the next pass
      from the stored low-dimensional moments (m, S) and the new Theta. This makes xi an
      exact deterministic function of stored n x K x (d + d(d+1)/2) moments, so no n x m
      xi array is needed in this mode.
Every step is a coordinate ascent step of L, so B(Theta, xi) must not decrease (checked).
`local_sweeps > 1` adds extra xi <-> q alternations at fixed Theta inside a pass.

Scale
-----
Rows are processed in chunks; per chunk and component the O(c m) elementwise work runs in
fused numba kernels and the O(c m d^2) work in BLAS products (`big` dtype, float32 on the
full data). d x d algebra and every objective sum are float64. In isotropic mode xi is a
K x n x m array kept in RAM in `xi_dtype` (float16 on the full data to fit K = 16 in the
memory budget). The bound is always evaluated exactly at the stored (rounded) xi, so the
rounding only makes the xi step slightly less than optimal; it does not invalidate the bound.

Shared numerics (sigmoid, softplus, pg_mean) come from ../../bernfac.py.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bernfac import pg_mean, sigmoid, softplus  # noqa: E402

PG_SMALL = 1e-4  # same branch point as bernfac.pg_mean


# ----------------------------------------------------------------------------
# Shared scalar utilities (handoff section 2.3)
# ----------------------------------------------------------------------------
def log_sigmoid(x):
    return -softplus(-np.asarray(x, dtype=float))


def log_2cosh_half(x):
    """log(2 cosh(x/2)) = |x|/2 + log1p(exp(-|x|))."""
    a = np.abs(np.asarray(x, dtype=float))
    return a / 2.0 + np.log1p(np.exp(-a))


def bound_constant(xi):
    """c(xi) = -log(2 cosh(xi/2)) + xi^2 pg_mean(xi) / 2."""
    xi = np.asarray(xi, dtype=float)
    return -log_2cosh_half(xi) + 0.5 * pg_mean(xi) * xi * xi


def quad_bound(y, psi, xi):
    """Quadratic lower bound on log p(y | psi): kappa psi - w psi^2 / 2 + c(xi)."""
    kappa = np.asarray(y, float) - 0.5
    return kappa * psi - 0.5 * pg_mean(xi) * psi * psi + bound_constant(xi)


def bernoulli_loglik(y, psi):
    """Exact log p(y | psi) = y psi - softplus(psi)."""
    return np.asarray(y, float) * psi - softplus(psi)


def logsumexp(a, axis=-1):
    a = np.asarray(a, dtype=float)
    mx = np.max(a, axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    out = np.log(np.sum(np.exp(a - mx), axis=axis, keepdims=True)) + mx
    return np.squeeze(out, axis=axis)


def gauss_hermite(Q):
    """Nodes x and normalised weights w (sum 1) so that E[f(X)], X ~ N(a, s^2), is
    approximated by sum_q w_q f(a + sqrt(2 s^2) x_q). Golub-Welsch on the Jacobi matrix
    with off-diagonal sqrt((1..Q-1)/2) (weight function exp(-x^2))."""
    off = np.sqrt(np.arange(1, Q) / 2.0)
    J = np.diag(off, 1) + np.diag(off, -1)
    x, V = np.linalg.eigh(J)
    w = V[0] ** 2
    return x, w / w.sum()


def expected_sigmoid(a, s2, Q=20):
    """E[sigmoid(X)], X ~ N(a, s2), by Gauss-Hermite (exact sigmoid(a) when s2 == 0)."""
    a, s2 = np.broadcast_arrays(np.asarray(a, float), np.asarray(s2, float))
    x, w = gauss_hermite(Q)
    sd = np.sqrt(2.0 * np.maximum(s2, 0.0))
    out = np.sum(w * sigmoid(a[..., None] + sd[..., None] * x), axis=-1)
    return np.where(s2 > 0, out, sigmoid(a))  # zero variance handled exactly


# ----------------------------------------------------------------------------
# Half-vectorisation helpers for symmetric d x d matrices
# ----------------------------------------------------------------------------
def tri(d):
    """Upper-triangle index pairs (incl. diagonal) and quadratic-form multiplicities."""
    i0, i1 = np.triu_indices(d)
    mult = np.where(i0 == i1, 1.0, 2.0)
    return i0, i1, mult


def vech(S):
    d = S.shape[-1]
    i0, i1, _ = tri(d)
    return S[..., i0, i1]


def unvech(v, d):
    i0, i1, _ = tri(d)
    out = np.zeros(v.shape[:-1] + (d, d), dtype=np.float64)
    out[..., i0, i1] = v
    out[..., i1, i0] = v
    return out


def bb_sym(B):
    """(m, p) rows vech(b_j b_j') (no doubling)."""
    i0, i1, _ = tri(B.shape[-1])
    return B[..., i0] * B[..., i1]


# ----------------------------------------------------------------------------
# Numba kernels (fused O(c m) elementwise work)
# ----------------------------------------------------------------------------
@njit(inline="always")
def _pgm(ax):
    if ax < PG_SMALL:
        s2 = ax * ax
        return 0.25 - s2 / 48.0 + s2 * s2 / 480.0
    return np.tanh(ax / 2.0) / (2.0 * ax)


@njit(parallel=True, cache=True)
def _k_local_terms(Y, X, XI, mu, t, WA, RA, IA, const):
    """Per observed cell: w = f(xi), a = 1 + t w; WA = w/a, RA = (kappa - w mu)/a, IA = 1/a.
    const_i = sum_j [c(xi) - log(a)/2 + (kappa mu - w mu^2/2 + t kappa^2/2)/a].
    Excluded cells get zeros and contribute nothing."""
    c, m = Y.shape
    for i in prange(c):
        acc = 0.0
        for j in range(m):
            if X[i, j]:
                WA[i, j] = 0.0
                RA[i, j] = 0.0
                IA[i, j] = 0.0
                continue
            xi = np.float64(XI[i, j])
            ax = abs(xi)
            w = _pgm(ax)
            a = 1.0 + t * w
            ia = 1.0 / a
            kap = np.float64(Y[i, j]) - 0.5
            mj = np.float64(mu[j])
            WA[i, j] = w * ia
            RA[i, j] = (kap - w * mj) * ia
            IA[i, j] = ia
            cxi = -(ax / 2.0 + np.log1p(np.exp(-ax))) + 0.5 * w * ax * ax
            acc += cxi - 0.5 * np.log(a) + (kap * mj - 0.5 * w * mj * mj + 0.5 * t * kap * kap) * ia
        const[i] = acc


@njit(parallel=True, fastmath=True, cache=True)
def _k_local_terms32(Y, X, XI, mu, t, WA, RA, IA, const):
    """float32 version of _k_local_terms (used when big_dtype='float32'): the same formulas
    in single precision with expm1 to avoid cancellation; per-row sums accumulate in float64."""
    c, m = Y.shape
    t32 = np.float32(t)
    one = np.float32(1.0)
    half = np.float32(0.5)
    for i in prange(c):
        acc = 0.0
        for j in range(m):
            ob = not X[i, j]
            ax = abs(np.float32(XI[i, j]))
            em = np.expm1(-ax)                      # e^-ax - 1  (<= 0)
            if ax < np.float32(1e-3):
                s2 = ax * ax
                w = np.float32(0.25) - s2 / np.float32(48.0) + s2 * s2 / np.float32(480.0)
            else:
                w = -em / ((np.float32(2.0) + em) * np.float32(2.0) * ax)
            tw = t32 * w
            ia = one / (one + tw)
            kap = np.float32(Y[i, j]) - half
            mj = np.float32(mu[j])
            WA[i, j] = w * ia if ob else np.float32(0.0)
            RA[i, j] = (kap - w * mj) * ia if ob else np.float32(0.0)
            IA[i, j] = ia if ob else np.float32(0.0)
            term = (-(half * ax + np.log1p(em + one)) + half * w * ax * ax - half * np.log1p(tw)
                    + (kap * mj - half * w * mj * mj + half * t32 * kap * kap) * ia)
            if ob:
                acc += np.float64(term)
        const[i] = acc


@njit(parallel=True, cache=True)
def _k_post_iso(Y, X, BM, Q, IA, mu, t, XI, PB, E2):
    """Observed-cell posterior moments under q(z, psi | c = l) and the xi update.
    psibar = (mu + b'm + t kappa)/a,  v = t/a + b'Sb/a^2,  xi = sqrt(psibar^2 + v)."""
    c, m = Y.shape
    for i in prange(c):
        for j in range(m):
            if X[i, j]:
                XI[i, j] = 0.0
                PB[i, j] = 0.0
                E2[i, j] = 0.0
                continue
            ia = np.float64(IA[i, j])
            kap = np.float64(Y[i, j]) - 0.5
            pb = (np.float64(mu[j]) + np.float64(BM[i, j]) + t * kap) * ia
            q = max(np.float64(Q[i, j]), 0.0)
            e2 = pb * pb + t * ia + q * ia * ia
            XI[i, j] = np.sqrt(e2)
            PB[i, j] = pb
            E2[i, j] = e2


@njit(parallel=True, cache=True)
def _k_xi_none(X, BM, Q, mu, XI):
    """t = 0: xi = sqrt(E[(mu + b'z)^2]) = sqrt((mu + b'm)^2 + b'Sb)."""
    c, m = X.shape
    for i in prange(c):
        for j in range(m):
            if X[i, j]:
                XI[i, j] = 0.0
                continue
            e = np.float64(mu[j]) + np.float64(BM[i, j])
            XI[i, j] = np.sqrt(e * e + max(np.float64(Q[i, j]), 0.0))


@njit(inline="always")
def _sig(x):
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    e = np.exp(x)
    return e / (1.0 + e)


@njit(parallel=True, cache=True)
def _k_predict_pairs(rows, cols, mu, B, t, r, mloc, Sv, i0, i1, mult, xq, wq, P, P1):
    """Missing-cell predictive: sum_l r_il E[sigmoid(X)], X ~ N(mu_lj + b_lj'm_il,
    t_l + b_lj' S_il b_lj), by Gauss-Hermite. P1 = 1 - P computed separately (no cancellation)."""
    N = rows.shape[0]
    K, _, d = B.shape
    p = i0.shape[0]
    Qn = xq.shape[0]
    for idx in prange(N):
        i = rows[idx]
        j = cols[idx]
        sp = 0.0
        sq = 0.0
        for l in range(K):
            rl = r[i, l]
            if rl == 0.0:
                continue
            a = mu[l, j]
            for k in range(d):
                a += B[l, j, k] * mloc[i, l, k]
            s2 = t[l]
            for k in range(p):
                s2 += mult[k] * Sv[i, l, k] * B[l, j, i0[k]] * B[l, j, i1[k]]
            if s2 <= 0.0:
                sp += rl * _sig(a)
                sq += rl * _sig(-a)
            else:
                sd = np.sqrt(2.0 * s2)
                ep = 0.0
                eq = 0.0
                for q in range(Qn):
                    x = a + sd * xq[q]
                    ep += wq[q] * _sig(x)
                    eq += wq[q] * _sig(-x)
                sp += rl * ep
                sq += rl * eq
        P[idx] = sp
        P1[idx] = sq


@njit(parallel=True, cache=True)
def _k_pair_moments(rows, cols, mu, B, t, mloc, Sv, i0, i1, mult, A, S2):
    """Per-component missing-cell mean and variance (for tests / diagnostics)."""
    N = rows.shape[0]
    K, _, d = B.shape
    p = i0.shape[0]
    for idx in prange(N):
        i = rows[idx]
        j = cols[idx]
        for l in range(K):
            a = mu[l, j]
            for k in range(d):
                a += B[l, j, k] * mloc[i, l, k]
            s2 = t[l]
            for k in range(p):
                s2 += mult[k] * Sv[i, l, k] * B[l, j, i0[k]] * B[l, j, i1[k]]
            A[idx, l] = a
            S2[idx, l] = s2


# ----------------------------------------------------------------------------
# Data handling
# ----------------------------------------------------------------------------
def prepare_data(Y, mask=None, excluded=None, allow_empty=False):
    """Return (Y uint8, X bool) where X marks cells that do NOT enter training.

    Y may be dense numeric/boolean (NaN = missing) or scipy.sparse (implicit zeros are
    OBSERVED zeros unless the mask says otherwise). `mask` (True = observed) and
    `excluded` (True = held out) are alternatives; both may be given.
    """
    try:
        import scipy.sparse as sp
        if sp.issparse(Y):
            Y = Y.toarray()
    except ImportError:  # pragma: no cover
        pass
    Y = np.asarray(Y)
    if Y.ndim != 2:
        raise ValueError("Y must be a 2-D matrix")
    if Y.dtype == np.uint8 or Y.dtype == bool:
        X = np.zeros(Y.shape, bool)
        Yu = Y.view(np.uint8) if Y.dtype == bool else Y
    else:
        Yf = Y.astype(float)
        X = np.isnan(Yf)  # NaN = missing; +-inf is rejected below if included
        Yu = None
    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != Y.shape:
            raise ValueError("mask shape must match Y")
        X = X | ~mask
    if excluded is not None:
        excluded = np.asarray(excluded, bool)
        if excluded.shape != Y.shape:
            raise ValueError("excluded shape must match Y")
        X = X | excluded if X.any() else excluded
    if Yu is None:
        inc = Yf[~X]
        if np.any(~np.isfinite(inc)) or np.any((inc != 0) & (inc != 1)):
            raise ValueError("every included entry of Y must be exactly 0 or 1")
        Yu = np.where(X, 0, np.nan_to_num(Yf, nan=0.0)).astype(np.uint8)
    else:
        if Yu.size and Yu.max() > 1:
            raise ValueError("every included entry of Y must be exactly 0 or 1")
    if X.all() and not allow_empty:
        raise ValueError("no training observations: every entry is excluded or missing")
    return Yu, X


# ----------------------------------------------------------------------------
# Model and options
# ----------------------------------------------------------------------------
@dataclass
class MFAOptions:
    residual_mode: str = "isotropic"      # 'isotropic' or 'none'
    residual_variance: float = 0.25       # starting / fixed t_l (isotropic)
    learn_residual_variance: bool = False
    a0: float = 2.0                       # inverse-gamma shape (learned t only)
    b0: float = 0.75                      # inverse-gamma scale
    lambda_mu: float = 1.0
    lambda_B: float = 1.0
    alpha: float = 1.1
    max_iter: int = 200
    tol: float = 1e-6                     # relative bound change for convergence
    local_sweeps: int = 1                 # xi <-> q alternations per pass
    seed: int = 1
    init: str = "random"                  # 'random' (handoff 3.9) or 'kmeans'
    init_mean_noise: float = 0.5
    init_loading_scale: float = 0.1       # loadings ~ scale/sqrt(d) * N(0,1) for 'random'
    quadrature_order: int = 20
    big_dtype: str = "float64"            # dtype of the O(c m) products ('float32' at scale)
    xi_dtype: str = "float64"             # storage dtype of xi (isotropic)
    chunk_budget_mb: float = 300.0
    mono_tol: float = 1e-9                # allowed relative bound decrease before flagging
    px: bool = False                      # parameter-expansion step after each M-step (see px_step)
    verbose: bool = False

    def validate(self):
        if self.residual_mode not in ("isotropic", "none"):
            raise ValueError("residual_mode must be 'isotropic' or 'none'")
        if self.residual_mode == "isotropic" and not self.residual_variance > 0:
            raise ValueError("isotropic mode needs a positive residual variance")
        if self.residual_mode == "none" and self.learn_residual_variance:
            raise ValueError("no residual variance to learn in residual_mode='none'")
        if not (self.lambda_mu > 0 and self.lambda_B > 0):
            raise ValueError("lambda_mu and lambda_B must be positive precisions")
        if not self.alpha > 1:
            raise ValueError("alpha must be > 1")
        if not (self.a0 > 0 and self.b0 > 0):
            raise ValueError("a0, b0 must be positive")
        if self.local_sweeps < 1:
            raise ValueError("local_sweeps >= 1")


@dataclass
class MFAModel:
    pi: np.ndarray        # (K,)
    mu: np.ndarray        # (K, m)
    B: np.ndarray         # (K, m, d)
    t: np.ndarray         # (K,) residual variances (0 in mode 'none')
    opts: MFAOptions = field(default_factory=MFAOptions)

    @property
    def K(self):
        return self.mu.shape[0]

    @property
    def m(self):
        return self.mu.shape[1]

    @property
    def d(self):
        return self.B.shape[2]

    def log_prior(self):
        o = self.opts
        lp = -0.5 * o.lambda_mu * float(np.sum(self.mu ** 2)) - 0.5 * o.lambda_B * float(np.sum(self.B ** 2))
        lp += (o.alpha - 1.0) * float(np.sum(np.log(self.pi)))
        if o.residual_mode == "isotropic" and o.learn_residual_variance:
            lp += float(np.sum(-(o.a0 + 1.0) * np.log(self.t) - o.b0 / self.t))
        return lp

    def save(self, path):
        np.savez(path, pi=self.pi, mu=self.mu, B=self.B, t=self.t,
                 opts=json.dumps(asdict(self.opts)))

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        return MFAModel(z["pi"].astype(float), z["mu"].astype(float), z["B"].astype(float),
                        z["t"].astype(float), MFAOptions(**json.loads(str(z["opts"]))))


@dataclass
class LocalState:
    r: np.ndarray        # (n, K) responsibilities
    m: np.ndarray        # (n, K, d) coordinate means
    Sv: np.ndarray       # (n, K, p) vech of coordinate covariances
    score: np.ndarray    # (n, K) l_il^bound
    row_bound: np.ndarray  # (n,) logsumexp_l(log pi + score)
    active: np.ndarray   # (n,) row has >= 1 training observation

    def S(self, rows=slice(None)):
        return unvech(self.Sv[rows], self.m.shape[-1])

    def save(self, path, dtype=np.float32):
        np.savez(path, r=self.r.astype(dtype), m=self.m.astype(dtype), Sv=self.Sv.astype(dtype),
                 score=self.score, row_bound=self.row_bound, active=self.active)


def _empty_state(n, K, d):
    p = d * (d + 1) // 2
    Sv = np.zeros((n, K, p))
    Sv[..., np.flatnonzero(tri(d)[0] == tri(d)[1])] = 1.0
    return LocalState(np.full((n, K), 1.0 / K), np.zeros((n, K, d)), Sv, np.zeros((n, K)),
                      np.zeros(n), np.ones(n, bool))


# ----------------------------------------------------------------------------
# Small per-row reference implementation (readable; used by tests and docs)
# ----------------------------------------------------------------------------
def local_row_reference(y, obs, mu, B, t, xi):
    """One row/component local Gaussian update (handoff section 3.4-3.5) in plain numpy.

    y, obs, mu, xi: (m,); B: (m, d). Uses only observed columns. Returns dict with m, S,
    psibar, vpsi (observed-cell posterior), Ezpsi (d, |O|), score (l^bound), A, h.
    """
    o = np.asarray(obs, bool)
    muo, Bo, xio = mu[o], B[o], xi[o]
    kap = y[o] - 0.5
    w = pg_mean(xio)
    a = 1.0 + t * w
    d = B.shape[1]
    A = np.eye(d) + Bo.T @ (Bo * (w / a)[:, None])
    h = Bo.T @ ((kap - w * muo) / a)
    L = np.linalg.cholesky(A)
    mz = np.linalg.solve(A, h)
    S = np.linalg.solve(A, np.eye(d))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    psibar = (muo + Bo @ mz + t * kap) / a
    vpsi = t / a + np.einsum("jp,pq,jq->j", Bo, S, Bo) / a ** 2
    Ezpsi = mz[:, None] * psibar[None, :] + (S @ Bo.T) / a[None, :]
    score = (np.sum(bound_constant(xio) - 0.5 * np.log(a)
                    + (kap * muo - 0.5 * w * muo ** 2 + 0.5 * t * kap ** 2) / a)
             - 0.5 * logdet + 0.5 * h @ mz)
    return dict(m=mz, S=S, psibar=psibar, vpsi=vpsi, Ezpsi=Ezpsi, score=float(score), A=A, h=h,
                w=w, a=a)


# ----------------------------------------------------------------------------
# Chunked local computations
# ----------------------------------------------------------------------------
class _Comp:
    """Per-component global arrays in the `big` dtype, built once per pass."""

    def __init__(self, model, l, big):
        self.mu = model.mu[l].astype(big)
        self.B = np.ascontiguousarray(model.B[l].astype(big))
        bs = bb_sym(model.B[l])
        self.BBs = np.ascontiguousarray(bs.astype(big))
        self.BBq = np.ascontiguousarray((bs * tri(model.d)[2]).astype(big))
        self.t = float(model.t[l])


def _gauss_from_stats(Avech, h, const, d):
    """A = I + unvech(Avech); m = A^{-1} h, S = A^{-1} (solves), score."""
    A = unvech(Avech.astype(np.float64), d) + np.eye(d)
    L = np.linalg.cholesky(A)
    hh = h.astype(np.float64)
    mz = np.linalg.solve(A, hh[..., None])[..., 0]
    S = np.linalg.solve(A, np.broadcast_to(np.eye(d), A.shape))
    S = 0.5 * (S + np.swapaxes(S, -1, -2))
    logdet = 2.0 * np.sum(np.log(np.diagonal(L, axis1=-2, axis2=-1)), -1)
    score = const - 0.5 * logdet + 0.5 * np.sum(hh * mz, -1)
    return mz, vech(S), score


def _component_local(Yc, Xc, xi, comp, d, big, sweeps, none_mode):
    """Run `sweeps` q-updates for one component on one row chunk, updating xi in place
    between them (coordinate ascent at fixed globals). Returns the moments at the final xi
    plus IA (=1/a) and WA (=w/a) at that xi."""
    c, m = Yc.shape
    WA = np.empty((c, m), big)
    RA = np.empty((c, m), big)
    IA = np.empty((c, m), big)
    const = np.empty(c)
    for s in range(sweeps):
        (_k_local_terms32 if big == np.float32 else _k_local_terms)(Yc, Xc, xi, comp.mu, comp.t, WA, RA, IA, const)
        Av = WA @ comp.BBs
        h = RA @ comp.B
        mz, Sv, score = _gauss_from_stats(Av, h, const, d)
        if s < sweeps - 1:
            BM = mz.astype(big) @ comp.B.T
            Q = Sv.astype(big) @ comp.BBq.T
            if none_mode:
                _k_xi_none(Xc, BM, Q, comp.mu, xi)
            else:
                PB = np.empty((c, m), big)
                E2 = np.empty((c, m), big)
                _k_post_iso(Yc, Xc, BM, Q, IA, comp.mu, comp.t, xi, PB, E2)
    return mz, Sv, score, IA, WA


def _chunk_rows(n, m, K, budget_mb, itemsize):
    c = int(budget_mb * 1e6 / max(1, (K + 8) * m * itemsize))
    return int(np.clip(c, 16, 4096))


def memory_estimate(n, m, K, d, opts):
    """Rough peak memory (GB) of a fit, excluding Y and the exclusion mask."""
    p, p1 = d * (d + 1) // 2, (d + 1) * (d + 2) // 2
    xi_b = np.dtype(opts.xi_dtype).itemsize
    big_b = np.dtype(opts.big_dtype).itemsize
    est = {}
    est["xi"] = K * n * m * xi_b if opts.residual_mode == "isotropic" else 0
    est["local_moments"] = n * K * (d + p + 2) * 8
    est["mstep_stats"] = K * m * (p1 + p + d + 2) * 8
    c = _chunk_rows(n, m, K, opts.chunk_budget_mb, big_b)
    est["chunk_work"] = (K + 10) * c * m * big_b
    est["params"] = K * m * (d + 1) * 8 * 3
    gb = {k: v / 1e9 for k, v in est.items()}
    gb["total"] = sum(gb.values())
    gb["chunk_rows"] = c
    return gb


# ----------------------------------------------------------------------------
# One pass: E-step (+ optional sufficient statistics and xi update)
# ----------------------------------------------------------------------------
def e_pass(Y, X, model, state, xi_store=None, accumulate=True, sweeps=1, rows=None, nobs=None):
    """Compute q* at the current (Theta, xi) for all rows, the bound, responsibilities,
    and (if accumulate) the frozen M-step statistics; in isotropic mode also write the
    xi update from the same q into xi_store.

    Returns (row_bound_sum, stats). `state` is updated in place.
    """
    o = model.opts
    big = np.dtype(o.big_dtype)
    n, m = Y.shape
    K, d = model.K, model.d
    p, p1 = d * (d + 1) // 2, (d + 1) * (d + 2) // 2
    none_mode = o.residual_mode == "none"
    comps = [_Comp(model, l, big) for l in range(K)]
    logpi = np.log(model.pi)
    hi0, hi1, _ = tri(d + 1)
    st = None
    if accumulate:
        st = dict(N=np.zeros(K), Neff=np.zeros(K), G=np.zeros((K, m, p1)),
                  zsum=np.zeros((K, d)), zz=np.zeros((K, d, d)),
                  gz=np.zeros((K, m, d + 1)))
        if not none_mode:
            st["gS"] = np.zeros((K, m, p))
            st["v"] = np.zeros((K, m))
        st["n_active"] = 0
    c = _chunk_rows(n, m, K, o.chunk_budget_mb, big.itemsize)
    total = 0.0
    for s in range(0, n, c):
        e = min(n, s + c)
        Yc, Xc = Y[s:e], X[s:e]
        cn = e - s
        nob = (~Xc).sum(1)
        act = nob > 0
        keep = np.empty((K, cn, m), big)
        mloc = np.empty((cn, K, d))
        Svl = np.empty((cn, K, p))
        sc = np.empty((cn, K))
        for l, comp in enumerate(comps):
            if none_mode:
                xi = np.empty((cn, m), big)
                BM = state.m[s:e, l].astype(big) @ comp.B.T
                Q = state.Sv[s:e, l].astype(big) @ comp.BBq.T
                _k_xi_none(Xc, BM, Q, comp.mu, xi)
                del BM, Q
            else:
                xi = xi_store[l, s:e].astype(big)
            mz, Sv, score, IA, WA = _component_local(Yc, Xc, xi, comp, d, big, sweeps, none_mode)
            if sweeps > 1 and not none_mode:
                xi_store[l, s:e] = xi  # extra local sweeps changed xi; keep it consistent
            mloc[:, l], Svl[:, l], sc[:, l] = mz, Sv, score
            keep[l] = WA if none_mode else IA
            del xi, IA, WA
        lp = logpi[None, :] + sc
        lse = logsumexp(lp, axis=1)
        r = np.exp(lp - lse[:, None])
        r[~act] = model.pi
        lse[~act] = 0.0
        total += float(lse[act].sum())
        state.r[s:e], state.m[s:e], state.Sv[s:e], state.score[s:e] = r, mloc, Svl, sc
        state.row_bound[s:e], state.active[s:e] = lse, act
        if not accumulate:
            continue
        ra = np.where(act[:, None], r, 0.0)
        st["n_active"] += int(act.sum())
        Mf = (~Xc).astype(big) if not none_mode else None
        Kc = ((Yc.astype(big) - 0.5) * (~Xc)) if none_mode else None
        for l, comp in enumerate(comps):
            rl = ra[:, l]
            ml = mloc[:, l]
            H = np.empty((cn, d + 1, d + 1))
            H[:, 0, 0] = 1.0
            H[:, 0, 1:] = ml
            H[:, 1:, 0] = ml
            H[:, 1:, 1:] = unvech(Svl[:, l], d) + ml[:, :, None] * ml[:, None, :]
            rH = (rl[:, None] * H[:, hi0, hi1]).astype(big)
            rz = (rl[:, None] * np.c_[np.ones(cn), ml]).astype(big)
            st["N"][l] += rl.sum()
            st["zsum"][l] += rl @ ml
            st["zz"][l] += np.einsum("i,ipq->pq", rl, H[:, 1:, 1:])
            st["Neff"][l] += float(rl @ nob)
            if none_mode:
                st["G"][l] += keep[l].T @ rH          # sum_i r w H over observed cells
                st["gz"][l] += Kc.T @ rz              # sum_i r kappa [1; m]
            else:
                BM = ml.astype(big) @ comp.B.T
                Q = Svl[:, l].astype(big) @ comp.BBq.T
                xin = np.empty((cn, m), big)
                PB = np.empty((cn, m), big)
                E2 = np.empty((cn, m), big)
                _k_post_iso(Yc, Xc, BM, Q, keep[l], comp.mu, comp.t, xin, PB, E2)
                xi_store[l, s:e] = xin                # xi update from this q (valid, see top)
                st["G"][l] += Mf.T @ rH
                st["gz"][l] += PB.T @ rz
                st["gS"][l] += keep[l].T @ (rl[:, None] * Svl[:, l]).astype(big)
                st["v"][l] += E2.T @ rl.astype(big)
                del BM, Q, xin, PB, E2
        del keep
    return total, st


# ----------------------------------------------------------------------------
# M-step
# ----------------------------------------------------------------------------
def px_step(model, state, st, min_mass=1.0):
    """Optional parameter-expansion (PX) step, taken after the M-step with q still frozen.

    Under q, fit the coordinate prior of component l as N(eta_l, Gamma_l) (responsibility-
    weighted mean and covariance of the coordinates), then map back to N(0, I) by
    z' = L^-1 (z - eta), B' = B L, mu' = mu + B eta (Gamma = L L'). This changes (q, Theta)
    jointly without changing any logit distribution, so all likelihood and PG-bound terms
    are unchanged; the coordinate-prior-plus-entropy term rises by
        N_l / 2 [tr(M2) - d - log det Gamma] >= 0,
    and the Gaussian priors on mu, B change by an exactly computed amount. The step is
    accepted per component only if the total change is positive, so it is a valid ascent
    step on the same variational bound (Liu, Rubin & Wu 1998 PX-EM idea, applied to the
    frozen variational objective). Not part of the handoff baseline; enabled by opts.px.
    Returns the list of accepted gains."""
    o = model.opts
    d = model.d
    gains = []
    for l in range(model.K):
        N = st["N"][l]
        if N < min_mass:
            gains.append(0.0)
            continue
        m1 = st["zsum"][l] / N
        M2 = st["zz"][l] / N
        Gam = M2 - np.outer(m1, m1)
        Gam = 0.5 * (Gam + Gam.T)
        try:
            L = np.linalg.cholesky(Gam)
        except np.linalg.LinAlgError:
            gains.append(0.0)
            continue
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        gain = 0.5 * N * (np.trace(M2) - d - logdet)
        Bn = model.B[l] @ L
        mun = model.mu[l] + model.B[l] @ m1
        dprior = (-0.5 * o.lambda_B * (np.sum(Bn ** 2) - np.sum(model.B[l] ** 2))
                  - 0.5 * o.lambda_mu * (np.sum(mun ** 2) - np.sum(model.mu[l] ** 2)))
        if gain + dprior <= 0:
            gains.append(0.0)
            continue
        model.B[l], model.mu[l] = Bn, mun
        Li = np.linalg.solve(L, np.eye(d))
        state.m[:, l] = (state.m[:, l] - m1) @ Li.T
        S = unvech(state.Sv[:, l], d)
        state.Sv[:, l] = vech(Li @ S @ Li.T)
        gains.append(float(gain + dprior))
    return gains


def m_step(model, st):
    """Exact conditional maximisation of the frozen bound over (mu, B), then pi, then t."""
    o = model.opts
    K, d = model.K, model.d
    Lam = np.diag(np.r_[o.lambda_mu, np.full(d, o.lambda_B)])
    info = dict(sse=[], sse_min=None)
    for l in range(K):
        G = unvech(st["G"][l], d + 1)
        if o.residual_mode == "none":
            theta = np.linalg.solve(G + Lam, st["gz"][l][..., None])[..., 0]
        else:
            g = st["gz"][l].copy()
            g[:, 1:] += np.einsum("jpq,jq->jp", unvech(st["gS"][l], d), model.B[l])  # S b_old / a
            tl = model.t[l]
            theta = np.linalg.solve(G + tl * Lam, g[..., None])[..., 0]
            if o.learn_residual_variance:
                sse = float(np.sum(st["v"][l]) - 2.0 * np.sum(theta * g)
                            + np.einsum("jp,jpq,jq->", theta, G, theta))
                scale = max(1.0, float(np.sum(st["v"][l])))
                if sse < -1e-6 * scale:
                    raise RuntimeError(f"negative SSE {sse:.3e} in component {l}: bug")
                sse = max(sse, 0.0)
                info["sse"].append(sse)
                model.t[l] = (o.b0 + 0.5 * sse) / (o.a0 + 1.0 + 0.5 * st["Neff"][l])
        model.mu[l] = theta[:, 0]
        model.B[l] = theta[:, 1:]
    na = st["n_active"]
    model.pi = (st["N"] + o.alpha - 1.0) / (na + K * (o.alpha - 1.0))
    model.pi = model.pi / model.pi.sum()
    return info


# ----------------------------------------------------------------------------
# Initialisation
# ----------------------------------------------------------------------------
def _col_logit(Y, X, rows=None):
    Yr = Y if rows is None else Y[rows]
    Xr = X if rows is None else X[rows]
    ones = np.zeros(Y.shape[1])
    tot = np.zeros(Y.shape[1])
    for s in range(0, Yr.shape[0], 2048):
        ob = ~Xr[s:s + 2048]
        ones += (Yr[s:s + 2048] * ob).sum(0)
        tot += ob.sum(0)
    zeros = tot - ones
    return np.log((ones + 0.5) / (zeros + 0.5))


def init_model(Y, X, K, d, opts, rng, pcs=None):
    """'random': smoothed column logit + N(0, init_mean_noise^2) per component, loadings
    init_loading_scale/sqrt(d) * N(0, 1) (handoff 3.9).
    'kmeans': k-means on training-data principal components (pcs, n x q, computed from
    training cells only) gives a hard partition; mu_l = smoothed column logit of the
    training cells of cluster l, B_l = top-d right singular vectors of that cluster's
    centred logit proxy 4(y - rate) (excluded cells 0), scaled to the proxy's per-row
    standard deviation along each direction and halved. Initialisation only; it does not
    change the model or the prior."""
    n, m = Y.shape
    base = _col_logit(Y, X)
    if opts.init == "random":
        noise = opts.init_mean_noise if K > 1 else 0.0
        mu = base[None, :] + noise * rng.standard_normal((K, m))
        B = opts.init_loading_scale / np.sqrt(d) * rng.standard_normal((K, m, d))
    elif opts.init == "kmeans":
        from sklearn.cluster import KMeans
        from sklearn.utils.extmath import randomized_svd
        if K > 1:
            if pcs is None:
                pcs = training_pcs(Y, X, 30, int(rng.integers(1 << 31)))
            lab = KMeans(K, n_init=3, random_state=int(rng.integers(1 << 31))).fit_predict(pcs)
        else:
            lab = np.zeros(n, int)
        mu = np.empty((K, m))
        B = np.empty((K, m, d))
        for l in range(K):
            idx = np.flatnonzero(lab == l)
            if idx.size < max(5, d + 1):
                mu[l] = base + opts.init_mean_noise * rng.standard_normal(m)
                B[l] = opts.init_loading_scale / np.sqrt(d) * rng.standard_normal((m, d))
                continue
            lg = _col_logit(Y, X, idx)
            mu[l] = lg + 0.05 * rng.standard_normal(m)
            rate = 1.0 / (1.0 + np.exp(-lg))
            Z = 4.0 * (Y[idx].astype(np.float32) - rate.astype(np.float32))
            Z[X[idx]] = 0.0
            _, sv, Vt = randomized_svd(Z, d, n_iter=3, random_state=int(rng.integers(1 << 31)))
            B[l] = 0.5 * Vt.T * (sv / np.sqrt(idx.size))[None, :]
            del Z
    else:
        raise ValueError(f"unknown init {opts.init!r}")
    pi = np.full(K, 1.0 / K)
    t = np.full(K, opts.residual_variance if opts.residual_mode == "isotropic" else 0.0)
    return MFAModel(pi, mu, B, t, opts)


def training_pcs(Y, X, q=30, seed=0, block=2048):
    """Top-q principal-component scores of the column-centred training matrix
    (excluded cells set to 0 after centring). Used only for the k-means initialisation."""
    from sklearn.utils.extmath import randomized_svd
    n, m = Y.shape
    rate = 1.0 / (1.0 + np.exp(-_col_logit(Y, X)))
    Z = np.empty((n, m), np.float32)
    for s in range(0, n, block):
        z = Y[s:s + block].astype(np.float32) - rate.astype(np.float32)
        z[X[s:s + block]] = 0.0
        Z[s:s + block] = z
    U, sv, _ = randomized_svd(Z, q, n_iter=4, random_state=seed)
    del Z
    return (U * sv).astype(np.float64)


def init_xi(model, n, dtype):
    """xi_ilj^2 = mu_lj^2 + ||b_lj||^2 + t_l (prior logit second moment), same for all rows."""
    K, m = model.K, model.m
    xi = np.empty((K, n, m), dtype)
    for l in range(K):
        row = np.sqrt(model.mu[l] ** 2 + np.sum(model.B[l] ** 2, 1) + model.t[l]).astype(dtype)
        xi[l] = row[None, :]
    return xi


# ----------------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------------
def fit(Y, K, d, mask=None, excluded=None, opts=None, model0=None, pcs=None, log=None,
        callback=None, **kw):
    """Fit the Bernoulli MFA by structured variational EM.

    Returns (model, state, info). `state` holds responsibilities, coordinate means and
    covariances computed at the returned globals and xi (consistent snapshot).
    """
    opts = opts or MFAOptions()
    for k_, v_ in kw.items():
        setattr(opts, k_, v_)
    opts.validate()
    Yu, X = prepare_data(Y, mask, excluded)
    n, m = Yu.shape
    rng = np.random.default_rng(opts.seed)
    t_init = time.time()
    model = model0 if model0 is not None else init_model(Yu, X, K, d, opts, rng, pcs)
    model.opts = opts
    state = _empty_state(n, K, d)
    xi_store = None
    if opts.residual_mode == "isotropic":
        xi_store = init_xi(model, n, np.dtype(opts.xi_dtype))
    info = dict(bound=[], loglik_bound=[], log_prior=[], rel=[], occupancy=[], t=[],
                param_change=[], seconds=[], n=n, m=m, K=K, d=d, n_observed=int((~X).sum()),
                memory_gb=memory_estimate(n, m, K, d, opts), init_seconds=time.time() - t_init)
    t0 = time.time()
    B_prev = None
    converged = False
    max_dec = 0.0
    for it in range(opts.max_iter + 1):
        lb, st = e_pass(Yu, X, model, state, xi_store, accumulate=True, sweeps=opts.local_sweeps)
        lpri = model.log_prior()
        Bk = lb + lpri
        info["bound"].append(Bk); info["loglik_bound"].append(lb); info["log_prior"].append(lpri)
        info["occupancy"].append((st["N"] / max(1, st["n_active"])).tolist())
        info["t"].append(model.t.tolist())
        info["seconds"].append(time.time() - t0)
        if callback is not None:
            callback(it, model, state, info)
        if B_prev is not None:
            rel = (Bk - B_prev) / max(1.0, abs(B_prev))
            info["rel"].append(rel)
            max_dec = max(max_dec, -rel)
            msg = f"  it={it:3d} B={Bk:.8e} rel={rel:.2e} t={time.time() - t0:.0f}s occ={np.round(st['N'] / st['n_active'], 3).tolist() if model.K <= 8 else ''}"
            if opts.verbose:
                print(msg, flush=True)
            if log is not None:
                print(msg, file=log, flush=True)
            if abs(rel) < opts.tol:
                converged = True
                break
        elif log is not None:
            print(f"  it=  0 B={Bk:.8e} t={time.time() - t0:.0f}s", file=log, flush=True)
        if it == opts.max_iter:
            break
        old = (model.mu.copy(), model.B.copy())
        m_step(model, st)
        if opts.px:
            info.setdefault("px_gain", []).append(float(np.sum(px_step(model, state, st))))
        info["param_change"].append(float(max(np.abs(model.mu - old[0]).max(),
                                               np.abs(model.B - old[1]).max())))
        B_prev = Bk
        del st
    info.update(converged=converged, iterations=it, max_rel_decrease=max_dec,
                monotone=bool(max_dec <= opts.mono_tol), fit_seconds=time.time() - t0)
    del xi_store
    return model, state, info


def fit_restarts(Y, K, d, n_restarts=3, mask=None, excluded=None, opts=None, **kw):
    """Several seeds; keep the restart with the largest final TRAINING bound."""
    opts = opts or MFAOptions()
    best = None
    finals = []
    for r in range(n_restarts):
        o = MFAOptions(**{**asdict(opts), **kw, "seed": opts.seed + r})
        res = fit(Y, K, d, mask=mask, excluded=excluded, opts=o)
        finals.append(res[2]["bound"][-1])
        if best is None or finals[-1] > best[2]["bound"][-1]:
            best = res
    best[2]["restart_bounds"] = finals
    return best


# ----------------------------------------------------------------------------
# New-row inference (globals frozen)
# ----------------------------------------------------------------------------
def infer_rows(model, Ynew, mask=None, excluded=None, max_sweeps=20, tol=1e-7):
    """Local variational inference for new rows given their available observations.

    Global parameters are not modified. xi starts at the prior second moment and
    alternates with q until every row/component score changes by < tol (relative) or
    max_sweeps is reached. Fully unobserved rows return r = pi, m = 0, S = I.
    """
    o = model.opts
    big = np.dtype(o.big_dtype)
    Yu, X = prepare_data(Ynew, mask, excluded, allow_empty=True)
    n, m = Yu.shape
    K, d = model.K, model.d
    state = _empty_state(n, K, d)
    none_mode = o.residual_mode == "none"
    comps = [_Comp(model, l, big) for l in range(K)]
    c = _chunk_rows(n, m, K, o.chunk_budget_mb, big.itemsize)
    sweeps_used = 0
    for s in range(0, n, c):
        e = min(n, s + c)
        Yc, Xc = Yu[s:e], X[s:e]
        act = (~Xc).sum(1) > 0
        for l, comp in enumerate(comps):
            xi = np.broadcast_to(np.sqrt(comp.mu.astype(float) ** 2 + np.sum(model.B[l] ** 2, 1)
                                         + comp.t), (e - s, m)).astype(big)
            prev = None
            for sw in range(max_sweeps):
                mz, Sv, score, IA, WA = _component_local(Yc, Xc, xi, comp, d, big, 1, none_mode)
                if prev is not None and np.all(np.abs(score - prev) <= tol * (1 + np.abs(score))):
                    break
                prev = score
                BM = mz.astype(big) @ comp.B.T
                Q = Sv.astype(big) @ comp.BBq.T
                if none_mode:
                    _k_xi_none(Xc, BM, Q, comp.mu, xi)
                else:
                    PB = np.empty_like(xi); E2 = np.empty_like(xi)
                    _k_post_iso(Yc, Xc, BM, Q, IA, comp.mu, comp.t, xi, PB, E2)
            sweeps_used = max(sweeps_used, sw + 1)
            state.m[s:e, l], state.Sv[s:e, l], state.score[s:e, l] = mz, Sv, score
        lp = np.log(model.pi)[None, :] + state.score[s:e]
        lse = logsumexp(lp, axis=1)
        r = np.exp(lp - lse[:, None])
        r[~act] = model.pi
        lse[~act] = 0.0
        state.m[s:e][~act] = 0.0
        state.r[s:e], state.row_bound[s:e], state.active[s:e] = r, lse, act
    return state, dict(sweeps=sweeps_used)


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def predict_pairs(model, state, rows, cols, Q=None, return_complement=False):
    """Missing-cell predictive probabilities for explicit (row, col) pairs.

    For each pair: sum_l r_il E[sigmoid(X)], X ~ N(mu_lj + b_lj' m_il, t_l + b_lj' S_il b_lj),
    by Gauss-Hermite quadrature (Q nodes, default model.opts.quadrature_order). This is the
    correct predictive for cells that were NOT used in training; for training cells it
    ignores the cell's own observation (use local_row_reference for the observed-cell
    posterior psibar, vpsi)."""
    Q = Q or model.opts.quadrature_order
    xq, wq = gauss_hermite(Q)
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    i0, i1, mult = tri(model.d)
    P = np.empty(rows.size)
    P1 = np.empty(rows.size)
    _k_predict_pairs(rows, cols, np.ascontiguousarray(model.mu, float), np.ascontiguousarray(model.B, float),
                     np.asarray(model.t, float), np.ascontiguousarray(state.r, float),
                     np.ascontiguousarray(state.m, float), np.ascontiguousarray(state.Sv, float),
                     i0.astype(np.int64), i1.astype(np.int64), mult, xq, wq, P, P1)
    return (P, P1) if return_complement else P


def pair_moments(model, state, rows, cols):
    """Per-component missing-cell predictive mean and variance, (N, K) each."""
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    i0, i1, mult = tri(model.d)
    A = np.empty((rows.size, model.K))
    S2 = np.empty((rows.size, model.K))
    _k_pair_moments(rows, cols, np.ascontiguousarray(model.mu, float), np.ascontiguousarray(model.B, float),
                    np.asarray(model.t, float), np.ascontiguousarray(state.m, float),
                    np.ascontiguousarray(state.Sv, float), i0.astype(np.int64), i1.astype(np.int64), mult, A, S2)
    return A, S2


def predict_rows(model, state, rows, Q=None):
    """Missing-cell predictive probabilities for whole rows (len(rows) x m)."""
    rows = np.asarray(rows, dtype=np.int64)
    m = model.m
    rr = np.repeat(rows, m)
    cc = np.tile(np.arange(m), rows.size)
    return predict_pairs(model, state, rr, cc, Q).reshape(rows.size, m)


def score_cells(Y, split, model, state, Q=None, block=1024, auc_sample=2_000_000, seed=0,
                n_cal_bins=10):
    """Held-out Bernoulli NLL / Brier / AUC (2 M subsample, same convention as
    bernfac_large.heldout_scores) and 10-bin calibration on cells where split is True.
    Log probabilities are taken from P and 1 - P computed separately; values below 1e-300
    are clipped (count reported)."""
    from sklearn.metrics import roc_auc_score
    nll = brier = 0.0
    cnt = 0
    clipped = 0
    ys, ps = [], []
    for s in range(0, Y.shape[0], block):
        Hb = np.asarray(split[s:s + block])
        if not Hb.any():
            continue
        ri, ci = np.nonzero(Hb)
        P, P1 = predict_pairs(model, state, ri + s, ci, Q, return_complement=True)
        y = np.asarray(Y[s:s + block])[ri, ci].astype(np.float64)
        lp = np.where(y > 0, P, P1)
        clipped += int(np.sum(lp < 1e-300))
        nll -= float(np.sum(np.log(np.maximum(lp, 1e-300))))
        brier += float(np.sum((P - y) ** 2))
        cnt += y.size
        ys.append(y.astype(np.int8)); ps.append(P.astype(np.float32))
    y, p = np.concatenate(ys), np.concatenate(ps)
    edges = np.linspace(0, 1, n_cal_bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, n_cal_bins - 1)
    cal = dict(count=np.bincount(b, minlength=n_cal_bins).tolist(),
               mean_p=(np.bincount(b, p, n_cal_bins) / np.maximum(1, np.bincount(b, minlength=n_cal_bins))).tolist(),
               mean_y=(np.bincount(b, y, n_cal_bins) / np.maximum(1, np.bincount(b, minlength=n_cal_bins))).tolist())
    rng = np.random.default_rng(seed)
    if y.size > auc_sample:
        idx = rng.choice(y.size, auc_sample, replace=False)
        ya, pa = y[idx], p[idx]
    else:
        ya, pa = y, p
    auc = float(roc_auc_score(ya, pa)) if 0 < ya.sum() < ya.size else float("nan")
    return dict(nll=nll / cnt, brier=brier / cnt, auc=auc, n_cells=cnt, n_clipped=clipped,
                calibration=cal)


# ----------------------------------------------------------------------------
# Rotation-invariant cell representation
# ----------------------------------------------------------------------------
def logit_mean_embedding(model, state, n_components=50):
    """Embedding of E-hat[psi_i] = sum_l r_il (mu_l + B_l m_il), centred over cells, projected
    on its top principal directions.

    The n x m matrix is never formed: it equals F G' with F = [r, r_il m_il] (n x K(d+1))
    and G = [mu; B] (m x K(d+1)), so the centred SVD is exact (thin QR of G, then an SVD of
    a n x K(d+1) matrix). Returns (U_k s_k, s, explained fraction); at most K(d+1) columns.
    """
    n, K, d = state.m.shape
    F = np.concatenate([state.r, (state.r[:, :, None] * state.m).reshape(n, K * d)], 1)
    G = np.concatenate([model.mu.T, np.transpose(model.B, (1, 0, 2)).reshape(model.m, K * d)], 1)
    Fc = F - F.mean(0, keepdims=True)
    Qg, Rg = np.linalg.qr(G)
    U, s, _ = np.linalg.svd(Fc @ Rg.T, full_matrices=False)
    k = min(n_components, int(np.sum(s > 1e-10 * max(s[0], 1e-300))))
    tot = float(np.sum(s ** 2))
    return (U[:, :k] * s[:k]).astype(np.float32), s, float(np.sum(s[:k] ** 2) / max(tot, 1e-300))
