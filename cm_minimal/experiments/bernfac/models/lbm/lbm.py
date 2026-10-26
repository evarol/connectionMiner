"""Bernoulli latent block model (LBM), handoff section 5. Python port of the MATLAB spec.

Model
-----
    c_i ~ Cat(pi)   (K_r row groups),      d_j ~ Cat(rho)  (K_c column groups)
    Y_ij | c_i, d_j, B ~ Bernoulli(sigmoid(B[c_i, d_j]))
    B_ab ~ N(0, 1/lambda_block),  pi ~ Dir(alpha_r 1),  rho ~ Dir(alpha_c 1)

Inference
---------
Mean-field q(c, d) = prod_i Cat(c_i; R_i) prod_j Cat(d_j; C_j) and MAP for B, pi, rho.
We maximise the label-variational objective (handoff eq. in 5.3)

    F = sum_{(i,j) in O} sum_ab R_ia C_jb [y_ij B_ab - softplus(B_ab)]
        + sum_ia R_ia (log pi_a - log R_ia) + sum_jb C_jb (log rho_b - log C_jb)
        - lambda/2 ||B||_F^2 + (alpha_r - 1) sum log pi + (alpha_c - 1) sum log rho.

Schedule per outer iteration (5.8):
  1. rows:    log R_ia = const + log pi_a + [(M.*Y) C B' - (M C) softplus(B)']_ia   (exact Bernoulli)
  2. columns: same with the NEW R
  3. pi, rho Dirichlet MAP
  4. N = R' M C,  T = R' (M.*Y) C   (observed zeros counted through M)
  5. PG block steps  B <- (T - N/2) / (lambda + N f(B_old)),  f = PG(1, .) mean
     (or a safeguarded Newton solve of the same scalar objective)
Every step is a coordinate ascent step on F, so F never decreases.

Everything is done with matrix products of the n x m data with n x K or m x K
responsibility matrices; no n x m x K tensor is ever formed.

Prediction is the mean-field posterior predictive  P = R sigmoid(B) C'
(never sigmoid(R B C')).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, xlogy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../bernfac
from bernfac import pg_mean, softplus  # noqa: E402  (shared stable utilities)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-softplus(-x))


def log_sigmoid(x):
    return -softplus(-np.asarray(x, dtype=float))


def log_softmax_rows(S):
    return S - logsumexp(S, axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# Data container: masked binary matrix with matrix-product access
# ----------------------------------------------------------------------------
class BinaryData:
    """Training view of a binary matrix.

    Stores Yt = M .* Ysafe and M (observation mask) in `dtype`. If every entry is
    observed, M is not stored and M @ C is computed as 1 (1' C).
    Excluded entries never contribute, whatever value is stored there.
    """

    def __init__(self, Y, mask=None, dtype=np.float64, _arrays=None):
        if _arrays is not None:  # fast path: (Yt, M) already prepared and validated by caller
            self.Yt, self.M = _arrays
        else:
            Y = np.asarray(Y)
            if Y.ndim != 2:
                raise ValueError("Y must be a 2-D matrix")
            Yf = Y.astype(np.float64, copy=False)
            if mask is None:
                M = ~np.isnan(Yf)
            else:
                M = np.asarray(mask, dtype=bool)
                if M.shape != Y.shape:
                    raise ValueError("mask shape does not match Y")
            vals = Yf[M]
            if not np.all(np.isfinite(vals)) or np.any((vals != 0) & (vals != 1)):
                raise ValueError("observed entries of Y must be exactly 0 or 1")
            if not M.any():
                raise ValueError("no observed training entries")
            Ysafe = np.where(M, np.nan_to_num(Yf, nan=0.0), 0.0)
            self.Yt = Ysafe.astype(dtype)
            self.M = None if M.all() else M.astype(dtype)
        self.n, self.m = self.Yt.shape
        self.dtype = self.Yt.dtype
        if self.M is None:
            self.row_obs = np.full(self.n, float(self.m))
            self.col_obs = np.full(self.m, float(self.n))
        else:
            self.row_obs = self.M.sum(1, dtype=np.float64)
            self.col_obs = self.M.sum(0, dtype=np.float64)
        self.active_rows = self.row_obs > 0
        self.active_cols = self.col_obs > 0
        self.n_obs = float(self.row_obs.sum())
        self.n_ones = float(self.Yt.sum(dtype=np.float64))

    @classmethod
    def from_train_arrays(cls, Yt, M):
        """Yt = M .* Y (float), M float mask (or None for fully observed)."""
        return cls(None, _arrays=(Yt, M))

    def right(self, C):
        """(M.*Y) C and M C, both n x K (float64)."""
        Cd = np.ascontiguousarray(C, dtype=self.dtype)
        YC = (self.Yt @ Cd).astype(np.float64)
        if self.M is None:
            MC = np.broadcast_to(C.sum(0), (self.n, C.shape[1])).astype(np.float64)
        else:
            MC = (self.M @ Cd).astype(np.float64)
        return YC, MC

    def left(self, R):
        """(M.*Y)' R and M' R, both m x K (float64)."""
        Rd = np.ascontiguousarray(R, dtype=self.dtype)
        YR = (Rd.T @ self.Yt).T.astype(np.float64)  # avoids a transposed copy of Yt
        if self.M is None:
            MR = np.broadcast_to(R.sum(0), (self.m, R.shape[1])).astype(np.float64)
        else:
            MR = (Rd.T @ self.M).T.astype(np.float64)
        return YR, MR


# ----------------------------------------------------------------------------
# Block statistics and the scalar block problem
# ----------------------------------------------------------------------------
def block_stats(data: BinaryData, R, C, right=None):
    """N = R' M C, T = R' (M.*Y) C. `right` = cached data.right(C)."""
    YC, MC = data.right(C) if right is None else right
    return R.T @ MC, R.T @ YC


def block_objective(N, T, B, lam):
    """Elementwise J_ab(b) = T b - N softplus(b) - lam/2 b^2."""
    return T * B - N * softplus(B) - 0.5 * lam * B * B


def pg_block_step(N, T, B, lam):
    """One PG tangent-quadratic ascent step; N = 0 gives the prior mode 0."""
    return (T - 0.5 * N) / (lam + N * pg_mean(B))


def solve_blocks_pg(N, T, B0, lam, steps=5, tol=0.0):
    B = np.array(B0, dtype=float)
    for _ in range(steps):
        Bn = pg_block_step(N, T, B, lam)
        done = np.max(np.abs(Bn - B)) <= tol if B.size else True
        B = Bn
        if done:
            break
    return B


def solve_blocks_newton(N, T, lam, tol=1e-12, max_iter=200):
    """Safeguarded (bracketed) Newton root of J'(b) = T - N sigmoid(b) - lam b.

    J' is strictly decreasing; the root lies in [(T-N)/lam, T/lam].
    """
    N = np.asarray(N, float); T = np.asarray(T, float)
    lo, hi = (T - N) / lam, T / lam
    b = np.zeros_like(N)
    for _ in range(max_iter):
        s = sigmoid(b)
        g = T - N * s - lam * b
        if np.all(np.abs(g) <= tol * (1.0 + N + np.abs(T))):
            break
        h = N * s * (1 - s) + lam
        lo = np.where(g > 0, b, lo)
        hi = np.where(g > 0, hi, b)
        bn = b + g / h
        bad = (bn <= lo) | (bn >= hi)
        b = np.where(bad, 0.5 * (lo + hi), bn)
    return b


# ----------------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------------
def elbo_parts(N, T, B, R, C, pi, rho, lam, alpha_r, alpha_c, active_rows=None, active_cols=None):
    Ra = R if active_rows is None else R[active_rows]
    Ca = C if active_cols is None else C[active_cols]
    ll = float(np.sum(T * B - N * softplus(B)))
    row_kl = float(Ra.sum(0) @ np.log(pi) - xlogy(Ra, Ra).sum())
    col_kl = float(Ca.sum(0) @ np.log(rho) - xlogy(Ca, Ca).sum())
    prior_B = float(-0.5 * lam * np.sum(B * B))
    prior_mix = float((alpha_r - 1) * np.log(pi).sum() + (alpha_c - 1) * np.log(rho).sum())
    total = ll + row_kl + col_kl + prior_B + prior_mix
    return dict(total=total, exp_loglik=ll, row_assign=row_kl, col_assign=col_kl,
                prior_B=prior_B, prior_mix=prior_mix)


def lbm_objective(data, model, lam=None, alpha_r=None, alpha_c=None):
    lam = model.lam if lam is None else lam
    ar = model.alpha_r if alpha_r is None else alpha_r
    ac = model.alpha_c if alpha_c is None else alpha_c
    N, T = block_stats(data, model.R, model.C)
    return elbo_parts(N, T, model.B, model.R, model.C, model.pi, model.rho, lam, ar, ac,
                      data.active_rows, data.active_cols)


# ----------------------------------------------------------------------------
# Coordinate updates
# ----------------------------------------------------------------------------
def row_scores(right, B, logpi):
    YC, MC = right
    return YC @ B.T - MC @ softplus(B).T + logpi


def col_scores(left, B, logrho):
    YR, MR = left
    return YR @ B - MR @ softplus(B) + logrho


def update_rows(data, C, B, pi, right=None):
    right = data.right(C) if right is None else right
    S = row_scores(right, B, np.log(pi))
    R = np.exp(log_softmax_rows(S))
    R[~data.active_rows] = pi  # wholly unobserved rows: prior
    return R


def update_cols(data, R, B, rho, left=None):
    left = data.left(R) if left is None else left
    S = col_scores(left, B, np.log(rho))
    C = np.exp(log_softmax_rows(S))
    C[~data.active_cols] = rho
    return C


def update_mix(R, C, alpha_r, alpha_c, active_rows, active_cols):
    Ra, Ca = R[active_rows], C[active_cols]
    Kr, Kc = R.shape[1], C.shape[1]
    pi = (Ra.sum(0) + alpha_r - 1) / (Ra.shape[0] + Kr * (alpha_r - 1))
    rho = (Ca.sum(0) + alpha_c - 1) / (Ca.shape[0] + Kc * (alpha_c - 1))
    return pi, rho


# ----------------------------------------------------------------------------
# Model, initialisation, fitting
# ----------------------------------------------------------------------------
@dataclass
class LBMModel:
    R: np.ndarray
    C: np.ndarray
    B: np.ndarray
    pi: np.ndarray
    rho: np.ndarray
    lam: float
    alpha_r: float
    alpha_c: float
    info: dict = field(default_factory=dict)

    @property
    def row_labels(self):
        return self.R.argmax(1)

    @property
    def col_labels(self):
        return self.C.argmax(1)


def random_responsibilities(n, K, rng, conc=1.0, floor=1e-6):
    X = rng.dirichlet(np.full(K, conc), size=n) + floor
    return X / X.sum(1, keepdims=True)


def hard_to_soft(labels, K, rng, weight=0.8):
    """Strictly positive responsibilities centred on given labels (for warm starts)."""
    X = (1 - weight) * random_responsibilities(len(labels), K, rng)
    X[np.arange(len(labels)), labels] += weight
    return X


def spectral_labels(data: BinaryData, Kr, Kc, rank=20, seed=0, svd=None):
    """Training-only spectral warm start: k-means on truncated-SVD row/column scores of
    the zero-filled proxy (M.*Y - M/2). Returns (row_labels, col_labels, svd)."""
    from sklearn.cluster import KMeans
    from sklearn.utils.extmath import randomized_svd
    if svd is None:
        Z = data.Yt - 0.5 * (1.0 if data.M is None else data.M)
        rank = min(rank, min(data.n, data.m) - 1)
        U, s, Vt = randomized_svd(np.asarray(Z, np.float32), rank, n_iter=4, random_state=seed)
        del Z
        svd = (U * s, Vt.T * s)
    Ur, Vc = svd
    lr = KMeans(Kr, n_init=1, random_state=seed).fit_predict(Ur) if Kr > 1 else np.zeros(len(Ur), int)
    lc = KMeans(Kc, n_init=1, random_state=seed).fit_predict(Vc) if Kc > 1 else np.zeros(len(Vc), int)
    return lr, lc, svd


def _fit_once(data, Kr, Kc, R0, C0, lam, alpha_r, alpha_c, max_iter, tol, label_sweeps,
              block_inner_steps, block_solver, monitor, monitor_every, verbose, mono_tol):
    t0 = time.time()
    ar_, ac_ = data.active_rows, data.active_cols
    R, C = np.array(R0, float), np.array(C0, float)
    pi, rho = update_mix(R, C, alpha_r, alpha_c, ar_, ac_)
    right = data.right(C)
    N, T = block_stats(data, R, C, right)
    B = solve_blocks_newton(N, T, lam) if block_solver == "newton" else \
        solve_blocks_pg(N, T, np.zeros((Kr, Kc)), lam, steps=100, tol=1e-12)
    parts = elbo_parts(N, T, B, R, C, pi, rho, lam, alpha_r, alpha_c, ar_, ac_)
    hist = dict(elbo=[parts["total"]], sub=[], monitor=[], time=[0.0])
    n_decrease, worst = 0, 0.0
    status = "max_iter"
    F_old = parts["total"]
    lp_pi = lambda p: (alpha_r - 1) * np.log(p).sum()  # noqa: E731
    for it in range(1, max_iter + 1):
        subs = []
        for _ in range(label_sweeps):
            # rows (C, B, pi fixed)
            S = row_scores(right, B, np.log(pi))
            R = np.exp(log_softmax_rows(S)); R[~ar_] = pi
            Ra = R[ar_]
            # F after row update (likelihood = sum_ia R_ia * (row lik score))
            ll = float(np.sum(Ra * (S[ar_] - np.log(pi))))
            Ca = C[ac_]
            f_row = ll + float(Ra.sum(0) @ np.log(pi) - xlogy(Ra, Ra).sum()) \
                + float(Ca.sum(0) @ np.log(rho) - xlogy(Ca, Ca).sum()) - 0.5 * lam * np.sum(B * B) \
                + lp_pi(pi) + (alpha_c - 1) * np.log(rho).sum()
            subs.append(f_row)
            # columns with the NEW R
            left = data.left(R)
            S = col_scores(left, B, np.log(rho))
            C = np.exp(log_softmax_rows(S)); C[~ac_] = rho
            Ca = C[ac_]
            ll = float(np.sum(Ca * (S[ac_] - np.log(rho))))
            f_col = ll + float(Ra.sum(0) @ np.log(pi) - xlogy(Ra, Ra).sum()) \
                + float(Ca.sum(0) @ np.log(rho) - xlogy(Ca, Ca).sum()) - 0.5 * lam * np.sum(B * B) \
                + lp_pi(pi) + (alpha_c - 1) * np.log(rho).sum()
            subs.append(f_col)
            right = data.right(C)  # cache for N, T and next row update
        pi, rho = update_mix(R, C, alpha_r, alpha_c, ar_, ac_)
        N, T = block_stats(data, R, C, right)
        subs.append(elbo_parts(N, T, B, R, C, pi, rho, lam, alpha_r, alpha_c, ar_, ac_)["total"])
        if block_solver == "newton":
            B = solve_blocks_newton(N, T, lam)
        else:
            B = solve_blocks_pg(N, T, B, lam, steps=block_inner_steps, tol=1e-12)
        parts = elbo_parts(N, T, B, R, C, pi, rho, lam, alpha_r, alpha_c, ar_, ac_)
        subs.append(parts["total"])
        seq = [F_old] + subs
        for a, b in zip(seq[:-1], seq[1:]):
            d = (a - b) / max(1.0, abs(a))
            if d > mono_tol:
                n_decrease += 1
            worst = max(worst, d)
        F = parts["total"]
        hist["elbo"].append(F); hist["sub"].append(subs); hist["time"].append(time.time() - t0)
        if monitor is not None and monitor_every and (it % monitor_every == 0):
            hist["monitor"].append((it, monitor(R, C, B)))
        if verbose:
            print(f"  it {it:4d}  F={F:.6e}  dF/|F|={(F - F_old) / abs(F_old):.2e}  "
                  f"t={time.time() - t0:.1f}s", flush=True)
        if abs(F - F_old) <= tol * max(1.0, abs(F_old)):
            status = "converged"
            F_old = F
            break
        F_old = F
    R[~ar_] = pi; C[~ac_] = rho  # prior assignments for wholly unobserved rows/columns
    occ_r, occ_c = R[ar_].sum(0), C[ac_].sum(0)
    info = dict(elbo=F_old, parts=parts, n_iter=it, status=status, wall=time.time() - t0,
                n_decrease=n_decrease, max_rel_decrease=worst, history=hist,
                occupancy_rows=occ_r, occupancy_cols=occ_c,
                n_empty_rows=int((occ_r < 1).sum()), n_empty_cols=int((occ_c < 1).sum()))
    return LBMModel(R, C, B, pi, rho, lam, alpha_r, alpha_c, info)


def fit_lbm(data, Kr, Kc, lam=1.0, alpha_r=1.1, alpha_c=1.1, max_iter=300, tol=1e-7,
            label_sweeps=1, block_inner_steps=5, block_solver="pg", seed=1, n_restarts=5,
            init="random", inits=None, monitor=None, monitor_every=0, verbose=False,
            mono_tol=1e-9, spectral_svd=None):
    """Fit with restarts; the returned model is the restart with the highest final F
    (training objective only). `init` = 'random' | 'spectral' | 'mixed' (restart 0 is
    spectral, the rest random). `inits` = explicit list of (R0, C0)."""
    if not isinstance(data, BinaryData):
        data = BinaryData(data)
    if lam <= 0 or alpha_r < 1 or alpha_c < 1:
        raise ValueError("need lambda_block > 0 and alpha >= 1")
    if Kr < 1 or Kc < 1:
        raise ValueError("K_r, K_c must be >= 1")
    rng = np.random.default_rng(seed)
    if inits is None:
        inits, svd = [], spectral_svd
        for r in range(n_restarts):
            if init == "spectral" or (init == "mixed" and r == 0):
                lr, lc, svd = spectral_labels(data, Kr, Kc, seed=seed + r, svd=svd)
                inits.append((hard_to_soft(lr, Kr, rng), hard_to_soft(lc, Kc, rng), "spectral"))
            else:
                inits.append((random_responsibilities(data.n, Kr, rng),
                              random_responsibilities(data.m, Kc, rng), "random"))
    best, runs = None, []
    for r, ini in enumerate(inits):
        R0, C0 = ini[0], ini[1]
        kind = ini[2] if len(ini) > 2 else "given"
        if verbose:
            print(f" restart {r} ({kind})", flush=True)
        mdl = _fit_once(data, Kr, Kc, R0, C0, lam, alpha_r, alpha_c, max_iter, tol, label_sweeps,
                        block_inner_steps, block_solver, monitor, monitor_every, verbose, mono_tol)
        mdl.info["restart"] = r; mdl.info["init"] = kind
        runs.append(mdl.info)
        if best is None or mdl.info["elbo"] > best.info["elbo"]:
            best = mdl
    best.info["restarts"] = [dict(restart=i["restart"], init=i["init"], elbo=i["elbo"],
                                  n_iter=i["n_iter"], status=i["status"], wall=i["wall"],
                                  n_decrease=i["n_decrease"], max_rel_decrease=i["max_rel_decrease"],
                                  elbo_trace=list(i["history"]["elbo"]),
                                  monitor=i["history"]["monitor"]) for i in runs]
    best.info["config"] = dict(Kr=Kr, Kc=Kc, lam=lam, alpha_r=alpha_r, alpha_c=alpha_c,
                               max_iter=max_iter, tol=tol, label_sweeps=label_sweeps,
                               block_inner_steps=block_inner_steps, block_solver=block_solver,
                               seed=seed, n_restarts=len(inits), init=init)
    return best


# ----------------------------------------------------------------------------
# Prediction and new row / column inference
# ----------------------------------------------------------------------------
def predict_pairs(model, rows, cols, chunk=1_000_000):
    """p_ij = sum_ab R_ia C_jb sigmoid(B_ab) for index pairs."""
    G = model.R @ sigmoid(model.B)
    out = np.empty(len(rows))
    for s in range(0, len(rows), chunk):
        out[s:s + chunk] = np.einsum("ij,ij->i", G[rows[s:s + chunk]], model.C[cols[s:s + chunk]])
    return out


def predict_matrix(model, rows=None):
    """Dense R sigmoid(B) C' for a row batch (small problems / chunks only)."""
    R = model.R if rows is None else model.R[rows]
    return R @ sigmoid(model.B) @ model.C.T


def predict_prior(model):
    return float(model.pi @ sigmoid(model.B) @ model.rho)


def infer_rows(model, Ynew, mask=None):
    """Responsibilities of new rows given their observed cells; C, B, pi fixed.
    Rows with no observed cell get R = pi."""
    d = BinaryData(Ynew, mask) if not isinstance(Ynew, BinaryData) else Ynew
    return update_rows(d, model.C, model.B, model.pi)


def infer_cols(model, Ynew, mask=None):
    """Responsibilities of new columns (Ynew is n x m_new); R, B, rho fixed."""
    d = BinaryData(Ynew, mask) if not isinstance(Ynew, BinaryData) else Ynew
    return update_cols(d, model.R, model.B, model.rho)


# ----------------------------------------------------------------------------
# Held-out scoring (Bernoulli NLL, Brier, AUC on a subsample, 10-bin calibration)
# ----------------------------------------------------------------------------
P_CLIP = 1e-12


def score_probs(y, p, auc_sample=2_000_000, seed=0):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    pc = np.clip(p, P_CLIP, 1 - P_CLIP)
    n_clip = int(np.sum((p < P_CLIP) | (p > 1 - P_CLIP)))
    nll = float(-np.mean(y * np.log(pc) + (1 - y) * np.log1p(-pc)))
    brier = float(np.mean((p - y) ** 2))
    ys, ps = y, p
    if y.size > auc_sample:
        idx = np.random.default_rng(seed).choice(y.size, auc_sample, replace=False)
        ys, ps = y[idx], p[idx]
    auc = float(roc_auc_score(ys, ps)) if 0 < ys.sum() < ys.size else float("nan")
    edges = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(p, edges) - 1, 0, 9)
    cnt = np.bincount(b, minlength=10)
    mp = np.bincount(b, p, 10) / np.maximum(cnt, 1)
    my = np.bincount(b, y, 10) / np.maximum(cnt, 1)
    ece = float(np.sum(cnt * np.abs(mp - my)) / cnt.sum())
    return dict(nll=nll, brier=brier, auc=auc, ece=ece, n=int(y.size), n_clipped=n_clip,
                calib=dict(count=cnt.tolist(), mean_p=mp.tolist(), mean_y=my.tolist()))


def heldout_probs(model, Y, mask, block=1024, dtype=np.float32):
    """Predicted probabilities and labels at the True entries of `mask`, row-chunked."""
    G = (model.R @ sigmoid(model.B)).astype(dtype)
    Ct = np.ascontiguousarray(model.C.T, dtype=dtype)
    ys, ps = [], []
    for s in range(0, Y.shape[0], block):
        Hb = mask[s:s + block]
        if not Hb.any():
            continue
        P = G[s:s + block] @ Ct
        ps.append(P[Hb].astype(np.float64)); ys.append(Y[s:s + block][Hb].astype(np.float64))
    return np.concatenate(ys), np.concatenate(ps)


def heldout_scores(model, Y, mask, **kw):
    y, p = heldout_probs(model, Y, mask)
    return score_probs(y, p, **kw)


# ----------------------------------------------------------------------------
# Utilities: permutation alignment
# ----------------------------------------------------------------------------
def align_labels(true, est, K_true, K_est):
    """Hungarian matching of estimated groups to true groups (by co-occurrence)."""
    from scipy.optimize import linear_sum_assignment
    M = np.zeros((K_true, K_est))
    np.add.at(M, (true, est), 1)
    r, c = linear_sum_assignment(-M)
    return dict(zip(c, r)), M
