"""Loaders for the fitted v3 probability models (R3-R9, Poisson) on the full fly preMN 24 h data.

Every model was fit on the train entries only (train = ~heldout). Each loader returns a
`Predictor` with

    block(rows, cols)  -> (P, Q)   P = Pr(y_ij = 1), Q = 1 - P (computed separately, no
                                   cancellation) for the len(rows) x len(cols) sub-block
    pairs(i, j)        -> (P, Q)   the same for explicit index pairs
    mu_block(rows, cols)           expected count (count models only)

No n x m array is ever formed: everything is evaluated per block or per pair.

standardized(code) (R4, R5, R6 only) returns the same kind of Predictor with the cell offset c_i
fixed to its median over cells: the depth-standardized probability p~ (see its docstring and
check_standardized.py).

Model definitions (codes from v3style.py)
    R3  BernFac logit                   p = sigmoid(u_i'v_j)
    R4  BernFac logit + offsets         p = sigmoid(a_i + b_j + u_i'v_j)
    R5  BernFac cloglog + offsets       p = 1 - exp(-exp(eta)), eta = a_i + b_j + u_i'v_j clipped
                                        to [-30, 4.4] as in bernfac_large.cll_terms
    R6  NB count factorization          mu = exp(o_i + a_i + b_j + u_i'v_j) (eta clipped to
                                        [-40, 15] as in countfac.entry_terms), o_i = log(train UMI of
                                        cell i) stored in the fit; p = 1 - (1 + mu/phi_j)^(-phi_j)
    PO  Poisson count factorization     same mu, p = 1 - exp(-mu)
    R7  mixture of local factors        p = sum_l r_il E[sigmoid(X)], X ~ N(mu_lj + b_lj'm_il,
                                        t_l + b_lj'S_il b_lj), Gauss-Hermite (mfa.predict_pairs,
                                        Q = model.opts.quadrature_order). The fit saved only r and m
                                        (float16), not the coordinate covariances S, so the local
                                        posteriors (m_i, S_i) are recomputed once with the globals
                                        frozen by mfa.infer_rows on the train entries (the same local
                                        update the fit uses, iterated to convergence) and cached in
                                        v3/cache/mfa_local_state.npz (n x d(d+1)/2, not n x m).
                                        The predictive integrates over q(z_i); for train entries it
                                        is the same posterior predictive (it does not condition on
                                        y_ij a second time).
    R8  neural manifold (affine head)   p = sigmoid(g(z_i)_j), plug-in at the joint MAP (map_plugin)
    R9  latent block model              p = sum_ab R_ia C_jb sigmoid(B_ab), i.e. R sigmoid(B) C'
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
BF = HERE.parent                                   # cm_minimal/experiments/bernfac
ROOT = BF.parents[2]
STUDY = ROOT / "outputs/bernfac_20261020"
DATA = STUDY / "full/data"
CACHE = STUDY / "v3/cache"
sys.path.insert(0, str(BF))
sys.path.insert(0, str(BF / "models/mfa"))
sys.path.insert(0, str(BF / "models/neural"))

CLL_MIN, CLL_MAX = -30.0, 4.4      # bernfac_large.ETA_MIN/ETA_MAX
CNT_MIN, CNT_MAX = -40.0, 15.0     # countfac.ETA_MIN/ETA_MAX

FITS = {
    "R3": STUDY / "full/fits/cv_k40_lam10.npz",
    "R4": STUDY / "full/fits/cv_off_k40_lam10.npz",
    "R5": STUDY / "full/fits/cv_cll_k40_lam30.npz",
    "R6": STUDY / "models/counts/fits/best_nb.npz",
    "PO": STUDY / "models/counts/fits/best_poisson.npz",
    "R7": STUDY / "models/mfa/fits/best.npz",
    "R8": STUDY / "models/neural/fits/best.pt",
    "R9": STUDY / "models/lbm/fits/best.npz",
}
# name in models/comparison.csv
COMPARISON_NAME = {
    "R3": "BernFac logit", "R4": "BernFac logit + offsets", "R5": "BernFac cloglog + offsets",
    "R6": "NB count factorization", "PO": "Poisson count factorization",
    "R7": "mixture of local factors", "R8": "neural manifold", "R9": "latent block model",
}


def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _sig_pair(eta):
    """sigmoid(eta), sigmoid(-eta), each accurate in its own tail."""
    e = np.exp(-np.abs(eta))
    big = 1.0 / (1.0 + e)
    small = e / (1.0 + e)
    pos = eta >= 0
    return np.where(pos, big, small), np.where(pos, small, big)


class Predictor:
    code = ""
    is_count = False

    def block(self, rows, cols):
        raise NotImplementedError

    def pairs(self, i, j):
        raise NotImplementedError

    def prob(self, rows, cols):
        return self.block(rows, cols)[0]

    def logpairs(self, i, j):
        """(log P, log Q) at index pairs; generic version floors at 1e-300, subclasses are exact."""
        P, Q = self.pairs(i, j)
        return np.log(np.maximum(P, 1e-300)), np.log(np.maximum(Q, 1e-300))


class _LinearEta(Predictor):
    """eta_ij = u_i'v_j (+ a_i + b_j (+ o_i)); subclasses map eta to (P, Q)."""

    def __init__(self, U, V, a=None, b=None, o=None):
        self.U, self.V = np.asarray(U, np.float64), np.asarray(V, np.float64)
        n, m = len(U), len(V)
        self.a = np.zeros(n) if a is None else np.asarray(a, np.float64)
        self.b = np.zeros(m) if b is None else np.asarray(b, np.float64)
        if o is not None:
            self.a = self.a + np.asarray(o, np.float64)

    def eta_block(self, rows, cols):
        rows, cols = np.asarray(rows), np.asarray(cols)
        return self.U[rows] @ self.V[cols].T + self.a[rows, None] + self.b[None, cols]

    def eta_pairs(self, i, j):
        i, j = np.asarray(i), np.asarray(j)
        out = np.empty(len(i))
        for s in range(0, len(i), 2_000_000):
            ii, jj = i[s:s + 2_000_000], j[s:s + 2_000_000]
            out[s:s + 2_000_000] = np.einsum("ik,ik->i", self.U[ii], self.V[jj]) + self.a[ii] + self.b[jj]
        return out

    def link(self, eta, cols):
        raise NotImplementedError

    def block(self, rows, cols):
        return self.link(self.eta_block(rows, cols), np.asarray(cols)[None, :])

    def pairs(self, i, j):
        return self.link(self.eta_pairs(i, j), np.asarray(j))


class Logit(_LinearEta):
    def link(self, eta, cols):
        return _sig_pair(eta)

    def logpairs(self, i, j):
        eta = self.eta_pairs(i, j)
        return -np.logaddexp(0.0, -eta), -np.logaddexp(0.0, eta)


class Cloglog(_LinearEta):
    def link(self, eta, cols):
        mu = np.exp(np.clip(eta, CLL_MIN, CLL_MAX))
        return -np.expm1(-mu), np.exp(-mu)

    def logpairs(self, i, j):
        mu = np.exp(np.clip(self.eta_pairs(i, j), CLL_MIN, CLL_MAX))
        return np.log(np.maximum(-np.expm1(-mu), 1e-300)), -mu


class Counts(_LinearEta):
    is_count = True

    def __init__(self, U, V, a, b, o, phi=None):
        super().__init__(U, V, a, b, o)
        self.phi = None if phi is None or np.size(phi) == 0 else np.asarray(phi, np.float64)

    def mu_from_eta(self, eta):
        return np.exp(np.clip(eta, CNT_MIN, CNT_MAX))

    def log_p0(self, eta, cols):
        mu = self.mu_from_eta(eta)
        return -mu if self.phi is None else -self.phi[cols] * np.log1p(mu / self.phi[cols])

    def link(self, eta, cols):
        lp0 = self.log_p0(eta, cols)
        return -np.expm1(lp0), np.exp(lp0)

    def logpairs(self, i, j):
        """Exact log Pr(y = 0) = -mu (Poisson) or -phi log(1 + mu/phi) (NB); it can be far below
        log(1e-300) when mu is huge, which matters for the Poisson fit (see validate_predictors)."""
        lp0 = self.log_p0(self.eta_pairs(i, j), np.asarray(j))
        return np.log(np.maximum(-np.expm1(lp0), 1e-300)), lp0

    def mu_block(self, rows, cols):
        return self.mu_from_eta(self.eta_block(rows, cols))

    def mu_pairs(self, i, j):
        return self.mu_from_eta(self.eta_pairs(i, j))


class LBM(Predictor):
    def __init__(self, R, C, B):
        self.R, self.C = np.asarray(R, np.float64), np.asarray(C, np.float64)
        self.G1 = self.R @ _sig(np.asarray(B, np.float64))         # n x Kc
        self.G0 = self.R @ _sig(-np.asarray(B, np.float64))

    def block(self, rows, cols):
        Ct = self.C[np.asarray(cols)].T
        return self.G1[rows] @ Ct, self.G0[rows] @ Ct

    def pairs(self, i, j):
        Cj = self.C[j]
        return np.einsum("ik,ik->i", self.G1[i], Cj), np.einsum("ik,ik->i", self.G0[i], Cj)


class Neural(Predictor):
    def __init__(self, path):
        import torch
        torch.set_num_threads(4)
        from bernoulli_neural import load_model
        self.torch = torch
        self.model, _, self.extra = load_model(path)
        self.model.eval()

    def logits_block(self, rows, cols):
        t = self.torch
        with t.no_grad():
            Z = self.model.Z[t.as_tensor(np.asarray(rows), dtype=t.long)]
            out = self.model.decoder(Z, t.as_tensor(np.asarray(cols), dtype=t.long))
        return out.to(t.float64).numpy()

    def block(self, rows, cols):
        return _sig_pair(self.logits_block(rows, cols))

    def pairs(self, i, j):
        from bernoulli_neural.predict import predict_pairs
        return _sig_pair(predict_pairs(self.model, np.asarray(i), np.asarray(j), logits=True))


class MFA(Predictor):
    def __init__(self, path, state_path=None, refresh=False):
        import mfa
        self.mfa = mfa
        self.model = mfa.MFAModel.load(path)
        state_path = Path(state_path or CACHE / "mfa_local_state.npz")
        if refresh or not state_path.exists():
            Y = np.load(DATA / "Y.npy")
            H = np.load(DATA / "heldout.npy")
            state, info = mfa.infer_rows(self.model, Y, excluded=H, max_sweeps=50, tol=1e-7)
            del Y, H
            state_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(state_path, r=state.r, m=state.m.astype(np.float32), Sv=state.Sv.astype(np.float32),
                     active=state.active, sweeps=info["sweeps"])
        z = np.load(state_path)
        K, d = self.model.K, self.model.d
        n = len(z["r"])
        self.state = mfa.LocalState(z["r"].astype(float), z["m"].astype(float), z["Sv"].astype(float),
                                    np.zeros((n, K)), np.zeros(n), z["active"])

    def pairs(self, i, j):
        P, P1 = self.mfa.predict_pairs(self.model, self.state, np.asarray(i), np.asarray(j),
                                       return_complement=True)
        return P, P1

    def block(self, rows, cols):
        rows, cols = np.asarray(rows), np.asarray(cols)
        P, Q = self.pairs(np.repeat(rows, len(cols)), np.tile(cols, len(rows)))
        return P.reshape(len(rows), len(cols)), Q.reshape(len(rows), len(cols))


def load(code):
    f = FITS[code]
    if code == "R3":
        z = np.load(f); p = Logit(z["U"], z["V"])
    elif code == "R4":
        z = np.load(f); p = Logit(z["U"], z["V"], z["a"], z["b"])
    elif code == "R5":
        z = np.load(f); p = Cloglog(z["U"], z["V"], z["a"], z["b"])
    elif code in ("R6", "PO"):
        z = np.load(f); p = Counts(z["U"], z["V"], z["a"], z["b"], z["o"], z["phi"])
    elif code == "R7":
        p = MFA(f)
    elif code == "R8":
        p = Neural(f)
    elif code == "R9":
        z = np.load(f); p = LBM(z["R"], z["C"], z["B"])
    else:
        raise KeyError(code)
    p.code = code
    return p


ALL = ["R3", "R4", "R5", "R6", "PO", "R7", "R8", "R9"]

# --------------------------------------------------------------------------- depth standardization
STD_MODELS = ("R4", "R5", "R6")


def cell_offset(pred):
    """Cell offset c_i of a loaded R4 / R5 / R6 predictor: a_i (R4, R5) or o_i + a_i (R6; the
    Counts class already stores the sum in .a)."""
    return pred.a.copy()


def standardized(code, ref="median"):
    """Depth-standardized predictor for R4, R5 or R6.

    Same model as load(code) with the cell offset c_i (a_i for R4/R5, o_i + a_i for R6) replaced
    by one reference value c_bar for every cell, keeping b_j and u_i'v_j:
        R4  p~ = sigmoid(c_bar + b_j + u_i'v_j)
        R5  p~ = 1 - exp(-exp(c_bar + b_j + u_i'v_j))
        R6  p~ = 1 - (1 + mu~/phi_j)^(-phi_j),  mu~ = exp(c_bar + b_j + u_i'v_j)
    i.e. "cell i's state at a typical sequencing depth". ref = "median" (default) or "mean" of
    c_i over all cells, or a number. p~ is NOT a predictor of observed detection (that is p^ from
    load(code), which is what was scored on held-out entries).

    The returned Predictor has the usual block / pairs / logpairs (/ mu_block) plus
      .c_obs  the original c_i (length n)     .c_ref  the reference value c_bar
    and .code = code + "~". load(code) is unchanged.
    """
    if code not in STD_MODELS:
        raise KeyError(f"{code} has no explicit cell offset; standardized() is defined for {STD_MODELS}")
    p = load(code)
    c = cell_offset(p)
    if isinstance(ref, str):
        cbar = {"median": np.median, "mean": np.mean}[ref](c)
    else:
        cbar = float(ref)
    p.c_obs, p.c_ref = c, float(cbar)
    p.a = np.full_like(c, cbar)
    p.code = code + "~"
    return p


def nll_on_mask(pred, Y, M, block=1024, per_entry=False):
    """Mean Bernoulli NLL over the True entries of M (row blocks; exact log P / log Q where the model
    allows it, else floored at 1e-300).
    Returns (nll, n, y, p[, per-entry nll]) with vectors in row-major order of M."""
    tot, cnt, ys, ps, ls = 0.0, 0, [], [], []
    for s in range(0, Y.shape[0], block):
        Mb = np.asarray(M[s:s + block])
        ri, ci = np.nonzero(Mb)
        if ri.size == 0:
            continue
        lP, lQ = pred.logpairs(ri + s, ci)
        P = np.exp(lP)
        y = np.asarray(Y[s:s + block])[ri, ci]
        ll = np.where(y > 0, lP, lQ)
        tot -= float(np.sum(ll))
        if per_entry:
            ls.append((-ll).astype(np.float32))
        cnt += y.size
        ys.append(y.astype(np.int8)); ps.append(P.astype(np.float32))
    if per_entry:
        return tot / cnt, cnt, np.concatenate(ys), np.concatenate(ps), np.concatenate(ls)
    return tot / cnt, cnt, np.concatenate(ys), np.concatenate(ps)
