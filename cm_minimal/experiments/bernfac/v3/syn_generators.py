"""Five synthetic worlds for the shared v3 benchmark (n = 2000 cells, m = 300 genes).

Every generator returns COUNTS X and the detection probability P_true = Pr(X_ij >= 1), so the
binary data Y = 1[X >= 1] and every count model can be fitted on the same dataset. A global
shift c of the linear predictor is set by bisection so that mean(P_true) = 0.18 (the full fly
data has detection density 0.181).

G1  low-rank logit       logit P = a_i + b_j + u_i'v_j (rank 8), a_i ~ N(0, 0.5^2) cell offset.
                         Counts: X = Y (1 + Poisson(1)), so Pr(X >= 1) = P exactly; the positive
                         part carries no information. Favours R4 (logit + offsets).
G2  NB counts + depth    mu_ij = s_i exp(c + b_j + u_i'v_j) (rank 8), log s_i ~ N(0, 0.5^2),
                         phi_j ~ logN(0, 1) clipped to [0.05, 100], X ~ NB(mu, phi),
                         P = 1 - (1 + mu/phi)^-phi. Favours R6 (NB factorization).
G3  10 cell types        log mu_ij = log s_i + c + b_j + super_{g(t_i) j} + type_{t_i j} + w_i'h_j:
                         3 super-classes x 10 types (unequal sizes), 25 marker genes per type
                         (+N(2, 0.5^2)), dense type effects N(0, 0.4^2), rank-2 within-type
                         variation (sd 0.3), lognormal depth, NB noise as G2. Looks like
                         hemilineages; favours none of the models exactly.
G4  nonlinear manifold   z_i ~ N(0, I_2); f_j(z) = a_j sin(q_j'z) + c_j cos(r_j'z) + e_j tanh(s_j'z)
                         (handoff 8.2), rescaled to sd 2.5; logit P = c + d_i + b_j + f_j(z_i) with
                         a depth offset d_i ~ N(0, 0.5^2). Counts as G1. Favours R8.
G5  blocks               8 row x 10 column groups (Dirichlet(2) proportions), block logits
                         N(c, 1.5^2); P = sigma(B)[r_i, c_j] (handoff 8.3). Counts as G1. Favours R9.

Splits: one uniform draw per entry, train < 0.70 <= val < 0.85 <= test; the same masks are
used by every method.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

N, M = 2000, 300
DENSITY = 0.18
DATASETS = ["G1", "G2", "G3", "G4", "G5"]
SEEDS = [0, 1, 2]
TITLES = {"G1": "G1 low-rank logit", "G2": "G2 NB counts", "G3": "G3 cell types",
          "G4": "G4 nonlinear manifold", "G5": "G5 blocks"}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate(pfun, lo=-15.0, hi=15.0, target=DENSITY, iters=60):
    """Bisection on the global shift c so that mean(pfun(c)) = target (pfun increasing in c)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if pfun(mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _nb_detect(mu, phi):
    return -np.expm1(-phi * np.log1p(mu / phi))


def _nb_draw(rng, mu, phi):
    """X ~ NB(mean mu, var mu + mu^2/phi) via the gamma-Poisson mixture."""
    lam = rng.gamma(shape=np.broadcast_to(phi, mu.shape), scale=mu / phi)
    return rng.poisson(lam)


def _lowrank(rng, n, m, k, sd):
    L = rng.standard_normal((n, k)) @ rng.standard_normal((m, k)).T
    return L * (sd / L.std())


def _binary_counts(rng, P):
    """Y ~ Bernoulli(P), X = Y (1 + Poisson(1)) so that Pr(X >= 1) = P exactly."""
    Y = rng.random(P.shape) < P
    return (Y * (1 + rng.poisson(1.0, P.shape))).astype(np.int64)


def generate(g, seed):
    rng = np.random.default_rng(10_000 * (DATASETS.index(g) + 1) + seed)
    n, m = N, M
    out = dict(labels=None, z=None, col_labels=None)
    if g == "G1":
        a = rng.normal(0, 0.5, n); b = rng.normal(0, 1.0, m)
        L = _lowrank(rng, n, m, 8, 1.5)
        eta = a[:, None] + b[None, :] + L
        c = _calibrate(lambda c: _sig(eta + c))
        P = _sig(eta + c)
        X = _binary_counts(rng, P)
        out["order"] = np.argsort(np.linalg.svd(L, full_matrices=False)[0][:, 0])
    elif g == "G2":
        logs = rng.normal(0, 0.5, n); logs -= logs.mean()
        b = rng.normal(0, 1.2, m)
        L = _lowrank(rng, n, m, 8, 1.0)
        phi = np.clip(np.exp(rng.normal(0, 1.0, m)), 0.05, 100.0)
        eta = logs[:, None] + b[None, :] + L
        c = _calibrate(lambda c: _nb_detect(np.exp(eta + c), phi))
        mu = np.exp(eta + c)
        P = _nb_detect(mu, phi)
        X = _nb_draw(rng, mu, phi)
        out["order"] = np.argsort(np.linalg.svd(L, full_matrices=False)[0][:, 0])
        out["phi"] = phi
    elif g == "G3":
        T, S = 10, 3
        prop = np.sort(rng.dirichlet(np.full(T, 3.0)))[::-1]
        t = np.sort(rng.choice(T, n, p=prop))
        sup = np.arange(T) % S
        super_eff = rng.normal(0, 0.6, (S, m))
        type_eff = rng.normal(0, 0.4, (T, m))
        for k in range(T):
            mk = rng.choice(m, 25, replace=False)
            type_eff[k, mk] += rng.normal(2.0, 0.5, 25)
        within = rng.normal(0, 0.3, (n, 2)) @ rng.standard_normal((m, 2)).T
        logs = rng.normal(0, 0.5, n); logs -= logs.mean()
        b = rng.normal(0, 1.2, m)
        phi = np.clip(np.exp(rng.normal(0, 1.0, m)), 0.05, 100.0)
        eta = logs[:, None] + b[None, :] + super_eff[sup[t]] + type_eff[t] + within
        c = _calibrate(lambda c: _nb_detect(np.exp(eta + c), phi))
        mu = np.exp(eta + c)
        P = _nb_detect(mu, phi)
        X = _nb_draw(rng, mu, phi)
        out["labels"] = t
        out["order"] = np.lexsort((logs, t))
        out["phi"] = phi
    elif g == "G4":
        z = rng.standard_normal((n, 2))
        q, r, s = (rng.standard_normal((m, 2)) * 1.3 for _ in range(3))
        a, cc, e = (rng.standard_normal(m) for _ in range(3))
        f = a * np.sin(z @ q.T) + cc * np.cos(z @ r.T) + e * np.tanh(z @ s.T)
        f = f * (2.5 / f.std())
        d = rng.normal(0, 0.5, n); b = rng.normal(0, 0.7, m)
        eta = d[:, None] + b[None, :] + f
        c = _calibrate(lambda c: _sig(eta + c))
        P = _sig(eta + c)
        X = _binary_counts(rng, P)
        from sklearn.cluster import KMeans
        out["z"] = z
        out["labels"] = KMeans(8, n_init=10, random_state=0).fit_predict(z)   # 8 regions of the latent plane
        out["order"] = np.argsort(np.arctan2(z[:, 1], z[:, 0]))
    elif g == "G5":
        Kr, Kc = 8, 10
        zr = np.sort(rng.choice(Kr, n, p=rng.dirichlet(np.full(Kr, 2.0))))
        zc = rng.choice(Kc, m, p=rng.dirichlet(np.full(Kc, 2.0)))
        B = rng.normal(0, 1.5, (Kr, Kc))
        c = _calibrate(lambda c: _sig(B + c)[zr][:, zc])
        P = _sig(B + c)[zr][:, zc]
        X = _binary_counts(rng, P)
        out["labels"] = zr
        out["col_labels"] = zc
        out["order"] = np.argsort(zr, kind="stable")
    else:
        raise ValueError(g)
    u = np.random.default_rng(seed + 77 + 1000 * DATASETS.index(g)).random((n, m))
    split = np.where(u < 0.70, 0, np.where(u < 0.85, 1, 2)).astype(np.uint8)   # 0 train, 1 val, 2 test
    out.update(X=X, P=P, split=split, shift=c)
    return out


def save(D, path):
    keep = {k: v for k, v in D.items() if v is not None}
    keep["X"] = D["X"].astype(np.uint16)
    keep["P"] = D["P"].astype(np.float32)
    np.savez_compressed(path, **keep)


def load(path):
    z = np.load(path)
    D = {k: z[k] for k in z.files}
    D["X"] = D["X"].astype(np.int64)
    D["P"] = D["P"].astype(np.float64)
    D["Y"] = (D["X"] >= 1).astype(np.uint8)
    for k in ("labels", "z", "col_labels"):
        D.setdefault(k, None)
    return D


def ensure(datadir, g, seed):
    p = Path(datadir) / f"{g}_seed{seed}.npz"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        save(generate(g, seed), p)
    return load(p)
