"""Tests for the Bernoulli mixture of local factor models (handoff sections 7.1 and 7.2).

Run as plain python:   python tests/test_mfa.py
Each test prints PASS/FAIL; the exit code is the number of failures.
"""
from __future__ import annotations

import io
import math
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mfa  # noqa: E402
from mfa import (MFAModel, MFAOptions, bound_constant, e_pass, gauss_hermite, local_row_reference,  # noqa: E402
                 m_step, pg_mean, prepare_data, quad_bound, sigmoid, softplus)


# ----------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------
def gen_mfa(n=60, m=25, K=3, d=2, t=0.25, seed=0, mean_shift=0.0, pi=(0.5, 0.3, 0.2)):
    rng = np.random.default_rng(seed)
    pi = np.asarray(pi[:K]) / np.sum(pi[:K])
    c = rng.choice(K, n, p=pi)
    mu = rng.normal(mean_shift, 1.5, (K, m))
    B = rng.normal(0, 1, (K, m, d))
    z = rng.normal(size=(n, d))
    psi = mu[c] + np.einsum("imd,id->im", B[c], z) + np.sqrt(t) * rng.normal(size=(n, m))
    Y = (rng.random((n, m)) < sigmoid(psi)).astype(float)
    return Y, c, psi


def random_model(K, m, d, t, rng, mode="isotropic", **kw):
    o = MFAOptions(residual_mode=mode, residual_variance=t if mode == "isotropic" else 0.25, **kw)
    return MFAModel(np.full(K, 1.0 / K), rng.normal(0, 1, (K, m)), rng.normal(0, 0.7, (K, m, d)),
                    np.full(K, t if mode == "isotropic" else 0.0), o)


def ref_pass(Y, X, model, xi):
    """Per-row/component reference moments at stored xi (K, n, m)."""
    n, m = Y.shape
    out = {}
    for i in range(n):
        for l in range(model.K):
            out[i, l] = local_row_reference(Y[i].astype(float), ~X[i], model.mu[l], model.B[l],
                                            model.t[l], xi[l, i])
    return out


# ----------------------------------------------------------------------------
# 7.1 shared mathematical tests
# ----------------------------------------------------------------------------
def test_pg_mean():
    x = np.array([0.0, 1e-8, -1e-8, 1.0, -1.0, 20.0, -20.0, 1000.0, -1000.0])
    w = pg_mean(x)
    assert np.all(np.isfinite(w)) and np.all(w > 0) and np.all(w <= 0.25)
    assert abs(pg_mean(0.0) - 0.25) < 1e-15
    assert np.allclose(pg_mean(x), pg_mean(-x), rtol=0, atol=0)
    g = np.linspace(-30, 30, 4001)
    assert np.allclose(pg_mean(g), pg_mean(-g))
    sw = np.array([1e-4 * (1 - 1e-9), 1e-4 * (1 + 1e-9)])
    assert abs(np.diff(pg_mean(sw))[0]) < 1e-12
    assert abs(pg_mean(1000.0) * 2000.0 - 1) < 1e-12
    assert abs(pg_mean(1.0) - math.tanh(0.5) / 2.0) < 1e-15
    small = 3e-6
    assert abs(pg_mean(small) - (0.25 - small ** 2 / 48)) < 1e-15


def test_quadratic_bound():
    psi = np.linspace(-12, 12, 241)
    for xi in [0.0, 1e-6, 0.3, 1.0, 4.0, 15.0]:
        for y in (0, 1):
            lb = quad_bound(y, psi, xi)
            ex = y * psi - softplus(psi)
            assert np.all(lb <= ex + 1e-12), (xi, y)
            for s in (xi, -xi):
                assert abs(quad_bound(y, s, xi) - (y * s - softplus(s))) < 1e-10
                h = 1e-5
                gl = (quad_bound(y, s + h, xi) - quad_bound(y, s - h, xi)) / (2 * h)
                ge = ((y * (s + h) - softplus(s + h)) - (y * (s - h) - softplus(s - h))) / (2 * h)
                assert abs(gl - ge) < 1e-6
    # bound constant matches its definition
    for xi in [0.0, 0.5, 30.0]:
        assert abs(bound_constant(xi) - (-math.log(2 * math.cosh(xi / 2)) + 0.5 * pg_mean(xi) * xi * xi)) < 1e-12


def test_stable_likelihood():
    x = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):  # underflow to 0 is benign
        sp = softplus(x)
        sg = sigmoid(x)
        ls = mfa.log_sigmoid(x)
        l2 = mfa.log_2cosh_half(x)
    assert np.all(np.isfinite(sp)) and np.all(np.isfinite(sg)) and np.all(np.isfinite(ls))
    assert abs(sp[1] - math.exp(-50)) < 1e-30 and abs(sp[3] - (50 + math.exp(-50))) < 1e-12
    assert abs(ls[0] + 1000) < 1e-12 and abs(l2[4] - 500) < 1e-12
    assert abs(sp[2] - math.log(2)) < 1e-15


def test_masks():
    Y, _, _ = gen_mfa(seed=1)
    rng = np.random.default_rng(3)
    mask = rng.random(Y.shape) < 0.8
    for mode in ("isotropic", "none"):
        kw = dict(residual_mode=mode, max_iter=15, tol=0, seed=4, K=2, d=2)
        Y2 = Y.copy()
        Y2[~mask] = 1 - Y2[~mask]           # change values at excluded cells
        Y3 = Y.copy()
        Y3[~mask] = np.nan
        r1 = mfa.fit(Y, mask=mask, **kw)
        r2 = mfa.fit(Y2, mask=mask, **kw)
        r3 = mfa.fit(Y3, **kw)
        assert np.allclose(r1[2]["bound"], r2[2]["bound"], rtol=0, atol=0)
        assert np.allclose(r1[2]["bound"], r3[2]["bound"], rtol=1e-12)
        assert np.array_equal(r1[0].B, r2[0].B)
        # an observed zero changes the fit
        i, j = np.argwhere(mask & (Y == 0))[0]
        Y4 = Y.copy(); Y4[i, j] = 1
        r4 = mfa.fit(Y4, mask=mask, **kw)
        assert abs(r4[2]["bound"][-1] - r1[2]["bound"][-1]) > 1e-6


def test_dense_sparse_parity():
    import scipy.sparse as sp
    Y, _, _ = gen_mfa(seed=2)
    mask = np.random.default_rng(5).random(Y.shape) < 0.85
    kw = dict(K=2, d=2, max_iter=8, tol=0, seed=1)
    a = mfa.fit(Y, mask=mask, **kw)
    b = mfa.fit(sp.csr_matrix(Y), mask=mask, **kw)
    assert np.allclose(a[2]["bound"], b[2]["bound"], rtol=1e-13)
    rows, cols = np.nonzero(~mask)
    assert np.allclose(mfa.predict_pairs(a[0], a[1], rows, cols), mfa.predict_pairs(b[0], b[1], rows, cols))


def test_invalid_inputs():
    Y, _, _ = gen_mfa(seed=0, n=10, m=6)
    bad = [
        lambda: mfa.fit(np.zeros(5), K=1, d=1),
        lambda: mfa.fit(np.where(Y > 0, 2.0, 0.0), K=1, d=1),
        lambda: mfa.fit(Y, K=1, d=1, lambda_mu=0.0),
        lambda: mfa.fit(Y, K=1, d=1, lambda_B=-1.0),
        lambda: mfa.fit(Y, K=1, d=1, residual_variance=0.0),
        lambda: mfa.fit(Y, K=1, d=1, residual_mode="none", learn_residual_variance=True),
        lambda: mfa.fit(Y, K=1, d=1, alpha=1.0),
        lambda: mfa.fit(Y, K=1, d=1, mask=np.zeros(Y.shape, bool)),
        lambda: mfa.fit(np.full(Y.shape, np.nan), K=1, d=1),
        lambda: mfa.fit(Y, K=1, d=1, mask=np.ones((3, 3), bool)),
        lambda: mfa.fit(np.where(Y > 0, np.inf, 0.0), K=1, d=1),
    ]
    for f in bad:
        try:
            f()
        except (ValueError, np.linalg.LinAlgError):
            continue
        raise AssertionError("invalid input accepted")
    # NaN at an excluded cell is fine
    Yn = Y.copy(); Yn[0, 0] = np.nan
    mfa.fit(Yn, K=1, d=1, max_iter=2)


def test_missing_rows_cols():
    Y, _, _ = gen_mfa(seed=3, n=40, m=15)
    mask = np.ones(Y.shape, bool)
    mask[7] = False          # fully unobserved row
    mask[:, 4] = False       # fully unobserved column
    model, state, info = mfa.fit(Y, K=3, d=2, mask=mask, max_iter=30, seed=2)
    assert np.allclose(state.r[7], model.pi) and np.allclose(state.m[7], 0)
    assert np.allclose(state.S(7), np.eye(2)[None].repeat(3, 0))
    assert not state.active[7] and info["n"] == 40
    # unobserved column: posterior mean of (mu, B) is the prior mode 0
    assert np.allclose(model.mu[:, 4], 0) and np.allclose(model.B[:, 4], 0)
    # prior predictive for the missing row: sum_l pi_l E sigma(N(mu_lj, t + |b_lj|^2))
    j = 2
    pr = sum(model.pi[l] * mfa.expected_sigmoid(model.mu[l, j], model.t[l] + model.B[l, j] @ model.B[l, j], 40)
             for l in range(3))
    assert abs(mfa.predict_pairs(model, state, [7], [j], Q=40)[0] - pr) < 1e-10
    # new-row inference on an all-missing row gives the same
    st2, _ = mfa.infer_rows(model, Y[[7]], mask=np.zeros((1, Y.shape[1]), bool))
    assert np.allclose(st2.r[0], model.pi) and np.allclose(st2.m[0], 0)


def test_probability_averaging():
    # asymmetric two-component example
    r = np.array([0.8, 0.2])
    a = np.array([-3.0, 4.0])
    s2 = np.array([4.0, 0.5])
    e_sig = np.sum(r * mfa.expected_sigmoid(a, s2, 60))
    assert abs(e_sig - sigmoid(np.sum(r * a))) > 0.05
    assert abs(e_sig - np.sum(r * sigmoid(a))) > 0.02
    # the model prediction implements the first
    rng = np.random.default_rng(0)
    model = random_model(2, 3, 1, 0.5, rng)
    model.mu[:, 0] = a; model.B[:, 0, 0] = 0.0; model.t[:] = 0.0
    model.t[:] = [4.0, 0.5]
    st = mfa._empty_state(1, 2, 1)
    st.r[0] = r
    assert abs(mfa.predict_pairs(model, st, [0], [0], Q=60)[0] - e_sig) < 1e-12


def test_serialization():
    Y, _, _ = gen_mfa(seed=4)
    model, state, _ = mfa.fit(Y, K=2, d=2, max_iter=10, learn_residual_variance=True)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(Path(tmp) / "m.npz")
        m2 = MFAModel.load(Path(tmp) / "m.npz")
    rows, cols = np.nonzero(np.ones(Y.shape, bool))
    assert np.array_equal(mfa.predict_pairs(model, state, rows, cols), mfa.predict_pairs(m2, state, rows, cols))
    assert m2.opts.learn_residual_variance and m2.opts == model.opts


# ----------------------------------------------------------------------------
# 7.2 model tests
# ----------------------------------------------------------------------------
def test_A_dense_gaussian_reference():
    rng = np.random.default_rng(11)
    m, d, t = 7, 3, 0.4
    y = (rng.random(m) < 0.5).astype(float)
    obs = rng.random(m) < 0.8
    mu, B, xi = rng.normal(size=m), rng.normal(size=(m, d)), np.abs(rng.normal(1, 1, m))
    ref = local_row_reference(y, obs, mu, B, t, xi)
    o = obs
    Bo, muo, kap, w = B[o], mu[o], y[o] - 0.5, pg_mean(xi[o])
    k = Bo.shape[0]
    P = np.block([[np.eye(d) + Bo.T @ Bo / t, -Bo.T / t], [-Bo / t, np.eye(k) / t + np.diag(w)]])
    hh = np.r_[-Bo.T @ muo / t, muo / t + kap]
    C = np.linalg.solve(P, np.eye(d + k))
    mean = C @ hh
    assert np.allclose(mean[:d], ref["m"], atol=1e-10)
    assert np.allclose(C[:d, :d], ref["S"], atol=1e-10)
    assert np.allclose(mean[d:], ref["psibar"], atol=1e-10)
    assert np.allclose(np.diag(C)[d:], ref["vpsi"], atol=1e-10)
    a = 1 + t * w
    assert np.allclose(C[:d, d:], ref["S"] @ Bo.T / a, atol=1e-10)            # Cov(z, psi_j) = S b_j / a_j
    assert np.allclose(ref["Ezpsi"], C[:d, d:] + np.outer(mean[:d], mean[d:]), atol=1e-10)
    # the chunked e_pass gives the same moments and score
    Y = np.vstack([y, 1 - y]).astype(np.uint8)
    X = np.vstack([~obs, ~obs])
    model = MFAModel(np.array([1.0]), mu[None], B[None], np.array([t]), MFAOptions(residual_variance=t))
    st = mfa._empty_state(2, 1, d)
    xs = np.broadcast_to(xi, (1, 2, m)).copy()
    e_pass(Y, X, model, st, xs.copy(), accumulate=False)
    assert np.allclose(st.m[0, 0], ref["m"], atol=1e-10) and np.allclose(st.S(0)[0], ref["S"], atol=1e-10)
    assert abs(st.score[0, 0] - ref["score"]) < 1e-9


def _log_gauss_integral(kap, w, cxi, mu, C):
    """log int prod_j exp(kappa psi - w psi^2/2 + c) N(psi; mu, C) dpsi (dense, small)."""
    Ci = np.linalg.inv(C)
    P = Ci + np.diag(w)
    lin = Ci @ mu + kap
    sgn1, ld1 = np.linalg.slogdet(C)
    sgn2, ld2 = np.linalg.slogdet(P)
    return float(np.sum(cxi) - 0.5 * ld1 - 0.5 * ld2 + 0.5 * lin @ np.linalg.solve(P, lin) - 0.5 * mu @ Ci @ mu)


def test_B_bound_normalisation():
    from scipy.integrate import quad
    rng = np.random.default_rng(12)
    m, d, t = 6, 2, 0.3
    y = (rng.random(m) < 0.4).astype(float)
    obs = np.ones(m, bool)
    mu, B, xi = rng.normal(size=m), rng.normal(size=(m, d)), np.abs(rng.normal(1, 1, m))
    ref = local_row_reference(y, obs, mu, B, t, xi)
    direct = _log_gauss_integral(y - 0.5, pg_mean(xi), bound_constant(xi), mu, B @ B.T + t * np.eye(m))
    assert abs(ref["score"] - direct) < 1e-9
    # 1-D: bound <= true logistic-normal evidence (positive and zero residual)
    for (t1, b1, mu1, y1) in [(0.25, 1.3, -0.5, 1.0), (1.0, 0.4, 2.0, 0.0), (0.0, 1.1, 0.3, 1.0)]:
        s2 = b1 * b1 + t1
        ev = math.log(quad(lambda p: math.exp(y1 * p - float(softplus(p))) * math.exp(-(p - mu1) ** 2 / (2 * s2))
                           / math.sqrt(2 * math.pi * s2), -60, 60, epsabs=1e-13, limit=400)[0])
        xi1 = np.array([0.7])
        best = -np.inf
        for _ in range(60):  # coordinate ascent in xi increases the bound towards the evidence
            rr = local_row_reference(np.array([y1]), np.array([True]), np.array([mu1]), np.array([[b1]]), t1, xi1)
            assert rr["score"] <= ev + 1e-10
            assert rr["score"] >= best - 1e-12
            best = rr["score"]
            xi1 = np.sqrt(rr["psibar"] ** 2 + rr["vpsi"])
        assert ev - best < 0.1  # tight-ish at the optimum, never above


def test_C_moment_sensitive_xi():
    # same posterior mean, different variances -> different xi and PG weights
    Y = np.array([[1, 1]], np.uint8)
    X = np.zeros((1, 2), bool)
    BM = np.array([[0.5, 0.5]])
    Q = np.array([[0.1, 3.0]])
    IA = np.ones((1, 2))
    XI, PB, E2 = np.empty((1, 2)), np.empty((1, 2)), np.empty((1, 2))
    mfa._k_post_iso(Y, X, BM, Q, IA, np.zeros(2), 0.0, XI, PB, E2)
    assert PB[0, 0] == PB[0, 1]
    assert XI[0, 1] > XI[0, 0] + 1.0 and np.allclose(XI ** 2, PB ** 2 + Q)
    assert pg_mean(XI[0, 0]) > pg_mean(XI[0, 1])
    assert not np.allclose(XI, np.abs(PB))
    # in a fit the stored xi after one pass equals sqrt(psibar^2 + v), not |psibar|
    Yd, _, _ = gen_mfa(seed=6, n=12, m=8)
    rng = np.random.default_rng(1)
    model = random_model(2, 8, 2, 0.5, rng)
    Yu, X = prepare_data(Yd)
    xi0 = mfa.init_xi(model, 12, np.float64)
    xs = xi0.copy()
    st = mfa._empty_state(12, 2, 2)
    e_pass(Yu, X, model, st, xs, accumulate=True)
    for (i, l), rr in ref_pass(Yu, X, model, xi0).items():
        assert np.allclose(xs[l, i], np.sqrt(rr["psibar"] ** 2 + rr["vpsi"]), atol=1e-12)
        assert not np.allclose(xs[l, i], np.abs(rr["psibar"]))


def _frozen_obj_iso(theta, l, j, refs, r, Y, X, t, Lam, Bold):
    """Frozen expected objective for column j of component l (positive residual)."""
    f = -0.5 * theta @ Lam @ theta
    for i in range(Y.shape[0]):
        if X[i, j]:
            continue
        rr = refs[i, l]
        jj = int(np.sum(~X[i, :j]))   # position of column j among observed columns
        H = np.block([[np.ones((1, 1)), rr["m"][None]], [rr["m"][:, None], rr["S"] + np.outer(rr["m"], rr["m"])]])
        e = np.r_[rr["psibar"][jj], rr["m"] * rr["psibar"][jj] + rr["S"] @ Bold[j] / rr["a"][jj]]
        E2 = rr["psibar"][jj] ** 2 + rr["vpsi"][jj]
        f += r[i, l] * (-(E2 - 2 * theta @ e + theta @ H @ theta) / (2 * t))
    return f


def _frozen_obj_none(theta, l, j, refs, r, Y, X, Lam):
    f = -0.5 * theta @ Lam @ theta
    for i in range(Y.shape[0]):
        if X[i, j]:
            continue
        rr = refs[i, l]
        jj = int(np.sum(~X[i, :j]))
        H = np.block([[np.ones((1, 1)), rr["m"][None]], [rr["m"][:, None], rr["S"] + np.outer(rr["m"], rr["m"])]])
        f += r[i, l] * ((Y[i, j] - 0.5) * theta @ np.r_[1.0, rr["m"]] - 0.5 * rr["w"][jj] * theta @ H @ theta)
    return f


def test_D_mstep_optimality():
    from scipy.optimize import minimize_scalar
    Yd, _, _ = gen_mfa(seed=7, n=20, m=6)
    mask = np.random.default_rng(2).random(Yd.shape) < 0.85
    Yu, X = prepare_data(Yd, mask)
    for mode in ("isotropic", "none"):
        rng = np.random.default_rng(3)
        model = random_model(2, 6, 2, 0.4, rng, mode=mode, lambda_mu=0.7, lambda_B=1.3,
                             learn_residual_variance=(mode == "isotropic"))
        n = Yu.shape[0]
        st = mfa._empty_state(n, 2, 2)
        if mode == "isotropic":
            xi = mfa.init_xi(model, n, np.float64)
            xs = xi.copy()
        else:
            xs = None
        _, stats = e_pass(Yu, X, model, st, xs, accumulate=True)
        if mode == "isotropic":
            refs = ref_pass(Yu, X, model, xi)
        else:
            # xi used in this pass: sqrt((mu + b'0)^2 + b'Ib) from the empty state
            xin = np.stack([np.broadcast_to(np.sqrt(model.mu[l] ** 2 + np.sum(model.B[l] ** 2, 1)), (n, 6))
                            for l in range(2)])
            refs = ref_pass(Yu, X, model, xin)
        r = st.r.copy()
        for (i, l), rr in refs.items():
            assert np.allclose(st.m[i, l], rr["m"], atol=1e-10)
        Bold, told = model.B.copy(), model.t.copy()
        Lam = np.diag([0.7, 1.3, 1.3])
        m_step(model, stats)
        for l in range(2):
            for j in range(6):
                th = np.r_[model.mu[l, j], model.B[l, j]]
                if mode == "isotropic":
                    f = lambda v: _frozen_obj_iso(v, l, j, refs, r, Yu, X, told[l], Lam, Bold[l])
                else:
                    f = lambda v: _frozen_obj_none(v, l, j, refs, r, Yu, X, Lam)
                h = 1e-5
                g = np.array([(f(th + h * e) - f(th - h * e)) / (2 * h) for e in np.eye(3)])
                assert np.max(np.abs(g)) < 1e-5 * max(1.0, abs(f(th))), (mode, l, j, g)
                # and it is a maximum: a perturbation lowers the frozen objective
                assert f(th) >= f(th + 1e-3 * rng.normal(size=3))
        if mode == "isotropic":
            # inverse-gamma t update vs 1-D optimisation of the frozen objective
            o = model.opts
            for l in range(2):
                sse, neff = 0.0, 0.0
                for (i, ll), rr in refs.items():
                    if ll != l:
                        continue
                    obs = np.flatnonzero(~X[i])
                    for jj, j in enumerate(obs):
                        th = np.r_[model.mu[l, j], model.B[l, j]]
                        H = np.block([[np.ones((1, 1)), rr["m"][None]], [rr["m"][:, None], rr["S"] + np.outer(rr["m"], rr["m"])]])
                        e = np.r_[rr["psibar"][jj], rr["m"] * rr["psibar"][jj] + rr["S"] @ Bold[l, j] / rr["a"][jj]]
                        sse += r[i, l] * (rr["psibar"][jj] ** 2 + rr["vpsi"][jj] - 2 * th @ e + th @ H @ th)
                        neff += r[i, l]
                obj = lambda lt: -(-(neff / 2 + o.a0 + 1) * lt - (sse / 2 + o.b0) * np.exp(-lt))
                res = minimize_scalar(obj, bounds=(-10, 5), method="bounded", options=dict(xatol=1e-12))
                assert abs(np.exp(res.x) - model.t[l]) < 1e-6 * model.t[l], (np.exp(res.x), model.t[l])


def test_E_monotonicity():
    worst = np.inf
    for seed in range(3):
        Yd, _, _ = gen_mfa(seed=20 + seed, n=50, m=20, mean_shift=-1.0 if seed == 2 else 0.0)
        mask = np.random.default_rng(seed).random(Yd.shape) < (0.7 + 0.1 * seed)
        for mode, learn in (("isotropic", False), ("isotropic", True), ("none", False)):
            for sweeps in (1, 3):
                for init in ("random", "kmeans"):
                    _, _, info = mfa.fit(Yd, K=3, d=2, mask=mask, residual_mode=mode, learn_residual_variance=learn,
                                         local_sweeps=sweeps, max_iter=40, tol=0, seed=seed, init=init)
                    b = np.array(info["bound"])
                    rel = np.diff(b) / np.abs(b[:-1])
                    worst = min(worst, rel.min())
                    assert rel.min() > -1e-12, (seed, mode, learn, sweeps, init, rel.min())
                    assert info["monotone"]
    # reduced-precision storage used on the full data: report, loose check
    Yd, _, _ = gen_mfa(seed=30, n=200, m=60)
    _, _, info = mfa.fit(Yd, K=3, d=2, max_iter=60, tol=0, big_dtype="float32", xi_dtype="float16", seed=1)
    b = np.array(info["bound"])
    rel = np.diff(b) / np.abs(b[:-1])
    print(f"    float64 smallest relative step {worst:.2e}; float32/float16 smallest step {rel.min():.2e}")
    assert rel.min() > -1e-6


def test_F_empty_components_and_missing_cells():
    Yd, _, _ = gen_mfa(seed=8, n=60, m=15, K=1, pi=(1.0,))
    rng = np.random.default_rng(1)
    for mode in ("isotropic", "none"):
        model = random_model(4, 15, 2, 0.3, rng, mode=mode)
        model.mu[3] = -60.0          # component that no row can use
        model.pi = np.array([0.33, 0.33, 0.33, 0.01])
        mod, st, info = mfa.fit(Yd, K=4, d=2, model0=model, opts=model.opts, max_iter=25, tol=0)
        assert np.all(np.isfinite(mod.mu)) and np.all(np.isfinite(mod.B)) and np.all(mod.pi > 0)
        assert mod.pi[3] < 0.05 and info["monotone"]
        assert np.all(np.isfinite(info["bound"]))
    # an exactly empty component in the M-step: finite prior-mode solution, positive weight
    for learn in (False, True):
        model = random_model(3, 15, 2, 0.3, rng, learn_residual_variance=learn)
        Yu, X = prepare_data(Yd)
        xs = mfa.init_xi(model, 60, np.float64)
        st = mfa._empty_state(60, 3, 2)
        _, stats = e_pass(Yu, X, model, st, xs, accumulate=True)
        for key in ("G", "gz", "gS", "v"):
            stats[key][2] = 0.0
        stats["N"][2] = 0.0; stats["Neff"][2] = 0.0
        m_step(model, stats)
        assert np.allclose(model.mu[2], 0) and np.allclose(model.B[2], 0) and model.pi[2] > 0
        assert np.isfinite(model.t).all() and model.t[2] > 0
    # missing-cell predictive variance is t + b'Sb, not the observed-cell posterior variance
    rng = np.random.default_rng(2)
    model = random_model(1, 6, 2, 0.5, rng)
    y = (rng.random(6) < 0.5).astype(float)
    xi = np.abs(rng.normal(1, 1, 6))
    obs = np.ones(6, bool); obs[5] = False
    rr = local_row_reference(y, obs, model.mu[0], model.B[0], 0.5, xi)
    st = mfa._empty_state(1, 1, 2)
    st.m[0, 0], st.Sv[0, 0] = rr["m"], mfa.vech(rr["S"])
    A, S2 = mfa.pair_moments(model, st, [0, 0], [5, 0])
    b5, b0 = model.B[0, 5], model.B[0, 0]
    assert abs(S2[0, 0] - (0.5 + b5 @ rr["S"] @ b5)) < 1e-12
    assert abs(A[0, 0] - (model.mu[0, 5] + b5 @ rr["m"])) < 1e-12
    assert abs(S2[1, 0] - (0.5 + b0 @ rr["S"] @ b0)) < 1e-12 and abs(S2[1, 0] - rr["vpsi"][0]) > 1e-3


def test_G_quadrature():
    rng = np.random.default_rng(0)
    for a, s2 in [(0.3, 0.5), (-2.0, 2.0), (1.5, 0.05), (0.0, 9.0), (3.0, 25.0)]:
        e20, e40 = mfa.expected_sigmoid(a, s2, 20), mfa.expected_sigmoid(a, s2, 40)
        mc = sigmoid(a + math.sqrt(s2) * rng.standard_normal(2_000_000)).mean()
        assert abs(e40 - mc) < 2e-3, (a, s2)
        if s2 <= 2:
            assert abs(e20 - e40) < 1e-6, (a, s2)
        if abs(e20 - e40) > 1e-6:   # large variances need more nodes: report, do not assume
            print(f"    note: 20 vs 40 Gauss-Hermite nodes differ by {abs(e20 - e40):.1e} at s2={s2}")
        assert abs(e40 - mc) < 2e-3 and abs(e20 - mc) < 1e-2
    assert mfa.expected_sigmoid(0.7, 0.0, 20) == sigmoid(np.array(0.7))
    x, w = gauss_hermite(20)
    assert abs(w.sum() - 1) < 1e-14 and abs(np.sum(w * x ** 2) - 0.5) < 1e-13
    # numba predictor == numpy quadrature
    model = random_model(2, 5, 2, 0.4, rng)
    st = mfa._empty_state(3, 2, 2)
    st.r[:] = [0.3, 0.7]
    st.m[:] = rng.normal(size=(3, 2, 2))
    P = mfa.predict_pairs(model, st, [0, 1, 2], [4, 0, 3])
    A, S2 = mfa.pair_moments(model, st, [0, 1, 2], [4, 0, 3])
    ref = np.sum(st.r[[0, 1, 2]] * mfa.expected_sigmoid(A, S2, 20), 1)
    assert np.allclose(P, ref, atol=1e-14)


def test_H_gauge_invariance():
    rng = np.random.default_rng(9)
    Yd, _, _ = gen_mfa(seed=9, n=15, m=10)
    Yu, X = prepare_data(Yd, np.random.default_rng(1).random(Yd.shape) < 0.8)
    model = random_model(2, 10, 3, 0.3, rng)
    Qm, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    m2 = MFAModel(model.pi.copy(), model.mu.copy(), model.B @ Qm, model.t.copy(), model.opts)
    xi = mfa.init_xi(model, 15, np.float64)
    s1, s2 = mfa._empty_state(15, 2, 3), mfa._empty_state(15, 2, 3)
    b1, _ = e_pass(Yu, X, model, s1, xi.copy(), accumulate=False)
    b2, _ = e_pass(Yu, X, m2, s2, xi.copy(), accumulate=False)
    assert abs(b1 - b2) < 1e-9 and abs(model.log_prior() - m2.log_prior()) < 1e-9
    assert np.allclose(s2.m, np.einsum("pq,ilp->ilq", Qm, s1.m), atol=1e-10)
    assert np.allclose(s2.S(), Qm.T @ s1.S() @ Qm, atol=1e-10)
    rows, cols = np.nonzero(np.ones(Yd.shape, bool))
    assert np.allclose(mfa.predict_pairs(model, s1, rows, cols), mfa.predict_pairs(m2, s2, rows, cols), atol=1e-12)
    e1 = mfa.logit_mean_embedding(model, s1, 5)[0]
    e2 = mfa.logit_mean_embedding(m2, s2, 5)[0]
    assert np.allclose(np.abs(e1), np.abs(e2), atol=1e-4)


def test_chunked_equals_reference_and_chunk_invariance():
    Yd, _, _ = gen_mfa(seed=13, n=40, m=12)
    Yu, X = prepare_data(Yd, np.random.default_rng(3).random(Yd.shape) < 0.75)
    rng = np.random.default_rng(4)
    for mode in ("isotropic", "none"):
        model = random_model(3, 12, 2, 0.35, rng, mode=mode)
        res = []
        for budget in (1e-4, 100.0):
            model.opts.chunk_budget_mb = budget
            st = mfa._empty_state(40, 3, 2)
            xi = mfa.init_xi(model, 40, np.float64) if mode == "isotropic" else None
            b, stats = e_pass(Yu, X, model, st, xi, accumulate=True)
            res.append((b, st, stats))
        assert abs(res[0][0] - res[1][0]) < 1e-9
        assert np.allclose(res[0][2]["G"], res[1][2]["G"]) and np.allclose(res[0][1].r, res[1][1].r)


def test_new_row_inference():
    Yd, _, _ = gen_mfa(seed=14, n=80, m=20)
    model, state, _ = mfa.fit(Yd[:60], K=3, d=2, max_iter=40, seed=1)
    snap = (model.mu.copy(), model.B.copy(), model.pi.copy(), model.t.copy())
    ctx = np.random.default_rng(0).random((20, 20)) < 0.5
    Ynew = Yd[60:].copy()
    st, inf = mfa.infer_rows(model, Ynew, mask=ctx)
    assert all(np.array_equal(a, b) for a, b in zip(snap, (model.mu, model.B, model.pi, model.t)))
    Ynew2 = Ynew.copy(); Ynew2[~ctx] = 1 - Ynew2[~ctx]   # held-out cells must not matter
    st2, _ = mfa.infer_rows(model, Ynew2, mask=ctx)
    assert np.allclose(st.r, st2.r) and np.allclose(st.m, st2.m)
    rows, cols = np.nonzero(~ctx)
    p = mfa.predict_pairs(model, st, rows, cols)
    assert np.all((p > 0) & (p < 1))


def test_float32_kernel_matches_float64():
    rng = np.random.default_rng(0)
    c, m = 7, 300
    Y = (rng.random((c, m)) < 0.4).astype(np.uint8)
    X = rng.random((c, m)) < 0.1
    XI = np.concatenate([np.abs(rng.normal(0, 3, (c, m - 20))), rng.uniform(0, 2e-3, (c, 20))], 1)
    mu = rng.normal(size=m)
    for t in (0.0, 0.25, 1.0):
        o64 = [np.empty((c, m)) for _ in range(3)]; c64 = np.empty(c)
        o32 = [np.empty((c, m), np.float32) for _ in range(3)]; c32 = np.empty(c)
        mfa._k_local_terms(Y, X, XI, mu, t, *o64, c64)
        mfa._k_local_terms32(Y, X, XI.astype(np.float32), mu.astype(np.float32), t, *o32, c32)
        for a, b in zip(o64, o32):
            assert np.allclose(a, b, rtol=2e-6, atol=1e-7)
        assert np.allclose(c64, c32, rtol=1e-6)


def test_px_step():
    """Optional PX step: logit distributions (hence predictions) unchanged, bound monotone."""
    Yd, _, _ = gen_mfa(seed=40, n=60, m=20)
    mask = np.random.default_rng(1).random(Yd.shape) < 0.8
    Yu, X = prepare_data(Yd, mask)
    rng = np.random.default_rng(2)
    for mode in ("isotropic", "none"):
        model = random_model(3, 20, 2, 0.3, rng, mode=mode)
        st = mfa._empty_state(60, 3, 2)
        xs = mfa.init_xi(model, 60, np.float64) if mode == "isotropic" else None
        _, stats = e_pass(Yu, X, model, st, xs, accumulate=True)
        m_step(model, stats)
        rows, cols = np.nonzero(~mask)
        p0 = mfa.predict_pairs(model, st, rows, cols)
        A0, S0 = mfa.pair_moments(model, st, rows, cols)
        gains = mfa.px_step(model, st, stats)
        assert max(gains) > 0
        A1, S1 = mfa.pair_moments(model, st, rows, cols)
        assert np.allclose(A0, A1, atol=1e-10) and np.allclose(S0, S1, atol=1e-10)
        assert np.allclose(p0, mfa.predict_pairs(model, st, rows, cols), atol=1e-12)
        for learn in ((False, True) if mode == "isotropic" else (False,)):
            _, _, info = mfa.fit(Yd, K=3, d=2, mask=mask, residual_mode=mode, learn_residual_variance=learn,
                                 px=True, max_iter=40, tol=0, seed=3)
            b = np.array(info["bound"])
            assert np.all(np.diff(b) / np.abs(b[:-1]) > -1e-12) and info["monotone"]


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    warnings.simplefilter("error", RuntimeWarning)
    fails = 0
    for f in TESTS:
        try:
            f()
            print(f"PASS {f.__name__}", flush=True)
        except Exception:
            fails += 1
            print(f"FAIL {f.__name__}", flush=True)
            traceback.print_exc()
    print(f"{len(TESTS) - fails}/{len(TESTS)} passed")
    sys.exit(fails)
