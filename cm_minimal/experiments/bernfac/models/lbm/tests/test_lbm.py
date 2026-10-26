"""Tests for the Bernoulli latent block model (handoff 7.4 A-H plus a few 7.1 checks).

Run as plain python:  python tests/test_lbm.py
"""
from __future__ import annotations

import itertools
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, xlogy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lbm  # noqa: E402
from lbm import (BinaryData, block_objective, block_stats, elbo_parts, fit_lbm,  # noqa: E402
                 pg_block_step, pg_mean, sigmoid, softplus, solve_blocks_newton, solve_blocks_pg)


def _rand_problem(rng, n=7, m=5, Kr=3, Kc=2, miss=0.3):
    Y = (rng.random((n, m)) < 0.4).astype(float)
    M = rng.random((n, m)) > miss
    R = lbm.random_responsibilities(n, Kr, rng)
    C = lbm.random_responsibilities(m, Kc, rng)
    B = rng.normal(size=(Kr, Kc))
    return Y, M, R, C, B


def _full_F(data, R, C, B, pi, rho, lam=1.0, ar=1.1, ac=1.1):
    N, T = block_stats(data, R, C)
    return elbo_parts(N, T, B, R, C, pi, rho, lam, ar, ac, data.active_rows, data.active_cols)["total"]


# ---------------------------------------------------------------- shared utilities
def test_utilities():
    x = np.array([0, 1e-8, -1e-8, 1, -1, 20, -20, 1000, -1000.0])
    f = pg_mean(x)
    assert np.all(np.isfinite(f)) and np.all(f > 0) and np.all(f <= 0.25)
    assert f[0] == 0.25 and np.allclose(pg_mean(x), pg_mean(-x))
    assert abs(pg_mean(np.array([1e3]))[0] * 2e3 - 1) < 1e-12
    xs = np.linspace(-1e-3, 1e-3, 101)  # continuity across the small-x branch
    assert np.max(np.abs(np.diff(pg_mean(xs)))) < 1e-8
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        s = softplus(np.array([-1000, 0, 1000.0]))
    assert np.allclose(s, [0, np.log(2), 1000])
    # quadratic bound below exact log-lik, tight at psi = +/- xi
    for xi in [0.0, 0.5, 3.0]:
        w = pg_mean(np.array([xi]))[0]
        c = -(xi / 2 + np.log1p(np.exp(-xi))) + 0.5 * w * xi ** 2
        for y in (0, 1):
            psi = np.linspace(-8, 8, 401)
            exact = y * psi - softplus(psi)
            bound = (y - 0.5) * psi - 0.5 * w * psi ** 2 + c
            assert np.all(bound <= exact + 1e-12)
            for p0 in (xi, -xi):
                assert abs(((y - 0.5) * p0 - 0.5 * w * p0 ** 2 + c) - (y * p0 - softplus(p0))) < 1e-12


def test_invalid_inputs():
    for bad in (np.array([[0, 2.0]]), np.array([[0, np.inf]]), np.zeros(3)):
        try:
            BinaryData(bad); raise AssertionError("accepted invalid input")
        except ValueError:
            pass
    try:
        BinaryData(np.zeros((2, 2)), mask=np.zeros((2, 2), bool)); raise AssertionError
    except ValueError:
        pass
    # NaN outside the mask is fine; inside the mask it is rejected
    Y = np.array([[0, np.nan], [1, 0]])
    BinaryData(Y)
    try:
        BinaryData(Y, mask=np.ones((2, 2), bool)); raise AssertionError
    except ValueError:
        pass
    try:
        fit_lbm(BinaryData(np.zeros((3, 3))), 1, 1, lam=0.0); raise AssertionError
    except ValueError:
        pass


# ---------------------------------------------------------------- A. sufficient statistics
def test_A_sufficient_stats():
    rng = np.random.default_rng(0)
    Y, M, R, C, B = _rand_problem(rng)
    d = BinaryData(Y, M)
    N, T = block_stats(d, R, C)
    Nl = np.zeros_like(N); Tl = np.zeros_like(T)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if M[i, j]:
                for a in range(R.shape[1]):
                    for b in range(C.shape[1]):
                        Nl[a, b] += R[i, a] * C[j, b]
                        Tl[a, b] += R[i, a] * C[j, b] * Y[i, j]
    assert np.allclose(N, Nl, atol=1e-12) and np.allclose(T, Tl, atol=1e-12)
    # fully observed: N = (R'1)(C'1)', observed zeros counted
    d2 = BinaryData(Y)
    N2, _ = block_stats(d2, R, C)
    assert np.allclose(N2, np.outer(R.sum(0), C.sum(0)))
    # value stored at an excluded cell is ignored
    Y3 = Y.copy(); Y3[~M] = np.nan
    N3, T3 = block_stats(BinaryData(Y3, M), R, C)
    assert np.allclose(N3, N) and np.allclose(T3, T)


# ---------------------------------------------------------------- B. PG scalar steps
def test_B_pg_vs_newton():
    rng = np.random.default_rng(1)
    N = rng.uniform(0, 50, size=(30, 20)); N[0, 0] = 0.0
    T = rng.uniform(0, 1, size=N.shape) * N
    T[1, :] = 0.0; T[2, :] = N[2, :]  # all failures / all successes
    for lam in (0.01, 0.1, 1.0, 10.0):
        B = rng.normal(0, 3, size=N.shape)
        J = block_objective(N, T, B, lam)
        for _ in range(2000):
            Bn = pg_block_step(N, T, B, lam)
            Jn = block_objective(N, T, Bn, lam)
            assert np.all(Jn >= J - 1e-10 * (1 + np.abs(J))), "PG step decreased J"
            B, J = Bn, Jn
        Bnt = solve_blocks_newton(N, T, lam)
        score = T - N * sigmoid(Bnt) - lam * Bnt
        assert np.max(np.abs(score)) < 1e-8
        assert np.max(np.abs(B - Bnt)) < 1e-6, (lam, np.max(np.abs(B - Bnt)))
        assert Bnt[0, 0] == 0.0 and pg_block_step(N, T, B, lam)[0, 0] == 0.0
    # unpenalised closed form check for interior blocks with tiny lambda
    b = solve_blocks_newton(np.array([10.0]), np.array([3.0]), 1e-10)
    assert abs(b[0] - np.log(3 / 7)) < 1e-6


# ---------------------------------------------------------------- C. assignment updates
def test_C_assignment_updates():
    rng = np.random.default_rng(2)
    for trial in range(20):
        Y, M, R, C, B = _rand_problem(rng, n=9, m=6, Kr=3, Kc=4)
        d = BinaryData(Y, M)
        pi = rng.dirichlet(np.ones(3)); rho = rng.dirichlet(np.ones(4))
        F0 = _full_F(d, R, C, B, pi, rho)
        R1 = lbm.update_rows(d, C, B, pi)
        assert np.allclose(R1.sum(1), 1)
        F1 = _full_F(d, R1, C, B, pi, rho)
        C1 = lbm.update_cols(d, R1, B, rho)
        assert np.allclose(C1.sum(1), 1)
        F2 = _full_F(d, R1, C1, B, pi, rho)
        assert F1 >= F0 - 1e-10 and F2 >= F1 - 1e-10
        # the row update is the exact coordinate maximiser: random perturbations do not help
        for _ in range(5):
            Rp = np.abs(R1 + 0.05 * rng.random(R1.shape)); Rp /= Rp.sum(1, keepdims=True)
            assert _full_F(d, Rp, C, B, pi, rho) <= F1 + 1e-10
        # explicit loop for one row score
        i = 0
        s = np.log(pi).copy()
        for a in range(3):
            for j in range(6):
                if M[i, j]:
                    s[a] += sum(C[j, b] * (Y[i, j] * B[a, b] - softplus(B[a, b])) for b in range(4))
        assert np.allclose(R1[0], np.exp(s - logsumexp(s)))


# ---------------------------------------------------------------- D. exact enumeration
def test_D_exact_enumeration():
    rng = np.random.default_rng(3)
    for trial in range(10):
        Y = (rng.random((2, 2)) < 0.5).astype(float)
        B = rng.normal(0, 2, (2, 2)); pi = rng.dirichlet([1, 1]); rho = rng.dirichlet([1, 1])
        terms = []
        for c in itertools.product(range(2), repeat=2):
            for dd in itertools.product(range(2), repeat=2):
                lp = sum(np.log(pi[a]) for a in c) + sum(np.log(rho[b]) for b in dd)
                lp += sum(Y[i, j] * B[c[i], dd[j]] - softplus(B[c[i], dd[j]]) for i in range(2) for j in range(2))
                terms.append(lp)
        logZ = logsumexp(terms)
        data = BinaryData(Y)
        # label ELBO (no parameter priors: lam=0 term, alpha=1) at optimised and random R, C
        def label_elbo(R, C):
            N, T = block_stats(data, R, C)
            return elbo_parts(N, T, B, R, C, pi, rho, 0.0, 1.0, 1.0)["total"]
        R = lbm.random_responsibilities(2, 2, rng); C = lbm.random_responsibilities(2, 2, rng)
        assert label_elbo(R, C) <= logZ + 1e-12
        for _ in range(50):
            R = lbm.update_rows(data, C, B, pi); C = lbm.update_cols(data, R, B, rho)
        e = label_elbo(R, C)
        assert e <= logZ + 1e-12, (e, logZ)
        assert logZ - e < 5.0  # sane gap


# ---------------------------------------------------------------- E. full schedule monotone
def _check_monotone(model, tol=1e-10):
    tr = model.info["history"]["elbo"]
    assert np.all(np.diff(tr) >= -tol * np.maximum(1, np.abs(tr[:-1]))), "ELBO decreased"
    for r in model.info["restarts"]:
        assert r["n_decrease"] == 0, r


def test_E_full_schedule():
    rng = np.random.default_rng(4)
    n, m = 40, 30
    zr = rng.integers(0, 3, n); zc = rng.integers(0, 2, m)
    P = np.array([[0.1, 0.8], [0.6, 0.2], [0.9, 0.9]])[zr][:, zc]
    Y = (rng.random((n, m)) < P).astype(float)
    cases = {
        "generic": (Y, None),
        "all_zero": (np.zeros((n, m)), None),
        "all_one": (np.ones((n, m)), None),
        "missing": (Y, rng.random((n, m)) > 0.4),
    }
    # empty observed block: rows 0..9 never observe columns 0..9; plus a fully missing row/column
    Mb = np.ones((n, m), bool); Mb[:10, :10] = False; Mb[-1, :] = False; Mb[:, -1] = False
    cases["empty_block_and_missing_row_col"] = (Y, Mb)
    for name, (Yc, Mc) in cases.items():
        for solver in ("pg", "newton"):
            d = BinaryData(Yc, Mc)
            mdl = fit_lbm(d, 3, 4, lam=1.0, max_iter=60, tol=1e-12, n_restarts=2, seed=5,
                          block_solver=solver, mono_tol=1e-10)
            _check_monotone(mdl)
            assert np.allclose(mdl.R.sum(1), 1) and np.allclose(mdl.C.sum(1), 1)
            assert np.all(np.isfinite(mdl.B))
            if name.startswith("empty"):
                assert np.allclose(mdl.R[-1], mdl.pi) and np.allclose(mdl.C[-1], mdl.rho)
    # spectral warm start also monotone
    mdl = fit_lbm(BinaryData(Y), 3, 2, max_iter=50, tol=1e-12, n_restarts=1, init="spectral")
    _check_monotone(mdl)
    # masks: changing excluded values does not change the fit; observed zeros do
    M = rng.random((n, m)) > 0.3
    Y2 = Y.copy(); Y2[~M] = 1 - Y2[~M]
    a = fit_lbm(BinaryData(Y, M), 3, 2, max_iter=30, n_restarts=1, seed=9)
    b = fit_lbm(BinaryData(Y2, M), 3, 2, max_iter=30, n_restarts=1, seed=9)
    assert np.allclose(a.B, b.B) and np.allclose(a.R, b.R)
    Y3 = Y.copy(); Y3[M & (Y == 0)] = 1
    c = fit_lbm(BinaryData(Y3, M), 3, 2, max_iter=30, n_restarts=1, seed=9)
    assert not np.allclose(a.B, c.B)


# ---------------------------------------------------------------- F. predictive mixture
def test_F_predictive_mixture():
    rng = np.random.default_rng(5)
    R = lbm.random_responsibilities(4, 2, rng); C = lbm.random_responsibilities(3, 2, rng)
    B = rng.normal(0, 3, (2, 2))
    mdl = lbm.LBMModel(R, C, B, np.ones(2) / 2, np.ones(2) / 2, 1.0, 1.1, 1.1)
    P = lbm.predict_matrix(mdl)
    Pl = np.zeros((4, 3))
    for i in range(4):
        for j in range(3):
            Pl[i, j] = sum(R[i, a] * C[j, b] * sigmoid(B[a, b]) for a in range(2) for b in range(2))
    assert np.allclose(P, Pl)
    rows, cols = np.repeat(np.arange(4), 3), np.tile(np.arange(3), 4)
    assert np.allclose(lbm.predict_pairs(mdl, rows, cols), Pl.ravel())
    # asymmetric two-component case: mixture of probabilities != sigmoid of mixed logit
    R2 = np.array([[0.5, 0.5]]); C2 = np.array([[1.0, 0.0]]); B2 = np.array([[-6.0, 0], [2.0, 0]])
    m2 = lbm.LBMModel(R2, C2, B2, R2[0], C2[0], 1.0, 1.1, 1.1)
    p_mix = lbm.predict_matrix(m2)[0, 0]
    p_bad = sigmoid(R2 @ B2 @ C2.T)[0, 0]
    assert abs(p_mix - p_bad) > 0.15, (p_mix, p_bad)
    # prior-only prediction
    assert np.isclose(lbm.predict_prior(mdl), mdl.pi @ sigmoid(B) @ mdl.rho)


# ---------------------------------------------------------------- G. permutations
def test_G_permutation_invariance():
    rng = np.random.default_rng(6)
    Y, M, R, C, B = _rand_problem(rng, n=8, m=7, Kr=3, Kc=4)
    d = BinaryData(Y, M)
    pi = rng.dirichlet(np.ones(3)); rho = rng.dirichlet(np.ones(4))
    F = _full_F(d, R, C, B, pi, rho)
    mdl = lbm.LBMModel(R, C, B, pi, rho, 1.0, 1.1, 1.1)
    P = lbm.predict_matrix(mdl)
    pr, pc = rng.permutation(3), rng.permutation(4)
    Fp = _full_F(d, R[:, pr], C[:, pc], B[pr][:, pc], pi[pr], rho[pc])
    Pp = lbm.predict_matrix(lbm.LBMModel(R[:, pr], C[:, pc], B[pr][:, pc], pi[pr], rho[pc], 1.0, 1.1, 1.1))
    assert np.isclose(F, Fp) and np.allclose(P, Pp)
    # updates are permutation-equivariant
    R1 = lbm.update_rows(d, C[:, pc], B[pr][:, pc], pi[pr])
    assert np.allclose(R1, lbm.update_rows(d, C, B, pi)[:, pr])


# ---------------------------------------------------------------- H. one block
def test_H_one_block():
    rng = np.random.default_rng(7)
    Y = (rng.random((20, 15)) < 0.3).astype(float)
    M = rng.random((20, 15)) > 0.2
    lam = 0.5
    mdl = fit_lbm(BinaryData(Y, M), 1, 1, lam=lam, max_iter=200, tol=1e-14, n_restarts=1,
                  block_inner_steps=50)
    n1, n = Y[M].sum(), M.sum()
    b = solve_blocks_newton(np.array([float(n)]), np.array([n1]), lam)[0]
    assert abs(mdl.B[0, 0] - b) < 1e-8
    # intercept-only penalised logistic regression via a direct 1-D optimiser
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda x: -(n1 * x - n * softplus(x) - 0.5 * lam * x * x), bounds=(-10, 10),
                        method="bounded", options=dict(xatol=1e-12))
    assert abs(r.x - b) < 1e-6
    assert np.allclose(lbm.predict_matrix(mdl), sigmoid(b))
    F = mdl.info["elbo"]
    assert np.isclose(F, n1 * b - n * softplus(b) - 0.5 * lam * b * b)  # entropy/mix terms vanish


# ---------------------------------------------------------------- new rows / columns
def test_new_rows_cols():
    rng = np.random.default_rng(8)
    n, m = 60, 40
    zr = rng.integers(0, 2, n); zc = rng.integers(0, 2, m)
    P = np.array([[0.05, 0.9], [0.9, 0.05]])[zr][:, zc]
    Y = (rng.random((n, m)) < P).astype(float)
    mdl = fit_lbm(BinaryData(Y[:50]), 2, 2, max_iter=100, n_restarts=3, seed=1)
    Rn = lbm.infer_rows(mdl, Y[50:])
    assert np.allclose(Rn.sum(1), 1)
    lab_tr = mdl.R.argmax(1)
    mp, _ = lbm.align_labels(zr[:50], lab_tr, 2, 2)
    pred = np.array([mp[k] for k in Rn.argmax(1)])
    assert np.mean(pred == zr[50:]) >= 0.9
    # a new row with no observations gets pi
    Ynan = np.full((1, m), np.nan)
    Rn0 = lbm.infer_rows(mdl, np.vstack([Y[50:51], Ynan]))
    assert np.allclose(Rn0[1], mdl.pi)
    Cn = lbm.infer_cols(mdl, Y[:50, :5])
    assert np.allclose(Cn.sum(1), 1)
    # serialisation round trip
    import io
    buf = io.BytesIO(); np.savez(buf, R=mdl.R, C=mdl.C, B=mdl.B, pi=mdl.pi, rho=mdl.rho); buf.seek(0)
    z = np.load(buf)
    m2 = lbm.LBMModel(z["R"], z["C"], z["B"], z["pi"], z["rho"], 1.0, 1.1, 1.1)
    assert np.allclose(lbm.predict_matrix(m2), lbm.predict_matrix(mdl))


def test_float32_data_path():
    """The float32 large-data path gives the same fit as float64 on a small problem."""
    rng = np.random.default_rng(9)
    Y = (rng.random((50, 40)) < 0.3).astype(float); M = rng.random((50, 40)) > 0.1
    d64 = BinaryData(Y, M)
    d32 = BinaryData.from_train_arrays((Y * M).astype(np.float32), M.astype(np.float32))
    a = fit_lbm(d64, 3, 3, max_iter=20, n_restarts=1, seed=3)
    b = fit_lbm(d32, 3, 3, max_iter=20, n_restarts=1, seed=3, mono_tol=1e-6)
    assert np.allclose(a.B, b.B, atol=1e-3)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS {name}")
        except Exception:
            failed += 1; print(f"FAIL {name}"); traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
