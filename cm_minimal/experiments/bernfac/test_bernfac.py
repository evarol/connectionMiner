"""Validation tests for BernFac (handoff section 11). Run:  python test_bernfac.py"""
import numpy as np

from bernfac import (fit, objective, pg_mean, predict, sigmoid, softplus,
                     surrogate_Q, update_rows)


def test_pg_mean():
    assert abs(pg_mean(0.0) - 0.25) < 1e-15
    x = np.linspace(-30, 30, 2001)
    w = pg_mean(x)
    assert np.allclose(w, pg_mean(-x)) and np.all(w > 0) and np.all(w <= 0.25)
    eps = np.array([1e-4 - 1e-12, 1e-4 + 1e-12])                   # continuity at switch
    assert abs(np.diff(pg_mean(eps))[0]) < 1e-10
    assert abs(pg_mean(200.0) * 2 * 200.0 - 1) < 1e-12               # ~1/(2|psi|)


def test_stable_functions():
    assert np.isfinite(sigmoid(np.array([-1000.0, 1000.0]))).all()
    assert abs(softplus(np.array([1000.0]))[0] - 1000.0) < 1e-9


def _tiny(seed=0, n=12, m=9, k=3):
    rng = np.random.default_rng(seed)
    Ut, Vt = rng.standard_normal((n, k)), rng.standard_normal((m, k))
    Y = (rng.random((n, m)) < sigmoid(Ut @ Vt.T)).astype(float)
    return Y, rng


def test_gradient_fd():
    Y, rng = _tiny()
    U, V = rng.standard_normal((12, 3)), rng.standard_normal((9, 3))
    J, _, gU, gV = objective(Y, U, V, 0.7, 1.3, grad=True)
    h = 1e-6
    for (A, g) in [(U, gU), (V, gV)]:
        for idx in [(0, 0), (3, 2), (5, 1)]:
            A[idx] += h; Jp, _ = objective(Y, U, V, 0.7, 1.3)
            A[idx] -= 2 * h; Jm, _ = objective(Y, U, V, 0.7, 1.3)
            A[idx] += h
            assert abs((Jp - Jm) / (2 * h) - g[idx]) < 1e-5


def test_block_optimality():
    """After an exact block solve the gradient of Q wrt that block is ~0."""
    Y, rng = _tiny(1)
    U, V = rng.standard_normal((12, 3)), rng.standard_normal((9, 3))
    W = pg_mean(U @ V.T); K = Y - 0.5
    U = update_rows(K, W, V, 0.5)
    gQ_U = (K - W * (U @ V.T)) @ V - 0.5 * U
    assert np.abs(gQ_U).max() < 1e-10
    V = update_rows(K.T, W.T, U, 0.8)
    gQ_V = (K - W * (U @ V.T)).T @ U - 0.8 * V
    assert np.abs(gQ_V).max() < 1e-10
    # and Q at the new point beats Q at random perturbations
    q0 = surrogate_Q(Y, U, V, W, 0.5, 0.8)
    for _ in range(5):
        assert q0 >= surrogate_Q(Y, U, V + 1e-3 * rng.standard_normal(V.shape), W, 0.5, 0.8)


def test_monotone():
    for s in range(5):
        Y, _ = _tiny(s, 30, 20, 4)
        r = fit(Y, 3, 0.5, 0.5, max_iter=300, tol=0, seed=s)
        assert np.all(np.diff(r.obj) > -1e-9 * max(1, abs(r.obj[0])))


def test_missing_and_zeros():
    Y, rng = _tiny(2)
    Yn = Y.copy(); Yn[0, :3] = np.nan
    r1 = fit(Yn, 2, seed=3, max_iter=50)
    mask = ~np.isnan(Yn)
    r2 = fit(np.nan_to_num(Yn, nan=1.0), 2, mask=mask, seed=3, max_iter=50)
    assert np.allclose(r1.U, r2.U) and r1.n_observed == Y.size - 3
    # observed zeros matter: flipping a 0 to 1 changes the fit
    Y2 = Y.copy(); j = np.argwhere(Y == 0)[0]; Y2[tuple(j)] = 1
    assert not np.allclose(fit(Y, 2, seed=3, max_iter=50).U, fit(Y2, 2, seed=3, max_iter=50).U)


def test_empty_row_col():
    Y, _ = _tiny(3)
    mask = np.ones_like(Y, bool); mask[4, :] = False; mask[:, 2] = False
    r = fit(Y, 2, mask=mask, max_iter=30)
    assert np.allclose(r.U[4], 0) and np.allclose(r.V[2], 0)


def test_shapes_and_seed():
    for (n, m, k) in [(10, 10, 1), (25, 8, 1), (8, 25, 4)]:
        Y, _ = _tiny(0, n, m, 2)
        r = fit(Y, k, max_iter=20, seed=7)
        assert r.U.shape == (n, k) and r.V.shape == (m, k)
        r2 = fit(Y, k, max_iter=20, seed=7)
        assert np.array_equal(r.U, r2.U)


def test_prior_shrinks():
    Y, _ = _tiny(4, 40, 30, 3)
    norms = [np.linalg.norm(predict(*(lambda r: (r.U, r.V))(fit(Y, 3, l, l, seed=1, max_iter=200))) - 0.5)
             for l in [0.1, 1, 10]]
    assert norms[0] > norms[1] > norms[2]


def test_zero_init_stuck():
    Y, _ = _tiny(5)
    r = fit(Y, 2, init="provided", U0=np.zeros((12, 2)), V0=np.zeros((9, 2)), max_iter=10)
    assert np.allclose(r.U, 0) and np.allclose(r.V, 0)
    assert abs(r.loglik[-1] + Y.size * np.log(2)) < 1e-9      # P = 0.5 baseline
    r = fit(Y, 2, max_iter=10)                                 # default init escapes
    assert np.abs(r.U).max() > 1e-3


def test_synthetic_recovery():
    rng = np.random.default_rng(1)
    n, m, k = 200, 150, 5
    Ut, Vt = 0.8 * rng.standard_normal((n, k)), 0.8 * rng.standard_normal((m, k))
    Pt = sigmoid(Ut @ Vt.T)
    Y = (rng.random((n, m)) < Pt).astype(float)
    r = fit(Y, k, 1.0, 1.0, max_iter=300, seed=1, n_restarts=3)
    corr = np.corrcoef(r.P.ravel(), Pt.ravel())[0, 1]
    assert corr > 0.85, corr
    assert r.obj[-1] >= r.obj[0]


if __name__ == "__main__":
    import sys
    fails = 0
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            try:
                f(); print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {e!r}")
    sys.exit(fails)
