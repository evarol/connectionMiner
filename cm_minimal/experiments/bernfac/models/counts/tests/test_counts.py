"""Tests for countfac (run with plain python: python tests/test_counts.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import countfac as cf  # noqa: E402


def synth(n=500, m=300, k=3, family="poisson", seed=1):
    rng = np.random.default_rng(seed)
    s = np.exp(rng.normal(np.log(3000), 0.4, n))
    a = rng.normal(0, 0.2, n)
    b = rng.normal(np.log(1.0 / m), 1.0, m)
    U, V = rng.normal(0, 0.6, (n, k)), rng.normal(0, 0.6, (m, k))
    mu = s[:, None] * np.exp(a[:, None] + b[None, :] + U @ V.T)
    phi = np.exp(rng.uniform(np.log(0.5), np.log(20), m))
    if family == "poisson":
        X = rng.poisson(mu)
    else:
        X = rng.poisson(rng.gamma(phi[None, :], mu / phi[None, :]))
    return X.astype(np.float32), dict(U=U, V=V, a=a, b=b, phi=phi, s=s, mu=mu)


def test_score_fisher_fd():
    rng = np.random.default_rng(0)
    eta = rng.normal(0, 1.5, 200).astype(np.float64)
    x = rng.poisson(np.exp(eta)).astype(np.float64)
    t = np.ones_like(x)
    eps = 1e-5
    for phi in (None, 2.5):
        def h(e, xx=x):
            ee = np.clip(e, cf.ETA_MIN, cf.ETA_MAX); mu = np.exp(ee)
            if phi is None:
                return xx * ee - mu
            L = np.log1p(mu / phi)
            return xx * (ee - L) - phi * L
        _, g, w = cf.entry_terms(eta, x, t, phi)
        g_fd = (h(eta + eps) - h(eta - eps)) / (2 * eps)
        assert np.allclose(g, g_fd, rtol=1e-6, atol=1e-6), f"score mismatch phi={phi}"
        # Fisher = E[-h''] ; -h'' is linear in x, so evaluating at x = mu gives the expectation
        mu = np.exp(eta)
        hess = (h(eta + eps, mu) - 2 * h(eta, mu) + h(eta - eps, mu)) / eps ** 2
        assert np.allclose(w, -hess, rtol=1e-4, atol=1e-4), f"Fisher mismatch phi={phi}"
        # h matches the full log-likelihood up to a phi-only constant
        full = cf.count_loglik(x, mu, phi)
        c = full - h(eta)
        c_expected = -cf.gammaln(x + 1) if phi is None else (cf.gammaln(x + phi) - cf.gammaln(phi)
                                                            - x * np.log(phi) - cf.gammaln(x + 1))
        assert np.allclose(c, c_expected, atol=1e-8)
    # dispersion derivatives
    mu = np.exp(rng.normal(0, 1, (3, 400)))
    xx = rng.poisson(mu).astype(np.float64); tt = np.ones_like(xx)
    jj, ii = np.nonzero(xx); xs = xx[jj, ii]; ws = np.ones_like(xs)
    phi = np.array([0.7, 3.0, 40.0])
    ll = lambda p: cf._gene_ll_dense(mu, xx, p) + cf._gene_ll_sparse(jj, xs, ws, p, 3)  # noqa: E731
    d1, d2 = cf._phi_derivs(mu, xx, phi, jj, xs, ws, 3)
    e = 1e-4 * phi
    d1_fd = (ll(phi + e) - ll(phi - e)) / (2 * e)
    d2_fd = (ll(phi + e) - 2 * ll(phi) + ll(phi - e)) / e ** 2
    assert np.allclose(d1, d1_fd, rtol=1e-5, atol=1e-6), (d1, d1_fd)
    assert np.allclose(d2, d2_fd, rtol=1e-3, atol=1e-5), (d2, d2_fd)
    print("ok  score / Fisher / dispersion derivatives match finite differences")


def _corr(A, B):
    return float(np.corrcoef(A.ravel(), B.ravel())[0, 1])


def test_recovery_and_monotone():
    for family in ("poisson", "nb"):
        X, tru = synth(family=family)
        H = np.random.default_rng(3).random(X.shape) < 0.05
        D = cf.CountData(X, H, depth="train")
        U, V, a, b, phi, tr = cf.fit(D, 3, 1.0, family=family, max_iter=150, tol=1e-9, block=128, verbose=False)
        d = np.diff(tr["obj"])
        assert np.all(d >= -1e-9 * abs(tr["obj"][0])), f"{family}: objective decreased"
        J_re, _ = cf.objective(D, U, V, a, b, 1.0, 1e-3, phi, block=128)
        assert abs(J_re - tr["obj"][-1]) < 1e-9 * abs(J_re), (J_re, tr["obj"][-1])   # tracked J is exact
        r = _corr(U @ V.T, tru["U"] @ tru["V"].T)
        logmu_hat = (U @ V.T + a[:, None] + b[None, :] + D.o[:, None])
        rmu = _corr(logmu_hat, np.log(tru["mu"]))
        msg = f"ok  {family}: {tr['iterations']} it, monotone, corr(UV', true)={r:.3f}, corr(log mu)={rmu:.3f}"
        assert r > 0.9 and rmu > 0.97, msg
        if family == "nb":
            rp = float(np.corrcoef(np.log(phi), np.log(tru["phi"]))[0, 1])
            med = float(np.median(np.abs(np.log(phi / tru["phi"]))))
            msg += f", corr(log phi)={rp:.3f}, median |log ratio|={med:.3f}"
            assert rp > 0.7 and med < 0.4, msg
        print(msg)


def test_mask_excludes_heldout():
    X, _ = synth(n=200, m=120, family="nb", seed=5)
    H = np.random.default_rng(4).random(X.shape) < 0.1
    X2 = X.copy(); X2[H] = X2[H] * 7 + 50
    for family in ("poisson", "nb"):
        fits = []
        for XX in (X, X2):
            D = cf.CountData(XX, H, depth="train")
            fits.append(cf.fit(D, 2, 1.0, family=family, max_iter=30, block=64, verbose=False))
        for A, B in zip(fits[0][:4], fits[1][:4]):
            assert np.array_equal(A, B), f"{family}: held-out counts changed the fit"
        if family == "nb":
            assert np.array_equal(fits[0][4], fits[1][4])
    print("ok  changing held-out counts leaves the fit bit-identical (Poisson and NB)")


def test_nb_to_poisson_limit():
    X, _ = synth(n=300, m=150, family="poisson", seed=8)
    D = cf.CountData(X, None, depth="train")
    P = cf.fit(D, 3, 1.0, family="poisson", max_iter=200, tol=1e-10, block=64, verbose=False)
    J_p, _ = cf.objective(D, *P[:4], 1.0, 1e-3, None)
    gaps = [abs(cf.objective(D, *P[:4], 1.0, 1e-3, np.full(D.m, ph))[0] - J_p) for ph in (1e4, 1e5, 1e6, 1e7)]
    ratios = np.array(gaps[:-1]) / np.array(gaps[1:])
    assert np.all((ratios > 8) & (ratios < 12)) and gaps[-1] / abs(J_p) < 1e-5, (gaps, ratios)  # gap ~ 1/phi
    big = np.full(D.m, 1e6)
    N = cf.fit(D, 3, 1.0, family="nb", phi=big, phi_steps=0, max_iter=200, tol=1e-10, block=64, verbose=False)
    ll_p, ll_n = P[5]["loglik"][-1], N[5]["loglik"][-1]
    r = _corr(P[0] @ P[1].T, N[0] @ N[1].T)
    assert abs(ll_p - ll_n) / abs(ll_p) < 1e-5 and r > 0.9999, (ll_p, ll_n, r)
    # letting phi move from a large start on Poisson data keeps phi large
    N2 = cf.fit(D, 3, 1.0, family="nb", init=P[:4], phi=np.full(D.m, 1e3), max_iter=20, block=64, verbose=False)
    assert np.median(N2[4]) > 50, np.median(N2[4])
    print(f"ok  NB objective -> Poisson as 1/phi (gaps {np.round(gaps, 3)}); NB(phi=1e6) fit matches Poisson: loglik {ll_n:.2f} vs {ll_p:.2f}, corr(UV')={r:.5f}; "
          f"free phi on Poisson data stays large (median {np.median(N2[4]):.3g})")


def test_heldout_scores_and_embedding():
    X, _ = synth(n=200, m=100, seed=9)
    H = np.random.default_rng(2).random(X.shape) < 0.2
    D = cf.CountData(X, H)
    U, V, a, b, phi, _ = cf.fit(D, 2, 1.0, max_iter=50, block=64, verbose=False)
    Y = (X > 0).astype(np.uint8)
    out, cal = cf.heldout_scores(Y, X, D.o, U, V, a, b, None, {"val": H}, block=64)
    mu = np.exp(U @ V.T + a[:, None] + b[None, :] + D.o[:, None])[H]
    y = Y[H]
    nll = -np.mean(np.where(y > 0, np.log(-np.expm1(-mu)), -mu))
    assert np.isclose(out["val"]["nll"], nll, rtol=1e-6)
    assert sum(c["count"] for c in cal) == H.sum()
    E, s = cf.embedding(U, V)
    A, S, _ = np.linalg.svd(U @ V.T, full_matrices=False)
    assert np.allclose(np.abs(E), np.abs(A[:, :2] * S[:2]), atol=1e-8)
    print("ok  held-out Bernoulli NLL and SVD embedding match direct computation")


if __name__ == "__main__":
    test_score_fisher_fd()
    test_mask_excludes_heldout()
    test_heldout_scores_and_embedding()
    test_nb_to_poisson_limit()
    test_recovery_and_monotone()
    print("all tests passed")
