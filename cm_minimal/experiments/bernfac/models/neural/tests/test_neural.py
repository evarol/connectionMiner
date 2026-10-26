"""Tests for bernoulli_neural (handoff 7.1 shared tests that apply to model 2, and 7.3 A-H).

Run:  python tests/test_neural.py        (plain runner; pytest also collects these functions)
All tests use float64 and tiny deterministic fixtures.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bernoulli_neural import (MaskedBinary, NeuralConfig, NeuralManifoldModel, ReducedAffineSurrogate,  # noqa: E402
                              compute_pg_weights, fit_bernoulli_neural_manifold, full_pass, infer_neural_rows,
                              initialize, lik_grad, load_model, neural_map_objective, pg_mean,
                              predict_bernoulli_neural_manifold, save_model, solve_affine_head)
from bernoulli_neural.fit import (_affine_pg_inner, _gradient_inner, _new_diag, fit_single,  # noqa: E402
                                  separable_armijo)
from bernoulli_neural.pg import bernoulli_loglik, bound_constant, pg_lower_bound, softplus  # noqa: E402

torch.set_num_threads(2)
F64 = torch.float64


def _data(n=14, m=9, seed=0, frac=0.8):
    rng = np.random.default_rng(seed)
    Y = (rng.random((n, m)) < 0.35).astype(float)
    mask = rng.random((n, m)) < frac
    return Y, mask


def _model(n, m, data, seed=3, **kw):
    base = dict(latent_dim=2, hidden_widths=(5, 4), n_restarts=1, max_outer=30, tol=1e-12)
    base.update(kw)
    cfg = NeuralConfig(**base)
    model = NeuralManifoldModel(cfg, n, m)
    initialize(model, data, seed=seed)
    with torch.no_grad():                         # make the head non-trivial
        model.decoder.V.normal_(0, 0.7, generator=torch.Generator().manual_seed(seed))
        model.Z.mul_(10)
    return model


def _total(model, data, kind, W=None):
    p = full_pass(model, data, W=W)
    return (p["S_lik"] if kind == "S" else p["F_lik"]) + float(model.penalty().detach())


def _autograd(model, data, kind, W=None):
    params = [model.Z] + list(model.decoder.parameters())
    lik_grad(model, data, kind, W, params)
    gl = [p.grad.clone() for p in params]
    for p in params:
        p.grad = None
    gp = torch.autograd.grad(model.penalty(), params)
    return params, [a + b for a, b in zip(gl, gp)]


# ------------------------------------------------------------------------------------------------
# 7.1 shared numerics
# ------------------------------------------------------------------------------------------------
def test_pg_mean_values():
    x = torch.tensor([0.0, 1e-8, -1e-8, 1.0, -1.0, 20.0, -20.0, 1000.0, -1000.0], dtype=F64)
    w = pg_mean(x)
    assert torch.all(torch.isfinite(w)) and torch.all(w > 0) and torch.all(w <= 0.25)
    assert w[0] == 0.25
    assert torch.allclose(w[1::2], w[2::2])                       # even
    assert abs(float(w[3]) - math.tanh(0.5) / 2) < 1e-15
    assert abs(float(w[7]) * 2000 - 1) < 1e-12                    # ~ 1/(2|x|)
    xs = torch.tensor([9.99e-6, 1.001e-5], dtype=F64)             # continuity across the branch
    assert abs(float(pg_mean(xs)[0] - pg_mean(xs)[1])) < 1e-12


def test_quadratic_bound():
    psi = torch.linspace(-12, 12, 241, dtype=F64)[:, None]
    xi = torch.tensor([0.0, 0.3, 1.0, 4.0, 15.0], dtype=F64)[None, :]
    for y in (0.0, 1.0):
        ll = bernoulli_loglik(psi, torch.full_like(psi, y))
        lb = pg_lower_bound(psi, torch.full_like(psi, y), xi)
        assert torch.all(lb <= ll + 1e-12)
        for s in (1, -1):                                           # touching and gradient contact
            p = (s * xi).clone().requires_grad_(True)
            a = bernoulli_loglik(p, torch.full_like(p, y))
            b = pg_lower_bound(p, torch.full_like(p, y), xi)
            assert torch.allclose(a, b, atol=1e-12)
            ga, = torch.autograd.grad(a.sum(), p)
            gb, = torch.autograd.grad(b.sum(), p)
            assert torch.allclose(ga, gb, atol=1e-12)


def test_stable_likelihood():
    x = torch.tensor([-1e4, -50.0, 0.0, 50.0, 1e4], dtype=F64)
    sp = softplus(x)
    assert torch.all(torch.isfinite(sp))
    assert torch.allclose(sp, torch.tensor([0.0, math.exp(-50), math.log(2), 50.0, 1e4], dtype=F64), rtol=1e-14)
    assert torch.all(torch.isfinite(bound_constant(x.abs())))


def test_invalid_inputs():
    Y, mask = _data()
    bad = Y.copy(); bad[0, 0] = 2
    mk = mask.copy(); mk[0, 0] = True
    for fn in (lambda: MaskedBinary(bad, mk),
               lambda: MaskedBinary(np.where(mk, np.nan, Y), mk),
               lambda: MaskedBinary(Y, np.zeros_like(mask)),
               lambda: MaskedBinary(Y, mask[:, :-1]),
               lambda: NeuralConfig(latent_dim=2, lambda_output=0.0),
               lambda: NeuralConfig(latent_dim=2, head_mode="relu"),
               lambda: fit_bernoulli_neural_manifold(Y, 3, NeuralConfig(latent_dim=2), observed_mask=mask)):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("invalid input accepted")
    d = MaskedBinary(np.where(mask, Y, np.nan))                   # NaN = unobserved when no mask given
    assert d.n_obs == mask.sum()


def test_masks_excluded_cells_ignored_and_observed_zeros_matter():
    Y, mask = _data(seed=1)
    cfg = NeuralConfig(latent_dim=2, hidden_widths=(4,), n_restarts=1, max_outer=8)
    m1, _ = fit_bernoulli_neural_manifold(Y, 2, cfg, observed_mask=mask)
    Y2 = Y.copy(); Y2[~mask] = 1 - Y2[~mask]
    m2, _ = fit_bernoulli_neural_manifold(Y2, 2, cfg, observed_mask=mask)
    assert torch.equal(m1.Z, m2.Z) and torch.equal(m1.decoder.V, m2.decoder.V)
    i, j = np.argwhere(~mask)[0]
    mask3 = mask.copy(); mask3[i, j] = True
    Y3 = Y.copy(); Y3[i, j] = 0
    m3, _ = fit_bernoulli_neural_manifold(Y3, 2, cfg, observed_mask=mask3)
    assert not torch.allclose(m1.Z, m3.Z)                         # an observed zero is data


def test_serialization_roundtrip():
    Y, mask = _data(seed=2)
    cfg = NeuralConfig(latent_dim=2, hidden_widths=(4, 3), head_mode="nonlinear", n_restarts=1, max_outer=5)
    model, hist = fit_bernoulli_neural_manifold(Y, 2, cfg, observed_mask=mask)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.pt")
        save_model(path, model, hist)
        m2, h2, _ = load_model(path)
    P1 = predict_bernoulli_neural_manifold(model)
    P2 = predict_bernoulli_neural_manifold(m2)
    assert np.array_equal(P1, P2)
    assert m2.config.head_mode == "nonlinear" and m2.config.hidden_widths == (4, 3)
    assert len(h2["original_map_loss"]) == len(hist["original_map_loss"])


# ------------------------------------------------------------------------------------------------
# 7.3 A: autodiff vs finite differences; F and S gradients agree at the touching point
# ------------------------------------------------------------------------------------------------
def test_A_autodiff_finite_differences():
    Y, mask = _data(n=7, m=5, seed=4)
    data = MaskedBinary(Y, mask)
    for head in ("affine", "nonlinear"):
        model = _model(7, 5, data, head_mode=head, hidden_widths=(4, 3))
        W, _ = compute_pg_weights(model, data)
        for kind in ("F", "S"):
            params, grads = _autograd(model, data, kind, W)
            rng = np.random.default_rng(0)
            for p, g in zip(params, grads):
                flat = p.data.view(-1)
                for k in rng.choice(flat.numel(), min(4, flat.numel()), replace=False):
                    old = float(flat[k]); h = 1e-6
                    flat[k] = old + h; fp = _total(model, data, kind, W)
                    flat[k] = old - h; fm = _total(model, data, kind, W)
                    flat[k] = old
                    fd = (fp - fm) / (2 * h)
                    ad = float(g.view(-1)[k])
                    assert abs(fd - ad) <= 1e-5 * max(1.0, abs(ad)), (head, kind, fd, ad)
        _, gF = _autograd(model, data, "F")
        _, gS = _autograd(model, data, "S", W)
        for a, b in zip(gF, gS):
            assert torch.allclose(a, b, atol=1e-10, rtol=1e-9)


# ------------------------------------------------------------------------------------------------
# 7.3 B: frozen PG weights have no gradient path and do not change inside an outer iteration
# ------------------------------------------------------------------------------------------------
def test_B_frozen_weights():
    Y, mask = _data(seed=5)
    data = MaskedBinary(Y, mask)
    for head in ("affine", "nonlinear"):
        model = _model(*Y.shape, data, head_mode=head)
        W, _ = compute_pg_weights(model, data)
        assert not W.requires_grad and W.grad_fn is None
        W0 = W.clone()
        state = dict(z_step=torch.ones(model.n, dtype=F64), head_step=torch.ones(model.m, dtype=F64), hid_step=1.0)
        diag = _new_diag()
        if head == "affine":
            _affine_pg_inner(model, data, W, state, model.config, diag)
        else:
            _gradient_inner(model, data, W, "S", state, model.config, diag)
        assert torch.equal(W, W0) and W.grad is None
        assert diag["z_counts"].get("unit_accepted", 0) > 0          # the inner loop really moved


# ------------------------------------------------------------------------------------------------
# 7.3 C: exact affine head solve; nonlinear head rejects it
# ------------------------------------------------------------------------------------------------
def test_C_head_solve():
    Y, mask = _data(seed=6)
    mask[:, 2] = False                                            # an all-missing column
    data = MaskedBinary(Y, mask)
    model = _model(*Y.shape, data, head_mode="affine")
    W, _ = compute_pg_weights(model, data)
    S0 = _total(model, data, "S", W)
    d = solve_affine_head(model, data, W, return_diagnostics=True)
    assert d["max_rel_residual"] < 1e-9
    assert _total(model, data, "S", W) <= S0
    assert torch.all(model.decoder.V[2] == 0) and model.decoder.b[2] == 0
    params, grads = _autograd(model, data, "S", W)
    gV = grads[[k for k, p in enumerate(params) if p is model.decoder.V][0]]
    gb = grads[[k for k, p in enumerate(params) if p is model.decoder.b][0]]
    assert float(gV.abs().max()) < 1e-9 and float(gb.abs().max()) < 1e-9
    # independent column-by-column reference with explicit sums
    with torch.no_grad():
        Ht = torch.cat([torch.ones(model.n, 1, dtype=F64), model.decoder.h(model.Z)], 1).numpy()
    Wn, K = W.numpy(), np.where(mask, Y - 0.5, 0.0)
    c = model.config
    j = 4
    A = np.diag([c.lambda_bias] + [c.lambda_output] * model.decoder.H)
    for i in range(model.n):
        A += Wn[i, j] * np.outer(Ht[i], Ht[i])
    th = np.linalg.solve(A, (K[:, j:j + 1] * Ht).sum(0))
    assert np.allclose(th, np.r_[float(model.decoder.b[j]), model.decoder.V[j].detach().numpy()], atol=1e-10)
    nl = _model(*Y.shape, data, head_mode="nonlinear")
    try:
        solve_affine_head(nl, data, W)
    except RuntimeError:
        pass
    else:
        raise AssertionError("nonlinear head accepted the linear head solve")


# ------------------------------------------------------------------------------------------------
# 7.3 D: monotonicity, oversized steps, rollback
# ------------------------------------------------------------------------------------------------
def test_D_monotonicity_and_rollback():
    Y, mask = _data(n=20, m=12, seed=7)
    data = MaskedBinary(Y, mask)
    for head in ("affine", "nonlinear"):
        for opt in ("pg_gem", "direct_bce"):
            model = _model(*Y.shape, data, head_mode=head, optimizer_mode=opt, max_outer=25)
            h = fit_single(model, data, model.config)
            Fh = np.array(h["original_map_loss"])
            assert np.all(np.diff(Fh) <= 1e-10 * np.abs(Fh[:-1])), (head, opt)
            if opt == "pg_gem":
                S0, S1 = np.array(h["surrogate_start"][1:]), np.array(h["surrogate_end"][1:])
                assert np.all(S1 <= S0 + 1e-10 * np.abs(S0))
                assert np.all(Fh[1:] - Fh[:-1] <= S1 - S0 + 1e-8 * np.abs(S0))   # F drop >= S drop
            assert Fh[-1] < Fh[0]
    # the handoff's suggested Armijo coefficient 1e-4 is also monotone
    model = _model(*Y.shape, data, armijo=1e-4, max_outer=25)
    h = fit_single(model, data, model.config)
    Fh = np.array(h["original_map_loss"])
    assert np.all(np.diff(Fh) <= 1e-10 * np.abs(Fh[:-1])) and h["termination"] != "monotonicity_violation"
    # deliberately oversized step: Armijo must backtrack, and with one trial nothing may move
    model = _model(*Y.shape, data)
    W, _ = compute_pg_weights(model, data)
    red = ReducedAffineSurrogate(model, data, W)
    lz = model.config.lambda_z
    hid = copy.deepcopy(model.decoder.hidden)
    Z = model.Z.detach().clone()

    def rows_fn(Zt, idx=None):
        zz = Zt if idx is None else Zt[idx]
        return red.row_lik(hid(zz), idx) + 0.5 * lz * (zz * zz).sum(1)

    Zv = Z.clone().requires_grad_(True)
    f = rows_fn(Zv)
    g, = torch.autograd.grad(f.sum(), Zv)
    f0 = f.detach()
    cnt = {}
    big = torch.full((model.n,), 1e8, dtype=F64)
    Zw = Z.clone()
    acc, step, fn = separable_armijo([Zw], [g], f0, lambda idx: rows_fn(Zw, idx), big, model.config, cnt)
    assert cnt["trials"] > 1 and bool(acc.all())
    assert torch.all(rows_fn(Zw).detach() < f0)
    one = copy.copy(model.config); one.max_trials = 1
    Zw = Z.clone()
    acc, _, _ = separable_armijo([Zw], [g], f0, lambda idx: rows_fn(Zw, idx), big, one, {})
    assert not bool(acc.any()) and torch.equal(Zw, Z)                 # rolled back bitwise
    # outer-level rollback: corrupt the parameters after the inner steps of iteration 3
    model = _model(*Y.shape, data, max_outer=10)
    saved = {}

    def cb(m, hist):
        saved["state"] = {k: v.clone() for k, v in m.state_dict().items()}

    def hook(m, q):
        if q == 3:
            with torch.no_grad():
                m.decoder.V.mul_(50.0)

    h = fit_single(model, data, model.config, outer_hook=hook, callback=cb)
    assert h["termination"] == "monotonicity_violation" and h["rejected_outer"] == 1
    assert len(h["original_map_loss"]) == 3                          # init + 2 accepted iterations
    for k, v in model.state_dict().items():
        assert torch.equal(v, saved["state"][k])


# ------------------------------------------------------------------------------------------------
# 7.3 E: linear-network limit g(z) = V z + b
# ------------------------------------------------------------------------------------------------
def test_E_linear_network_limit():
    Y, mask = _data(n=16, m=10, seed=8)
    data = MaskedBinary(Y, mask)
    model = _model(*Y.shape, data, hidden_widths=(), latent_dim=3)
    Z, V, b = (t.detach().numpy() for t in (model.Z, model.decoder.V, model.decoder.b))
    psi = Z @ V.T + b
    c = model.config
    ref = (np.where(mask, np.logaddexp(0, psi) - Y * psi, 0).sum() + 0.5 * c.lambda_z * (Z ** 2).sum()
           + 0.5 * c.lambda_output * (V ** 2).sum() + 0.5 * c.lambda_bias * (b ** 2).sum())
    assert abs(neural_map_objective(model, data=data)["original_map_loss"] - ref) < 1e-10 * abs(ref)
    # coordinate MAP update under frozen weights, including the -w b term
    W, _ = compute_pg_weights(model, data)
    Wn, K = W.numpy(), np.where(mask, Y - 0.5, 0.0)
    Zstar = np.stack([np.linalg.solve(c.lambda_z * np.eye(3) + (V.T * Wn[i]) @ V, ((K[i] - Wn[i] * b)[:, None] * V).sum(0))
                      for i in range(model.n)])
    with torch.no_grad():
        model.Z.copy_(torch.from_numpy(Zstar))
    params, grads = _autograd(model, data, "S", W)
    assert float(grads[0].abs().max()) < 1e-10                     # stationary point of S in Z
    # the package's per-row Armijo on the reduced surrogate converges to the same z_i
    with torch.no_grad():
        model.Z.zero_()
    red = ReducedAffineSurrogate(model, data, W)
    Zw = model.Z.detach().clone()
    step = torch.ones(model.n, dtype=F64)
    for _ in range(3000):
        Zv = Zw.clone().requires_grad_(True)
        f = red.row_lik(Zv) + 0.5 * c.lambda_z * (Zv * Zv).sum(1)
        g, = torch.autograd.grad(f.sum(), Zv)
        acc, st, _ = separable_armijo([Zw], [g], f.detach(), lambda idx: red.row_lik(Zw[idx], idx)
                                      + 0.5 * c.lambda_z * (Zw[idx] ** 2).sum(1), step, c)
        step = torch.where(acc, st * 2, st)
        if not bool(acc.any()):
            break
    assert np.allclose(Zw.numpy(), Zstar, atol=1e-6), np.abs(Zw.numpy() - Zstar).max()


# ------------------------------------------------------------------------------------------------
# 7.3 F: output-rank behaviour
# ------------------------------------------------------------------------------------------------
def test_F_rank_bound():
    n, m, H = 80, 60, 6
    rng = np.random.default_rng(9)
    data = MaskedBinary((rng.random((n, m)) < 0.4).astype(float))
    for head, expect_low in (("affine", True), ("nonlinear", False)):
        model = _model(n, m, data, head_mode=head, hidden_widths=(12, H), latent_dim=4, seed=11)
        with torch.no_grad():
            model.decoder.V.mul_(4.0)
            model.Z.copy_(torch.randn(n, 4, generator=torch.Generator().manual_seed(1), dtype=F64))
            P = model.decoder(model.Z).numpy()
        s = np.linalg.svd(P, compute_uv=False)
        rank = int((s > 1e-9 * s[0]).sum())
        if expect_low:
            assert rank <= H + 1, rank
        else:
            assert rank > H + 1, rank


# ------------------------------------------------------------------------------------------------
# 7.3 G: new-row inference leaves the decoder bitwise unchanged; held-out cells never enter
# ------------------------------------------------------------------------------------------------
def test_G_new_row_inference():
    Y, mask = _data(n=20, m=12, seed=10)
    cfg = NeuralConfig(latent_dim=2, hidden_widths=(5, 4), n_restarts=1, max_outer=10)
    model, _ = fit_bernoulli_neural_manifold(Y, 2, cfg, observed_mask=mask)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = np.random.default_rng(1)
    Ynew = (rng.random((6, 12)) < 0.4).astype(float)
    ctx = rng.random((6, 12)) < 0.5
    ctx[5] = False                                                 # a row with no context
    for method in ("pg_gem", "direct_bce"):
        Z1, d1 = infer_neural_rows(model, Ynew, ctx, method=method, n_starts=3)
        for k, v in model.state_dict().items():
            assert torch.equal(v, before[k]), k
        Y2 = Ynew.copy(); Y2[~ctx] = 1 - Y2[~ctx]                     # change only evaluated cells
        Z2, _ = infer_neural_rows(model, Y2, ctx, method=method, n_starts=3)
        assert np.array_equal(Z1, Z2)
        assert np.all(Z1[5] == 0) and d1["rows_without_context"] == 1
    assert all(p.grad is None for p in model.decoder.parameters())


# ------------------------------------------------------------------------------------------------
# 7.3 H: matched optimizer benchmark (PG vs direct BCE), and a convex special case
# ------------------------------------------------------------------------------------------------
def test_H_matched_optimizers():
    Y, mask = _data(n=30, m=15, seed=12)
    data = MaskedBinary(Y, mask)
    res = {}
    for opt in ("pg_gem", "direct_bce"):
        model = _model(*Y.shape, data, optimizer_mode=opt, max_outer=150, tol=1e-10, seed=5)
        F0 = _total(model, data, "F")
        t = time.time()
        h = fit_single(model, data, model.config)
        res[opt] = (F0, h["final_F"], time.time() - t, h["iteration"][-1])
    assert res["pg_gem"][0] == res["direct_bce"][0]                 # identical initialisation
    for opt, (F0, F1, sec, it) in res.items():
        assert F1 < F0
        print(f"    {opt:10s} F0={F0:.4f} F_final={F1:.4f} iters={it} {sec:.2f}s")
    # fixed linear decoder: the row problem is strictly convex -> both optimisers agree
    model = _model(*Y.shape, data, hidden_widths=(), latent_dim=2)
    rng = np.random.default_rng(3)
    Ynew = (rng.random((5, 15)) < 0.4).astype(float)
    Za, _ = infer_neural_rows(model, Ynew, None, method="pg_gem", n_starts=2, max_iter=3000, tol=1e-14)
    Zb, _ = infer_neural_rows(model, Ynew, None, method="direct_bce", n_starts=2, max_iter=3000, tol=1e-14)
    assert np.allclose(Za, Zb, atol=1e-6), np.abs(Za - Zb).max()


# ------------------------------------------------------------------------------------------------
def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        t = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t:.1f}s)", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {name}", flush=True)
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
