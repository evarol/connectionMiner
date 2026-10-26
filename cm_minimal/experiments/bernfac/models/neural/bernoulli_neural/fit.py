"""Joint-MAP fitting: PG generalized EM ('pg_gem') and direct full-batch BCE ('direct_bce').

pg_gem, one outer iteration (handoff 4.5-4.8):
  1. W = f(psi_old) from the parameter snapshot (detached, stored, never differentiated).
  2. affine head: exact per-column ridge solve of S over (V, b)            [block minimisation]
     Z: per-row Armijo gradient steps on S; hidden: global Armijo steps on S (rows of S are
     separable in z_i given the decoder, so per-row step sizes are a valid checked descent).
     The Z / hidden steps use the exact sufficient-statistic form of S (ReducedAffineSurrogate).
     Optionally re-solve the head.
     nonlinear head: per-column Armijo head steps, per-row Z steps, global hidden steps, all
     on S evaluated by full streamed passes.
  3. Check S_end <= S_start and F_new <= F_old (precision-aware tolerance); on a violation the
     whole outer iteration is rolled back and the fit stops with a diagnostic.
direct_bce: the same blocks and line searches applied to F itself (no surrogate).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from .config import NeuralConfig
from .data import MaskedBinary
from .decoder import NeuralManifoldModel, initialize
from .objectives import (F64, ReducedAffineSurrogate, full_pass, lik_grad, solve_affine_head,
                         unit_values)


# ---------------------------------------------------------------------------------------------
# checked line searches
# ---------------------------------------------------------------------------------------------
def separable_armijo(xs, gs, f0, values, step0, cfg, counters=None):
    """Per-unit Armijo backtracking for an objective that is a sum of independent unit terms.

    xs: list of tensors (modified in place) sharing the leading unit dimension; gs: matching
    gradients; f0: current per-unit objective (float64); values(idx): per-unit objective for
    units idx at the current xs. Units with no acceptable step within max_trials are restored.
    Returns (accepted bool mask, step tensor, f_new).
    """
    U = xs[0].shape[0]
    g2 = sum(g.reshape(U, -1).to(F64).pow(2).sum(1) for g in gs)
    x0 = [x.detach().clone() for x in xs]
    step = step0.clone()
    acc = torch.zeros(U, dtype=torch.bool)
    f = f0.clone()
    todo = torch.nonzero(g2 > cfg.grad_tol).squeeze(1)
    trials = 0
    for _ in range(cfg.max_trials):
        if todo.numel() == 0:
            break
        trials += 1
        st = step[todo]
        with torch.no_grad():
            for x, x00, g in zip(xs, x0, gs):
                shape = (-1,) + (1,) * (x.dim() - 1)
                x[todo] = x00[todo] - st.view(shape).to(x.dtype) * g[todo]
        ft = values(todo)
        ok = torch.isfinite(ft) & (ft <= f0[todo] - cfg.armijo * st * g2[todo])
        acc[todo[ok]] = True
        f[todo[ok]] = ft[ok]
        todo = todo[~ok]
        step[todo] *= cfg.backtrack
    if todo.numel():
        with torch.no_grad():
            for x, x00 in zip(xs, x0):
                x[todo] = x00[todo]
    if counters is not None:
        counters["trials"] = counters.get("trials", 0) + trials
        counters["unit_accepted"] = counters.get("unit_accepted", 0) + int(acc.sum())
        counters["unit_rejected"] = counters.get("unit_rejected", 0) + int(((g2 > cfg.grad_tol) & ~acc).sum())
    return acc, step, f


def global_armijo(params, grads, f0, value_fn, step0, cfg, counters=None):
    """Plain Armijo backtracking on one block. Returns (accepted, step, f_new)."""
    g2 = float(sum(g.to(F64).pow(2).sum() for g in grads))
    if not (g2 > cfg.grad_tol):
        return False, step0, f0
    x0 = [p.detach().clone() for p in params]
    step = step0
    for _ in range(cfg.max_trials):
        if counters is not None:
            counters["trials"] = counters.get("trials", 0) + 1
        with torch.no_grad():
            for p, x, g in zip(params, x0, grads):
                p.copy_(x - step * g.to(p.dtype))
        f = value_fn()
        if math.isfinite(f) and f <= f0 - cfg.armijo * step * g2:
            if counters is not None:
                counters["accepted"] = counters.get("accepted", 0) + 1
            return True, step, f
        step *= cfg.backtrack
    with torch.no_grad():
        for p, x in zip(params, x0):
            p.copy_(x)
    if counters is not None:
        counters["rejected"] = counters.get("rejected", 0) + 1
    return False, step, f0


def _next_step(step, accepted, cfg):
    """Adaptive initial step for the next search (2x the last accepted step)."""
    if not cfg.adaptive_step:
        return torch.full_like(step, cfg.step_init) if torch.is_tensor(step) else cfg.step_init
    if torch.is_tensor(step):
        return torch.clamp(step / cfg.backtrack, max=cfg.step_max)
    return min(step / cfg.backtrack, cfg.step_max) if accepted else step


# ---------------------------------------------------------------------------------------------
# inner blocks
# ---------------------------------------------------------------------------------------------
def _affine_pg_inner(model, data, W, state, cfg, diag):
    """Affine head pg_gem: exact head solve, then Z / hidden Armijo steps on the reduced S."""
    t0 = time.time()
    solve_affine_head(model, data, W)
    diag["head_solves"] += 1
    diag["t_head"] += time.time() - t0
    t1 = time.time()
    red = ReducedAffineSurrogate(model, data, W)
    hidden64 = copy.deepcopy(model.decoder.hidden).to(F64)
    hpar = list(hidden64.parameters())
    Z64 = model.Z.detach().to(F64).clone()
    lz, lh = cfg.lambda_z, cfg.lambda_hidden

    def hid_pen():
        return 0.5 * lh * sum((p * p).sum() for p in hpar) if hpar else torch.zeros((), dtype=F64)

    zc, hc = diag["z_counts"], diag["hidden_counts"]
    for _ in range(cfg.inner_rounds):
        # ---- Z block: per-row Armijo on S_i(z_i) + lambda_z/2 |z_i|^2
        for _k in range(cfg.inner_steps):
            Zv = Z64.clone().requires_grad_(True)
            for p in hpar:
                p.requires_grad_(False)
            rows = red.row_lik(hidden64(Zv)) + 0.5 * lz * (Zv * Zv).sum(1)
            g, = torch.autograd.grad(rows.sum(), Zv)
            f0 = rows.detach()

            def values(idx):
                with torch.no_grad():
                    zi = Z64[idx]
                    return red.row_lik(hidden64(zi), idx) + 0.5 * lz * (zi * zi).sum(1)

            acc, step, _ = separable_armijo([Z64], [g], f0, values, state["z_step"], cfg, zc)
            state["z_step"] = torch.where(acc, _next_step(step, True, cfg), step)
            zc["steps"] = zc.get("steps", 0) + 1
            if not bool(acc.any()):
                break
        # ---- hidden block: global Armijo on sum_i S_i + lambda_hidden/2 |eta|^2
        if hpar:
            for p in hpar:
                p.requires_grad_(True)
            for _k in range(cfg.inner_steps):
                tot = red.row_lik(hidden64(Z64)).sum() + hid_pen()
                grads = torch.autograd.grad(tot, hpar)

                def value_fn():
                    with torch.no_grad():
                        return float(red.row_lik(hidden64(Z64)).sum() + hid_pen())

                ok, step, _ = global_armijo(hpar, grads, float(tot.detach()), value_fn, state["hid_step"], cfg, hc)
                state["hid_step"] = _next_step(step, ok, cfg) if ok else max(step, 1e-12)
                if not ok:
                    break
    with torch.no_grad():
        model.Z.copy_(Z64.to(model.Z.dtype))
        for p, q in zip(model.decoder.hidden.parameters(), hpar):
            p.copy_(q.detach().to(p.dtype))
    del red
    diag["t_inner"] += time.time() - t1
    if cfg.resolve_head:
        t2 = time.time()
        solve_affine_head(model, data, W)
        diag["head_solves"] += 1
        diag["t_head"] += time.time() - t2


def _gradient_inner(model, data, W, kind, state, cfg, diag):
    """Checked gradient blocks on S (kind='S', frozen W) or on F (kind='F').

    head: per-column Armijo (columns are separable given Z, eta)
    Z:    per-row Armijo (rows are separable given the decoder)
    hidden: global Armijo
    """
    dec = model.decoder
    lz, lv, lb, lh = cfg.lambda_z, cfg.lambda_output, cfg.lambda_bias, cfg.lambda_hidden
    t0 = time.time()
    # ---- head block
    for _k in range(cfg.head_steps):
        _, _, cols = lik_grad(model, data, kind, W, [dec.V, dec.b], per_col=True)
        gV = dec.V.grad + lv * dec.V.detach()
        gb = dec.b.grad + lb * dec.b.detach()
        dec.V.grad = None
        dec.b.grad = None
        with torch.no_grad():
            f0 = cols + 0.5 * lv * (dec.V.detach().to(F64) ** 2).sum(1) + 0.5 * lb * dec.b.detach().to(F64) ** 2
            h_cache = torch.cat([dec.h(model.Z[s:s + cfg.chunk_rows]) for s in range(0, model.n, cfg.chunk_rows)])

        def values(idx):
            v = unit_values(model, data, kind, W, 1, idx, h_cache)
            return v + 0.5 * lv * (dec.V.detach()[idx].to(F64) ** 2).sum(1) + 0.5 * lb * dec.b.detach()[idx].to(F64) ** 2

        with torch.no_grad():
            acc, step, _ = separable_armijo([dec.V.data, dec.b.data], [gV, gb], f0, values, state["head_step"], cfg,
                                            diag["head_counts"])
        state["head_step"] = torch.where(acc, _next_step(step, True, cfg), step)
        diag["head_counts"]["steps"] = diag["head_counts"].get("steps", 0) + 1
        if not bool(acc.any()):
            break
    diag["t_head"] += time.time() - t0
    t1 = time.time()
    # ---- Z block
    for _k in range(cfg.inner_steps):
        _, rows, _ = lik_grad(model, data, kind, W, [model.Z], per_row=True)
        gZ = model.Z.grad + lz * model.Z.detach()
        model.Z.grad = None
        f0 = rows + 0.5 * lz * (model.Z.detach().to(F64) ** 2).sum(1)

        def values(idx):
            v = unit_values(model, data, kind, W, 0, idx)
            return v + 0.5 * lz * (model.Z.detach()[idx].to(F64) ** 2).sum(1)

        acc, step, _ = separable_armijo([model.Z.data], [gZ], f0, values, state["z_step"], cfg, diag["z_counts"])
        state["z_step"] = torch.where(acc, _next_step(step, True, cfg), step)
        diag["z_counts"]["steps"] = diag["z_counts"].get("steps", 0) + 1
        if not bool(acc.any()):
            break
    # ---- hidden block
    hpar = dec.hidden_parameters()
    if hpar:
        def pen_h():
            return 0.5 * lh * float(sum((p.detach().to(F64) ** 2).sum() for p in hpar))

        for _k in range(cfg.inner_steps):
            lik, _, _ = lik_grad(model, data, kind, W, hpar)
            grads = [p.grad + lh * p.detach() for p in hpar]
            for p in hpar:
                p.grad = None
            f0 = lik + pen_h()

            def value_fn():
                p = full_pass(model, data, W=W if kind == "S" else None)
                return (p["S_lik"] if kind == "S" else p["F_lik"]) + pen_h()

            ok, step, _ = global_armijo([p.data for p in hpar], grads, f0, value_fn, state["hid_step"], cfg,
                                        diag["hidden_counts"])
            state["hid_step"] = _next_step(step, ok, cfg) if ok else max(step, 1e-12)
            if not ok:
                break
    diag["t_inner"] += time.time() - t1


# ---------------------------------------------------------------------------------------------
# one restart
# ---------------------------------------------------------------------------------------------
def _new_diag():
    return dict(head_solves=0, t_head=0.0, t_inner=0.0, z_counts={}, hidden_counts={}, head_counts={})


def grad_norm_F(model, data):
    """||grad F|| over all parameters (likelihood + priors), streamed."""
    params = [model.Z] + list(model.decoder.parameters())
    lik_grad(model, data, "F", None, params)
    with torch.enable_grad():
        for p in params:
            p.requires_grad_(True)
        pen = model.penalty()
        gp = torch.autograd.grad(pen, params, allow_unused=True)
    tot = 0.0
    for p, g in zip(params, gp):
        gg = p.grad if p.grad is not None else torch.zeros_like(p)
        if g is not None:
            gg = gg + g
        tot += float(gg.to(F64).pow(2).sum())
        p.grad = None
    return math.sqrt(tot)


def fit_single(model, data, cfg: NeuralConfig, eval_sets=None, log=None, outer_hook=None,
               grad_norm_every=1, callback=None):
    """Run one restart in place. Returns the history dict.

    outer_hook(model, q): test-only hook called after the inner blocks and before the
    monotonicity check (used to exercise the rollback path).
    """
    n, m = model.n, model.m
    dt = model.Z.dtype
    pg = cfg.optimizer_mode == "pg_gem"
    state = dict(z_step=torch.full((n,), cfg.step_init, dtype=F64),
                 head_step=torch.full((m,), cfg.step_init, dtype=F64),
                 hid_step=cfg.step_init)
    W = torch.empty(n, m, dtype=dt) if pg else None
    p = full_pass(model, data, W_out=W, eval_sets=eval_sets)
    pen = sum(model.penalty_parts().values())
    F_old = p["F_lik"] + pen
    S_start = p["S_lik_new"] + pen if pg else math.nan
    hist = dict(iteration=[0], original_nll=[p["F_lik"]], original_map_loss=[F_old], surrogate_start=[S_start],
                surrogate_end=[math.nan], grad_norm=[grad_norm_F(model, data) if grad_norm_every else math.nan],
                seconds=[0.0], val_nll=[p.get("val_nll", math.nan)], z_unit_accepted=[0], z_unit_rejected=[0],
                hidden_accepted=[0], hidden_rejected=[0], head_unit_accepted=[0], head_unit_rejected=[0],
                head_solves=[0], line_search_trials=[0], t_head=[0.0], t_inner=[0.0], z_step_median=[cfg.step_init],
                hid_step=[cfg.step_init], monotone=[True])
    t_start = time.time()
    termination = "max_outer"
    rejected_outer = 0
    stochastic = False
    for q in range(1, cfg.max_outer + 1):
        snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
        state_snap = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()}
        diag = _new_diag()
        if pg and cfg.head_mode == "affine":
            _affine_pg_inner(model, data, W, state, cfg, diag)
        elif pg:
            _gradient_inner(model, data, W, "S", state, cfg, diag)
        else:
            _gradient_inner(model, data, None, "F", state, cfg, diag)
        if outer_hook is not None:
            outer_hook(model, q)
        do_val = eval_sets is not None and (q % cfg.val_every == 0)
        if pg:
            p = full_pass(model, data, W=W, W_out=W, eval_sets=eval_sets if do_val else None)
        else:
            p = full_pass(model, data, eval_sets=eval_sets if do_val else None)
        pen = sum(model.penalty_parts().values())
        F_new = p["F_lik"] + pen
        S_end = p["S_lik"] + pen if pg else math.nan
        tolF = cfg.monotone_rtol * max(1.0, abs(F_old))
        viol_S = pg and S_end > S_start + cfg.monotone_rtol * max(1.0, abs(S_start))
        viol_F = (F_new > F_old + tolF) or not math.isfinite(F_new)
        if viol_S or viol_F:
            rejected_outer += 1
            model.load_state_dict(snap)
            state.update(state_snap)
            if pg:
                full_pass(model, data, W_out=W)
            msg = (f"outer {q}: monotonicity violation rejected and rolled back "
                   f"(S_end-S_start={S_end - S_start:.3e}, F_new-F_old={F_new - F_old:.3e})")
            hist.setdefault("violations", []).append(msg)
            if log:
                log(msg)
            termination = "monotonicity_violation"
            break
        zc, hc, hdc = diag["z_counts"], diag["hidden_counts"], diag["head_counts"]
        gn = grad_norm_F(model, data) if (grad_norm_every and q % grad_norm_every == 0) else math.nan
        rel = (F_old - F_new) / max(1.0, abs(F_old))
        for k, v in dict(iteration=q, original_nll=p["F_lik"], original_map_loss=F_new, surrogate_start=S_start,
                         surrogate_end=S_end, grad_norm=gn, seconds=time.time() - t_start,
                         val_nll=p.get("val_nll", math.nan),
                         z_unit_accepted=zc.get("unit_accepted", 0), z_unit_rejected=zc.get("unit_rejected", 0),
                         hidden_accepted=hc.get("accepted", 0), hidden_rejected=hc.get("rejected", 0),
                         head_unit_accepted=hdc.get("unit_accepted", 0), head_unit_rejected=hdc.get("unit_rejected", 0),
                         head_solves=diag["head_solves"],
                         line_search_trials=zc.get("trials", 0) + hc.get("trials", 0) + hdc.get("trials", 0),
                         t_head=diag["t_head"], t_inner=diag["t_inner"],
                         z_step_median=float(state["z_step"].median()), hid_step=float(state["hid_step"]),
                         monotone=True).items():
            hist[k].append(v)
        if log and (q <= 3 or q % 5 == 0 or cfg.verbose):
            log(f"  it {q:4d} F={F_new:.8e} rel={rel:.2e} S:{S_start:.6e}->{S_end:.6e} "
                f"val={p.get('val_nll', float('nan')):.5f} |g|={gn:.3e} t={time.time() - t_start:.0f}s "
                f"(head {diag['t_head']:.1f}s inner {diag['t_inner']:.1f}s)")
        if callback is not None:
            callback(model, hist)
        F_old = F_new
        if pg:
            S_start = p["S_lik_new"] + pen
        if rel < cfg.tol:
            termination = "tol"
            break
        if time.time() - t_start > cfg.max_seconds:
            termination = "time_limit"
            break
    hist["termination"] = termination
    hist["rejected_outer"] = rejected_outer
    hist["stochastic"] = stochastic
    hist["final_F"] = F_old
    hist["monotone_F"] = bool(np.all(np.diff(hist["original_map_loss"]) <= cfg.monotone_rtol * np.maximum(
        1.0, np.abs(hist["original_map_loss"][:-1]))))
    hist["fit_seconds"] = time.time() - t_start
    return hist


# ---------------------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------------------
def fit_bernoulli_neural_manifold(Y, latent_dim=None, config: NeuralConfig = None, observed_mask=None,
                                  data=None, init_Z=None, eval_sets=None, log=None, grad_norm_every=1):
    """Joint MAP fit with n_restarts seeded starts; the restart with the lowest final training
    objective F is returned (never chosen by held-out scores).

    Returns (model, history). history['restarts'] lists every restart's summary.
    """
    if config is None:
        raise ValueError("a NeuralConfig is required")
    if latent_dim is not None and int(latent_dim) != config.latent_dim:
        raise ValueError(f"latent_dim argument ({latent_dim}) != config.latent_dim ({config.latent_dim})")
    if observed_mask is None:
        observed_mask = config.observed_mask
    if data is None:
        data = MaskedBinary(Y, observed_mask)
    torch.manual_seed(config.seed)
    best, best_hist, summaries = None, None, []
    for r in range(config.n_restarts):
        model = NeuralManifoldModel(config, data.n, data.m)
        initialize(model, data, seed=config.seed + r, init_Z=init_Z)
        if log:
            log(f"restart {r} (seed {config.seed + r})")
        hist = fit_single(model, data, config, eval_sets=eval_sets, log=log, grad_norm_every=grad_norm_every)
        hist["seed"] = config.seed + r
        summaries.append(dict(restart=r, seed=config.seed + r, final_F=hist["final_F"],
                              iterations=hist["iteration"][-1], termination=hist["termination"],
                              seconds=hist["fit_seconds"]))
        if best is None or hist["final_F"] < best_hist["final_F"]:
            best, best_hist = model, hist
        else:
            del model
    best_hist["restarts"] = summaries
    best_hist["best_restart"] = int(np.argmin([s["final_F"] for s in summaries]))
    best.meta.update(dict(n_observed=data.n_obs, torch_version=str(torch.__version__), seed=config.seed,
                          best_restart=best_hist["best_restart"]))
    return best, best_hist
