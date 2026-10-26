"""Original MAP objective F, frozen PG surrogate S, streamed passes and the affine-head solve.

    F(Z, eta, V, b) = sum_O [softplus(psi) - y psi] + lambda_z/2 |Z|^2 + lambda_hidden/2 |eta|^2
                      + lambda_output/2 |V|^2 + lambda_bias/2 |b|^2                  (handoff 4.3)
    S(Z, eta, V, b; w) = sum_O [w psi^2 / 2 - kappa psi] + the same penalties          (handoff 4.5)

The likelihood part of F is computed with torch.nn.functional.binary_cross_entropy_with_logits
(reduction='none'), masked to training cells and summed. Everything is streamed over row chunks
and accumulated exactly in float64; the PG weights W used by one outer iteration are a stored,
detached tensor computed from one parameter snapshot.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as Fnn

from .pg import pg_mean

F64 = torch.float64


# ---------------------------------------------------------------------------------------------
# elementwise terms
# ---------------------------------------------------------------------------------------------
def bce_terms(psi, y, mk):
    return torch.where(mk, Fnn.binary_cross_entropy_with_logits(psi, y, reduction="none"),
                       torch.zeros((), dtype=psi.dtype))


def surrogate_terms(psi, w, kappa):
    """w and kappa are already zero off the training mask."""
    return 0.5 * w * psi * psi - kappa * psi


def lik_terms(kind, psi, y, mk, kappa, w):
    if kind == "F":
        return bce_terms(psi, y, mk)
    if kind == "S":
        return surrogate_terms(psi, w, kappa)
    raise ValueError(kind)


@contextlib.contextmanager
def grad_only(model, params):
    """Enable requires_grad only for `params` (identity-compared) during a pass."""
    keep = {id(p) for p in params}
    saved = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for p, _ in saved:
            p.requires_grad_(id(p) in keep)
        yield
    finally:
        for p, rg in saved:
            p.requires_grad_(rg)


# ---------------------------------------------------------------------------------------------
# streamed passes
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def full_pass(model, data, W=None, W_out=None, eval_sets=None, per_row=False, per_col=False):
    """One streamed pass over all rows at the current parameters.

    Returns F_lik (sum of BCE over training cells), and if W is given S_lik (frozen surrogate
    under W). If W_out is given, the PG weights of the CURRENT parameters are written into it
    (it may be the same tensor as W: each chunk's S under the old W is computed before that
    chunk is overwritten) and S_lik_new (surrogate at its own touching point) is returned.
    eval_sets: {name: (Ytrue uint8 tensor n x m, mask bool tensor n x m)} -> NLL sums/counts.
    """
    dt = model.Z.dtype
    n, chunk = model.n, model.config.chunk_rows
    out = dict(F_lik=0.0)
    if W is not None:
        out["S_lik"] = 0.0
    if W_out is not None:
        out["S_lik_new"] = 0.0
    if per_row:
        out["F_rows"] = torch.zeros(n, dtype=F64)
        if W is not None:
            out["S_rows"] = torch.zeros(n, dtype=F64)
    if per_col:
        out["F_cols"] = torch.zeros(model.m, dtype=F64)
        if W is not None:
            out["S_cols"] = torch.zeros(model.m, dtype=F64)
    ev = {k: [0.0, 0] for k in (eval_sets or {})}
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        psi = model.logits_rows(s, e)
        y, mk, kappa = data.rows(s, e, dt)
        fl = bce_terms(psi, y, mk)
        out["F_lik"] += float(fl.sum(dtype=F64))
        if per_row:
            out["F_rows"][s:e] = fl.sum(1, dtype=F64)
        if per_col:
            out["F_cols"] += fl.sum(0, dtype=F64)
        if W is not None:
            sl = surrogate_terms(psi, W[s:e], kappa)
            out["S_lik"] += float(sl.sum(dtype=F64))
            if per_row:
                out["S_rows"][s:e] = sl.sum(1, dtype=F64)
            if per_col:
                out["S_cols"] += sl.sum(0, dtype=F64)
        if W_out is not None:
            wn = torch.where(mk, pg_mean(psi), torch.zeros((), dtype=dt))
            out["S_lik_new"] += float(surrogate_terms(psi, wn, kappa).sum(dtype=F64))
            W_out[s:e] = wn
        for k, (Yt, Mt) in (eval_sets or {}).items():
            Mb = Mt[s:e]
            if Mb.any():
                eta = psi[Mb].to(F64)
                yt = Yt[s:e][Mb].to(F64)
                ev[k][0] += float((Fnn.softplus(eta) - yt * eta).sum())
                ev[k][1] += int(yt.numel())
    for k, (tot, cnt) in ev.items():
        out[f"{k}_nll"] = tot / max(cnt, 1)
    return out


def lik_grad(model, data, kind, W, params, per_row=False, per_col=False):
    """Likelihood part of F (kind='F') or S (kind='S') and its gradient wrt `params` (.grad).

    Gradients are accumulated exactly over row chunks; priors are NOT included here.
    """
    dt = model.Z.dtype
    n, chunk = model.n, model.config.chunk_rows
    for p in params:
        p.grad = None
    tot = 0.0
    rows = torch.zeros(n, dtype=F64) if per_row else None
    cols = torch.zeros(model.m, dtype=F64) if per_col else None
    with grad_only(model, params), torch.enable_grad():
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            psi = model.logits_rows(s, e)
            y, mk, kappa = data.rows(s, e, dt)
            terms = lik_terms(kind, psi, y, mk, kappa, None if W is None else W[s:e])
            loss = terms.sum(dtype=F64)
            loss.backward()
            tot += float(loss.detach())
            if per_row:
                rows[s:e] = terms.detach().sum(1, dtype=F64)
            if per_col:
                cols += terms.detach().sum(0, dtype=F64)
    return tot, rows, cols


@torch.no_grad()
def unit_values(model, data, kind, W, axis, idx, h_cache=None):
    """Likelihood part per row (axis=0) or per column (axis=1) for units `idx` only."""
    dt = model.Z.dtype
    chunk = model.config.chunk_rows
    dec = model.decoder
    out = torch.zeros(len(idx), dtype=F64)
    if axis == 0:
        for a in range(0, len(idx), chunk):
            r = idx[a:a + chunk]
            psi = dec(model.Z[r])
            y, mk, kappa = data.rows_idx(r, dt)
            out[a:a + chunk] = lik_terms(kind, psi, y, mk, kappa, None if W is None else W[r]).sum(1, dtype=F64)
        return out
    allc = len(idx) == model.m
    for s in range(0, model.n, chunk):
        e = min(model.n, s + chunk)
        h = h_cache[s:e] if h_cache is not None else dec.h(model.Z[s:e])
        psi = dec.link(dec.pre(h, None if allc else idx))
        y, mk, kappa = data.rows(s, e, dt)
        w = None if W is None else W[s:e]
        if not allc:
            y, mk, kappa = y[:, idx], mk[:, idx], kappa[:, idx]
            w = None if w is None else w[:, idx]
        out += lik_terms(kind, psi, y, mk, kappa, w).sum(0, dtype=F64)
    return out


def neural_map_objective(model, Y=None, observed_mask=None, data=None):
    """Original objective parts at the current parameters (handoff 4.11 API)."""
    from .data import MaskedBinary
    if data is None:
        data = MaskedBinary(Y, observed_mask)
    p = full_pass(model, data)
    pen = model.penalty_parts()
    total_pen = sum(pen.values())
    return dict(original_nll=p["F_lik"], penalty=total_pen, penalty_parts=pen,
                original_map_loss=p["F_lik"] + total_pen, n_observed=data.n_obs)


@torch.no_grad()
def compute_pg_weights(model, data):
    """Detached PG weights f(psi_old) at the current parameters, zero off the training mask."""
    W = torch.empty(model.n, model.m, dtype=model.Z.dtype)
    p = full_pass(model, data, W_out=W)
    return W, p


# ---------------------------------------------------------------------------------------------
# exact affine-head update (handoff 4.6)
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def solve_affine_head(model, data, W, col_block=4096, return_diagnostics=False):
    """Per column j solve [Lambda_H + sum_i w_ij h~_i h~_i'] theta_j = sum_i kappa_ij h~_i,
    h~_i = [1; h_i], theta_j = [b_j; v_j], Lambda_H = diag(lambda_bias, lambda_output I_H).

    Exact block-coordinate minimiser of the frozen surrogate S over (V, b) with Z and the hidden
    network fixed. Cholesky solves; no inverse is formed. All-missing columns give theta_j = 0.
    """
    cfg = model.config
    if cfg.head_mode != "affine":
        raise RuntimeError("the exact linear head solve is only valid for head_mode='affine'; "
                           "the nonlinear head must be optimised by gradients")
    dec = model.decoder
    dt = model.Z.dtype
    n, m, H = model.n, model.m, dec.H
    H1 = H + 1
    chunk = cfg.chunk_rows
    Ht = torch.empty(n, H1, dtype=dt)
    Ht[:, 0] = 1.0
    for s in range(0, n, chunk):
        Ht[s:s + chunk, 1:] = dec.h(model.Z[s:s + chunk])
    G = (Ht[:, :, None] * Ht[:, None, :]).reshape(n, H1 * H1)          # rows vec(h~ h~')
    reg = torch.cat([torch.tensor([cfg.lambda_bias], dtype=F64), torch.full((H,), cfg.lambda_output, dtype=F64)])
    rhs = torch.zeros(m, H1, dtype=F64)
    for s in range(0, n, chunk):
        _, _, kappa = data.rows(s, s + chunk, dt)
        rhs += (kappa.T @ Ht[s:s + chunk]).to(F64)
    theta = torch.empty(m, H1, dtype=F64)
    max_res = 0.0
    for c0 in range(0, m, col_block):
        c1 = min(m, c0 + col_block)
        A = (W[:, c0:c1].T @ G).to(F64).reshape(-1, H1, H1)
        A.diagonal(dim1=1, dim2=2).add_(reg)
        L, info = torch.linalg.cholesky_ex(A)
        if int(info.max()) > 0:
            raise RuntimeError("head normal matrix not positive definite (check priors / weights)")
        th = torch.cholesky_solve(rhs[c0:c1, :, None], L)[..., 0]
        theta[c0:c1] = th
        if return_diagnostics:
            res = (A @ th[..., None])[..., 0] - rhs[c0:c1]
            rel = res.norm(dim=1) / rhs[c0:c1].norm(dim=1).clamp_min(1e-300)
            max_res = max(max_res, float(rel.max()))
        del A, L
    dec.b.copy_(theta[:, 0].to(dt))
    dec.V.copy_(theta[:, 1:].to(dt))
    if return_diagnostics:
        return dict(max_rel_residual=max_res, theta=theta)
    return None


# ---------------------------------------------------------------------------------------------
# reduced (sufficient-statistic) form of S for the affine head with (V, b) held fixed
# ---------------------------------------------------------------------------------------------
class ReducedAffineSurrogate:
    """S_lik(Z, eta) = sum_i [ h_i' M_i h_i / 2 + r_i' h_i + c_i ]  exactly, with
       M_i = sum_j w_ij v_j v_j',  r_i = sum_j (w_ij b_j - kappa_ij) v_j,
       c_i = sum_j (w_ij b_j^2 / 2 - kappa_ij b_j),
    for frozen W and the CURRENT (fixed) affine head. An algebraic identity, not an approximation;
    it makes Z / hidden inner steps cost O(n H^2) instead of O(n m H). Stored in float64.
    """

    @torch.no_grad()
    def __init__(self, model, data, W):
        dec = model.decoder
        dt = model.Z.dtype
        V, b = dec.V.detach(), dec.b.detach()
        n, m, H = model.n, model.m, dec.H
        chunk = model.config.chunk_rows
        GV = (V[:, :, None] * V[:, None, :]).reshape(m, H * H)
        self.M = torch.empty(n, H, H, dtype=F64)
        self.r = torch.empty(n, H, dtype=F64)
        self.c = torch.empty(n, dtype=F64)
        bb = 0.5 * b * b
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            Wc = W[s:e]
            _, _, kappa = data.rows(s, e, dt)
            self.M[s:e] = (Wc @ GV).to(F64).reshape(-1, H, H)
            self.r[s:e] = ((Wc * b - kappa) @ V).to(F64)
            self.c[s:e] = (Wc @ bb - kappa @ b).to(F64)
        del GV

    def row_lik(self, h, idx=None):
        """h: float64 (len(idx) or n) x H."""
        if idx is not None and len(idx) > self.M.shape[0] // 4:
            # gathering most of M (n x H x H) costs more than evaluating every row
            full = torch.zeros(self.M.shape[0], h.shape[1], dtype=h.dtype)
            full[idx] = h
            return self.row_lik(full)[idx]
        M = self.M if idx is None else self.M[idx]
        r = self.r if idx is None else self.r[idx]
        c = self.c if idx is None else self.c[idx]
        Mh = torch.bmm(M, h.unsqueeze(2)).squeeze(2)
        return 0.5 * (h * Mh).sum(1) + (r * h).sum(1) + c
