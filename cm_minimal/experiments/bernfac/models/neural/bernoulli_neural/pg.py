"""Stable scalar utilities and the Polya-Gamma quadratic bound (handoff section 2).

pg_mean(x)            = tanh(x/2) / (2x), 1/4 at 0  (PG(1, x) mean; the frozen surrogate weight)
bound_constant(xi)    = -log(2 cosh(xi/2)) + 0.5 * pg_mean(xi) * xi^2
log p(y | psi) >= kappa psi - 0.5 w psi^2 + c(xi),  w = pg_mean(xi), kappa = y - 1/2,
with equality at psi = +-xi. For point-MAP (model 2), xi = |psi_old|.
"""
from __future__ import annotations

import torch

SMALL = 1e-5


def softplus(x):
    """log(1 + e^x) = max(x, 0) + log1p(exp(-|x|)); no overflow.

    Evaluated with torch.logaddexp(x, 0), which computes exactly this stable form but also has
    the correct derivative sigmoid(x) at x = 0 (the literal max/abs expression gives autograd
    the subgradient 1 there).
    """
    return torch.logaddexp(x, torch.zeros_like(x))


def log_sigmoid(x):
    return -softplus(-x)


def log_2cosh_half(x):
    """log(2 cosh(x/2)) = |x|/2 + log1p(exp(-|x|))."""
    a = torch.abs(x)
    return 0.5 * a + torch.log1p(torch.exp(-a))


def pg_mean(x):
    """tanh(x/2)/(2x) with an indexed series branch for |x| < 1e-5 (no 0/0 is ever formed).

    Positive, even, at most 1/4, ~ 1/(2|x|) for large |x|.
    """
    a = torch.abs(x)
    small = a < SMALL
    a_safe = torch.where(small, torch.ones_like(a), a)
    big = torch.tanh(0.5 * a_safe) / (2.0 * a_safe)
    s2 = a * a
    series = 0.25 - s2 / 48.0 + s2 * s2 / 480.0
    return torch.where(small, series, big)


def bound_constant(xi):
    return -log_2cosh_half(xi) + 0.5 * pg_mean(xi) * xi * xi


def bernoulli_loglik(psi, y):
    """y psi - softplus(psi), elementwise."""
    return y * psi - softplus(psi)


def pg_lower_bound(psi, y, xi):
    """kappa psi - 0.5 f(xi) psi^2 + c(xi) <= log p(y | psi)."""
    w = pg_mean(xi)
    return (y - 0.5) * psi - 0.5 * w * psi * psi + bound_constant(xi)
