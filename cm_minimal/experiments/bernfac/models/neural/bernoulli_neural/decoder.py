"""Decoder g_theta(z) = head(b + V h_eta(z)) and the auto-decoder model (handoff 4.1-4.2).

Parameter groups are kept separate so each Gaussian prior is counted exactly once:
  hidden  eta = weights and biases of the tanh MLP z -> h  (lambda_hidden)
  output  V   (m x H)                                      (lambda_output)
  bias    b   (m)                                          (lambda_bias)
  rows    Z   (n x d)                                      (lambda_z)

Rank note (affine head): Psi = H_Z V' + 1 b', so rank(Psi) <= min(n, m, H + 1) whatever the
latent dimension d. A width-H affine head confines all logits to an (H+1)-dimensional affine
span. The nonlinear head psi = t + rho*tanh(t) removes that particular bound but loses the
exact per-column ridge solve. Neither head makes the latent map injective or fixes the
intrinsic dimension at d.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .config import NeuralConfig


class NeuralDecoder(nn.Module):
    def __init__(self, cfg: NeuralConfig, m: int):
        super().__init__()
        dt = cfg.torch_dtype
        layers = []
        prev = cfg.latent_dim
        for w in cfg.hidden_widths:
            layers += [nn.Linear(prev, w, dtype=dt), nn.Tanh()]
            prev = w
        self.hidden = nn.Sequential(*layers)          # identity when hidden_widths == ()
        self.H = prev
        self.m = m
        self.head_mode = cfg.head_mode
        self.rho = float(cfg.nonlinear_head_rho)
        self.V = nn.Parameter(torch.zeros(m, self.H, dtype=dt))
        self.b = nn.Parameter(torch.zeros(m, dtype=dt))

    # -- parameter groups ---------------------------------------------------
    def hidden_parameters(self):
        return list(self.hidden.parameters())

    def output_parameters(self):
        return [self.V, self.b]

    # -- forward ------------------------------------------------------------
    def h(self, z):
        return self.hidden(z)

    def pre(self, h, cols=None):
        """t = b + V h (rows of h), optionally restricted to a column index tensor."""
        if cols is None:
            return torch.addmm(self.b, h, self.V.T)
        return torch.addmm(self.b[cols], h, self.V[cols].T)

    def link(self, t):
        if self.head_mode == "affine":
            return t
        return t + self.rho * torch.tanh(t)

    def forward(self, z, cols=None):
        return self.link(self.pre(self.h(z), cols))


class NeuralManifoldModel(nn.Module):
    """Row coordinates Z (optimised variables, no encoder) + shared decoder."""

    def __init__(self, cfg: NeuralConfig, n: int, m: int):
        super().__init__()
        self.config = cfg
        self.decoder = NeuralDecoder(cfg, m)
        self.Z = nn.Parameter(torch.zeros(n, cfg.latent_dim, dtype=cfg.torch_dtype))
        self.row_ids = None          # optional external identifiers, original order preserved
        self.col_ids = None
        self.meta = {}

    @property
    def n(self):
        return self.Z.shape[0]

    @property
    def m(self):
        return self.decoder.m

    def logits_rows(self, s, e, cols=None):
        return self.decoder(self.Z[s:e], cols)

    def penalty(self):
        """Sum of the four Gaussian log-prior penalties (each group counted once)."""
        c = self.config
        hid = sum((p * p).sum() for p in self.decoder.hidden_parameters()) if self.decoder.hidden_parameters() \
            else torch.zeros((), dtype=self.Z.dtype)
        return (0.5 * c.lambda_z * (self.Z * self.Z).sum()
                + 0.5 * c.lambda_hidden * hid
                + 0.5 * c.lambda_output * (self.decoder.V * self.decoder.V).sum()
                + 0.5 * c.lambda_bias * (self.decoder.b * self.decoder.b).sum())

    def penalty_parts(self):
        c = self.config
        with torch.no_grad():
            hid = sum(float((p * p).sum()) for p in self.decoder.hidden_parameters())
            return dict(z=0.5 * c.lambda_z * float((self.Z * self.Z).sum()),
                        hidden=0.5 * c.lambda_hidden * hid,
                        output=0.5 * c.lambda_output * float((self.decoder.V ** 2).sum()),
                        bias=0.5 * c.lambda_bias * float((self.decoder.b ** 2).sum()))


def initialize(model: NeuralManifoldModel, data, seed: int, init_Z=None):
    """Nonzero, seeded initialisation from training data only (handoff 4.9).

    Hidden layers: PyTorch default (uniform +-1/sqrt(fan_in)). V ~ N(0, head_init_scale^2).
    b: training-only smoothed column logits (or zero). Z ~ N(0, z_init_scale^2) unless init_Z
    (e.g. a training-only SVD start) is given; then Z = init_Z + 1e-3 N(0,1) jitter per seed.
    """
    cfg = model.config
    g = torch.Generator().manual_seed(int(seed))
    dt = cfg.torch_dtype
    with torch.no_grad():
        for mod in model.decoder.hidden:
            if isinstance(mod, nn.Linear):
                bound = 1.0 / math.sqrt(mod.in_features)
                mod.weight.copy_((torch.rand(mod.weight.shape, generator=g, dtype=torch.float64) * 2 - 1) * bound)
                mod.bias.copy_((torch.rand(mod.bias.shape, generator=g, dtype=torch.float64) * 2 - 1) * bound)
        model.decoder.V.copy_(cfg.head_init_scale * torch.randn(model.decoder.V.shape, generator=g,
                                                                dtype=torch.float64))
        if cfg.bias_init == "column_logit":
            r = data.column_rates(cfg.bias_pseudocount)
            t = torch.log(r) - torch.log1p(-r)
            if cfg.head_mode == "nonlinear" and cfg.nonlinear_head_rho > 0:
                t = invert_link(t, cfg.nonlinear_head_rho)
            model.decoder.b.copy_(t.to(dt))
        else:
            model.decoder.b.zero_()
        if init_Z is None:
            Z0 = cfg.z_init_scale * torch.randn(model.Z.shape, generator=g, dtype=torch.float64)
        else:
            Z0 = torch.as_tensor(init_Z, dtype=torch.float64)
            if Z0.shape != model.Z.shape:
                raise ValueError(f"init_Z shape {tuple(Z0.shape)} != {tuple(model.Z.shape)}")
            Z0 = Z0 + 1e-3 * torch.randn(Z0.shape, generator=g, dtype=torch.float64)
        model.Z.copy_(Z0.to(dt))
    return model


def invert_link(psi, rho, iters=60):
    """Solve t + rho tanh(t) = psi elementwise (monotone; Newton from t = psi)."""
    t = psi.clone()
    for _ in range(iters):
        th = torch.tanh(t)
        t = t - (t + rho * th - psi) / (1 + rho * (1 - th * th))
    return t
