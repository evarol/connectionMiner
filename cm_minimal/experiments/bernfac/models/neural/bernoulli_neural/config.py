"""Typed configuration for the neural-manifold auto-decoder (handoff section 4.11)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Optional

import torch

HEAD_MODES = ("affine", "nonlinear")
OPTIMIZER_MODES = ("pg_gem", "direct_bce")
DTYPES = {"float64": torch.float64, "float32": torch.float32}


@dataclass
class NeuralConfig:
    """All hyperparameters are explicit. Nothing is changed silently during a fit.

    Model:   z_i ~ N(0, I/lambda_z),  h_i = h_eta(z_i) (tanh MLP),  t_i = b + V h_i
             psi_i = t_i                      (head_mode='affine')
             psi_i = t_i + rho * tanh(t_i)    (head_mode='nonlinear')
             Y_ij ~ Bernoulli(sigmoid(psi_ij)) on training cells.
    Priors:  eta ~ N(0, I/lambda_hidden), V ~ N(0, I/lambda_output), b ~ N(0, I/lambda_bias).
    Fit:     joint MAP over (Z, eta, V, b). Not a VAE, not a marginal likelihood.
    """

    latent_dim: int
    hidden_widths: tuple = (64, 64)       # () gives the linear-network limit h = z
    activation: str = "tanh"
    head_mode: str = "affine"
    nonlinear_head_rho: float = 1.0
    optimizer_mode: str = "pg_gem"
    dtype: str = "float64"
    device: str = "cpu"

    lambda_z: float = 1.0
    lambda_hidden: float = 0.01
    lambda_output: float = 1.0
    lambda_bias: float = 0.01

    max_outer: int = 200
    tol: float = 1e-6                     # stop when (F_old - F_new) / max(1, |F_old|) < tol
    max_seconds: float = math.inf         # wall-clock budget (termination reason 'time_limit')

    # inner schedule (per outer iteration)
    inner_steps: int = 5                  # accepted gradient steps per block (Z, hidden) per round
    inner_rounds: int = 1                 # Z/hidden alternations per outer iteration
    head_steps: int = 5                   # gradient head steps (nonlinear head or direct_bce)
    resolve_head: bool = True             # affine pg_gem: re-solve the head after the Z/hidden steps

    # Armijo backtracking
    step_init: float = 1.0
    adaptive_step: bool = True            # start each search from 2x the last accepted step
    step_max: float = 1e6
    backtrack: float = 0.5
    armijo: float = 0.25                  # handoff suggests 1e-4; with the adaptive (doubling) initial
                                          # step that accepts near-2/L oscillating steps on stiff rows,
                                          # so 0.25 is the default here (configurable; tests cover both)
    max_trials: int = 30
    grad_tol: float = 1e-30               # squared-gradient-norm threshold for "numerically zero"

    # monotonicity check: a violation is F_new > F_old + monotone_rtol * max(1, |F_old|)
    monotone_rtol: Optional[float] = None  # None -> 1e-10 (float64) or 1e-7 (float32)

    seed: int = 1
    n_restarts: int = 3
    z_init_scale: float = 0.1             # Z ~ N(0, scale^2) unless init_Z is supplied
    head_init_scale: float = 0.01         # V ~ N(0, scale^2)
    bias_init: str = "column_logit"       # training-only smoothed column logits, or 'zero'
    bias_pseudocount: float = 0.5

    observed_mask: Optional[object] = field(default=None, repr=False)  # optional training mask (n x m bool)
    chunk_rows: int = 4096                # rows per streamed block (exact accumulation)
    val_every: int = 1                    # validation NLL every k outer iterations (if a mask is given)
    verbose: bool = False

    def __post_init__(self):
        self.hidden_widths = tuple(int(w) for w in self.hidden_widths)
        if int(self.latent_dim) != self.latent_dim or self.latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
        if any(w < 1 for w in self.hidden_widths):
            raise ValueError("hidden widths must be positive")
        if self.activation != "tanh":
            raise ValueError("only the smooth 'tanh' hidden activation is implemented")
        if self.head_mode not in HEAD_MODES:
            raise ValueError(f"head_mode must be one of {HEAD_MODES}")
        if self.optimizer_mode not in OPTIMIZER_MODES:
            raise ValueError(f"optimizer_mode must be one of {OPTIMIZER_MODES}")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {tuple(DTYPES)}")
        for name in ("lambda_z", "lambda_hidden", "lambda_output", "lambda_bias"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be a strictly positive finite precision, got {v!r}")
        if not (0 < self.backtrack < 1):
            raise ValueError("backtrack must be in (0, 1)")
        if not (0 < self.armijo < 1):
            raise ValueError("armijo must be in (0, 1)")
        if self.nonlinear_head_rho < 0:
            raise ValueError("nonlinear_head_rho must be >= 0")
        if self.bias_init not in ("column_logit", "zero"):
            raise ValueError("bias_init must be 'column_logit' or 'zero'")
        if self.monotone_rtol is None:
            self.monotone_rtol = 1e-10 if self.dtype == "float64" else 1e-7

    @property
    def torch_dtype(self):
        return DTYPES[self.dtype]

    @property
    def H(self):
        return self.hidden_widths[-1] if self.hidden_widths else self.latent_dim

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "observed_mask"}
        d["hidden_widths"] = list(self.hidden_widths)
        return d

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
