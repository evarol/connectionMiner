"""bernoulli_neural: nonlinear latent manifold with a neural decoder for a binary matrix.

Auto-decoder joint MAP (no encoder): row coordinates Z and decoder parameters are optimised
together. Two optimisers for the same objective: 'pg_gem' (frozen Polya-Gamma quadratic
surrogate with checked steps, exact affine-head ridge solve) and 'direct_bce'.
See ../README.md.
"""
from .config import NeuralConfig
from .data import MaskedBinary
from .decoder import NeuralDecoder, NeuralManifoldModel, initialize
from .fit import fit_bernoulli_neural_manifold, fit_single
from .io import load_model, save_model
from .objectives import (ReducedAffineSurrogate, compute_pg_weights, full_pass, lik_grad,
                         neural_map_objective, solve_affine_head)
from .pg import bound_constant, log_2cosh_half, log_sigmoid, pg_mean, softplus
from .predict import (PREDICTION_MODE, heldout_scores, infer_neural_rows, predict_bernoulli_neural_manifold,
                      predict_logits, predict_pairs, prior_predictive_mc, scores_from_logit_fn)

__all__ = [n for n in dir() if not n.startswith("_")]
