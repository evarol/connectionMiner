"""Shared names, colours and paths for the v3 deck figures (one source of truth for all workers)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "outputs/bernfac_20261020"
V3 = STUDY / "v3"
FIGS = V3 / "figures"

# representation code -> (short name, long name, colour)
METHODS = {
    "R1": ("processed counts", "processed counts (log-norm, HVG, PCA)", "#0F4D92"),
    "R2": ("binary PCA", "binary Y (PCA)", "#9E9E9E"),
    "R3": ("BernFac logit", "BernFac logit (UVᵀ)", "#B64342"),
    "R4": ("BernFac + offsets", "BernFac logit + cell/gene offsets", "#42949E"),
    "R5": ("BernFac cloglog", "BernFac cloglog + offsets", "#9A4D8E"),
    "R6": ("NB counts", "negative-binomial count factorization", "#E69F00"),
    "R7": ("local-factor mixture", "mixture of local factor models", "#009E73"),
    "R8": ("neural manifold", "neural manifold auto-decoder", "#CC79A7"),
    "R9": ("latent blocks", "Bernoulli latent block model", "#56B4E9"),
}
ORDER = list(METHODS)
PROB_MODELS = ["R3", "R4", "R5", "R6", "R7", "R8", "R9"]   # models that output p_ij


def name(code, long=False):
    return METHODS[code][1 if long else 0]


def colour(code):
    return METHODS[code][2]
