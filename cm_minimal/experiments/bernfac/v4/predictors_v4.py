"""Loaders for the W2 (v4) refits, wrapping v3/predictors.py (unchanged) with the same conventions:
train = ~heldout, Predictor.block / pairs / logpairs, exact log P(y=0) for count models, no n x m array.

    import predictors_v4 as P4
    p = P4.load_fit("poisson_k120_lam10_from80")   # any v4 fit by tag (see P4.list_fits())
    p = P4.load("R6")                              # v4-selected config per family (falls back to v3)
    lP, lQ = p.logpairs(i, j)                      # score any index pairs, e.g. W1's bootstrap
    P4.nll_on_mask(p, Y, M)                        # = v3 predictors.nll_on_mask

Tags
    poisson_k{K}_lam{L}_{start}.npz, nb_k{K}_lam{L}_{start}.npz   -> v3 Counts (PO / R6)
    off_k{K}_lam{L}_{start}.npz                                   -> v3 Logit with offsets (R4)
    neural/{name}.pt                                              -> v3 Neural (R8)
start = from{K0} (warm start: K0 fit padded with N(0, 0.01^2) columns) or cold.
SELECTED is filled from v4/tables/comparison_v4.csv (validation-only selection) when it exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v3"))
import numpy as np  # noqa: E402
import predictors as P3  # noqa: E402

V4 = P3.STUDY / "v4"
FITS = V4 / "fits"
nll_on_mask = P3.nll_on_mask
standardized = P3.standardized          # v3 selected fits (R4/R5/R6) only


def list_fits():
    return sorted([p.stem for p in FITS.glob("*.npz")] + [f"neural/{p.stem}" for p in (FITS / "neural").glob("*.pt")])


def load_fit(tag):
    if tag.startswith("neural/"):
        p = P3.Neural(FITS / f"{tag}.pt"); p.code = "R8"; p.tag = tag; return p
    z = np.load(FITS / f"{tag}.npz")
    if tag.startswith(("poisson_", "nb_")):
        p = P3.Counts(z["U"], z["V"], z["a"], z["b"], z["o"], z["phi"])
        p.code = "PO" if tag.startswith("poisson_") else "R6"
    elif tag.startswith("off_"):
        p = P3.Logit(z["U"], z["V"], z["a"], z["b"]); p.code = "R4"
    else:
        raise KeyError(tag)
    p.tag = tag
    return p


def selected():
    """code -> source of the v4-selected config ('v3' or a v4 tag), from comparison_v4.csv."""
    f = V4 / "tables/comparison_v4.csv"
    if not f.exists():
        return {}
    import pandas as pd
    d = pd.read_csv(f)
    return {r.code: r.source for r in d.itertuples() if isinstance(r.source, str) and r.source != "v3"}


def load(code):
    """v4-selected fit for a family if a v4 refit won validation, else the v3 fit (P3.load)."""
    sel = selected()
    if code in sel:
        p = load_fit(sel[code]); p.code = code; return p
    return P3.load(code)


ALL = P3.ALL
