"""Shared helpers for the BernFac real-data study (outputs/bernfac_20261020/real).

- paths and data loading
- the fixed cell / gene display ordering used by every heatmap
- the rotation-invariant cell embedding used for all decoding analyses
- the house figure style (scientific-figure-making skill) and export helper
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "outputs/bernfac_20261020/data"
OUT = ROOT / "outputs/bernfac_20261020/real"
FITS = OUT / "fits"
TABLES = OUT / "tables"
FIGS = OUT / "figures"

# Study grid (primary binarization Y_ge1)
LAMBDAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
KS = [1, 2, 3, 5, 10, 20]
# Sensitivity grid (Y_ge2)
LAMBDAS_GE2 = [0.3, 1.0, 3.0]
KS_GE2 = [2, 5, 10]
FIT_KW = dict(max_iter=500, tol=1e-7)
N_RANDOM_RESTARTS = 3
INIT_SCALE = 0.1

HEMILINEAGES = ["14A", "04B", "09A", "12B", "03A"]
LIBRARIES = ["SITTA9", "SITTB9", "SITTG8", "SITTC8"]


def fit_tag(dataset, lam, k):
    return f"{dataset}_lam{lam:g}_k{k}"


def load_data():
    """Return dict with counts, Y_ge1, Y_ge2 (float arrays), cells, genes."""
    return dict(
        counts=np.load(DATA / "counts.npy").astype(float),
        Y_ge1=np.load(DATA / "Y_ge1.npy").astype(float),
        Y_ge2=np.load(DATA / "Y_ge2.npy").astype(float),
        cells=pd.read_csv(DATA / "cells.csv"),
        genes=pd.read_csv(DATA / "genes.csv"),
    )


def display_order(cells, Y):
    """Cells sorted by hemilineage then library (then total UMI); genes ordered
    by average-linkage hierarchical clustering of their Y_ge1 columns
    (correlation distance). Fixed for every heatmap in the study."""
    from scipy.cluster.hierarchy import leaves_list, linkage

    c = cells.copy()
    c["_h"] = c["hemilineage"].map({h: i for i, h in enumerate(HEMILINEAGES)})
    c["_l"] = c["library"].map({l: i for i, l in enumerate(LIBRARIES)})
    row = c.sort_values(["_h", "_l", "total_umi"]).index.to_numpy()
    col = leaves_list(linkage(Y.T, method="average", metric="correlation"))
    return row, col


def cell_embedding(U, V):
    """Rotation-invariant cell representation from a fit.

    Psi = U V' = A S B' (thin SVD). We use E = A S (n x k): the rows of E have
    exactly the pairwise Euclidean distances of the rows of logit(P) = Psi, and
    E is unchanged by the gauge (U R, V R) up to column signs (which neither
    kNN nor logistic regression cares about). Raw U is never used directly.
    """
    A, s, _ = np.linalg.svd(U @ V.T, full_matrices=False)
    k = U.shape[1]
    return A[:, :k] * s[:k]


def load_fit(dataset, lam, k):
    z = np.load(FITS / f"{fit_tag(dataset, lam, k)}.npz")
    return {key: z[key] for key in z.files}


# ----------------------------------------------------------------------------
# House figure style (scientific-figure-making skill) -- see figstyle.py
# ----------------------------------------------------------------------------
from figstyle import (DEFAULT_COLORS, PALETTE, FigureStyle,  # noqa: E402,F401
                      apply_publication_style)
from figstyle import finalize_figure as _finalize  # noqa: E402

# categorical annotation colours (hemilineage / library / sex)
HEMI_COLORS = dict(zip(HEMILINEAGES, [PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"],
                                      PALETTE["teal"], PALETTE["violet"]]))
LIB_COLORS = dict(zip(LIBRARIES, ["#E69F00", "#F0C987", "#555555", "#AAAAAA"]))
SEX_COLORS = {"F": "#D55E8B", "M": "#3A9BD5"}


def finalize_figure(fig, name, dpi=200, pad=0.05):
    """Save figures/<name>.png (dpi 200) and .pdf; warn if PNG > 1.5 MB."""
    return _finalize(fig, FIGS / name, dpi=dpi, pad=pad)
