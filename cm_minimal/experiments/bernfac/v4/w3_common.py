"""W3 (v4 revision): shared paths, data loading and the representation registry.

Representations (raw embedding -> centred SVD -> principal coordinates, see w3_build.py)
  R1  processed counts: log-norm, 2,000 HVGs (seurat_v3), scale, PCA 50   (v3 cache, unchanged)
  R2  binary Y, PCA 50 of column-centred Y                                 (v3 cache, unchanged)
  R3  BernFac logit         UV' of the TRAIN-ONLY fit cv_k40_lam10         (v3 used the all-entry refit)
  R4  BernFac + offsets     UV' of the TRAIN-ONLY fit cv_off_k40_lam10
  R5  BernFac cloglog       UV' of the TRAIN-ONLY fit cv_cll_k40_lam30
  R6  NB counts k=120       UV' (models/counts/embedding_nb.npy, train-only fit)
  R7  local-factor mixture  (models/mfa/embedding.npy, train-only fit)
  R8  neural manifold Z     (models/neural/embedding.npy; HOOK: --r8 / env W3_R8_EMBEDDING)
  R9  latent blocks R B     (models/lbm/embedding.npy)
  R10 analytic Pearson residuals (NB, theta=100) on the R1 HVGs, PCA 50    (new comparator)
  R11 scVI latent mean, 20 dims, raw counts of the R1 HVGs                 (new comparator)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
BF = HERE.parent
sys.path.insert(0, str(BF))
ROOT = BF.parents[2]
STUDY = ROOT / "outputs/bernfac_20261020"
V4 = STUDY / "v4"
FIGS, TABS, CACHE = V4 / "figures", V4 / "tables", V4 / "cache"
for _d in (FIGS, TABS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)
V3CACHE = STUDY / "full/umap/cache"
COUNTS = ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h/G_prime.npz"
GENES = ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h/genes.csv"
CELLS = ROOT / "outputs/connectionminer_ready_inputs/bundles/motor/premn_24h/cells.csv.gz"
KEPT = STUDY / "full/data/genes_kept.npy"
FITS = STUDY / "full/fits"
MODELS = STUDY / "models"

# R8 hook: W2 may deliver a better-converged neural fit. Point W3_R8_EMBEDDING (or --r8) at its Z.
R8_DEFAULT = MODELS / "neural/embedding.npy"


def r8_path():
    return Path(os.environ.get("W3_R8_EMBEDDING", R8_DEFAULT))


TRAIN_ONLY = {"R3": FITS / "cv_k40_lam10.npz", "R4": FITS / "cv_off_k40_lam10.npz",
              "R5": FITS / "cv_cll_k40_lam30.npz"}

METHODS = {
    "R1": ("processed counts", "#0F4D92"),
    "R2": ("binary PCA", "#9E9E9E"),
    "R3": ("BernFac logit", "#B64342"),
    "R4": ("BernFac + offsets", "#42949E"),
    "R5": ("BernFac cloglog", "#9A4D8E"),
    "R6": ("NB counts", "#E69F00"),
    "R7": ("local-factor mixture", "#009E73"),
    "R8": ("neural manifold", "#CC79A7"),
    "R9": ("latent blocks", "#56B4E9"),
    "R10": ("Pearson residuals", "#8C6D31"),
    "R11": ("scVI", "#2B2B2B"),
}
ORDER = list(METHODS)


def name(r):
    return METHODS[r][0]


def colour(r):
    return METHODS[r][1]


def load_counts():
    """Raw counts on the 19,543 kept genes (same column set as every model)."""
    X = sp.load_npz(COUNTS).tocsr()
    kept = np.load(KEPT)
    X = X[:, kept].tocsr()
    X.sort_indices()
    return X, kept


def gene_names(kept=None):
    g = pd.read_csv(GENES)
    col = [c for c in g.columns if "symbol" in c.lower() or "name" in c.lower() or "gene" in c.lower()][0]
    names = g[col].astype(str).values
    return names if kept is None else names[kept]


def load_cells(X=None):
    c = pd.read_csv(CELLS)
    c["birthtime"] = c["birthtime"].fillna("precursor")
    if X is None:
        X, _ = load_counts()
    c["log10_umi"] = np.log10(np.asarray(X.sum(1)).ravel())
    c["stratum"] = c["hemilineage"].astype(str) + "|" + c["birthtime"].astype(str)
    return c


def hvg_index(X):
    """The R1 HVG set (seurat_v3 on raw counts, 2,000 genes), cached as column indices into X."""
    f = CACHE / "w3_hvg_idx.npy"
    if f.exists():
        return np.load(f)
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(X.astype(np.float32))
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=2000)
    idx = np.flatnonzero(a.var["highly_variable"].values)
    np.save(f, idx)
    return idx
