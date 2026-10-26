"""Shared loading for the full-data LBM runs (16,460 cells x 19,543 genes)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lbm  # noqa: E402

ROOT = HERE.parents[4]
DATA = ROOT / "outputs/bernfac_20261020/full/data"
OUT = ROOT / "outputs/bernfac_20261020/models/lbm"
CELLS = ROOT / "outputs/connectionminer_ready_inputs/bundles/motor/premn_24h/cells.csv.gz"


def load():
    """Y (uint8), val/test masks, and the training BinaryData (float32, train = ~heldout)."""
    Y = np.load(DATA / "Y.npy")
    H = np.load(DATA / "heldout.npy")
    val = np.load(DATA / "val.npy"); test = np.load(DATA / "test.npy")
    assert not (val & test).any() and np.array_equal(val | test, H)
    M = (~H).astype(np.float32)
    del H
    Yt = Y.astype(np.float32); Yt *= M
    data = lbm.BinaryData.from_train_arrays(Yt, M)
    return Y, val, test, data


def spectral_svd(data, rank=20, seed=0):
    """Truncated SVD of the training-only proxy M.*(Y - 1/2), cached on disk."""
    f = OUT / "cache_spectral_svd.npz"
    if f.exists():
        z = np.load(f); return z["U"], z["V"]
    from sklearn.utils.extmath import randomized_svd
    Z = data.Yt - 0.5 * data.M
    U, s, Vt = randomized_svd(Z, rank, n_iter=4, random_state=seed)
    del Z
    U, V = (U * s).astype(np.float32), (Vt.T * s).astype(np.float32)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, U=U, V=V)
    return U, V
