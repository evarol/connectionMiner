"""Input validation and a streamed view of a masked binary matrix (handoff section 1).

Y is stored once as uint8 with excluded cells set to 0 (Ysafe); a boolean mask marks the
training observations. Excluded cells never contribute, whatever value is stored there.
Without a mask, every non-NaN entry is observed. Implicit zeros of a sparse matrix are
observed zeros.
"""
from __future__ import annotations

import numpy as np
import torch


def _to_dense(Y):
    if hasattr(Y, "toarray"):                      # scipy.sparse: implicit zeros are observed zeros
        return np.asarray(Y.toarray())
    if isinstance(Y, torch.Tensor):
        return Y.detach().cpu().numpy()
    return np.asarray(Y)


class MaskedBinary:
    """Validated binary matrix + training mask, read in row chunks."""

    def __init__(self, Y, observed_mask=None, allow_empty_rows=True):
        Y = _to_dense(Y)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D, got shape {Y.shape}")
        n, m = Y.shape
        if observed_mask is None:
            if Y.dtype.kind == "f":
                mask = ~np.isnan(Y)
            else:
                mask = np.ones((n, m), bool)
        else:
            mask = _to_dense(observed_mask)
            if mask.shape != (n, m):
                raise ValueError(f"observed_mask shape {mask.shape} does not match Y {Y.shape}")
            if mask.dtype != bool:
                raise ValueError("observed_mask must be boolean")
        if not mask.any():
            raise ValueError("no training observations: the observed mask is empty")
        if Y.dtype == np.uint8 or Y.dtype == bool:
            bad = mask & (Y > 1)
            if bad.any():
                raise ValueError("included entries of Y must be exactly 0 or 1")
            Ysafe = np.where(mask, Y, 0).astype(np.uint8)
        else:
            Yf = np.asarray(Y, dtype=np.float64) if Y.dtype.kind != "f" else Y
            vals = Yf[mask]
            if not np.all(np.isfinite(vals)):
                raise ValueError("included entries of Y must be finite (NaN/inf found at an observed cell)")
            if not np.all((vals == 0) | (vals == 1)):
                raise ValueError("included entries of Y must be exactly 0 or 1")
            Ysafe = np.where(mask, np.nan_to_num(Yf, nan=0.0), 0.0).astype(np.uint8)
        self.Y = torch.from_numpy(np.ascontiguousarray(Ysafe))
        self.mask = torch.from_numpy(np.ascontiguousarray(mask))
        self.n, self.m = n, m
        self.n_obs = int(mask.sum())

    @classmethod
    def from_trusted(cls, Y_uint8, mask_bool):
        """Fast path for large, already validated uint8 / bool arrays (no copies beyond Ysafe)."""
        obj = cls.__new__(cls)
        if Y_uint8.dtype != np.uint8 or mask_bool.dtype != bool or Y_uint8.shape != mask_bool.shape:
            raise ValueError("from_trusted needs uint8 Y and a bool mask of the same shape")
        Ysafe = Y_uint8.copy()
        Ysafe[~mask_bool] = 0
        if Ysafe.max() > 1:
            raise ValueError("Y must be binary")
        obj.Y, obj.mask = torch.from_numpy(Ysafe), torch.from_numpy(np.ascontiguousarray(mask_bool))
        obj.n, obj.m = Ysafe.shape
        obj.n_obs = int(mask_bool.sum())
        if obj.n_obs == 0:
            raise ValueError("no training observations")
        return obj

    def rows(self, s, e, dtype):
        """(y, mask, kappa) for rows s:e; kappa = y - 1/2 on training cells, 0 elsewhere."""
        y = self.Y[s:e].to(dtype)
        mk = self.mask[s:e]
        kappa = torch.where(mk, y - 0.5, torch.zeros((), dtype=dtype))
        return y, mk, kappa

    def rows_idx(self, idx, dtype):
        y = self.Y[idx].to(dtype)
        mk = self.mask[idx]
        kappa = torch.where(mk, y - 0.5, torch.zeros((), dtype=dtype))
        return y, mk, kappa

    def column_rates(self, pseudocount=0.5):
        """Training-only smoothed column detection rates (pseudocount on both outcomes)."""
        ones = torch.zeros(self.m, dtype=torch.float64)
        obs = torch.zeros(self.m, dtype=torch.float64)
        for s in range(0, self.n, 4096):
            ones += self.Y[s:s + 4096].to(torch.float64).sum(0)
            obs += self.mask[s:s + 4096].to(torch.float64).sum(0)
        return (ones + pseudocount) / (obs + 2 * pseudocount)
