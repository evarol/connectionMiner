from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from .models import CmResult, PrepData, RawData


def cm_build_type_gene_probabilities(
    raw: RawData,
    prep: PrepData,
    cm: CmResult,
    cfg: dict[str, Any],
) -> None:
    G_cells = raw.G_cells
    n_cells, ng_shared = G_cells.shape

    meta_sizes = prep.meta_sizes.astype(float)
    K = meta_sizes.size
    cell_to_metacell = prep.cell_to_metacell.astype(int)

    M = sparse.csr_matrix(
        (np.ones(n_cells), (np.arange(n_cells), cell_to_metacell)),
        shape=(n_cells, K),
    )

    G_bin = G_cells > 0
    G_metacell_counts_all = M.T @ G_bin.astype(float)
    G_metacell_p_all = G_metacell_counts_all / np.maximum(meta_sizes[:, None], 1.0)

    min_samples_prior = int(cfg.get("metacell", {}).get("min_samples_prior", 0) or 0)
    if min_samples_prior > 0:
        p_global = np.mean(G_bin, axis=0)
        for k in range(K):
            n_k = meta_sizes[k]
            if n_k < min_samples_prior:
                G_metacell_p_all[k, :] = (
                    n_k * G_metacell_p_all[k, :] + (min_samples_prior - n_k) * p_global
                ) / min_samples_prior

    P = cm.P
    G_type_prob_full = P @ G_metacell_p_all

    W_meta = M.T.multiply(1.0 / np.maximum(meta_sizes[:, None], 1.0))
    P_all = P @ W_meta
    if sparse.issparse(P_all):
        P_all_arr = P_all.toarray()
    else:
        P_all_arr = np.asarray(P_all, dtype=float)

    type_mass = np.sum(P_all_arr, axis=1)
    identifiable = type_mass > 1e-10

    col_sums = np.sum(P_all_arr, axis=0)
    P_all_col_norm = np.zeros_like(P_all_arr)
    pos = col_sums > 0
    if np.any(pos):
        P_all_col_norm[:, pos] = P_all_arr[:, pos] / col_sums[pos]
    cell_contributions = np.sum(P_all_col_norm, axis=1)

    cell_type = np.zeros(n_cells, dtype=int)
    for c in range(n_cells):
        k = cell_to_metacell[c]
        cell_type[c] = int(np.argmax(P[:, k]))
    n_cells_type = np.bincount(cell_type, minlength=P.shape[0])

    cm.meta["G_type_prob_full"] = G_type_prob_full
    cm.meta["identifiable_type"] = identifiable
    cm.meta["cell_contributions"] = cell_contributions
    cm.meta["n_cells_type"] = n_cells_type
