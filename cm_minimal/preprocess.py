from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .models import PrepData, RawData


def cm_preprocess_binary(raw: RawData, cfg: dict[str, Any]) -> PrepData:
    rng = np.random.default_rng(cfg["seed"])

    X_bin = (raw.G_cells > 0).astype(float)
    ng_shared = raw.genes_shared.size

    n_genes_use = min(cfg.get("binary", {}).get("n_genes_use", 4000), ng_shared)
    min_cells = max(int(cfg.get("binary", {}).get("min_cells", 5)), 1)

    nz = np.sum(X_bin, axis=0)
    cand = np.where(nz >= min_cells)[0]
    if cand.size == 0:
        cand = np.arange(ng_shared)

    mu = np.mean(X_bin[:, cand], axis=0)
    var = mu * (1.0 - mu)
    ord_idx = np.argsort(-var)
    ng_use = min(n_genes_use, cand.size)
    solver_gene_idx = cand[ord_idx[:ng_use]]

    X_feat = X_bin[:, solver_gene_idx]
    cell_to_metacell, meta_sizes, signature_table = _build_metacells_from_features(
        X_feat,
        raw.P_constraints_cells,
        cfg["metacell"],
        cfg["seed"],
    )

    K = int(meta_sizes.size)
    M = sparse.csr_matrix(
        (np.ones(cell_to_metacell.size), (np.arange(cell_to_metacell.size), cell_to_metacell)),
        shape=(cell_to_metacell.size, K),
    )

    P_constraints_metacell = ((raw.P_constraints_cells @ M) > 0).astype(float).toarray()

    G_metacell_counts = M.T @ X_bin[:, solver_gene_idx]
    G_metacell_p = G_metacell_counts / np.maximum(meta_sizes[:, None], 1.0)

    min_samples_prior = int(cfg["metacell"].get("min_samples_prior", 0) or 0)
    if min_samples_prior > 0:
        p_global = np.mean(X_bin[:, solver_gene_idx], axis=0)
        small = np.where(meta_sizes < min_samples_prior)[0]
        for k in small:
            n_k = meta_sizes[k]
            G_metacell_p[k, :] = (
                n_k * G_metacell_p[k, :] + (min_samples_prior - n_k) * p_global
            ) / min_samples_prior
        if small.size:
            print(
                f"  Mean-cell prior (binary): shrunk {small.size} metacells with size < {min_samples_prior} toward global mean"
            )

    meta = dict(raw.meta)
    meta["N_metacells"] = int(K)
    meta["Ng_solver"] = int(solver_gene_idx.size)

    return PrepData(
        G_cells=raw.G_cells,
        genes_shared=raw.genes_shared,
        P_constraints_cells=raw.P_constraints_cells,
        C_counts=raw.C_counts,
        C_mask=raw.C_mask,
        umap_xy=raw.umap_xy,
        meta=meta,
        solver_gene_idx=solver_gene_idx,
        genes_solver=raw.genes_shared[solver_gene_idx],
        cell_to_metacell=cell_to_metacell,
        meta_sizes=meta_sizes,
        signature_table=signature_table,
        P_constraints_metacell=P_constraints_metacell,
        G_metacell_p=G_metacell_p,
        G_metacell_p_solve=G_metacell_p,
    )


def cm_preprocess_pg(raw: RawData, cfg: dict[str, Any]) -> PrepData:
    rng = np.random.default_rng(cfg["seed"])

    G = raw.G_cells.astype(float)
    ng_shared = raw.genes_shared.size

    n_genes_use = min(cfg.get("pg", {}).get("n_genes_use", ng_shared), ng_shared)
    min_cells = max(int(cfg.get("pg", {}).get("min_cells", 5)), 1)

    nz = np.sum(G > 0, axis=0)
    cand = np.where(nz >= min_cells)[0]
    if cand.size == 0:
        cand = np.arange(ng_shared)

    Gc = G[:, cand]
    mu = np.mean(Gc, axis=0)
    mu2 = np.mean(Gc ** 2, axis=0)
    var = mu2 - mu ** 2
    ord_idx = np.argsort(-var)
    ng_use = min(n_genes_use, cand.size)
    solver_gene_idx = cand[ord_idx[:ng_use]]

    X_feat = G[:, solver_gene_idx]
    cell_to_metacell, meta_sizes, signature_table = _build_metacells_from_features(
        X_feat,
        raw.P_constraints_cells,
        cfg["metacell"],
        cfg["seed"],
    )

    K = int(meta_sizes.size)
    M = sparse.csr_matrix(
        (np.ones(cell_to_metacell.size), (np.arange(cell_to_metacell.size), cell_to_metacell)),
        shape=(cell_to_metacell.size, K),
    )

    P_constraints_metacell = ((raw.P_constraints_cells @ M) > 0).astype(float).toarray()

    G_metacell_sum = M.T @ G[:, solver_gene_idx]
    G_metacell_p = G_metacell_sum / np.maximum(meta_sizes[:, None], 1.0)

    min_samples_prior = int(cfg["metacell"].get("min_samples_prior", 0) or 0)
    if min_samples_prior > 0:
        p_global = np.mean(G[:, solver_gene_idx], axis=0)
        small = np.where(meta_sizes < min_samples_prior)[0]
        for k in small:
            n_k = meta_sizes[k]
            G_metacell_p[k, :] = (
                n_k * G_metacell_p[k, :] + (min_samples_prior - n_k) * p_global
            ) / min_samples_prior
        if small.size:
            print(
                f"  Mean-cell prior (PG): shrunk {small.size} metacells with size < {min_samples_prior} toward global mean"
            )

    meta = dict(raw.meta)
    meta["N_metacells"] = int(K)
    meta["Ng_solver"] = int(solver_gene_idx.size)

    return PrepData(
        G_cells=raw.G_cells,
        genes_shared=raw.genes_shared,
        P_constraints_cells=raw.P_constraints_cells,
        C_counts=raw.C_counts,
        C_mask=raw.C_mask,
        umap_xy=raw.umap_xy,
        meta=meta,
        solver_gene_idx=solver_gene_idx,
        genes_solver=raw.genes_shared[solver_gene_idx],
        cell_to_metacell=cell_to_metacell,
        meta_sizes=meta_sizes,
        signature_table=signature_table,
        P_constraints_metacell=P_constraints_metacell,
        G_metacell_p=G_metacell_p,
        G_metacell_p_solve=G_metacell_p,
    )


def _build_metacells_from_features(
    X_cells: np.ndarray,
    P_constraints_cells: sparse.csr_matrix,
    metacell_cfg: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    n_cells = X_cells.shape[0]
    cell_to_metacell = np.full(n_cells, -1, dtype=int)

    groups = _signature_groups(P_constraints_cells)
    signature_rows = []

    next_meta = 0
    for sig_id, idx_cells in enumerate(groups, start=1):
        n_group = len(idx_cells)
        if n_group == 0:
            signature_rows.append((sig_id, 0, 0))
            continue

        K = max(1, int(round(n_group / metacell_cfg["target_size"])))
        if K == 1:
            cell_to_metacell[np.asarray(idx_cells, dtype=int)] = next_meta
            next_meta += 1
            signature_rows.append((sig_id, n_group, 1))
            continue

        X = X_cells[np.asarray(idx_cells, dtype=int), :].astype(float)

        score = _pca_features(X, metacell_cfg.get("n_pcs", 50), seed)
        kmeans = KMeans(
            n_clusters=K,
            n_init=int(metacell_cfg.get("kmeans_reps", 5)),
            max_iter=int(metacell_cfg.get("kmeans_maxiter", 200)),
            random_state=seed,
        )
        labels = kmeans.fit_predict(score)
        labels = _merge_tiny_clusters(labels, score, int(metacell_cfg["min_size"]))

        n_local = int(labels.max()) + 1
        for k in range(n_local):
            idx_local = np.asarray(idx_cells, dtype=int)[labels == k]
            cell_to_metacell[idx_local] = next_meta + k

        signature_rows.append((sig_id, n_group, n_local))
        next_meta += n_local

    if np.any(cell_to_metacell < 0):
        missing = int(np.sum(cell_to_metacell < 0))
        raise RuntimeError(f"Metacell assignment failed for {missing} cells.")

    meta_sizes = np.bincount(cell_to_metacell, minlength=next_meta).astype(float)
    signature_table = pd.DataFrame(
        signature_rows,
        columns=["signature_id", "n_cells", "n_metacells"],
    )
    return cell_to_metacell, meta_sizes, signature_table


def _pca_features(X: np.ndarray, n_pcs: int, seed: int) -> np.ndarray:
    if X.shape[0] <= 2 or X.shape[1] <= 1:
        return X

    n_comp = min(X.shape[0] - 1, X.shape[1], max(2, n_pcs))
    pca = PCA(n_components=n_comp, random_state=seed)
    score = pca.fit_transform(X)
    explained = np.cumsum(pca.explained_variance_ratio_) * 100.0
    keep = np.where(explained <= 95.0)[0]
    if keep.size == 0:
        keep = np.array([0], dtype=int)
    keep = keep[: min(keep.size, n_pcs)]
    return score[:, keep]


def _merge_tiny_clusters(labels: np.ndarray, score: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return labels

    labels = labels.copy()
    while True:
        K = int(labels.max()) + 1
        sizes = np.bincount(labels, minlength=K)
        small = np.where(sizes < min_size)[0]
        if small.size == 0 or K == 1:
            break

        k_small = int(small[0])
        centers = np.zeros((K, score.shape[1]), dtype=float)
        for k in range(K):
            centers[k, :] = np.mean(score[labels == k, :], axis=0)

        d = np.sum((centers - centers[k_small, :]) ** 2, axis=1)
        d[k_small] = np.inf
        k_merge = int(np.argmin(d))
        labels[labels == k_small] = k_merge

        unique = np.unique(labels)
        remap = {old: new for new, old in enumerate(unique.tolist())}
        labels = np.asarray([remap[int(x)] for x in labels], dtype=int)

    return labels


def _signature_groups(P_constraints_cells: sparse.csr_matrix) -> list[list[int]]:
    P_csc = P_constraints_cells.tocsc()
    groups: dict[tuple[int, ...], list[int]] = {}
    for c in range(P_csc.shape[1]):
        start, end = P_csc.indptr[c], P_csc.indptr[c + 1]
        key = tuple(P_csc.indices[start:end].tolist())
        groups.setdefault(key, []).append(c)
    return list(groups.values())
