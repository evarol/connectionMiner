from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse


@dataclass
class RawData:
    G_cells: np.ndarray
    genes_shared: np.ndarray
    P_constraints_cells: sparse.csr_matrix
    C_counts: np.ndarray
    C_mask: np.ndarray
    umap_xy: np.ndarray
    raw_cluster_id: np.ndarray
    meta: dict[str, Any]


@dataclass
class PrepData:
    G_cells: np.ndarray
    genes_shared: np.ndarray
    P_constraints_cells: sparse.csr_matrix
    C_counts: np.ndarray
    C_mask: np.ndarray
    umap_xy: np.ndarray
    meta: dict[str, Any]
    solver_gene_idx: np.ndarray
    genes_solver: np.ndarray
    cell_to_metacell: np.ndarray
    meta_sizes: np.ndarray
    signature_table: Any
    P_constraints_metacell: np.ndarray
    G_metacell_p: np.ndarray
    G_metacell_p_solve: np.ndarray


@dataclass
class CmResult:
    P: np.ndarray
    beta: np.ndarray
    G_proj: np.ndarray
    loss: np.ndarray
    obj_beta: np.ndarray
    obj_P_fit: np.ndarray
    obj_P_ent: np.ndarray
    P_constraints: np.ndarray
    C: np.ndarray
    C_mask: np.ndarray
    C_recon: np.ndarray
    elapsed_sec: float
    Ng_solve: int
    Ng_eff: int
    is_low_rank: bool
    meta: dict[str, Any]
