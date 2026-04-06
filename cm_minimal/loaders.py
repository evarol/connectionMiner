from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg, sparse

from .models import RawData
from .utils import cell_to_num, find_column_ci, regex_last_int, robust_zscore


def cm_load_inputs(cfg: dict[str, Any]) -> RawData:
    """Load and align ConnectionMiner raw inputs (binary/SCT path)."""
    print("=== LOAD ===")
    preMN_genes, preMN_cells, preMN_expr = _read_expression_csv(
        cfg["paths"]["preMN_counts"], has_header=True, skip_cols=1
    )
    print(f"  preMN expression: {preMN_expr.shape[0]} genes x {preMN_expr.shape[1]} cells")

    mn_label_cols, _, mn_expr = _read_expression_csv_multi_label(cfg["paths"]["MN_counts"])
    mn_genes = _pick_best_gene_column(mn_label_cols, preMN_genes)
    print(f"  MN expression: {mn_expr.shape[0]} genes x {mn_expr.shape[1]} cells")

    preMN_cluster_ids, preMN_cluster_labels = _read_cell_cluster(cfg["paths"]["preMN_clusters"])
    preMN_umap_ids, preMN_umap_xy = _read_umap(cfg["paths"]["preMN_umap"])

    mn_clusters_mat = _read_numeric_matrix(cfg["paths"]["MN_clusters"])
    mn_umap_xy = _read_numeric_matrix(cfg["paths"]["MN_umap"])

    mn_cov_header, mn_cov_data = _read_xlsx_cells(cfg["paths"]["MN_covariates"])
    conn_header, conn_data = _read_xlsx_cells(cfg["paths"]["preMN_MN_connections"])

    print("=== ALIGN ===")
    preMN_cluster_labels, preMN_umap_xy = _align_premn(
        preMN_cells,
        preMN_cluster_ids,
        preMN_cluster_labels,
        preMN_umap_ids,
        preMN_umap_xy,
    )

    if mn_expr.shape[1] != mn_clusters_mat.shape[0]:
        raise ValueError(
            f"MN expression columns ({mn_expr.shape[1]}) do not match matched_clusters rows ({mn_clusters_mat.shape[0]})."
        )
    if mn_expr.shape[1] != mn_umap_xy.shape[0]:
        raise ValueError(
            f"MN expression columns ({mn_expr.shape[1]}) do not match MN UMAP rows ({mn_umap_xy.shape[0]})."
        )

    idx_keep = mn_clusters_mat[:, 0] == cfg["timepoint_filter"]
    mn_expr = mn_expr[:, idx_keep]
    mn_clusters_mat = mn_clusters_mat[idx_keep, :]
    mn_umap_xy = mn_umap_xy[idx_keep, :]
    print(
        f"  MN timepoint filter (Var1=={cfg['timepoint_filter']}): {idx_keep.size} -> {int(idx_keep.sum())} cells"
    )

    return _build_raw_struct(
        preMN_genes=preMN_genes,
        preMN_expr=preMN_expr,
        preMN_cluster_labels=preMN_cluster_labels,
        preMN_umap_xy=preMN_umap_xy,
        mn_genes=mn_genes,
        mn_expr=mn_expr,
        mn_clusters_mat=mn_clusters_mat,
        mn_umap_xy=mn_umap_xy,
        mn_cov_header=mn_cov_header,
        mn_cov_data=mn_cov_data,
        conn_header=conn_header,
        conn_data=conn_data,
    )


def cm_load_inputs_pg(cfg: dict[str, Any]) -> RawData:
    """Load and align PG-corrected inputs."""
    print("=== LOAD (PG) ===")
    preMN_genes, preMN_cells, preMN_expr = _read_expression_csv(
        cfg["paths"]["preMN_counts_pg"], has_header=True, skip_cols=1
    )
    print(f"  preMN expression (PG): {preMN_expr.shape[0]} genes x {preMN_expr.shape[1]} cells")

    mn_label_cols, _, mn_expr = _read_expression_csv_multi_label(cfg["paths"]["MN_counts_pg"])
    mn_genes = _pick_best_gene_column(mn_label_cols, preMN_genes)
    print(f"  MN expression (PG): {mn_expr.shape[0]} genes x {mn_expr.shape[1]} cells")

    preMN_cluster_ids, preMN_cluster_labels = _read_cell_cluster(cfg["paths"]["preMN_clusters"])
    preMN_umap_ids, preMN_umap_xy = _read_umap(cfg["paths"]["preMN_umap"])

    mn_clusters_mat = _read_numeric_matrix(cfg["paths"]["MN_clusters_pg"])
    mn_umap_xy = _read_numeric_matrix(cfg["paths"]["MN_umap_pg"])

    mn_cov_header, mn_cov_data = _read_xlsx_cells(cfg["paths"]["MN_covariates"])
    conn_header, conn_data = _read_xlsx_cells(cfg["paths"]["preMN_MN_connections"])

    print("=== ALIGN (PG) ===")
    preMN_cluster_labels, preMN_umap_xy = _align_premn(
        preMN_cells,
        preMN_cluster_ids,
        preMN_cluster_labels,
        preMN_umap_ids,
        preMN_umap_xy,
    )

    if mn_expr.shape[1] != mn_clusters_mat.shape[0]:
        raise ValueError(
            f"MN expression columns ({mn_expr.shape[1]}) do not match matched_clusters_pg rows ({mn_clusters_mat.shape[0]})."
        )
    if mn_expr.shape[1] != mn_umap_xy.shape[0]:
        raise ValueError(
            f"MN expression columns ({mn_expr.shape[1]}) do not match matched_umap_pg rows ({mn_umap_xy.shape[0]})."
        )

    return _build_raw_struct(
        preMN_genes=preMN_genes,
        preMN_expr=preMN_expr,
        preMN_cluster_labels=preMN_cluster_labels,
        preMN_umap_xy=preMN_umap_xy,
        mn_genes=mn_genes,
        mn_expr=mn_expr,
        mn_clusters_mat=mn_clusters_mat,
        mn_umap_xy=mn_umap_xy,
        mn_cov_header=mn_cov_header,
        mn_cov_data=mn_cov_data,
        conn_header=conn_header,
        conn_data=conn_data,
    )


def _build_raw_struct(
    *,
    preMN_genes: np.ndarray,
    preMN_expr: np.ndarray,
    preMN_cluster_labels: np.ndarray,
    preMN_umap_xy: np.ndarray,
    mn_genes: np.ndarray,
    mn_expr: np.ndarray,
    mn_clusters_mat: np.ndarray,
    mn_umap_xy: np.ndarray,
    mn_cov_header: list[str],
    mn_cov_data: np.ndarray,
    conn_header: list[str],
    conn_data: np.ndarray,
) -> RawData:
    print("=== GENE INTERSECT ===")
    genes_shared, idx_pre, idx_mn = _stable_intersect(preMN_genes, mn_genes)
    print(
        f"  Ng_preMN={preMN_genes.size}, Ng_MN={mn_genes.size}, Ng_shared={genes_shared.size}"
    )
    if genes_shared.size == 0:
        raise ValueError("No shared genes between preMN and MN.")

    preMN_expr = preMN_expr[idx_pre, :].T
    mn_expr = mn_expr[idx_mn, :].T

    print("=== CONSTRAINTS ===")
    pre_constraints, pre_names = _premn_constraints(conn_header, conn_data, preMN_cluster_labels)
    print(f"  preMN constraints: {pre_constraints.shape[0]} types x {pre_constraints.shape[1]} cells")

    mn_constraints, mn_names, mn_cov_mat_out = _mn_constraints(
        mn_cov_header,
        mn_cov_data,
        mn_clusters_mat,
    )
    print(f"  MN constraints: {mn_constraints.shape[0]} types x {mn_constraints.shape[1]} cells")

    unassigned_mn_cells = np.where(np.all(mn_constraints == 0, axis=0))[0]
    unassigned_mn_types = np.where(np.all(mn_constraints == 0, axis=1))[0]
    if unassigned_mn_cells.size and unassigned_mn_types.size:
        mn_constraints[np.ix_(unassigned_mn_types, unassigned_mn_cells)] = 1
        print(
            f"  MN legacy fill: linked {unassigned_mn_cells.size} unassigned cells to {unassigned_mn_types.size} unassigned types"
        )

    all_constraints = linalg.block_diag(pre_constraints, mn_constraints)
    n_cells_orig = all_constraints.shape[1]

    keep_cells = np.where(np.any(all_constraints > 0, axis=0))[0]
    all_constraints = all_constraints[:, keep_cells]
    n_cells_excluded = n_cells_orig - keep_cells.size
    if n_cells_excluded > 0:
        print(f"  Excluded {n_cells_excluded} cells with no type assignment from pipeline")

    unassigned_types = np.where(np.all(all_constraints == 0, axis=1))[0]
    n_types_no_cells = unassigned_types.size

    print("=== CONNECTOME ===")
    C_counts, C_mask = _build_connectome(
        conn_header,
        conn_data,
        mn_cov_header,
        mn_cov_data,
        pre_names,
        mn_names,
    )

    if n_types_no_cells > 0:
        C_mask[unassigned_types, :] = 0
        C_mask[:, unassigned_types] = 0
        print(f"  {n_types_no_cells} types with no cells: excluded from connectome loss")

    all_umap = np.vstack(
        [
            robust_zscore(preMN_umap_xy),
            robust_zscore(mn_umap_xy) + np.array([6.0, 0.0]),
        ]
    )
    all_G = np.vstack([preMN_expr, mn_expr])
    all_cluster_id = np.concatenate([preMN_cluster_labels.astype(float), mn_clusters_mat[:, 1].astype(float)])

    n_cells_pre_orig = preMN_expr.shape[0]
    is_mn_cell_full = np.concatenate(
        [np.zeros(n_cells_pre_orig, dtype=bool), np.ones(mn_expr.shape[0], dtype=bool)]
    )

    raw = RawData(
        G_cells=all_G[keep_cells, :],
        genes_shared=genes_shared,
        P_constraints_cells=sparse.csr_matrix(all_constraints),
        C_counts=C_counts,
        C_mask=C_mask,
        umap_xy=all_umap[keep_cells, :],
        raw_cluster_id=all_cluster_id[keep_cells],
        meta={},
    )

    raw.meta["Ncells_preMN"] = int(np.sum(keep_cells < n_cells_pre_orig))
    raw.meta["Ncells_MN"] = int(keep_cells.size - raw.meta["Ncells_preMN"])
    raw.meta["Ncells"] = int(keep_cells.size)
    raw.meta["Ntypes_preMN"] = int(pre_constraints.shape[0])
    raw.meta["Ntypes_MN"] = int(mn_constraints.shape[0])
    raw.meta["Ntypes"] = int(raw.meta["Ntypes_preMN"] + raw.meta["Ntypes_MN"])
    raw.meta["Ng_shared"] = int(genes_shared.size)
    raw.meta["MN_covariates_mat"] = mn_cov_mat_out
    raw.meta["preMN_names"] = pre_names
    raw.meta["MN_names"] = mn_names
    raw.meta["all_names"] = np.concatenate([pre_names, mn_names])
    raw.meta["isMN_type"] = np.concatenate(
        [np.zeros(pre_names.size, dtype=bool), np.ones(mn_names.size, dtype=bool)]
    )

    all_lineage, all_motor_pool = _lineage_motor_pool_from_conn(conn_header, conn_data, pre_names, mn_names.size)
    raw.meta["all_lineage"] = all_lineage
    raw.meta["all_motor_pool"] = all_motor_pool

    raw.meta["isMN_cell"] = is_mn_cell_full[keep_cells]
    raw.meta["n_cells_excluded_unassigned"] = int(n_cells_excluded)
    raw.meta["n_types_with_no_cells"] = int(n_types_no_cells)
    raw.meta["type_idx_with_no_cells"] = unassigned_types.astype(int)

    print(
        f"  Combined: {raw.meta['Ncells']} cells, {raw.meta['Ng_shared']} genes, {raw.meta['Ntypes']} types"
    )

    return raw


def smoke_subsample(raw: RawData, cfg: dict[str, Any]) -> RawData:
    rng = np.random.default_rng(cfg["seed"])
    orig_preMN = raw.meta["Ncells_preMN"]

    idx_genes = np.arange(raw.meta["Ng_shared"])
    if raw.meta["Ng_shared"] > cfg["smoke_test"]["max_genes"]:
        idx_genes = np.sort(rng.choice(raw.meta["Ng_shared"], size=cfg["smoke_test"]["max_genes"], replace=False))

    raw.G_cells = raw.G_cells[:, idx_genes]
    raw.genes_shared = raw.genes_shared[idx_genes]
    raw.meta["Ng_shared"] = int(idx_genes.size)

    idx_cells = np.arange(raw.meta["Ncells"])
    if raw.meta["Ncells"] > cfg["smoke_test"]["max_cells"]:
        idx_cells = np.sort(rng.choice(raw.meta["Ncells"], size=cfg["smoke_test"]["max_cells"], replace=False))

    raw.G_cells = raw.G_cells[idx_cells, :]
    raw.P_constraints_cells = raw.P_constraints_cells[:, idx_cells]
    raw.umap_xy = raw.umap_xy[idx_cells, :]
    raw.raw_cluster_id = raw.raw_cluster_id[idx_cells]

    raw.meta["Ncells"] = int(idx_cells.size)
    raw.meta["Ncells_preMN"] = int(np.sum(idx_cells < orig_preMN))
    raw.meta["Ncells_MN"] = int(raw.meta["Ncells"] - raw.meta["Ncells_preMN"])

    if "isMN_cell" in raw.meta:
        raw.meta["isMN_cell"] = raw.meta["isMN_cell"][idx_cells]

    return raw


def _read_expression_csv(path: str, has_header: bool, skip_cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if has_header:
        df = pd.read_csv(path, low_memory=False)
        genes = df.iloc[:, 0].astype(str).to_numpy()
        cells = df.columns[skip_cols:].astype(str).to_numpy()
        X = df.iloc[:, skip_cols:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    else:
        df = pd.read_csv(path, header=None, low_memory=False)
        genes = df.iloc[:, 0].astype(str).to_numpy()
        cells = np.array([], dtype=str)
        X = df.iloc[:, skip_cols:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return genes, cells, X


def _read_expression_csv_multi_label(path: str) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().rstrip("\n")
    fields = first_line.split(",")

    n_label_cols = 0
    for field in fields:
        try:
            float(field)
            break
        except ValueError:
            n_label_cols += 1

    if n_label_cols == 0:
        raise ValueError("MN expression: no leading string columns detected.")

    df = pd.read_csv(path, header=None, low_memory=False)
    label_cols = [df.iloc[:, j].astype(str).to_numpy() for j in range(n_label_cols)]
    X = df.iloc[:, n_label_cols:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return label_cols, np.array([], dtype=str), X


def _pick_best_gene_column(label_cols: list[np.ndarray], ref_genes: np.ndarray) -> np.ndarray:
    ref_set = set(ref_genes.tolist())
    best_overlap = -1
    best_idx = 0
    for j, col in enumerate(label_cols):
        ov = sum(g in ref_set for g in col)
        print(f"  MN label col {j + 1}: {ov} / {len(col)} genes overlap with preMN")
        if ov > best_overlap:
            best_overlap = ov
            best_idx = j
    if best_overlap <= 0:
        raise ValueError("No MN gene column overlaps with preMN genes.")
    print(f"  -> Using MN label column {best_idx + 1} ({best_overlap} overlapping genes)")
    return label_cols[best_idx].astype(str)


def _read_cell_cluster(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(path, header=None, dtype=object)
    df = df.dropna(how="all")
    if df.shape[1] < 2:
        raise ValueError(f"Cell cluster file has fewer than 2 columns: {path}")
    maybe_header = df.iloc[0, 1]
    if isinstance(maybe_header, str):
        try:
            float(maybe_header)
        except ValueError:
            df = df.iloc[1:, :]
    cluster_ids = df.iloc[:, 0].astype(str).to_numpy()
    cluster_labels = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    return cluster_ids, cluster_labels


def _read_umap(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None, dtype=object)
    df = df.dropna(how="all")
    if df.shape[1] < 3:
        raise ValueError(f"UMAP file has fewer than 3 columns: {path}")
    maybe_header = df.iloc[0, 1]
    if isinstance(maybe_header, str):
        try:
            float(maybe_header)
        except ValueError:
            df = df.iloc[1:, :]
    umap_ids = df.iloc[:, 0].astype(str).to_numpy()
    umap_xy = df.iloc[:, 1:3].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return umap_ids, umap_xy


def _read_numeric_matrix(path: str) -> np.ndarray:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, header=None, dtype=object)
    else:
        df = pd.read_excel(path, header=None, dtype=object)
    num = df.apply(pd.to_numeric, errors="coerce")
    num = num.dropna(how="all")
    return num.to_numpy(dtype=float)


def _read_xlsx_cells(path: str) -> tuple[list[str], np.ndarray]:
    df = pd.read_excel(path, header=None, dtype=object)
    df = df.dropna(how="all")
    if df.empty:
        return [], np.empty((0, 0), dtype=object)
    header = ["" if pd.isna(v) else str(v) for v in df.iloc[0, :].tolist()]
    data = df.iloc[1:, :].to_numpy(dtype=object)
    return header, data


def _align_premn(
    cell_ids: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_labels: np.ndarray,
    umap_ids: np.ndarray,
    umap_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cluster_map = {str(k): v for k, v in zip(cluster_ids, cluster_labels)}
    out_cluster = np.empty(cell_ids.shape[0], dtype=float)
    missing = []
    for i, cid in enumerate(cell_ids):
        key = str(cid)
        if key not in cluster_map:
            missing.append(key)
            out_cluster[i] = np.nan
        else:
            out_cluster[i] = cluster_map[key]
    if missing:
        raise ValueError(f"PreMN alignment: {len(missing)} expression cells not found in Cell_Cluster.xlsx.")
    print(f"  preMN cluster alignment: {cell_ids.size} / {cell_ids.size} matched")

    umap_map = {str(k): i for i, k in enumerate(umap_ids)}
    out_umap = np.empty((cell_ids.size, 2), dtype=float)
    missing_umap = []
    for i, cid in enumerate(cell_ids):
        key = str(cid)
        idx = umap_map.get(key)
        if idx is None:
            missing_umap.append(key)
            out_umap[i, :] = np.nan
        else:
            out_umap[i, :] = umap_xy[idx, :]
    if missing_umap:
        raise ValueError(f"PreMN alignment: {len(missing_umap)} expression cells not found in umapCoord_vnc.csv.")
    print(f"  preMN umap alignment: {cell_ids.size} / {cell_ids.size} matched")
    return out_cluster, out_umap


def _stable_intersect(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b_map: dict[str, int] = {}
    for j, g in enumerate(b.astype(str)):
        if g not in b_map:
            b_map[g] = j

    out_genes = []
    idx_a = []
    idx_b = []
    for i, g in enumerate(a.astype(str)):
        j = b_map.get(g)
        if j is not None:
            out_genes.append(g)
            idx_a.append(i)
            idx_b.append(j)

    return (
        np.asarray(out_genes, dtype=str),
        np.asarray(idx_a, dtype=int),
        np.asarray(idx_b, dtype=int),
    )


def _premn_constraints(conn_header: list[str], conn_data: np.ndarray, cluster_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tc_indices = []
    tc_nums = []
    for i, h in enumerate(conn_header):
        h_low = str(h).lower()
        if "transcriptional" in h_low and "cluster" in h_low:
            n = regex_last_int(str(h))
            if n is None:
                raise ValueError(f"Cannot parse cluster number from: {h}")
            tc_indices.append(i)
            tc_nums.append(n)

    if not tc_indices:
        raise ValueError('No "Transcriptional cluster" columns found in connection table.')

    order = np.argsort(np.asarray(tc_nums))
    tc_indices = [tc_indices[i] for i in order]
    tc_nums_sorted = np.asarray([tc_nums[i] for i in order], dtype=int)
    print(
        f"  Detected {len(tc_indices)} transcriptional cluster columns (clusters {tc_nums_sorted.min()}-{tc_nums_sorted.max()})"
    )

    V = cell_to_num(conn_data[:, tc_indices])
    V = np.nan_to_num(V, nan=0.0)

    labels = cluster_labels.astype(float)
    U = np.zeros((labels.size, tc_nums_sorted.size), dtype=float)
    tc_lookup = {int(tc): j for j, tc in enumerate(tc_nums_sorted.tolist())}
    for i, label in enumerate(labels):
        if np.isnan(label):
            continue
        col = tc_lookup.get(int(label))
        if col is not None:
            U[i, col] = 1.0

    pre_constraints = (V @ U.T > 0).astype(float)

    name_col = find_column_ci(conn_header, "Name")
    if name_col is None:
        raise ValueError("Name column not found in connection table.")
    pre_names = np.asarray([str(x) for x in conn_data[:, name_col]], dtype=str)

    return pre_constraints, pre_names


def _mn_constraints(
    header: list[str],
    data: np.ndarray,
    mn_clusters_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names_raw = data[:, 0]
    valid_rows = []
    for i, v in enumerate(names_raw):
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        s = str(v).strip()
        if s != "":
            valid_rows.append(i)

    data = data[valid_rows, :]
    mn_names = np.asarray([str(v) for v in data[:, 0]], dtype=str)
    print(f"  MN covariates: {mn_names.size} named types")

    cluster_col = find_column_ci(header, "Adult MNs clusters (MN3)")
    if cluster_col is None:
        raise ValueError('"Adult MNs clusters (MN3)" column not found.')

    raw_labels = data[:, cluster_col]
    mn_cluster_labels = np.full(raw_labels.shape[0], np.nan, dtype=float)
    for i, v in enumerate(raw_labels):
        if v is None:
            continue
        try:
            mn_cluster_labels[i] = float(v)
        except Exception:
            try:
                mn_cluster_labels[i] = float(str(v))
            except Exception:
                mn_cluster_labels[i] = np.nan

    mn_cov_mat = cell_to_num(data[:, 2:])
    mn_cov_mat = np.nan_to_num(mn_cov_mat, nan=0.0)

    cell_clusters = mn_clusters_mat[:, 1]
    mn_constraints = np.zeros((mn_cluster_labels.size, cell_clusters.size), dtype=float)
    for i, lbl in enumerate(mn_cluster_labels):
        if not np.isnan(lbl):
            mn_constraints[i, :] = cell_clusters == lbl

    return mn_constraints, mn_names, mn_cov_mat


def _build_connectome(
    conn_header: list[str],
    conn_data: np.ndarray,
    mn_header: list[str],
    mn_data: np.ndarray,
    premn_names: np.ndarray,
    mn_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_premn = premn_names.size
    n_mn = mn_names.size

    mn_header_map = {str(h): i for i, h in enumerate(mn_header)}

    shared_cols = []
    idx_mn = []
    for i, h in enumerate(conn_header):
        h_str = str(h)
        h_low = h_str.lower()
        if "transcriptional" in h_low and "cluster" in h_low:
            continue
        j = mn_header_map.get(h_str)
        if j is not None:
            shared_cols.append(i)
            idx_mn.append(j)

    if not shared_cols:
        print("Warning: no shared columns between connection table and MN covariates for connectome mapping.")
        z = np.zeros((n_premn + n_mn, n_premn + n_mn), dtype=float)
        return z.copy(), z.copy()

    premn_mn_raw = cell_to_num(conn_data[:, shared_cols])
    premn_mn_raw = np.nan_to_num(premn_mn_raw, nan=0.0)

    mn_cmap = cell_to_num(mn_data[:n_mn, idx_mn])
    mn_cmap = np.nan_to_num(mn_cmap, nan=0.0)

    col_sums = mn_cmap.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    mn_cmap_norm = mn_cmap / col_sums
    premn_mn_full = premn_mn_raw @ mn_cmap_norm.T

    unmeasured_mn = np.all(mn_cmap == 0, axis=1)
    premn_mn_mask = np.ones_like(premn_mn_full)
    premn_mn_mask[:, unmeasured_mn] = 0.0
    print(
        f"  preMN-MN connectome: {premn_mn_full.shape[0]}x{premn_mn_full.shape[1]}, {int(unmeasured_mn.sum())} unmeasured MN types"
    )

    premn_name_to_col = {name: i for i, name in enumerate(premn_names.tolist())}
    premn_premn = np.zeros((n_premn, n_premn), dtype=float)

    n_matched_cols = 0
    for col_idx, h in enumerate(conn_header):
        target = premn_name_to_col.get(str(h))
        if target is None:
            continue
        col_data = cell_to_num(conn_data[:, col_idx:col_idx + 1]).ravel()
        col_data = np.nan_to_num(col_data, nan=0.0)
        premn_premn[:, target] = col_data
        n_matched_cols += 1

    premn_premn_mask = np.ones_like(premn_premn)
    print(
        f"  preMN-preMN connectome: {premn_premn.shape[0]}x{premn_premn.shape[1]}, {n_matched_cols} matched columns of {n_premn} preMN types"
    )

    zeros_mn_mn = np.zeros((n_mn, n_mn), dtype=float)
    zeros_mn_premn = np.zeros((n_mn, n_premn), dtype=float)

    C_counts = np.block(
        [
            [premn_premn, premn_mn_full],
            [zeros_mn_premn, zeros_mn_mn],
        ]
    )
    C_mask = np.block(
        [
            [premn_premn_mask, premn_mn_mask],
            [np.zeros_like(zeros_mn_premn), zeros_mn_mn],
        ]
    )

    print(f"  Full connectome: {C_counts.shape[0]}x{C_counts.shape[1]}")
    return C_counts, C_mask


def _lineage_motor_pool_from_conn(
    conn_header: list[str],
    conn_data: np.ndarray,
    premn_names: np.ndarray,
    n_mn: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = premn_names.size + n_mn
    lineage = np.full(total, "NA", dtype=object)
    motor_pool = np.full(total, "NA", dtype=object)

    hl_col = find_column_ci(conn_header, "Hemilineage")
    mp_col = find_column_ci(conn_header, "MotorPool")

    for i in range(premn_names.size):
        if hl_col is not None and hl_col < conn_data.shape[1]:
            v = conn_data[i, hl_col]
            if v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip() != "":
                lineage[i] = str(v)
        if mp_col is not None and mp_col < conn_data.shape[1]:
            v = conn_data[i, mp_col]
            if v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip() != "":
                motor_pool[i] = str(v)

    return lineage, motor_pool
