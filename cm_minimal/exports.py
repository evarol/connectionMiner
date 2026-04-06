from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind

from .models import CmResult, PrepData, RawData


SYNAPTIC_COLUMNS = [
    "SynapseName",
    "preSynapseName",
    "postSynapseName",
    "preSynapseType",
    "postSynapseType",
    "preSynapseLineage",
    "postSynapseLineage",
    "preSynapseMotorPool",
    "postSynapseMotorPool",
    "interactionName",
    "preInteractionName",
    "postInteractionName",
    "synapseStrength",
    "geneCoExp",
    "preGeneExp",
    "postGeneExp",
    "interactionScore",
    "effectSize",
    "preEffectSize",
    "postEffectSize",
    "pValue",
    "prePvalue",
    "postPvalue",
]


def cm_export_type_gene_probabilities(raw: RawData, cm: CmResult, cfg: dict[str, Any]) -> None:
    run_dir = cfg.get("run_dir")
    if not run_dir:
        print("Warning: cfg.run_dir is empty, skipping type_gene_probabilities export.")
        return

    G_type_prob_full = cm.meta.get("G_type_prob_full")
    if G_type_prob_full is None:
        print("Warning: cm.G_type_prob_full missing, nothing to export.")
        return

    output_path = Path(run_dir) / "type_gene_probabilities.xlsx"

    Ntypes = G_type_prob_full.shape[0]
    genes = raw.genes_shared.astype(str)

    type_names = raw.meta.get("all_names", np.array([], dtype=str)).astype(str)
    if type_names.size < Ntypes:
        type_names = np.pad(type_names, (0, Ntypes - type_names.size), constant_values="")

    n_cells = cm.meta.get("n_cells_type", np.full(Ntypes, np.nan))
    cell_contributions = cm.meta.get("cell_contributions", np.full(Ntypes, np.nan))
    identifiable = cm.meta.get("identifiable_type", np.zeros(Ntypes, dtype=bool)).astype(bool)

    export_mat = G_type_prob_full.copy()
    export_mat[~identifiable, :] = np.nan

    base_df = pd.DataFrame(
        {
            "type_name": type_names[:Ntypes],
            "n_cells": np.asarray(n_cells)[:Ntypes],
            "cell_contributions": np.asarray(cell_contributions)[:Ntypes],
            "identifiable": identifiable[:Ntypes].astype(int),
        }
    )

    gene_df = pd.DataFrame(
        export_mat,
        columns=[f"{gene}_prob" for gene in genes.tolist()],
    )
    df = pd.concat([base_df, gene_df], axis=1)

    df.to_excel(output_path, index=False)
    print(f"  Exported type x gene probabilities to {output_path}")


def cm_export_synaptic_interaction_table(
    raw: RawData,
    prep: PrepData,
    cm: CmResult,
    cfg: dict[str, Any],
) -> None:
    run_dir = Path(cfg["run_dir"])

    ng_solver = prep.meta["Ng_solver"]
    G = cm.meta.get("G_type_prob")
    if G is None:
        G = cm.P @ prep.G_metacell_p
    G = np.asarray(G, dtype=float)[:, :ng_solver]

    gene_names_full = prep.genes_solver.astype(str)
    all_names = raw.meta["all_names"].astype(str)
    is_mn_name = raw.meta["isMN_type"].astype(bool)
    all_lineage = raw.meta["all_lineage"].astype(str)
    all_motor_pool = raw.meta["all_motor_pool"].astype(str)

    W = ((prep.C_mask > 0) & (~np.isnan(prep.C_counts))).astype(float)
    C = (prep.C_counts > 0).astype(float)
    C[W == 0] = 0.0
    C_cont = prep.C_counts.astype(float).copy()
    C_cont[W == 0] = 0.0

    gene_list_dir = Path(cfg["paths"]["data_root"]) / "Genes list"
    interactome_path = gene_list_dir / "Interactome_v3.xlsx"
    if not interactome_path.exists():
        interactome_path = gene_list_dir / "Interactome_v2.xlsx"
    if not interactome_path.exists():
        print("Warning: Interactome file not found; skipping synaptic table export.")
        return

    T = pd.read_excel(interactome_path)
    p1_col, p2_col, adhesive_col, could_col = _resolve_interactome_columns(T.columns.tolist())
    if p1_col is None or p2_col is None:
        print("Warning: Interactome table missing Partner1/Partner2 columns; skipping.")
        return

    directed_pairs = _build_directed_interactome_pairs(T, p1_col, p2_col, adhesive_col, could_col)
    gene_to_idx = {g: i for i, g in enumerate(gene_names_full.tolist())}
    pair_idx = [(gene_to_idx[a], gene_to_idx[b]) for (a, b) in directed_pairs if a in gene_to_idx and b in gene_to_idx]

    if len(pair_idx) < 2:
        print("Warning: Fewer than 2 solver genes in interactome; skipping.")
        return

    pairsA = np.asarray([p[0] for p in pair_idx], dtype=int)
    pairsB = np.asarray([p[1] for p in pair_idx], dtype=int)

    effect_focal, pval = gene_combination_similarity_ordering(C, G, pairsA, pairsB)

    s, t = np.where(C > 0)
    if s.size == 0:
        out_path = run_dir / "synaptic_interaction_table.xlsx"
        pd.DataFrame(columns=SYNAPTIC_COLUMNS).to_excel(out_path, index=False)
        print(f"  Wrote {out_path} (empty: no connections)")
        return

    neuron_type = np.where(is_mn_name, "MN", "preMN")

    n_syn = s.size
    n_pairs = pairsA.size
    pair_rep = np.tile(np.arange(n_pairs), n_syn)
    s_rep = np.repeat(s, n_pairs)
    t_rep = np.repeat(t, n_pairs)

    pre_g_idx = pairsA[pair_rep]
    post_g_idx = pairsB[pair_rep]

    synapse_name = all_names[s_rep] + "->" + all_names[t_rep]
    interaction_name = gene_names_full[pre_g_idx] + "->" + gene_names_full[post_g_idx]

    GG = G[s_rep, pre_g_idx] * G[t_rep, post_g_idx]
    pre_G = G[s_rep, pre_g_idx]
    post_G = G[t_rep, post_g_idx]

    pre_E = effect_focal[s_rep, pair_rep]
    post_E = effect_focal[t_rep, pair_rep]
    E = pre_E + post_E

    pre_P = np.clip(np.nan_to_num(pval[s_rep, pair_rep], nan=0.5), 1e-10, 1 - 1e-10)
    post_P = np.clip(np.nan_to_num(pval[t_rep, pair_rep], nan=0.5), 1e-10, 1 - 1e-10)
    P_comb = norm.cdf((norm.ppf(pre_P) + norm.ppf(post_P)) / np.sqrt(2.0))

    syn_strength = C_cont[s_rep, t_rep]
    interaction_score = GG * E

    full_table = pd.DataFrame(
        {
            "SynapseName": synapse_name,
            "preSynapseName": all_names[s_rep],
            "postSynapseName": all_names[t_rep],
            "preSynapseType": neuron_type[s_rep],
            "postSynapseType": neuron_type[t_rep],
            "preSynapseLineage": all_lineage[s_rep],
            "postSynapseLineage": all_lineage[t_rep],
            "preSynapseMotorPool": all_motor_pool[s_rep],
            "postSynapseMotorPool": all_motor_pool[t_rep],
            "interactionName": interaction_name,
            "preInteractionName": gene_names_full[pre_g_idx],
            "postInteractionName": gene_names_full[post_g_idx],
            "synapseStrength": syn_strength,
            "geneCoExp": GG,
            "preGeneExp": pre_G,
            "postGeneExp": post_G,
            "interactionScore": interaction_score,
            "effectSize": E,
            "preEffectSize": pre_E,
            "postEffectSize": post_E,
            "pValue": P_comb,
            "prePvalue": pre_P,
            "postPvalue": post_P,
        }
    )

    out_table = _prune_synaptic_table(full_table, k_prune=3)

    EXCEL_MAX_ROWS = 1_048_576
    if len(out_table) > EXCEL_MAX_ROWS - 1:
        out_table = out_table.iloc[: EXCEL_MAX_ROWS - 1, :]
        print(f"Warning: synaptic table truncated to {EXCEL_MAX_ROWS - 1} rows due to Excel limit.")

    out_path = run_dir / "synaptic_interaction_table.xlsx"
    out_table.to_excel(out_path, index=False)
    print(f"  Wrote {out_path} ({len(out_table)} rows, pruned from {len(full_table)})")


def _build_directed_interactome_pairs(
    T: pd.DataFrame,
    p1_col: str,
    p2_col: str,
    adhesive_col: str | None,
    could_col: str | None,
) -> set[tuple[str, str]]:
    if adhesive_col is not None and could_col is not None:
        sel = (T[adhesive_col] == 1) | (T[could_col] == 0)
        S = T.loc[sel, [p1_col, p2_col]]
    else:
        S = T[[p1_col, p2_col]]

    pairs: set[tuple[str, str]] = set()
    for a, b in S.itertuples(index=False):
        if pd.isna(a) or pd.isna(b):
            continue
        sa = str(a)
        sb = str(b)
        pairs.add((sa, sb))
        pairs.add((sb, sa))
    return pairs


def _resolve_interactome_columns(
    columns: list[str],
) -> tuple[str | None, str | None, str | None, str | None]:
    norm = {_norm_col(c): c for c in columns}
    p1_col = norm.get("partner1")
    p2_col = norm.get("partner2")
    adhesive_col = norm.get("adhesive")
    could_col = norm.get("couldbeboth")
    return p1_col, p2_col, adhesive_col, could_col


def _norm_col(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _prune_synaptic_table(df: pd.DataFrame, k_prune: int = 3) -> pd.DataFrame:
    keep_idx = []
    for _, grp in df.groupby("SynapseName", sort=False):
        idx = set()
        idx.update(grp.nlargest(k_prune, "interactionScore").index.tolist())
        idx.update(grp.nsmallest(k_prune, "interactionScore").index.tolist())
        idx.update(grp.nsmallest(k_prune, "pValue").index.tolist())
        keep_idx.extend(sorted(idx))

    keep_idx = sorted(set(keep_idx))
    return df.loc[keep_idx, :].reset_index(drop=True)


def gene_combination_similarity_ordering(
    C: np.ndarray,
    G: np.ndarray,
    pairsA: np.ndarray,
    pairsB: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    N = C.shape[0]
    A = C > 0
    n_pairs = pairsA.size

    effect = np.full((N, n_pairs), np.nan, dtype=float)
    pval = np.full((N, n_pairs), np.nan, dtype=float)

    for i in range(N):
        non_post, syn_post = find_indirect_targets_via_shared_source(A, i)
        non_pre, syn_pre = find_indirect_targets_via_shared_source(A.T, i)

        syn_samples_1 = _samples_post(G, i, syn_post, pairsA, pairsB)
        syn_samples_2 = _samples_pre(G, i, syn_pre, pairsA, pairsB)
        non_samples_1 = _samples_post(G, i, non_post, pairsA, pairsB)
        non_samples_2 = _samples_pre(G, i, non_pre, pairsA, pairsB)

        syn = _vstack_nonempty([syn_samples_1, syn_samples_2])
        non = _vstack_nonempty([non_samples_1, non_samples_2])

        if syn.size == 0 or non.size == 0:
            continue

        try:
            stat = ttest_ind(syn, non, axis=0, equal_var=False, nan_policy="omit")
            pval[i, :] = stat.pvalue
            effect[i, :] = np.nanmean(syn, axis=0) - np.nanmean(non, axis=0)
        except Exception:
            pass

    return effect, pval


def _samples_post(
    G: np.ndarray,
    i: int,
    idx_rows: np.ndarray,
    pairsA: np.ndarray,
    pairsB: np.ndarray,
) -> np.ndarray:
    if idx_rows.size == 0:
        return np.empty((0, pairsA.size), dtype=float)
    return G[i, pairsA][None, :] * G[idx_rows[:, None], pairsB]


def _samples_pre(
    G: np.ndarray,
    i: int,
    idx_rows: np.ndarray,
    pairsA: np.ndarray,
    pairsB: np.ndarray,
) -> np.ndarray:
    if idx_rows.size == 0:
        return np.empty((0, pairsA.size), dtype=float)
    return G[idx_rows[:, None], pairsA] * G[i, pairsB][None, :]


def _vstack_nonempty(parts: list[np.ndarray]) -> np.ndarray:
    kept = [p for p in parts if p.size > 0]
    if not kept:
        return np.empty((0, 0), dtype=float)
    return np.vstack(kept)


def find_indirect_targets_via_shared_source(A: np.ndarray, v_idx: int) -> tuple[np.ndarray, np.ndarray]:
    N, M = A.shape
    if N != M:
        raise ValueError("Adjacency matrix A must be square.")
    if v_idx < 0 or v_idx >= N:
        raise ValueError("Query node index v_idx is out of bounds.")

    A_logical = A > 0

    is_Z = A_logical[v_idx, :]
    Z = np.where(is_Z)[0]
    if not np.any(is_Z):
        return np.array([], dtype=int), Z

    is_W = (A_logical.T @ is_Z.astype(int)) > 0
    if not np.any(is_W):
        return np.array([], dtype=int), Z

    is_U_candidate = (is_W.astype(int) @ A_logical.astype(int)) > 0
    is_U = is_U_candidate & (~is_Z)
    is_U[v_idx] = False

    U = np.where(is_U)[0]
    return U, Z
