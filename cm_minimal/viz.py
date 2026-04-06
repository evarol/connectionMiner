from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import CmResult, PrepData, RawData


def cm_viz_constraint_diagnostics(raw: RawData, cfg: dict[str, Any]) -> None:
    viz_dir = _viz_dir(cfg)

    n_excl = int(raw.meta.get("n_cells_excluded_unassigned", 0))
    n_empty = int(raw.meta.get("n_types_with_no_cells", 0))
    type_idx = np.asarray(raw.meta.get("type_idx_with_no_cells", np.array([], dtype=int)), dtype=int)

    lines = [
        "Constraint diagnostics",
        f"  Cells excluded from metacell clustering (unassigned): {n_excl}",
        f"  Types with no cells (excluded from connectome loss): {n_empty}",
        "",
    ]

    if n_empty > 0 and "all_names" in raw.meta and type_idx.size > 0:
        names = raw.meta["all_names"][type_idx].astype(str)
        lines.append("  Type names (excluded from loss):")
        for name in names[:50]:
            lines.append(f"    {name}")
        if names.size > 50:
            lines.append(f"    ... and {names.size - 50} more")
        lines.append("")

    out_txt = viz_dir / "constraint_diagnostics.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved {out_txt}")

    P = raw.P_constraints_cells.toarray()
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    im = ax.imshow(P, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Cells (unassigned cells excluded from pipeline)")
    ax.set_ylabel("Types")
    ax.set_title("P_constraints (types x cells); zero rows = types excluded from connectome loss")
    out_png = viz_dir / "constraint_P_constraints.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved {out_png}")


def cm_viz_metacell_heatmap(prep: PrepData, cfg: dict[str, Any]) -> None:
    viz_dir = _viz_dir(cfg)

    n_hvg = int(cfg.get("viz", {}).get("n_hvg", 150))

    R = prep.G_cells[:, prep.solver_gene_idx].astype(float)
    ng = R.shape[1]
    n_hvg = min(n_hvg, ng)
    if n_hvg < 1:
        return

    v = np.var(R, axis=0)
    idx_hvg = np.argsort(-v)[:n_hvg]

    K = prep.meta["N_metacells"]
    meta_sizes = prep.meta_sizes.astype(float)
    M = np.zeros((prep.cell_to_metacell.size, K), dtype=float)
    M[np.arange(prep.cell_to_metacell.size), prep.cell_to_metacell] = 1.0

    before = prep.G_metacell_p[:, idx_hvg]
    after = (M.T @ (R[:, idx_hvg] > 0).astype(float)) / np.maximum(meta_sizes[:, None], 1.0)

    ord_rows = np.arange(K)
    try:
        from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
        from scipy.spatial.distance import pdist

        D = pdist(after, metric="euclidean")
        if D.size > 0:
            tree = linkage(D, method="average")
            tree = optimal_leaf_ordering(tree, D)
            ord_rows = leaves_list(tree)
    except Exception:
        pass

    before_ord = before[ord_rows, :]
    after_ord = after[ord_rows, :]

    ord_by_size = np.argsort(meta_sizes)
    before_by_size = before[ord_by_size, :]
    after_by_size = after[ord_by_size, :]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=140)
    im1 = axes[0].imshow(before_ord, aspect="auto", interpolation="nearest")
    axes[0].set_title("Before (metacell detection rate)")
    axes[0].set_xlabel("HVG")
    axes[0].set_ylabel("Metacell")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(after_ord, aspect="auto", interpolation="nearest")
    axes[1].set_title("After (feature > 0 fraction)")
    axes[1].set_xlabel("HVG")
    axes[1].set_ylabel("Metacell")
    fig.colorbar(im2, ax=axes[1])

    out_png = viz_dir / "metacell_HVG_heatmap_before_after.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved {out_png}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=140)
    im1 = axes[0].imshow(before_by_size, aspect="auto", interpolation="nearest")
    axes[0].set_title("Before; metacells ordered by size, small at bottom")
    axes[0].set_xlabel("HVG")
    axes[0].set_ylabel("Metacell")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(after_by_size, aspect="auto", interpolation="nearest")
    axes[1].set_title("After; metacells ordered by size, small at bottom")
    axes[1].set_xlabel("HVG")
    axes[1].set_ylabel("Metacell")
    fig.colorbar(im2, ax=axes[1])

    out_png = viz_dir / "metacell_HVG_heatmap_by_size.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved {out_png}")


def cm_viz_umap_four_panels(raw: RawData, prep: PrepData, cm: CmResult, cfg: dict[str, Any]) -> None:
    viz_dir = _viz_dir(cfg)
    seed = int(cfg.get("seed", 1))

    umap = raw.umap_xy
    n_cells = umap.shape[0]

    raw_cluster = raw.raw_cluster_id.astype(float).copy()
    finite_vals = raw_cluster[np.isfinite(raw_cluster)]
    fill_val = (np.max(finite_vals) + 1) if finite_vals.size else 0
    raw_cluster[~np.isfinite(raw_cluster)] = fill_val

    P = raw.P_constraints_cells.tocsc()
    sig_keys = []
    for c in range(P.shape[1]):
        s, e = P.indptr[c], P.indptr[c + 1]
        sig_keys.append(tuple(P.indices[s:e].tolist()))
    _, constraint_id = np.unique(np.asarray(sig_keys, dtype=object), return_inverse=True)

    meta_id = prep.cell_to_metacell.astype(int)

    cell_type = np.zeros(n_cells, dtype=int)
    for c in range(n_cells):
        k = prep.cell_to_metacell[c]
        cell_type[c] = int(np.argmax(cm.P[:, k]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=140)

    cmap_raw = _rand_colormap(max(int(np.max(raw_cluster)) + 1, 2), seed)
    cmap_constraint = _rand_colormap(max(int(np.max(constraint_id)) + 2, 2), seed + 1)
    cmap_meta = _rand_colormap(max(int(np.max(meta_id)) + 2, 2), seed + 2)
    cmap_type = _rand_colormap(max(int(np.max(cell_type)) + 2, 2), seed + 3)

    axes[0, 0].scatter(umap[:, 0], umap[:, 1], s=5, c=raw_cluster, cmap=cmap_raw)
    axes[0, 0].set_title("Raw clusters")
    axes[0, 0].set_xlabel("UMAP1")
    axes[0, 0].set_ylabel("UMAP2")

    axes[0, 1].scatter(umap[:, 0], umap[:, 1], s=5, c=constraint_id, cmap=cmap_constraint)
    axes[0, 1].set_title("Constraint sets")
    axes[0, 1].set_xlabel("UMAP1")
    axes[0, 1].set_ylabel("UMAP2")

    axes[1, 0].scatter(umap[:, 0], umap[:, 1], s=5, c=meta_id, cmap=cmap_meta)
    axes[1, 0].set_title("Metacells")
    axes[1, 0].set_xlabel("UMAP1")
    axes[1, 0].set_ylabel("UMAP2")

    axes[1, 1].scatter(umap[:, 0], umap[:, 1], s=5, c=cell_type, cmap=cmap_type)
    axes[1, 1].set_title("Inferred type (1 of 730)")
    axes[1, 1].set_xlabel("UMAP1")
    axes[1, 1].set_ylabel("UMAP2")

    out_png = viz_dir / "umap_four_panels.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved {out_png}")


def cm_viz_metacell_diagnostics(raw: RawData, prep: PrepData, cm: CmResult, cfg: dict[str, Any]) -> None:
    viz_dir = _viz_dir(cfg)

    Npre = int(raw.meta["Ncells_preMN"])
    Ncells = prep.cell_to_metacell.size
    sizes = prep.meta_sizes.astype(float)
    K = sizes.size

    n_below_10 = int(np.sum(sizes < 10))
    n_below_25 = int(np.sum(sizes < 25))
    n_in_10_25 = int(np.sum((sizes >= 10) & (sizes <= 25)))

    preMN_only = np.zeros(K, dtype=bool)
    MN_only = np.zeros(K, dtype=bool)
    mixed = np.zeros(K, dtype=bool)
    n_pre_in_k = np.zeros(K, dtype=int)
    n_mn_in_k = np.zeros(K, dtype=int)

    for k in range(K):
        idx = prep.cell_to_metacell == k
        n_pre_in_k[k] = int(np.sum(idx[:Npre]))
        n_mn_in_k[k] = int(np.sum(idx[Npre:Ncells]))
        if n_mn_in_k[k] == 0:
            preMN_only[k] = True
        elif n_pre_in_k[k] == 0:
            MN_only[k] = True
        else:
            mixed[k] = True

    n_meta_premn_only = int(np.sum(preMN_only))
    n_meta_mn_only = int(np.sum(MN_only))
    n_meta_mixed = int(np.sum(mixed))

    cell_type = np.zeros(Ncells, dtype=int)
    for c in range(Ncells):
        k = prep.cell_to_metacell[c]
        cell_type[c] = int(np.argmax(cm.P[:, k]))
    n_types_preMN = np.unique(cell_type[:Npre]).size
    n_types_MN = np.unique(cell_type[Npre:Ncells]).size

    n_excl_cells = int(raw.meta.get("n_cells_excluded_unassigned", 0))
    n_empty_types = int(raw.meta.get("n_types_with_no_cells", 0))

    mean_detection = np.mean(prep.G_metacell_p, axis=1)
    corr = np.corrcoef(sizes, mean_detection)[0, 1] if K > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0

    size_thresh = 10
    n_small = int(np.sum(sizes < size_thresh))
    mean_det_small = float(np.mean(mean_detection[sizes < size_thresh])) if np.any(sizes < size_thresh) else 0.0
    mean_det_rest = float(np.mean(mean_detection[sizes >= size_thresh])) if np.any(sizes >= size_thresh) else 0.0

    summary = (
        "Metacell diagnostics\n"
        f"  Cells excluded from metacell clustering (unassigned): {n_excl_cells}\n"
        f"  Types with no cells (excluded from connectome loss): {n_empty_types}\n"
        f"  Total metacells: {K}\n"
        f"  Size: min={int(np.min(sizes))}, max={int(np.max(sizes))}, median={int(np.median(sizes))}, mean={float(np.mean(sizes)):.1f}\n"
        f"  Below 10 cells: {n_below_10} ({100.0 * n_below_10 / max(K, 1):.1f}%)\n"
        f"  Below 25 cells: {n_below_25} ({100.0 * n_below_25 / max(K, 1):.1f}%)\n"
        f"  In [10,25]: {n_in_10_25}\n"
        f"  preMN-only metacells: {n_meta_premn_only}, MN-only: {n_meta_mn_only}, mixed: {n_meta_mixed}\n"
        f"  Metacells containing any MN cell: {n_meta_mn_only + n_meta_mixed} (of these, MN-only={n_meta_mn_only}, mixed={n_meta_mixed})\n"
        f"  Unique inferred types for preMN cells: {n_types_preMN} (of 730)\n"
        f"  Unique inferred types for MN cells: {n_types_MN} (of 730)\n"
        f"  Mean detection rate vs size: corr(size, mean_detection)={corr:.4f}\n"
        f"  Metacells with size < {size_thresh}: {n_small}; their mean detection={mean_det_small:.4f} vs rest={mean_det_rest:.4f}\n"
    )

    print(f"\n=== Metacell diagnostics ===\n{summary}")
    (viz_dir / "metacell_diagnostics.txt").write_text(summary, encoding="utf-8")
    print(f"  Saved {viz_dir / 'metacell_diagnostics.txt'}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=140)
    axes[0].hist(sizes, bins=50)
    yl = axes[0].get_ylim()
    axes[0].plot([10, 10], yl, "r--", linewidth=1.5)
    axes[0].plot([25, 25], yl, "m--", linewidth=1.5)
    axes[0].set_xlabel("Metacell size")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Metacell sizes (n={K}); below 10: {n_below_10}, below 25: {n_below_25}")

    axes[1].bar([0, 1, 2], [n_meta_premn_only, n_meta_mn_only, n_meta_mixed])
    axes[1].set_xticks([0, 1, 2], ["preMN only", "MN only", "mixed"])
    axes[1].set_ylabel("Number of metacells")
    axes[1].set_title("Metacells by batch content")

    out_png = viz_dir / "metacell_size_and_batch.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved {out_png}")

    lib = np.sum(raw.G_cells, axis=1) + 1.0
    log10_lib = np.log10(lib)
    mean_depth_per_meta = np.zeros(K, dtype=float)
    for k in range(K):
        idx = prep.cell_to_metacell == k
        mean_depth_per_meta[k] = np.mean(log10_lib[idx])

    batch_color = np.where(preMN_only, 1, np.where(MN_only, 2, 3))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=140)
    cmap = np.array([[0.2, 0.5, 0.8], [0.9, 0.3, 0.2], [0.5, 0.5, 0.5]])

    axes[0].scatter(sizes, mean_depth_per_meta, s=20, c=cmap[batch_color - 1])
    axes[0].set_xlabel("Metacell size")
    axes[0].set_ylabel("Mean log10(UMI+1)")
    axes[0].set_title("Metacell size vs mean depth (color: preMN/MN/mixed)")

    axes[1].scatter(sizes, mean_detection, s=20, c=cmap[batch_color - 1])
    axes[1].set_xlabel("Metacell size")
    axes[1].set_ylabel("Mean detection rate")
    axes[1].set_title(f"Size vs mean detection (r={corr:.3f})")

    out_png = viz_dir / "metacell_size_vs_depth.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved {out_png}")

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=140)
    ax.scatter(sizes, mean_detection, s=20, c=cmap[batch_color - 1])
    ax.set_xlabel("Metacell size")
    ax.set_ylabel("Mean detection rate")
    ax.set_title(f"Size vs mean detection (r={corr:.3f}); size < 10 may show artifact")
    out_png = viz_dir / "metacell_size_vs_mean_detection.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved {out_png}")


def cm_viz_connectome_fit(prep: PrepData, cm: CmResult, cfg: dict[str, Any]) -> None:
    viz_dir = _viz_dir(cfg)

    C = cm.C
    W = cm.C_mask
    C_hat = cm.P @ cm.G_proj @ cm.beta @ cm.G_proj.T @ cm.P.T

    idx = np.where(W > 0)
    c_obs = C[idx]
    c_pred = C_hat[idx]
    resid = c_obs - c_pred

    n_masked = c_obs.size
    if n_masked == 0:
        print("  Connectome fit: no masked entries, skipping.")
        return

    corr = np.corrcoef(c_obs, c_pred)[0, 1] if n_masked > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((c_obs - np.mean(c_obs)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, np.finfo(float).eps)
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    summary = (
        f"Connectome fit (masked entries: {n_masked})\n"
        f"  Correlation(obs, pred): {corr:.4f}\n"
        f"  R^2: {r2:.4f}\n"
        f"  RMSE: {rmse:.4f}\n"
    )
    print(f"\n=== Connectome fit ===\n{summary}")
    (viz_dir / "connectome_fit_summary.txt").write_text(summary, encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=140)
    axes[0].scatter(c_obs, c_pred, s=10, color="b")
    mx = float(np.max(np.concatenate([c_obs, c_pred])))
    axes[0].plot([0, mx], [0, mx], "r--", linewidth=1)
    axes[0].set_xlabel("Observed C")
    axes[0].set_ylabel("Predicted C_hat")
    axes[0].set_title(f"Masked entries (n={n_masked}); r={corr:.3f}, R2={r2:.3f}")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].hist(resid, bins=50)
    axes[1].set_xlabel("Residual (C - C_hat)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Residuals; RMSE={rmse:.4f}")

    out_png = viz_dir / "connectome_fit.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved {out_png}")

    c_display = C.copy()
    c_display[W == 0] = np.nan
    c_hat_display = C_hat.copy()
    c_hat_display[W == 0] = np.nan

    finite = np.concatenate([c_display[np.isfinite(c_display)], c_hat_display[np.isfinite(c_hat_display)]])
    vmax = float(np.max(finite)) if finite.size else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=140)
    im = axes[0].imshow(c_display, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[0].set_title("Target C (fitted only; types with no cells excluded from loss)")
    fig.colorbar(im, ax=axes[0])

    im = axes[1].imshow(c_hat_display, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[1].set_title("Predicted C_hat (fitted only; types with no cells excluded from loss)")
    fig.colorbar(im, ax=axes[1])

    out_png = viz_dir / "connectome_matrices.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved {out_png}")

    if cm.loss.size:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=140)
        ax.plot(np.arange(1, cm.loss.size + 1), cm.loss, "b-o", markersize=4)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Solver convergence")
        ax.grid(True)
        out_png = viz_dir / "solver_loss.png"
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
        print(f"  Saved {out_png}")


def cm_viz_identifiability(raw: RawData, prep: PrepData, cm: CmResult, cfg: dict[str, Any]) -> None:
    viz_dir = _viz_dir(cfg)

    Ntypes = cm.P.shape[0]
    type_names = raw.meta.get("all_names", np.array([], dtype=str)).astype(str)

    if "cell_contributions" in cm.meta and "identifiable_type" in cm.meta:
        cell_contrib = np.asarray(cm.meta["cell_contributions"], dtype=float)
        identifiable = np.asarray(cm.meta["identifiable_type"], dtype=bool)
    else:
        threshold = int(cfg.get("viz", {}).get("min_cells_identifiable", 5))
        n_cells = np.zeros(Ntypes, dtype=float)
        for c in range(prep.cell_to_metacell.size):
            k = prep.cell_to_metacell[c]
            t = int(np.argmax(cm.P[:, k]))
            n_cells[t] += 1
        cell_contrib = n_cells
        identifiable = n_cells >= threshold

    idx_id = np.where(identifiable)[0]
    idx_not = np.where(~identifiable)[0]

    out_txt = viz_dir / "identifiability_summary.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("Identifiability (non-zero type mass)\n")
        f.write(f"  Identifiable types: {idx_id.size} (of {Ntypes})\n")
        f.write(f"  Non-identifiable types: {idx_not.size}\n\n")
        f.write("Identifiable (type_index, cell_contributions, name):\n")
        for t in idx_id[:200]:
            nm = type_names[t] if t < type_names.size else ""
            f.write(f"  {t + 1}\t{cell_contrib[t]:.4f}\t{nm}\n")
        if idx_id.size > 200:
            f.write(f"  ... and {idx_id.size - 200} more\n")

        f.write("\nNon-identifiable (type_index, cell_contributions, name):\n")
        for t in idx_not[:200]:
            nm = type_names[t] if t < type_names.size else ""
            f.write(f"  {t + 1}\t{cell_contrib[t]:.4f}\t{nm}\n")
        if idx_not.size > 200:
            f.write(f"  ... and {idx_not.size - 200} more\n")

    print(
        f"  Identifiable: {idx_id.size} types (mass>0), non-identifiable: {idx_not.size}. Saved {out_txt}"
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=140)
    x = np.arange(1, Ntypes + 1)
    if Ntypes <= 300:
        colors = np.where(identifiable, "#33b266", "#e64d33")
        ax.bar(x, cell_contrib, color=colors, width=1.0)
    else:
        ax.plot(x[~identifiable], cell_contrib[~identifiable], ".", color="#e64d33", markersize=2)
        ax.plot(x[identifiable], cell_contrib[identifiable], ".", color="#33b266", markersize=2)

    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.set_xlabel("Type index")
    ax.set_ylabel("Cell contributions")
    ax.set_title(
        f"Identifiable (green, mass>0) vs not (red); {idx_id.size} identifiable, {idx_not.size} not"
    )
    ax.set_xlim(0.5, Ntypes + 0.5)

    out_png = viz_dir / "identifiability.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved {out_png}")


def cm_viz_type_heatmap(raw: RawData, prep: PrepData, cm: CmResult, cfg: dict[str, Any]) -> None:
    viz_dir = _viz_dir(cfg)
    k_stair = int(cfg.get("viz", {}).get("staircaser_k", 8))

    Ntypes = cm.P.shape[0]
    if "identifiable_type" in cm.meta:
        identifiable = np.asarray(cm.meta["identifiable_type"], dtype=bool)
    else:
        threshold = int(cfg.get("viz", {}).get("min_cells_identifiable", 5))
        n_cells = np.zeros(Ntypes, dtype=int)
        for c in range(prep.cell_to_metacell.size):
            k = prep.cell_to_metacell[c]
            t = int(np.argmax(cm.P[:, k]))
            n_cells[t] += 1
        identifiable = n_cells >= threshold

    idx_id = np.where(identifiable)[0]
    if idx_id.size == 0:
        print("  Type heatmap: no identifiable types, skipping.")
        return

    G = prep.G_metacell_p_solve
    T = cm.P @ G
    T_sub = T[idx_id, :]

    gene_idx, _ = cm_staircaser_genes(T_sub, k_stair)
    if gene_idx.size == 0:
        print("  Type heatmap: no genes selected by staircaser, skipping.")
        return

    H = T[idx_id[:, None], gene_idx]
    genes_use = prep.genes_solver[gene_idx].astype(str)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=140)
    im = ax.imshow(H, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Type")
    ax.set_title(f"Identifiable types x staircaser genes (mass>0, k={k_stair})")

    ax.set_xticks(np.arange(gene_idx.size))
    ax.set_xticklabels(genes_use, rotation=90, fontsize=6)

    type_names = raw.meta.get("all_names", np.array([], dtype=str)).astype(str)
    if type_names.size >= Ntypes:
        ax.set_yticks(np.arange(idx_id.size))
        ax.set_yticklabels(type_names[idx_id], fontsize=6)

    out_png = viz_dir / "type_heatmap_identifiable.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved {out_png}")


def cm_staircaser_genes(T: np.ndarray, k: int) -> tuple[np.ndarray, list[np.ndarray]]:
    Ntypes, Ng = T.shape
    if Ng == 0:
        return np.array([], dtype=int), [np.array([], dtype=int) for _ in range(Ntypes)]

    ord_col = np.argsort(T, axis=0)
    clust = ord_col[-1, :]
    max1 = np.max(T, axis=0)

    T2 = T.copy()
    T2[clust, np.arange(Ng)] = -np.inf
    max2 = np.max(T2, axis=0)
    gap = max1 - max2
    gap[np.isnan(gap)] = 0.0

    slots_left = [t for t in range(Ntypes) for _ in range(k)]
    gene_for_clusters: list[list[int]] = [[] for _ in range(Ntypes)]

    for g in np.argsort(-gap):
        t = int(clust[g])
        if t in slots_left and gap[g] > 0:
            slots_left.remove(t)
            gene_for_clusters[t].append(int(g))
        if not slots_left:
            break

    gene_idx: list[int] = []
    for t in range(Ntypes):
        gene_idx.extend(gene_for_clusters[t])

    if gene_idx:
        _, unique_idx = np.unique(np.asarray(gene_idx, dtype=int), return_index=True)
        gene_idx_arr = np.asarray(gene_idx, dtype=int)[np.sort(unique_idx)]
    else:
        gene_idx_arr = np.array([], dtype=int)

    return gene_idx_arr, [np.asarray(v, dtype=int) for v in gene_for_clusters]


def _viz_dir(cfg: dict[str, Any]) -> Path:
    run_dir = Path(cfg["run_dir"])
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    return viz_dir


def _rand_colormap(n: int, seed: int):
    rng = np.random.default_rng(seed)
    base = plt.cm.hsv(np.linspace(0, 1, n, endpoint=False))
    from matplotlib.colors import ListedColormap

    return ListedColormap(base[rng.permutation(n)])
