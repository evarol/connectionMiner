from __future__ import annotations

from pathlib import Path


def cm_get_paths() -> dict:
    """Default paths for ConnectionMiner data and repo layout."""
    repo_root = Path(__file__).resolve().parent.parent
    repo_name = repo_root.name
    if repo_name == "connectionMiner":
        data_root = Path("/Users/erdem/Dropbox/scRNAseq_DGE")
    else:
        data_root = repo_root.parent

    return {
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "preMN_counts": str(data_root / "scRNAseq PreMNs" / "counts_cg_corrected.txt"),
        "MN_counts": str(data_root / "Matrix and umap raw files" / "Merged" / "matched_gene_expression_cg_corrected.txt"),
        "preMN_clusters": str(data_root / "scRNAseq PreMNs" / "Cell_Cluster.xlsx"),
        "preMN_umap": str(data_root / "scRNAseq PreMNs" / "umapCoord_vnc.csv"),
        "MN_clusters": str(data_root / "Matrix and umap raw files" / "Merged" / "matched_clusters.xlsx"),
        "MN_umap": str(data_root / "Matrix and umap raw files" / "Merged" / "matched_umap_coordinates_time_specific.xlsx"),
        "MN_covariates": str(data_root / "Matrix and umap raw files" / "MNs_detailed_info_matrix format_with_developmental_age.xlsx"),
        "preMN_MN_connections": str(data_root / "scRNAseq PreMNs" / "PreMNs-MNs connection_20250107.xlsx"),
    }
