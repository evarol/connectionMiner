from __future__ import annotations

from typing import Any

import numpy as np

from .models import CmResult, PrepData, RawData


def cm_validate(obj: Any, cfg: dict[str, Any], stage: str) -> None:
    if stage == "raw":
        _assert_fields(obj, ["G_cells", "P_constraints_cells", "C_counts", "C_mask", "umap_xy", "meta"])

        if obj.G_cells.shape[0] != obj.P_constraints_cells.shape[1]:
            raise AssertionError(
                f"Cell count mismatch: G_cells has {obj.G_cells.shape[0]} rows, constraints has {obj.P_constraints_cells.shape[1]} cols."
            )

        if not cfg.get("smoke_test", {}).get("enabled", False):
            if obj.meta.get("Ntypes_preMN") != 701:
                raise AssertionError(f"Expected 701 preMN types, got {obj.meta.get('Ntypes_preMN')}.")
            if obj.meta.get("Ntypes_MN") != 29:
                raise AssertionError(f"Expected 29 MN types, got {obj.meta.get('Ntypes_MN')}.")
            if obj.meta.get("Ntypes") != 730:
                raise AssertionError(f"Expected 730 total types, got {obj.meta.get('Ntypes')}.")

        if obj.C_counts.shape[0] != obj.meta["Ntypes"] or obj.C_counts.shape[1] != obj.meta["Ntypes"]:
            raise AssertionError("Connectome size mismatch with Ntypes.")
        if obj.C_counts.shape != obj.C_mask.shape:
            raise AssertionError("C_counts and C_mask size mismatch.")
        if obj.umap_xy.shape != (obj.meta["Ncells"], 2):
            raise AssertionError("UMAP dimensions mismatch.")

        if np.isnan(obj.G_cells).any() or np.isinf(obj.G_cells).any():
            raise AssertionError("NaN/Inf in G_cells.")
        if np.isnan(obj.C_counts).any():
            raise AssertionError("NaNs in C_counts.")
        if obj.P_constraints_cells.nnz == 0:
            raise AssertionError("No constrained cells.")

    elif stage == "prep":
        _assert_fields(
            obj,
            [
                "G_metacell_p",
                "P_constraints_metacell",
                "cell_to_metacell",
                "meta_sizes",
            ],
        )

        K = obj.meta_sizes.size
        if obj.P_constraints_metacell.shape[1] != K:
            raise AssertionError("Metacell constraint column mismatch.")
        if obj.G_metacell_p.shape[0] != K:
            raise AssertionError("G_metacell_p row mismatch.")
        if np.any(obj.meta_sizes <= 0):
            raise AssertionError("Zero-size metacells found.")
        if np.isnan(obj.G_metacell_p).any() or np.isinf(obj.G_metacell_p).any():
            raise AssertionError("NaN/Inf in G_metacell_p.")

        if obj.solver_gene_idx.size == 0:
            raise AssertionError("solver_gene_idx is missing or empty.")
        if obj.G_metacell_p_solve.shape[0] != K:
            raise AssertionError("G_metacell_p_solve row mismatch.")

    elif stage == "prep_pg":
        _assert_fields(
            obj,
            [
                "G_metacell_p",
                "P_constraints_metacell",
                "cell_to_metacell",
                "meta_sizes",
                "solver_gene_idx",
                "G_metacell_p_solve",
            ],
        )

        K = obj.meta_sizes.size
        if obj.P_constraints_metacell.shape[1] != K:
            raise AssertionError("Metacell constraint column mismatch.")
        if obj.G_metacell_p.shape[0] != K:
            raise AssertionError("G_metacell_p row mismatch.")

    elif stage == "cm":
        _assert_fields(obj, ["P", "beta", "C", "C_mask", "loss"])

        if np.isnan(obj.P).any() or np.isinf(obj.P).any():
            raise AssertionError("NaN/Inf in P.")
        if np.isnan(obj.beta).any() or np.isinf(obj.beta).any():
            raise AssertionError("NaN/Inf in beta.")
        if obj.C.shape != obj.C_mask.shape:
            raise AssertionError("C and mask size mismatch.")
        if np.any(obj.loss < 0):
            raise AssertionError("Negative loss values found.")
    else:
        raise ValueError(f"Unknown validation stage: {stage}")

    print(f"  Validation [{stage}]: PASSED")


def _assert_fields(obj: Any, fields: list[str]) -> None:
    for field in fields:
        if not hasattr(obj, field):
            raise AssertionError(f"Missing required field: {field}")
