from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import default_config, ensure_run_dir, merge_config
from .exports import cm_export_synaptic_interaction_table, cm_export_type_gene_probabilities
from .loaders import cm_load_inputs, cm_load_inputs_pg, smoke_subsample
from .models import CmResult, PrepData, RawData
from .postprocess import cm_build_type_gene_probabilities
from .preprocess import cm_preprocess_binary, cm_preprocess_pg
from .solver import cm_solve
from .utils import save_struct_mat, write_manifest
from .validate import cm_validate
from .viz import (
    cm_viz_connectome_fit,
    cm_viz_constraint_diagnostics,
    cm_viz_identifiability,
    cm_viz_metacell_diagnostics,
    cm_viz_metacell_heatmap,
    cm_viz_type_heatmap,
    cm_viz_umap_four_panels,
)


def cm_run(pg_run_dir: str | None = None, cfg_overrides: dict[str, Any] | None = None) -> Path:
    input_mode = "binary"
    if pg_run_dir:
        if isinstance(pg_run_dir, str) and pg_run_dir.lower() == "sct":
            input_mode = "sct"
        elif Path(pg_run_dir).is_dir():
            input_mode = "pg"
        else:
            input_mode = "binary"

    cfg = default_config(input_mode=input_mode)
    cfg = merge_config(cfg, cfg_overrides)

    if input_mode == "pg":
        pg_dir = Path(pg_run_dir).resolve()
        cfg["pg_run_dir"] = str(pg_dir)
        cfg["paths"]["preMN_counts_pg"] = str(pg_dir / "counts_cg_corrected_pg.txt")
        cfg["paths"]["MN_counts_pg"] = str(pg_dir / "matched_gene_expression_cg_corrected_pg.txt")
        cfg["paths"]["MN_clusters_pg"] = str(pg_dir / "matched_clusters_pg.csv")
        cfg["paths"]["MN_umap_pg"] = str(pg_dir / "matched_umap_pg.csv")

    run_prefix = "run_pg" if input_mode == "pg" else "run"
    run_dir = ensure_run_dir(cfg, run_tag_prefix=run_prefix)

    if input_mode == "sct":
        raise NotImplementedError(
            "Python port currently supports binary and PG modes. SCT mode remains MATLAB-only due R/sctransform dependency."
        )

    if input_mode == "pg":
        _assert_pg_inputs(cfg)
        raw = cm_load_inputs_pg(cfg)
        if cfg["smoke_test"].get("enabled", False):
            raw = smoke_subsample(raw, cfg)
        cm_validate(raw, cfg, "raw")

        prep = cm_preprocess_pg(raw, cfg)
        cm_validate(prep, cfg, "prep_pg")
    else:
        raw = cm_load_inputs(cfg)
        if cfg["smoke_test"].get("enabled", False):
            raw = smoke_subsample(raw, cfg)
        cm_validate(raw, cfg, "raw")

        prep = cm_preprocess_binary(raw, cfg)
        cm_validate(prep, cfg, "prep_pg")

    cm = cm_solve(prep, cfg)
    cm_validate(cm, cfg, "cm")

    cm.meta["G_type_prob"] = cm.P @ prep.G_metacell_p

    if cfg.get("compute_type_gene_probabilities", True):
        cm_build_type_gene_probabilities(raw, prep, cm, cfg)

    if cfg.get("export_type_gene_probabilities", True):
        try:
            cm_export_type_gene_probabilities(raw, cm, cfg)
        except Exception as exc:
            print(f"Warning: cm_export_type_gene_probabilities failed: {exc}")

    if cfg.get("export_synaptic_interaction_table", True):
        try:
            cm_export_synaptic_interaction_table(raw, prep, cm, cfg)
        except Exception as exc:
            print(f"Warning: cm_export_synaptic_interaction_table failed: {exc}")

    if input_mode == "pg":
        save_struct_mat(run_dir / "prep_pg.mat", {"prep": _to_dict(prep)})
        save_struct_mat(run_dir / "cm_pg.mat", {"cm": _to_dict(cm)})
    else:
        save_struct_mat(run_dir / "prep.mat", {"prep": _to_dict(prep)})
        save_struct_mat(run_dir / "cm.mat", {"cm": _to_dict(cm)})

    print("=== Viz: metacell heatmap & UMAP panels ===")
    cm_viz_constraint_diagnostics(raw, cfg)
    cm_viz_metacell_heatmap(prep, cfg)
    cm_viz_umap_four_panels(raw, prep, cm, cfg)
    cm_viz_metacell_diagnostics(raw, prep, cm, cfg)
    cm_viz_connectome_fit(prep, cm, cfg)
    cm_viz_identifiability(raw, prep, cm, cfg)
    cm_viz_type_heatmap(raw, prep, cm, cfg)

    manifest = _build_manifest(cfg, raw, prep)
    write_manifest(run_dir, manifest)

    print(f"=== Pipeline complete: {run_dir} ===")
    return run_dir


def cm_run_pg(pg_run_dir: str, cfg_overrides: dict[str, Any] | None = None) -> Path:
    if not pg_run_dir:
        raise ValueError("You must provide the PG run directory from cm_batch_pg_preprocessing.")
    if not Path(pg_run_dir).is_dir():
        raise FileNotFoundError(f"PG run directory not found: {pg_run_dir}")
    return cm_run(pg_run_dir=pg_run_dir, cfg_overrides=cfg_overrides)


def _assert_pg_inputs(cfg: dict[str, Any]) -> None:
    required = [
        cfg["paths"]["preMN_counts_pg"],
        cfg["paths"]["MN_counts_pg"],
        cfg["paths"]["MN_clusters_pg"],
        cfg["paths"]["MN_umap_pg"],
    ]
    for p in required:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Expected PG file not found: {p}")


def _build_manifest(cfg: dict[str, Any], raw: RawData, prep: PrepData) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().strftime("%Y%m%dT%H%M%S"),
        "paths": cfg["paths"],
        "cfg": cfg,
        "meta": {
            "Ncells": raw.meta["Ncells"],
            "Ncells_preMN": raw.meta["Ncells_preMN"],
            "Ncells_MN": raw.meta["Ncells_MN"],
            "Ntypes_preMN": raw.meta["Ntypes_preMN"],
            "Ntypes_MN": raw.meta["Ntypes_MN"],
            "Ntypes": raw.meta["Ntypes"],
            "Ng_shared": raw.meta["Ng_shared"],
            "N_metacells": prep.meta["N_metacells"],
            "Ng_solver": prep.meta["Ng_solver"],
            "timepoint_filter": cfg["timepoint_filter"],
        },
    }


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_dict(v) for v in obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="ConnectionMiner Python pipeline")
    parser.add_argument("--mode", choices=["binary", "pg", "sct"], default="binary")
    parser.add_argument("--pg-run-dir", default="", help="Path to PG preprocessing output directory")
    parser.add_argument("--smoke", action="store_true", help="Enable smoke test subsampling")
    parser.add_argument("--max-cells", type=int, default=500)
    parser.add_argument("--max-genes", type=int, default=500)
    parser.add_argument("--num-iter", type=int, default=None)
    parser.add_argument("--no-type-export", action="store_true")
    parser.add_argument("--no-syn-export", action="store_true")
    args = parser.parse_args()

    overrides: dict[str, Any] = {
        "smoke_test": {
            "enabled": args.smoke,
            "max_cells": args.max_cells,
            "max_genes": args.max_genes,
        },
        "export_type_gene_probabilities": not args.no_type_export,
        "export_synaptic_interaction_table": not args.no_syn_export,
    }
    if args.num_iter is not None:
        overrides["solver"] = {"num_iter": args.num_iter}

    if args.mode == "pg":
        cm_run_pg(args.pg_run_dir, overrides)
    elif args.mode == "sct":
        cm_run("sct", overrides)
    else:
        cm_run(None, overrides)


if __name__ == "__main__":
    main()
