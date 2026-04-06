from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .paths import cm_get_paths


def default_config(input_mode: str = "binary") -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "seed": 750,
        "timepoint_filter": 3,
        "input_mode": input_mode,
        "paths": cm_get_paths(),
        "load": {
            "read_size": 1000,
            "numeric_class": "float32",
            "allow_reorder": True,
        },
        "sct": {
            "clip": 10,
            "n_genes_use": 4000,
            "min_cells": 5,
            "verbosity": 1,
            "method": "poisson",
            "gene_select": "variance",
            "future_max_gb": 8,
            "regress_batch": True,
            "rscript": "/Library/Frameworks/R.framework/Resources/bin/Rscript",
        },
        "pg": {
            "n_genes_use": 4000,
            "min_cells": 5,
        },
        "binary": {
            "n_genes_use": 4000,
            "min_cells": 5,
        },
        "metacell": {
            "target_size": 10,
            "min_size": 5,
            "min_samples_prior": 10,
            "n_pcs": 50,
            "kmeans_reps": 5,
            "kmeans_maxiter": 200,
        },
        "compute_type_gene_probabilities": True,
        "export_type_gene_probabilities": True,
        "export_synaptic_interaction_table": True,
        "solver": {
            "num_iter": 100,
            "lambda_sparsity": 0.001,
            "optimal_transport_epsilon": 1e-12,
            "optimal_transport_step": 0.04,
            "optimal_transport_iterations": 50,
            "regression_iterations": 50,
            "use_binary_connectome": True,
            "beta_rank": 0,
            "interactome_constraint": "none",
            "use_complement": True,
            "P_init": "random_proportional",
            "beta_init": "random",
            "time_limit_per_step": 30,
        },
        "smoke_test": {
            "enabled": False,
            "max_cells": 500,
            "max_genes": 500,
        },
        "viz": {
            "min_cells_identifiable": 5,
            "staircaser_k": 8,
            "n_hvg": 150,
        },
    }
    return cfg


def merge_config(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-merge override into base and return a new config."""
    if override is None:
        return base

    out = copy.deepcopy(base)

    def _merge(a: dict[str, Any], b: dict[str, Any]) -> None:
        for key, value in b.items():
            if isinstance(value, dict) and isinstance(a.get(key), dict):
                _merge(a[key], value)
            else:
                a[key] = value

    _merge(out, override)
    return out


def ensure_run_dir(cfg: dict[str, Any], run_tag_prefix: str = "run") -> Path:
    repo_root = Path(cfg["paths"]["repo_root"])
    run_root = repo_root / "cm_minimal" / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / f"{run_tag_prefix}_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["run_dir"] = str(run_dir)
    return run_dir
