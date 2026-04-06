from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import default_config, ensure_run_dir, merge_config
from .exports import cm_export_synaptic_interaction_table
from .loaders import cm_load_inputs
from .postprocess import cm_build_type_gene_probabilities
from .preprocess import cm_preprocess_binary
from .solver import cm_solve
from .validate import cm_validate


EXPECTED_VARS = [
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


def test_export_one_iter() -> None:
    cfg = default_config(input_mode="binary")
    cfg = merge_config(
        cfg,
        {
            "compute_type_gene_probabilities": True,
            "export_type_gene_probabilities": False,
            "export_synaptic_interaction_table": True,
            "solver": {
                "num_iter": 1,
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
            },
            "metacell": {
                "target_size": 10,
                "min_size": 5,
                "n_pcs": 50,
                "kmeans_reps": 2,
                "kmeans_maxiter": 50,
                "min_samples_prior": 10,
            },
            "smoke_test": {"enabled": False},
        },
    )

    run_dir = ensure_run_dir(cfg, run_tag_prefix="test_export")
    print(f"=== Test run (1 iter) => {run_dir} ===")

    raw = cm_load_inputs(cfg)
    cm_validate(raw, cfg, "raw")

    prep = cm_preprocess_binary(raw, cfg)
    cm_validate(prep, cfg, "prep_pg")

    cm = cm_solve(prep, cfg)
    cm_validate(cm, cfg, "cm")

    cm.meta["G_type_prob"] = cm.P @ prep.G_metacell_p
    if cfg["compute_type_gene_probabilities"]:
        cm_build_type_gene_probabilities(raw, prep, cm, cfg)

    cm_export_synaptic_interaction_table(raw, prep, cm, cfg)

    out_path = Path(cfg["run_dir"]) / "synaptic_interaction_table.xlsx"
    if not out_path.exists():
        raise FileNotFoundError(f"Expected file not created: {out_path}")

    T = pd.read_excel(out_path)
    print(f"  Table: {len(T)} rows, {len(T.columns)} columns")

    missing = sorted(set(EXPECTED_VARS).difference(set(T.columns.tolist())))
    if missing:
        raise AssertionError("Missing columns: " + ", ".join(missing))

    if len(T) > 1_048_575:
        raise AssertionError(f"Table has {len(T)} rows (Excel max 1_048_576).")

    print("  OK: synaptic_interaction_table.xlsx valid and within Excel row limit.")


if __name__ == "__main__":
    test_export_one_iter()
