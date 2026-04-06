from __future__ import annotations

from .config import default_config, merge_config
from .loaders import cm_load_inputs, smoke_subsample
from .preprocess import cm_preprocess_binary
from .validate import cm_validate


def test_binary_smoke() -> None:
    cfg = default_config(input_mode="binary")
    cfg = merge_config(
        cfg,
        {
            "smoke_test": {"enabled": True, "max_cells": 80, "max_genes": 200},
        },
    )

    raw = cm_load_inputs(cfg)
    raw = smoke_subsample(raw, cfg)
    cm_validate(raw, cfg, "raw")

    prep = cm_preprocess_binary(raw, cfg)
    cm_validate(prep, cfg, "prep_pg")

    print(
        f"Binary preprocess: N_metacells={prep.meta['N_metacells']}, Ng_solver={prep.meta['Ng_solver']}"
    )
    print(
        f"all_lineage present: {int('all_lineage' in raw.meta)}, isMN_type present: {int('isMN_type' in raw.meta)}"
    )
    print("Done.")


if __name__ == "__main__":
    test_binary_smoke()
