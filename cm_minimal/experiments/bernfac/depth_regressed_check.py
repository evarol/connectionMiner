"""Does the sequencing-depth axis explain why BernFac P (R3) and binary Y (R2) decode
hemilineage worse than processed counts (R1)?

For each cached embedding E (R1/R2/R3) we regress every column on log total UMI
(linear, with intercept) and recompute the main biology / batch metrics with the
same functions umap_compare.py uses. Output: umap/metrics_depth_regressed.csv
"""
import numpy as np
import pandas as pd

import umap_compare as uc


def main():
    X, cells, _ = uc.load_data()
    logumi = np.log10(np.asarray(X.sum(1)).ravel())
    rows = []
    for name in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]:
        E = np.load(uc.CACHE / f"{name}_embedding.npz")["E"].astype(np.float64)
        D = np.column_stack([np.ones_like(logumi), logumi])
        E_res = E - D @ np.linalg.lstsq(D, E, rcond=None)[0]
        for tag, M in [("as is", E), ("depth regressed", E_res)]:
            m, _ = uc.metrics_for(M, cells, logumi, f"{name} {tag}")
            rows.append(dict(rep=name, variant=tag, **{k: m[k] for k in
                        ["hl_knn_balacc", "hl_logreg_balacc", "hl_silhouette_cos", "hl_leiden_best_ARI",
                         "hl_nn_purity", "lib_knn_balacc", "lib_mix_rel", "depth_max_abs_top2"]}))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(uc.OUT / "metrics_depth_regressed.csv", index=False)


if __name__ == "__main__":
    main()
