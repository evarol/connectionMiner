"""Step 2: extract a 1000-cell x 100-gene count subset and binarize it.

Source: motor premn_24h (24h APF preMN, raw UMI counts, 16460 x 21050).
Design: 4 libraries (2 F, 2 M) x 5 hemilineages x 50 cells, a balanced crossed
design so that biology (hemilineage) and batch (library / sex) are orthogonal.
Genes: the top-ranked HVGs whose detection rate in the subset lies in [5%, 95%].
Binarization: Y = 1[count >= threshold]. Primary threshold = 1 (detected);
threshold = 2 ("more than one UMI") is saved as a sensitivity variant.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "outputs/connectionminer_ready_inputs/bundles/motor/premn_24h"
COUNTS = ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h/G_prime.npz"
OUT = ROOT / "outputs/bernfac_20261020/data"

LIBRARIES = ["SITTA9", "SITTB9", "SITTG8", "SITTC8"]
HEMILINEAGES = ["14A", "04B", "09A", "12B", "03A"]
CELLS_PER_CELL = 50
N_GENES = 100


def main(seed=0):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cells = pd.read_csv(BUNDLE / "cells.csv.gz")
    pick = []
    for lib in LIBRARIES:
        for hl in HEMILINEAGES:
            idx = np.flatnonzero((cells.library == lib) & (cells.hemilineage == hl))
            pick.append(rng.choice(idx, CELLS_PER_CELL, replace=False))
    pick = np.sort(np.concatenate(pick))
    meta = cells.iloc[pick].reset_index(drop=True)
    rows = meta.input_expression_row_0based.to_numpy()

    X = sp.load_npz(COUNTS).tocsr()[rows]
    hvg = pd.read_csv(BUNDLE / "hvg_genes.csv")
    det = np.asarray((X[:, hvg.source_index_0based.to_numpy()] > 0).mean(0)).ravel()
    keep = hvg[(det >= 0.05) & (det <= 0.95)].head(N_GENES)
    genes = keep.original_symbol.to_numpy()
    C = X[:, keep.source_index_0based.to_numpy()].toarray().astype(np.int32)

    # library size over ALL genes (for the depth/batch diagnostics)
    meta["total_umi"] = np.asarray(X.sum(1)).ravel()
    np.save(OUT / "counts.npy", C)
    np.save(OUT / "Y_ge1.npy", (C >= 1).astype(np.int8))
    np.save(OUT / "Y_ge2.npy", (C >= 2).astype(np.int8))
    meta.to_csv(OUT / "cells.csv", index=False)
    pd.DataFrame({"gene": genes, "source_index_0based": keep.source_index_0based,
                  "hvg_rank": keep.hvg_index}).to_csv(OUT / "genes.csv", index=False)
    print(f"counts {C.shape}  density(>=1)={np.mean(C >= 1):.3f}  density(>=2)={np.mean(C >= 2):.3f}")


if __name__ == "__main__":
    main()
