"""Shared data contract for the model comparison (handoff section 6).

The existing 5% held-out entries (seed 2026, used by every BernFac fit so far) are split
at random into a VALIDATION half (model/hyperparameter selection) and a TEST half
(final reporting only). train = everything else. Also defines a new-row split:
10% of cells (stratified by hemilineage) as test rows, each with a fixed 50% context
column mask.

Outputs (outputs/bernfac_20261020/full/data/):
  val.npy, test.npy                 boolean n x m (disjoint, union = heldout.npy)
  newrow_test_rows.npy, newrow_context.npy (rows x m boolean)

"""
from pathlib import Path

import numpy as np
import pandas as pd


D = Path(__file__).resolve().parents[3] / "outputs/bernfac_20261020/full/data"
CELLS = Path(__file__).resolve().parents[3] / "outputs/connectionminer_ready_inputs/bundles/motor/premn_24h/cells.csv.gz"


def main(seed=7):
    Y = np.load(D / "Y.npy")
    H = np.load(D / "heldout.npy")
    rng = np.random.default_rng(seed)
    coin = rng.random(H.shape, dtype=np.float32) < 0.5
    val, test = H & coin, H & ~coin
    np.save(D / "val.npy", val); np.save(D / "test.npy", test)
    cells = pd.read_csv(CELLS)
    rows = np.concatenate([rng.choice(g.index.to_numpy(), max(1, round(0.1 * len(g))), replace=False)
                           for _, g in cells.groupby("hemilineage")])
    rows.sort()
    ctx = rng.random((len(rows), Y.shape[1])) < 0.5
    np.save(D / "newrow_test_rows.npy", rows); np.save(D / "newrow_context.npy", ctx)
    print(f"train {(~H).sum():,} val {val.sum():,} test {test.sum():,}; new-row test rows {len(rows)}")


if __name__ == "__main__":
    main()
