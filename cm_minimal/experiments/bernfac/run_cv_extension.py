"""Step 4a (follow-up): the main CV grid put its held-out-NLL optimum on the
grid corner (lambda = 10, k = 20). This extends the same CV protocol
(run_cv.job: identical folds, restarts and metrics) to stronger shrinkage and
higher rank to check whether the optimum is interior:

    lambda in {10, 30, 100} x k in {10, 20, 40}   (lambda=10, k in {10,20} rerun
                                                   for an exact like-for-like row)
    refinement: lambda in {5, 15, 20} at k = 40   (k large enough that the prior,
                                                   not k, sets the effective rank)
Already-finished (lambda, k, fold) rows in tables/cv_extension_folds.csv are reused.

Outputs: tables/cv_extension_folds.csv, tables/cv_extension_summary.csv,
         tables/cv_extension_calibration.csv
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from multiprocessing import Pool

import pandas as pd

from common import TABLES
from run_cv import N_FOLDS, job

LAMBDAS_EXT = [10.0, 30.0, 100.0]
KS_EXT = [10, 20, 40]
REFINE = [(5.0, 40), (15.0, 40), (20.0, 40)]


def main():
    settings = [(l, k) for l in LAMBDAS_EXT for k in KS_EXT] + REFINE
    jobs = [("Y_ge1", l, k, f) for f in range(N_FOLDS) for l, k in settings]
    old_f, old_c = TABLES / "cv_extension_folds.csv", TABLES / "cv_extension_calibration.csv"
    prev = pd.read_csv(old_f) if old_f.exists() else pd.DataFrame(columns=["lam", "k", "fold"])
    prev_c = pd.read_csv(old_c) if old_c.exists() else pd.DataFrame()
    done = set(zip(prev.lam.astype(float), prev.k.astype(int), prev.fold.astype(int)))
    jobs = [j for j in jobs if (j[1], j[2], j[3]) not in done]
    jobs.sort(key=lambda j: j[1])
    with Pool(int(os.environ.get("NPROC", 11))) as pool:
        res = pool.map(job, jobs, chunksize=1)
    folds = pd.concat([prev, pd.DataFrame([r for rs, _ in res for r in rs])], ignore_index=True)
    cal = pd.concat([prev_c, pd.DataFrame([c for _, cs in res for c in cs])], ignore_index=True)
    cal.to_csv(old_c, index=False)
    folds.to_csv(old_f, index=False)
    summ = folds.groupby(["dataset", "model", "lam", "k"])[["nll", "auc", "brier", "train_nll"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(TABLES / "cv_extension_summary.csv", index=False)
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
