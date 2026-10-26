"""Step 3: full-data BernFac grid fits on the real 1000 x 100 subset.

For each (dataset, lambda, k), with lambda_u = lambda_v = lambda:
  * one fit from init='svd' and three from init='random' (seeds 1..3,
    init_scale 0.1); max_iter 500, tol 1e-7;
  * the restart with the largest final MAP objective J is kept;
  * all four final J's (and which init won) are recorded.

Grids: Y_ge1 lambda in {0.03,...,10} x k in {1,2,3,5,10,20};
       Y_ge2 lambda in {0.3,1,3} x k in {2,5,10} (sensitivity).

Outputs (outputs/bernfac_20261020/real/):
  fits/<dataset>_lam<l>_k<k>.npz   U, V, P, obj trace, loglik trace
  tables/grid_results.csv          one tidy row per fit
  tables/grid_results.json         same + all restart J's

Run:  OMP_NUM_THREADS=1 python run_grid.py
"""
from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import bernfac as bf
from common import (FIT_KW, FITS, INIT_SCALE, KS, KS_GE2, LAMBDAS, LAMBDAS_GE2,
                    N_RANDOM_RESTARTS, TABLES, fit_tag, load_data)


def fit_best(Y, k, lam, mask=None):
    """SVD init + N_RANDOM_RESTARTS random inits; return (best, list of all)."""
    runs = [("svd", 0, bf.fit(Y, k, lam, lam, mask=mask, init="svd", seed=0, **FIT_KW))]
    for s in range(1, N_RANDOM_RESTARTS + 1):
        runs.append(("random", s, bf.fit(Y, k, lam, lam, mask=mask, init="random",
                                         init_scale=INIT_SCALE, seed=s, **FIT_KW)))
    best = max(runs, key=lambda r: r[2].obj[-1])
    return best, runs


def job(args):
    dataset, lam, k = args
    Y = load_data()[dataset]
    t0 = time.time()
    (init, seed, r), runs = fit_best(Y, k, lam)
    obj = np.asarray(r.obj)
    np.savez_compressed(FITS / f"{fit_tag(dataset, lam, k)}.npz", U=r.U, V=r.V, P=r.P,
                        obj=obj, loglik=np.asarray(r.loglik))
    n_obs = Y.size
    row = dict(
        dataset=dataset, lam=lam, k=k, best_init=init, best_seed=seed,
        J=float(obj[-1]), loglik=float(r.loglik[-1]), loglik_per_entry=float(r.loglik[-1]) / n_obs,
        penalty=float(r.loglik[-1] - obj[-1]), iterations=r.iterations, converged=r.converged,
        grad_norm=r.grad_norm, monotone=bool(np.all(np.diff(obj) >= -1e-9 * max(1, abs(obj[-1])))),
        U_fro=float(np.linalg.norm(r.U)), V_fro=float(np.linalg.norm(r.V)),
        restart_J=[float(x[2].obj[-1]) for x in runs],
        restart_init=[f"{x[0]}{x[1]}" for x in runs],
        restart_iters=[int(x[2].iterations) for x in runs],
        restart_converged=[bool(x[2].converged) for x in runs],
        seconds=time.time() - t0,
    )
    Js = np.asarray(row["restart_J"])
    row["restart_J_spread"] = float(Js.max() - Js.min())
    row["restart_J_spread_rel"] = float((Js.max() - Js.min()) / abs(Js.max()))
    row["n_restarts_within_1e-6"] = int(np.sum((Js.max() - Js) / abs(Js.max()) < 1e-6))
    print(f"{dataset} lam={lam:g} k={k:2d}  J={row['J']:.2f} it={r.iterations} "
          f"conv={r.converged} best={init}{seed} spread={row['restart_J_spread']:.3g} "
          f"({row['seconds']:.0f}s)", flush=True)
    return row


def main():
    FITS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    jobs = [("Y_ge1", l, k) for l in LAMBDAS for k in KS]
    jobs += [("Y_ge2", l, k) for l in LAMBDAS_GE2 for k in KS_GE2]
    jobs.sort(key=lambda j: -j[2] / j[1] ** 0.5)  # slowest (big k, small lambda) first
    with Pool(int(os.environ.get("NPROC", 11))) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    rows.sort(key=lambda r: (r["dataset"], r["lam"], r["k"]))
    with open(TABLES / "grid_results.json", "w") as f:
        json.dump(rows, f, indent=1)
    df = pd.DataFrame(rows)
    for c in ["restart_J", "restart_init", "restart_iters", "restart_converged"]:
        df[c] = df[c].map(lambda v: ";".join(str(x) for x in v))
    df.to_csv(TABLES / "grid_results.csv", index=False)


if __name__ == "__main__":
    main()
