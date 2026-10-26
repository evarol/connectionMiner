"""Step 4e: identifiability / gauge study on the real Y_ge1 subset.

For fixed (lambda, k) we fit from 5 random initialisations (seeds 0..4,
init_scale 0.1, one restart each, max_iter 3000, tol 1e-9 so the runs are
well converged) and compare every seed with seed 0:

  P        : max / mean |P_s - P_0|                    (should be ~0)
  U raw    : ||U_s - U_0||_F / ||U_0||_F              (arbitrary, rotated)
  U aligned: same after orthogonal Procrustes R_s = argmin ||U_s R - U_0||
  Gram     : ||U_s U_s' - U_0 U_0'||_F / ||U_0 U_0'||_F (rotation-invariant)
  balance  : ||U'U - V'V||_F / ||U'U||_F               (MAP with lambda_u = lambda_v
             makes the factors balanced, so only orthogonal R remains free)

Gauge demonstration on the seed-0 solution:
  (U R, V R)        R random orthogonal  -> same P, same J
  (U A, V A^{-T})   A random invertible   -> same P, larger penalty (lower J):
                    the Gaussian prior breaks GL(k) down to O(k).

Outputs: tables/identifiability.csv, tables/gauge_demo.csv,
         fits/identifiability_lam<l>_k<k>.npz (U, V per seed, aligned U)
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

import bernfac as bf
from common import FITS, TABLES, load_data

SETTINGS = [(1.0, 5), (0.3, 10), (3.0, 3), (10.0, 20)]  # last = CV-best grid point
SEEDS = range(5)


def one(args):
    lam, k, seed = args
    Y = load_data()["Y_ge1"]
    r = bf.fit(Y, k, lam, lam, init="random", init_scale=0.1, seed=seed,
               max_iter=3000, tol=1e-9)
    print(f"lam={lam} k={k} seed={seed} J={r.obj[-1]:.4f} it={r.iterations} conv={r.converged}", flush=True)
    return lam, k, seed, r.U, r.V, r.obj[-1], r.iterations, r.converged


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    Y = load_data()["Y_ge1"]
    jobs = [(l, k, s) for l, k in SETTINGS for s in SEEDS]
    with Pool(11) as pool:
        res = pool.map(one, jobs, chunksize=1)
    rows, gauge = [], []
    rng = np.random.default_rng(7)
    for lam, k in SETTINGS:
        rs = [r for r in res if r[0] == lam and r[1] == k]
        U0, V0 = rs[0][3], rs[0][4]
        P0, G0 = bf.predict(U0, V0), U0 @ U0.T
        aligned = []
        for _, _, s, U, V, J, it, conv in rs:
            R, _ = orthogonal_procrustes(U, U0)
            aligned.append(U @ R)
            P = bf.predict(U, V)
            rows.append(dict(lam=lam, k=k, seed=s, J=J, J_minus_J0=J - rs[0][5], iterations=it,
                             converged=conv, P_maxabs=float(np.abs(P - P0).max()),
                             P_meanabs=float(np.abs(P - P0).mean()),
                             U_raw_rel=rel(U, U0), U_procrustes_rel=rel(U @ R, U0),
                             V_procrustes_rel=rel(V @ R, V0), gram_rel=rel(U @ U.T, G0),
                             balance_rel=rel(U.T @ U, V.T @ V)))
        np.savez_compressed(FITS / f"identifiability_lam{lam:g}_k{k}.npz",
                            U=np.stack([r[3] for r in rs]), V=np.stack([r[4] for r in rs]),
                            U_aligned=np.stack(aligned))
        # gauge demo on seed 0
        J0, _ = bf.objective(Y, U0, V0, lam, lam)
        R, _ = np.linalg.qr(rng.standard_normal((k, k)))
        A = R @ np.diag(np.exp(rng.uniform(-0.5, 0.5, k)))   # invertible, not orthogonal
        for name, (Ug, Vg) in {"identity": (U0, V0), "orthogonal R": (U0 @ R, V0 @ R),
                               "invertible A (UA, VA^-T)": (U0 @ A, V0 @ np.linalg.inv(A).T)}.items():
            Jg, llg = bf.objective(Y, Ug, Vg, lam, lam)
            gauge.append(dict(lam=lam, k=k, transform=name, J=Jg, J_minus_J0=Jg - J0, loglik=llg,
                              P_maxabs_vs_orig=float(np.abs(bf.predict(Ug, Vg) - P0).max()),
                              U_rel_change=rel(Ug, U0)))
    pd.DataFrame(rows).to_csv(TABLES / "identifiability.csv", index=False)
    pd.DataFrame(gauge).to_csv(TABLES / "gauge_demo.csv", index=False)
    print(pd.DataFrame(rows).to_string())
    print(pd.DataFrame(gauge).to_string())


if __name__ == "__main__":
    main()
