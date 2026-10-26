"""W2 (v4) count-factorization refits, reusing models/counts/run_counts.run_config unchanged
(same objective, MAX_ITER=60, tol=1e-5 relative J change, same val/test scoring), but writing
fits/configs to outputs/bernfac_20261020/v4/{fits,configs} so nothing existing is overwritten.

  python run_counts_v4.py pad FAMILY K LAM K_FROM SRC   # warm start: SRC fit (FAMILY_k{K_FROM}_lam{LAM})
                                                         # padded with N(0, 0.01^2) columns; SRC = old|v4
  python run_counts_v4.py cold FAMILY K LAM             # cold start (init_offsets, seed 0)
Tags get a suffix describing the start (e.g. nb_k160_lam10_from120).
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "models/counts"))
import run_counts as rc  # noqa: E402

OLD = rc.OUT
V4 = rc.ROOT / "outputs/bernfac_20261020/v4"
rc.OUT = V4                     # run_config writes V4/fits, V4/configs
_orig_tag = rc.tag


def main():
    cmd, fam, k, lam = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    D, Xs, _ = rc.load()
    if cmd == "pad":
        k_from, src = int(sys.argv[5]), sys.argv[6]
        base = OLD if src == "old" else V4
        name = _orig_tag(fam, k_from, lam) + ("" if src == "old" else f"_from{sys.argv[7]}" if len(sys.argv) > 7 else "")
        z = np.load(base / f"fits/{name}.npz")
        rng = np.random.default_rng(k)
        U = np.c_[z["U"], 0.01 * rng.standard_normal((D.n, k - k_from))].astype(np.float64)
        V = np.c_[z["V"], 0.01 * rng.standard_normal((D.m, k - k_from))].astype(np.float64)
        init, phi = (U, V, z["a"].astype(np.float64), z["b"].astype(np.float64)), (z["phi"] if fam == "nb" else None)
        suffix = f"_from{k_from}"
    else:
        init, phi, suffix = None, None, "_cold"
    rc.tag = lambda f, kk, ll, depth="train": _orig_tag(f, kk, ll) + suffix
    with open(V4 / f"logs/{fam}_k{k}_lam{lam:g}{suffix}.log", "a") as log:
        print(f"start {cmd} {sys.argv[2:]}", file=log, flush=True)
        rc.run_config(D, Xs, fam, k, lam, init, phi, log)


if __name__ == "__main__":
    main()
