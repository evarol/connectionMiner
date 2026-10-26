"""W2 (v4) R4 (BernFac logit + offsets) grid-edge fits, using bernfac_large.fit_offsets unchanged
(PG-EM, lam0 = 1e-3, max_iter 250, tol 1e-6, train = ~heldout) and the rescore_val_test scoring
(val/test NLL, Brier, AUC on a 2M subsample with rng seed 0).

  python run_offsets_v4.py K LAM [INIT]   INIT = cold (default: additive_init + N(0, 0.1^2) factors, seed 0,
                                           i.e. how the first, strongest lambda of every original path started)
                                           or pad:K0:LAM0 (v3 fit cv_off_k{K0}_lam{LAM0} padded with N(0, 0.01^2))
Writes v4/fits/off_k{K}_lam{LAM}_{init}.npz and v4/configs/off_*.json.
"""
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bernfac_large as bl  # noqa: E402
from rescore_val_test import scores_from_logit  # noqa: E402

ROOT = HERE.parents[3]
FULL = ROOT / "outputs/bernfac_20261020/full"
V4 = ROOT / "outputs/bernfac_20261020/v4"


def main(k, lam, init="cold"):
    tag = f"off_k{k}_lam{lam:g}_{init.replace(':', '')}"
    if (V4 / f"configs/{tag}.json").exists():
        print("skip", tag); return
    Y = np.load(FULL / "data/Y.npy"); H = np.load(FULL / "data/heldout.npy")
    ini = None
    if init.startswith("pad"):
        _, k0, l0 = init.split(":")
        z = np.load(FULL / f"fits/cv_off_k{k0}_lam{l0}.npz")
        rng = np.random.default_rng(k)
        ini = (np.c_[z["U"], 0.01 * rng.standard_normal((Y.shape[0], k - int(k0)))],
               np.c_[z["V"], 0.01 * rng.standard_normal((Y.shape[1], k - int(k0)))], z["a"], z["b"])
    t0 = time.time()
    with open(V4 / f"logs/{tag}.log", "a") as log:
        U, V, a, b, tr = bl.fit_offsets(Y, k, lam, heldout=H, init=ini, max_iter=250, tol=1e-6, log=log)
    secs = time.time() - t0
    del H
    U32, V32, a32, b32 = (x.astype(np.float32) for x in (U, V, a, b))
    fn = lambda r: U32[r] @ V32.T + a32[r, None] + b32[None, :]
    rv = scores_from_logit(Y, np.load(FULL / "data/val.npy"), fn)
    rt = scores_from_logit(Y, np.load(FULL / "data/test.npy"), fn)
    res = dict(model="logit + offsets", k=k, lam=lam, init=init, **{f"val_{x}": rv[x] for x in rv},
               **{f"test_{x}": rt[x] for x in rt}, J=tr["obj"][-1], iterations=tr["iterations"],
               converged=bool(tr["converged"]), monotone=bool(np.all(np.diff(tr["obj"]) > -1e-7 * abs(tr["obj"][0]))),
               seconds=secs)
    np.savez(V4 / f"fits/{tag}.npz", U=U32, V=V32, a=a, b=b, obj=np.array(tr["obj"]))
    (V4 / f"configs/{tag}.json").write_text(json.dumps(res, indent=1))
    print(res, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "cold")
