"""W2 (v4) R8 neural manifold refits with an ITERATION/TOLERANCE budget instead of 1,500 s wall clock.

Reuses models/neural/run_full.run_config unchanged (same model, defaults, SVD init, scoring), with
  max_outer = MAX_OUTER (250), tol = 2e-5 (relative F decrease, the study default),
  max_seconds = SAFETY (4,200 s) only as a safety cap; if it is hit, termination says 'time_limit'.
Outputs go to outputs/bernfac_20261020/v4/{fits,logs}/neural and v4/tables/neural_v4.csv.
The d=20 SVD start is the cached training-only one from models/neural/fits; the d=40 start is
computed the same way (run_full.svd_init with dmax=40) and stored in v4/fits/neural.

  python run_neural_v4.py NAME [NAME ...]     e.g. pg_affine_d20_w64_lv1_r1
"""
import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "models/neural"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import run_full as rf  # noqa: E402

torch.set_num_threads(4)
MAX_OUTER, TOL, SAFETY = 250, 2e-5, 4200.0
OLD_FITS = rf.FITS
V4 = rf.ROOT / "outputs/bernfac_20261020/v4"
rf.RES, rf.FITS, rf.LOGS = V4 / "fits/neural/results", V4 / "fits/neural", V4 / "logs/neural"
for p in (rf.RES, rf.FITS, rf.LOGS):
    p.mkdir(parents=True, exist_ok=True)
_svd = rf.svd_init


def svd_init(Y, data, dmax=20):
    if (OLD_FITS / "svd_init_d20.npy").exists() and _CUR["d"] <= 20:
        return np.load(OLD_FITS / "svd_init_d20.npy")
    return _svd(Y, data, dmax=max(dmax, _CUR["d"]))      # cached in v4/fits/neural


_CUR = {"d": 20}
rf.svd_init = svd_init


def table():
    import pandas as pd
    rows = []
    for f in sorted(rf.RES.glob("*.json")):
        r = json.loads(f.read_text())
        for k in ("val_calibration", "test_calibration", "config"):
            r.pop(k, None)
        rows.append(r)
    d = pd.DataFrame(rows)
    d["converged"] = d["termination"] == "tol"
    d.to_csv(V4 / "tables/neural_v4.csv", index=False)
    return d


if __name__ == "__main__":
    for name in sys.argv[1:]:
        _CUR["d"] = rf.parse(name)["latent_dim"]
        rf.run_config(name, max_seconds=SAFETY, overrides=dict(max_outer=MAX_OUTER, tol=TOL))
        print(table().to_string(), flush=True)
