"""Recompute each loader's test NLL and compare with models/comparison.csv; cache the test-entry
predictions (vectors over test entries only) for the calibration / depth figures."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import predictors as pr

Y = np.load(pr.DATA / "Y.npy")
T = np.load(pr.DATA / "test.npy")
comp = pd.read_csv(pr.STUDY / "models/comparison.csv").set_index("model")
rows = []
codes = sys.argv[1:] or pr.ALL
for code in codes:
    t0 = time.time()
    p = pr.load(code)
    t1 = time.time()
    nll, n, y, pv, le = pr.nll_on_mask(p, Y, T, per_entry=True)
    ref = float(comp.loc[pr.COMPARISON_NAME[code], "test_nll"])
    print(f"{code}: recomputed {nll:.6f}  reported {ref:.6f}  diff {nll - ref:+.2e}  n={n}  load {t1-t0:.0f}s score {time.time()-t1:.0f}s", flush=True)
    rows.append(dict(code=code, model=pr.COMPARISON_NAME[code], test_nll_recomputed=nll, test_nll_reported=ref,
                     diff=nll - ref, n_test=n))
    np.save(pr.CACHE / f"test_p_{code}.npy", pv)
    np.save(pr.CACHE / f"test_nll_{code}.npy", le)
    if not (pr.CACHE / "test_y.npy").exists():
        np.save(pr.CACHE / "test_y.npy", y)
    del p
out = pr.STUDY / "v3/tables/loader_validation.csv"
old = pd.read_csv(out) if out.exists() else pd.DataFrame()
new = pd.concat([old[~old.code.isin(codes)] if len(old) else old, pd.DataFrame(rows)]).sort_values("code")
new.to_csv(out, index=False)
print(new.to_string())
