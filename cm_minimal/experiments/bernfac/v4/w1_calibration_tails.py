"""W1 (v4): calibration tail facts on the 8,044,655 test entries.

Per model: fraction of test predictions p < 1e-2, 1e-3, 1e-4 and p > 0.99, 0.999 (with the observed
detection rate and mean p inside each tail), and where |mean p - observed| peaks over 20 quantile
bins of p (as in v3 calibration_bins.csv) and over 20 equal-width bins.
Output: v4/tables/calibration_tails.csv, v4/tables/calibration_peak.csv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STUDY = Path(__file__).resolve().parents[4] / "outputs/bernfac_20261020"
C3 = STUDY / "v3/cache"
TAB = STUDY / "v4/tables"
CODES = ["R3", "R4", "R5", "R6", "PO", "R7", "R8", "R9"]
NB = 20


def main():
    y = np.load(C3 / "test_y.npy").astype(np.float64)
    rows, peaks = [], []
    for c in CODES:
        p = np.load(C3 / f"test_p_{c}.npy").astype(np.float64)
        r = dict(code=c, n=len(p), mean_p=p.mean(), mean_y=y.mean())
        for name, s in [("lt_1e-2", p < 1e-2), ("lt_1e-3", p < 1e-3), ("lt_1e-4", p < 1e-4),
                        ("gt_0.99", p > 0.99), ("gt_0.999", p > 0.999)]:
            r[f"frac_{name}"] = s.mean()
            r[f"obs_rate_{name}"] = y[s].mean() if s.any() else np.nan
            r[f"mean_p_{name}"] = p[s].mean() if s.any() else np.nan
        rows.append(r)
        for scheme in ("quantile", "width"):
            if scheme == "quantile":
                e = np.quantile(p, np.linspace(0, 1, NB + 1))
            else:
                e = np.linspace(0, 1, NB + 1)
            b = np.clip(np.searchsorted(e, p, side="right") - 1, 0, NB - 1)
            cnt = np.bincount(b, minlength=NB)
            mp = np.bincount(b, p, NB) / np.maximum(cnt, 1)
            my = np.bincount(b, y, NB) / np.maximum(cnt, 1)
            diff = np.where(cnt > 0, mp - my, 0)
            k = int(np.argmax(np.abs(diff)))
            peaks.append(dict(code=c, binning=scheme, peak_bin=k + 1, p_lo=e[k], p_hi=e[k + 1], n_bin=int(cnt[k]),
                              mean_p=mp[k], obs_rate=my[k], error_pp=100 * diff[k],
                              bins_overpredicting=int(((diff > 0) & (cnt > 0)).sum()),
                              first_overpredicting_bin_p_lo=float(e[np.argmax(diff > 0)]),
                              all_errors_pp=";".join(f"{100 * x:.2f}" for x in diff)))
    d = pd.DataFrame(rows)
    d.to_csv(TAB / "calibration_tails.csv", index=False)
    pk = pd.DataFrame(peaks)
    pk.to_csv(TAB / "calibration_peak.csv", index=False)
    pd.set_option("display.width", 250)
    print(d[["code"] + [c for c in d.columns if c.startswith("frac_")]].to_string())
    print(d[["code"] + [c for c in d.columns if c.startswith("obs_rate_")]].to_string())
    print(pk.drop(columns="all_errors_pp").to_string())
    print(pk[pk.code == "PO"].all_errors_pp.to_string())


if __name__ == "__main__":
    main()
