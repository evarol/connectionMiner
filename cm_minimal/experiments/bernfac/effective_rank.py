"""Effective rank of each full-data fit.

With lambda_u = lambda_v = lambda, min over factorisations of
lambda/2 (||U||^2 + ||V||^2) subject to U V' = Psi equals lambda ||Psi||_* (nuclear
norm), so BernFac's MAP is nuclear-norm-penalised logistic PCA restricted to rank
<= k: the prior soft-thresholds singular values and can switch factors off.
We report the singular values of Psi = U V' and the number above 1% of the
largest (and above an absolute 1e-3).

Output: tables/effective_rank.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import KS, KS_GE2, LAMBDAS, LAMBDAS_GE2, TABLES, load_fit


def main():
    rows = []
    for ds, lams, ks in [("Y_ge1", LAMBDAS, KS), ("Y_ge2", LAMBDAS_GE2, KS_GE2)]:
        for lam in lams:
            for k in ks:
                f = load_fit(ds, lam, k)
                s = np.linalg.svd(f["U"] @ f["V"].T, compute_uv=False)[:k]
                rows.append(dict(dataset=ds, lam=lam, k=k, eff_rank_rel1pct=int(np.sum(s > 0.01 * s[0])),
                                 eff_rank_abs1e3=int(np.sum(s > 1e-3)), sv_max=float(s[0]),
                                 sv_min=float(s[-1]), nuclear_norm=float(s.sum()),
                                 singular_values=";".join(f"{x:.4g}" for x in s)))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "effective_rank.csv", index=False)
    print(df.drop(columns="singular_values").to_string())


if __name__ == "__main__":
    main()
