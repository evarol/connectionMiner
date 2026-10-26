"""Checks for predictors.standardized() (depth-standardized p~ for R4, R5, R6).

(a) a cell whose c_i equals the reference gives p~ = p^ exactly (tested with ref = c_k for a
    chosen cell k, and reported for the cell closest to the median);
(b) p~ depends on the cell only through u_i: eta~_ij - u_i'v_j = c_bar + b_j for every i, and
    giving cell i the factor u_k reproduces cell k's p~ row;
(c) load(code) is unchanged: test NLL recomputed for R3, R4, R5, R6, PO equals
    v3/tables/loader_validation.csv (R7, R8, R9 code paths are untouched).
Also reports c_bar, the spread of c_i and its Spearman rho with log total UMI.
Output: v3/tables/std_offsets.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import predictors as pr

rng = np.random.default_rng(1)
depth = np.load(pr.CACHE / "cell_depth.npy")
ld = np.log(depth)
n = len(depth)
rows = np.sort(rng.choice(n, 200, replace=False))
cols = np.sort(rng.choice(19543, 2000, replace=False))
recs = []
for code in pr.STD_MODELS:
    ph = pr.load(code)
    pt = pr.standardized(code)
    c, cbar = pt.c_obs, pt.c_ref
    assert np.array_equal(ph.a, c), "load() must keep each cell's own offset"
    # (a) exact equality when c_k = reference
    k = int(rows[0])
    pk = pr.standardized(code, ref=float(c[k]))
    Ph, Qh = ph.block([k], cols)
    Pk, Qk = pk.block([k], cols)
    a_exact = bool(np.array_equal(Ph, Pk) and np.array_equal(Qh, Qk))
    kmed = int(np.argmin(np.abs(c - cbar)))
    a_med = float(np.abs(ph.block([kmed], cols)[0] - pt.block([kmed], cols)[0]).max())
    # (b) eta~ - UV' is constant down each column, and swapping u_i reproduces another row
    E = pt.eta_block(rows, cols) - pt.U[rows] @ pt.V[cols].T
    b_const = float(np.abs(E - (cbar + pt.b[cols])[None]).max())
    U0 = pt.U.copy()
    i, j = int(rows[1]), int(rows[2])
    pt.U[i] = U0[j]
    b_swap = bool(np.array_equal(pt.block([i], cols)[0], pt.block([j], cols)[0]))
    pt.U = U0
    rho = spearmanr(c, ld)[0]
    rec = dict(code=code, c_ref_median=cbar, c_mean=c.mean(), c_sd=c.std(), c_q05=np.quantile(c, 0.05),
               c_q25=np.quantile(c, 0.25), c_q75=np.quantile(c, 0.75), c_q95=np.quantile(c, 0.95),
               spearman_c_logUMI=rho, check_a_exact=a_exact, check_a_nearest_median_maxdiff=a_med,
               check_b_eta_minus_UV_const_maxdev=b_const, check_b_swap_u_reproduces_row=b_swap)
    if code == "R6":
        a_only = c - np.log(depth)    # o_i is log train UMI, close to but not exactly log total UMI
        rec.update(spearman_a_only_logUMI=spearmanr(a_only, ld)[0], sd_a_only=a_only.std())
    recs.append(rec)
    print({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in rec.items()}, flush=True)
    del ph, pt, pk

# (c) unchanged loaders
Y = np.load(pr.DATA / "Y.npy")
T = np.load(pr.DATA / "test.npy")
ref = pd.read_csv(pr.STUDY / "v3/tables/loader_validation.csv").set_index("code")
for code in ["R3", "R4", "R5", "R6", "PO"]:
    nll = pr.nll_on_mask(pr.load(code), Y, T)[0]
    d = nll - ref.loc[code, "test_nll_recomputed"]
    print(f"(c) {code}: test NLL {nll:.9f}  loader_validation {ref.loc[code, 'test_nll_recomputed']:.9f}  diff {d:+.1e}")
    for r in recs:
        if r["code"] == code:
            r["check_c_test_nll_diff"] = d
out = pr.STUDY / "v3/tables/std_offsets.csv"
pd.DataFrame(recs).to_csv(out, index=False)
print(pd.DataFrame(recs).T.to_string())
