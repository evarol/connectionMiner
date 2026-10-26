"""W1 (v4): diagnose the Poisson test entries with NLL > 20 nats.

For each such entry: cell, gene, observed count, Poisson eta (before the [-40, 15] clip) and mu,
whether eta hit the clip, NB (R6) mu / p for the same entry, the gene's train detection rate and
mean train count in detected cells, the cell's train UMI, and the gene's train count range.
Output: v4/tables/poisson_outliers.csv (+ printed summary).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v3"))
import predictors as pr  # noqa: E402

C3 = pr.STUDY / "v3/cache"
TAB = pr.STUDY / "v4/tables"
BUNDLE = pr.ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h"


def main():
    T = np.load(pr.DATA / "test.npy", mmap_mode="r")
    I, J = [], []
    for s in range(0, T.shape[0], 2048):
        r, c = np.nonzero(np.asarray(T[s:s + 2048]))
        I.append(r + s); J.append(c)
    I = np.concatenate(I); J = np.concatenate(J)
    L = np.load(C3 / "test_nll_PO.npy")
    yb = np.load(C3 / "test_y.npy")
    sel = np.where(L > 20)[0]
    i, j = I[sel], J[sel]
    po, nb = pr.load("PO"), pr.load("R6")
    eta = po.eta_pairs(i, j)
    mu = np.exp(np.clip(eta, pr.CNT_MIN, pr.CNT_MAX))
    eta_nb = nb.eta_pairs(i, j)
    mu_nb = np.exp(np.clip(eta_nb, pr.CNT_MIN, pr.CNT_MAX))
    P_nb, _ = nb.pairs(i, j)
    import scipy.sparse as sp
    keep = np.load(pr.DATA / "genes_kept.npy")
    X = sp.load_npz(BUNDLE / "G_prime.npz").tocsc()[:, keep]          # cells x kept genes, raw counts
    H = np.load(pr.DATA / "heldout.npy", mmap_mode="r")
    ycount = np.asarray(X[i, j]).ravel()
    # gene-level train stats for the genes involved (column reads)
    genes = np.unique(j)
    gstat = {}
    for g in genes:
        col = X[:, g].toarray().ravel().astype(np.float64); tr = ~np.asarray(H[:, g])
        x = col[tr]
        det = x > 0
        gstat[g] = dict(gene_detect_rate_train=det.mean(), gene_mean_count_if_detected=x[det].mean() if det.any() else 0,
                        gene_max_count_train=x.max(), gene_n_cells_count_ge_100=int((x >= 100).sum()))
    gtab = pd.read_csv(BUNDLE / "genes.csv")
    names = gtab.iloc[keep, 0].astype(str).to_numpy()
    cell_umi = np.load(C3 / "cell_depth.npy")
    rows = []
    for k in range(len(sel)):
        g = j[k]
        rows.append(dict(cell=int(i[k]), gene=int(g), gene_name=str(names[g]), y_count=int(ycount[k]),
                         y_detect=int(yb[sel[k]]), nll_poisson=float(L[sel[k]]), eta_poisson=eta[k],
                         eta_clipped=bool(eta[k] >= pr.CNT_MAX or eta[k] <= pr.CNT_MIN), mu_poisson=mu[k],
                         log_offset_o_plus_a=po.a[i[k]], gene_offset_b=po.b[g], uv=eta[k] - po.a[i[k]] - po.b[g],
                         mu_nb=mu_nb[k], phi_nb=nb.phi[g], p_detect_nb=P_nb[k], cell_train_umi=cell_umi[i[k]],
                         **gstat[g]))
    d = pd.DataFrame(rows).sort_values("nll_poisson", ascending=False)
    d.to_csv(TAB / "poisson_outliers.csv", index=False)
    pd.set_option("display.width", 250)
    print(d.head(25).to_string())
    print("n", len(d), "y_detect counts", d.y_detect.value_counts().to_dict(), "eta clipped", d.eta_clipped.sum())
    print("genes", d.gene_name.value_counts().head(15).to_dict())
    print("cells", d.cell.nunique(), d.cell.value_counts().head(10).to_dict())
    print("mu quantiles", np.percentile(d.mu_poisson, [0, 50, 90, 100]))
    print("share of total PO test NLL", d.nll_poisson.sum() / L.astype(np.float64).sum())
    print("NB p_detect quantiles", np.percentile(d.p_detect_nb, [0, 50, 100]))


if __name__ == "__main__":
    main()
