import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "4"
import numpy as np, json, sys
S = "/Users/erdem/Documents/github/connectionMiner/outputs/bernfac_20261020"
C = S + "/v3/cache/"
OUT = sys.argv[1]
T = np.load(S + "/full/data/test.npy", mmap_mode="r")
n, m = T.shape
I, J = [], []
for s in range(0, n, 1024):
    r, c = np.nonzero(np.asarray(T[s:s + 1024]))
    I.append((r + s).astype(np.int32)); J.append(c.astype(np.int32))
I = np.concatenate(I); J = np.concatenate(J)
N = len(I); print("n test", N, flush=True)
codes = ["R3", "R4", "R5", "R6", "R7", "R8", "R9", "PO"]
L = {c: np.load(C + f"test_nll_{c}.npy").astype(np.float64) for c in codes}
print({c: L[c].mean() for c in codes}, flush=True)
pairs = [("R6", "R4"), ("R4", "R3"), ("R7", "R4"), ("R8", "R4"), ("R5", "R4"), ("R7", "R6"), ("R5", "R3"), ("R8", "R5")]
rng = np.random.default_rng(0)
B = 400
res = {}
ncell = np.bincount(I, minlength=n); ngene = np.bincount(J, minlength=m)
for a, b in pairs:
    d = L[a] - L[b]
    mean = d.mean()
    iid_se = d.std() / np.sqrt(N)
    # cell sums / gene sums
    cs = np.bincount(I, d, minlength=n); gs = np.bincount(J, d, minlength=m)
    reps = {"cell": [], "gene": [], "twoway": []}
    for _ in range(B):
        wc = rng.poisson(1.0, n).astype(np.float64)
        wg = rng.poisson(1.0, m).astype(np.float64)
        reps["cell"].append((wc @ cs) / (wc @ ncell))
        reps["gene"].append((wg @ gs) / (wg @ ngene))
        w = wc[I] * wg[J]
        reps["twoway"].append((w @ d) / w.sum())
    out = {"mean": mean, "iid_se": iid_se}
    for k, v in reps.items():
        v = np.array(v)
        out[k + "_se"] = v.std(ddof=1)
        out[k + "_ci"] = [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    # fraction of cells / genes favouring a
    pc = cs / np.maximum(ncell, 1); pg = gs / np.maximum(ngene, 1)
    out["frac_cells_a_better"] = float((pc[ncell > 0] < 0).mean())
    out["frac_genes_a_better"] = float((pg[ngene > 0] < 0).mean())
    # how concentrated: share of total gain from top 1% genes
    gsort = np.sort(gs)
    out["gene_gain_top1pct_share"] = float(gsort[: max(1, m // 100)].sum() / gs.sum())
    res[f"{a}-{b}"] = out
    print(a, b, json.dumps(out), flush=True)
json.dump(res, open(OUT, "w"), indent=1, default=float)
