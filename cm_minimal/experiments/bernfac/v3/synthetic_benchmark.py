"""Shared synthetic benchmark for the v3 deck: every method on the same simulated datasets.

    python synthetic_benchmark.py data            # generate and save the 15 datasets
    python synthetic_benchmark.py run [--jobs 4]  # all method x dataset x seed tasks (resumable)
    python synthetic_benchmark.py one G3 0 R7     # a single task (debugging / timing)
    python synthetic_benchmark.py collect         # results.csv, summary.csv, calibration.csv

Design (see syn_generators.py for the five worlds G1-G5): n = 2000 cells, m = 300 genes,
3 seeds per world, counts X with known P_true = Pr(X >= 1), Y = 1[X >= 1], one 70/15/15 entry
split shared by every method. Each method fits a small grid on train entries and keeps the
configuration with the lowest validation Bernoulli NLL (syn_methods.py). On TEST entries we
report Bernoulli NLL, AUC, Brier, the 10-bin expected calibration error, and the oracle
metrics against P_true: mean KL(P_true || P_hat) and RMSE(P_hat - P_true). Reliability curves
(10 equal-width bins of P_hat on test entries) store both the observed rate and the mean
P_true in each bin.

Cell-embedding recovery (G3 true cell type, G4 eight k-means regions of the true 2-D latent
plus kNN regression R^2 of the latent coordinates, G5 true row block): 5-fold kNN (k = 15)
balanced accuracy, ARI of k-means with the true number of groups, and the best ARI of Leiden
over resolutions {0.25, 0.5, 1, 2} (15-NN graph), as in umap_compare.py.

Tasks run in a pool of 4 single-threaded processes; each writes runs/<G>_s<seed>_<method>.json.
P_hat matrices are never written to disk.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import syn_generators as sg  # noqa: E402
from v3style import V3  # noqa: E402

OUT = V3 / "synthetic"
DATA, RUNS = OUT / "data", OUT / "runs"
METHODS = ["R3", "R4", "R5", "R6", "PO", "R7", "R8", "R9", "BASE"]   # BASE = B0, B1, oracle, R1, R2
EMBED_SETS = ("G3", "G4", "G5")
EPS = 1e-12


# ----------------------------------------------------------------------------- metrics
def bern_kl(p, q):
    q = np.clip(q, EPS, 1 - EPS)
    p = np.clip(p, 0.0, 1.0)
    from scipy.special import xlogy
    return xlogy(p, p) - xlogy(p, q) + xlogy(1 - p, 1 - p) - xlogy(1 - p, 1 - q)


def prob_metrics(P_hat, D):
    from sklearn.metrics import roc_auc_score
    te = D["split"] == 2
    p = P_hat[te]; y = D["Y"][te].astype(float); pt = D["P"][te]
    pc = np.clip(p, EPS, 1 - EPS)
    edges = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(p, edges) - 1, 0, 9)
    cnt = np.bincount(b, minlength=10)
    mp = np.bincount(b, p, 10) / np.maximum(cnt, 1)
    my = np.bincount(b, y, 10) / np.maximum(cnt, 1)
    mt = np.bincount(b, pt, 10) / np.maximum(cnt, 1)
    out = dict(test_nll=float(-np.mean(y * np.log(pc) + (1 - y) * np.log1p(-pc))),
               test_auc=float(roc_auc_score(y, p)), test_brier=float(np.mean((p - y) ** 2)),
               test_ece=float(np.sum(cnt * np.abs(mp - my)) / cnt.sum()),
               kl=float(np.mean(bern_kl(pt, p))), rmse=float(np.sqrt(np.mean((p - pt) ** 2))),
               val_nll=float(-np.mean(np.where(D["Y"][D["split"] == 1] > 0, np.log(np.clip(P_hat[D["split"] == 1], EPS, 1)),
                                               np.log(np.clip(1 - P_hat[D["split"] == 1], EPS, 1))))))
    cal = [dict(bin=i, mean_pred=mp[i], frac_pos=my[i], mean_ptrue=mt[i], count=int(cnt[i])) for i in range(10)]
    return out, cal


def embed_metrics(E, D, g):
    """kNN balanced accuracy, k-means ARI and best Leiden ARI vs the known groups (+ latent R^2 on G4)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    E = np.asarray(E, np.float64)
    E = E[:, E.std(0) > 1e-12]
    lab = D["labels"]
    K = len(np.unique(lab))
    pred = cross_val_predict(KNeighborsClassifier(15), E, lab, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    out = dict(embed_dim=int(E.shape[1]), knn_balacc=float(balanced_accuracy_score(lab, pred)),
               kmeans_ari=float(adjusted_rand_score(lab, KMeans(K, n_init=10, random_state=0).fit_predict(E))))
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(np.zeros((E.shape[0], 1), np.float32)); a.obsm["E"] = E.astype(np.float32)
    sc.pp.neighbors(a, n_neighbors=15, use_rep="E", random_state=0)
    best = -1.0
    for r in (0.25, 0.5, 1.0, 2.0):
        sc.tl.leiden(a, resolution=r, key_added="cl", random_state=0, flavor="igraph", n_iterations=2, directed=False)
        best = max(best, adjusted_rand_score(lab, a.obs["cl"].values))
    out["leiden_best_ari"] = float(best)
    if g == "G4":
        zp = cross_val_predict(KNeighborsRegressor(15), E, D["z"], cv=KFold(5, shuffle=True, random_state=0))
        z = D["z"]
        out["latent_r2"] = float(1 - np.sum((zp - z) ** 2) / np.sum((z - z.mean(0)) ** 2))
    return out


# ----------------------------------------------------------------------------- tasks
def task_path(g, seed, method):
    return RUNS / f"{g}_s{seed}_{method}.json"


def run_task(g, seed, method, force=False):
    import syn_methods as sm
    f = task_path(g, seed, method)
    if f.exists() and not force:
        return f"skip {f.name}"
    D = sg.ensure(DATA, g, seed)
    rows, cals = [], []
    base = dict(dataset=g, seed=seed)
    t0 = time.time()

    def record(code, P, emb, config, grid, secs):
        r = dict(base, method=code, config=config, seconds=secs, n_grid=len(grid))
        if P is not None:
            pm, cal = prob_metrics(P, D)
            r.update(pm)
            cals.extend(dict(base, method=code, **c) for c in cal)
        if emb is not None and g in EMBED_SETS:
            r.update(embed_metrics(emb, D, g))
        r["grid"] = json.dumps([[c, None if not np.isfinite(v) else round(v, 6), round(s, 2), it, cv]
                                for c, v, s, it, cv in grid])
        rows.append(r)

    try:
        if method == "BASE":
            for code in ("B0", "B1"):
                res = sm.RUNNERS[code](D)
                record(code, res["P"], None, res["config"], res["grid"], res["fit_seconds"])
            record("oracle", D["P"], None, "P_true", [], 0.0)
            for code, fn in (("R1", sm.embed_R1), ("R2", sm.embed_R2)):
                t1 = time.time(); E = fn(D)
                record(code, None, E, "PCA-50", [], time.time() - t1)
        else:
            res = sm.RUNNERS[method](D)
            record(method, res["P"], res["emb"], res["config"], res["grid"], res["fit_seconds"])
    except Exception:
        err = traceback.format_exc()
        (RUNS / f"{g}_s{seed}_{method}.error.txt").write_text(err)
        return f"FAILED {g} s{seed} {method}: {err.splitlines()[-1]}"
    f.write_text(json.dumps(dict(rows=rows, calibration=cals, wall=time.time() - t0), default=float))
    sel = rows[0]
    return (f"done {g} s{seed} {method:4s} {time.time() - t0:7.1f}s  cfg={sel.get('config')}  "
            f"KL={sel.get('kl', np.nan):.4f}  testNLL={sel.get('test_nll', np.nan):.4f}")


def _star(args):
    t = time.time()
    msg = run_task(*args)
    return f"[{time.strftime('%H:%M:%S')}] {msg}  (task wall {time.time() - t:.0f}s)"


def run_all(jobs=4, methods=None, datasets=None):
    from multiprocessing import get_context
    RUNS.mkdir(parents=True, exist_ok=True)
    for g in sg.DATASETS:
        for s in sg.SEEDS:
            sg.ensure(DATA, g, s)
    # slowest methods first so the pool stays busy at the end
    order = ["R8", "R6", "R7", "PO", "R9", "R5", "R4", "R3", "BASE"]
    tasks = [(g, s, mth) for mth in order if (methods is None or mth in methods)
             for g in (datasets or sg.DATASETS) for s in sg.SEEDS if not task_path(g, s, mth).exists()]
    print(f"{len(tasks)} tasks, {jobs} workers", flush=True)
    with get_context("spawn").Pool(jobs, maxtasksperchild=4) as pool:
        for msg in pool.imap_unordered(_star, tasks):
            print(msg, flush=True)


# ----------------------------------------------------------------------------- collect
def collect():
    rows, cals = [], []
    for f in sorted(RUNS.glob("*.json")):
        z = json.loads(f.read_text())
        rows += z["rows"]; cals += z["calibration"]
    df = pd.DataFrame(rows)
    order = ["oracle", "B0", "B1", "R1", "R2", "R3", "R4", "R5", "R6", "PO", "R7", "R8", "R9"]
    df["method"] = pd.Categorical(df["method"], order, ordered=True)
    df = df.sort_values(["dataset", "method", "seed"])
    first = ["method", "dataset", "seed", "config", "test_nll", "test_auc", "test_brier", "test_ece", "kl", "rmse",
             "val_nll", "knn_balacc", "kmeans_ari", "leiden_best_ari", "latent_r2", "embed_dim", "seconds", "n_grid"]
    df = df[[c for c in first if c in df] + [c for c in df if c not in first]]
    df.to_csv(OUT / "results.csv", index=False)
    pd.DataFrame(cals).to_csv(OUT / "calibration.csv", index=False)
    metrics = [c for c in ["kl", "rmse", "test_nll", "test_auc", "test_brier", "test_ece", "knn_balacc", "kmeans_ari",
                           "leiden_best_ari", "latent_r2", "seconds"] if c in df]
    summ = df.groupby(["dataset", "method"], observed=True)[metrics].agg(["mean", "std", "count"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ["configs"] = df.groupby(["dataset", "method"], observed=True)["config"].agg(lambda s: "; ".join(map(str, s))).values
    summ.to_csv(OUT / "summary.csv", index=False)
    piv = summ.pivot(index="method", columns="dataset", values="kl_mean")
    print("mean KL(P_true || P_hat) on test entries\n", piv.round(4).to_string())
    print("\ntest NLL\n", summ.pivot(index="method", columns="dataset", values="test_nll_mean").round(4).to_string())
    return df, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "run", "one", "collect"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "data":
        for g in sg.DATASETS:
            for s in sg.SEEDS:
                D = sg.ensure(DATA, g, s)
                print(g, s, f"density {D['Y'].mean():.3f}", f"meanP {D['P'].mean():.3f}")
    elif a.cmd == "run":
        run_all(a.jobs, a.methods, a.datasets)
    elif a.cmd == "one":
        RUNS.mkdir(parents=True, exist_ok=True)
        print(run_task(a.args[0], int(a.args[1]), a.args[2], force=a.force))
    else:
        collect()


if __name__ == "__main__":
    main()
