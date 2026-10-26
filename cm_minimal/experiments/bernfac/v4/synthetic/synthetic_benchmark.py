"""v4 synthetic benchmark (review W4). Same 15 datasets and the same split as v3 (read from
outputs/bernfac_20261020/v3/synthetic/data), wider grids, per-config test scoring, R8 with a fixed
iteration budget, R1/R2 on train entries, a depth-standardization ground-truth check and two
genuine-count variants of G1.

    python synthetic_benchmark.py run [--jobs 4] [--methods ...] [--datasets ...]
    python synthetic_benchmark.py one G3 0 R4          # single task (R8: one G3 0 R8 4  -> config index 4)
    python synthetic_benchmark.py collect              # tables in v4/synthetic and v4/tables

Per task, EVERY grid config is scored on validation (selection) and test (reporting) entries:
val NLL, test NLL, test AUC, KL(P_true || P_hat) and RMSE on test entries. The selected config is
the lowest validation NLL; the oracle config is the lowest test KL (shown only to measure selection
regret). Embedding metrics (G3-G5) are computed for the selected and the oracle config (all
configs for R8, whose configs are separate tasks). Depth standardization (G1-G4, which have a
cell offset in the generator) is scored for the selected and oracle configs of R4, R5, R6 and
Poisson (see depth_metrics). P_hat matrices are never written to disk.
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
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import syn_generators as sg  # noqa: E402
from v3style import STUDY  # noqa: E402

V4 = STUDY / "v4"
OUT = V4 / "synthetic"
DATA = STUDY / "v3" / "synthetic" / "data"          # reused unchanged (same datasets, same split)
RUNS = OUT / "runs"
TABLES = V4 / "tables"
WORLDS = list(sg.DATASETS)
VARIANTS = list(sg.G1_VARIANTS)                    # G1zP, G1zNB: count models only
EMBED_SETS = ("G3", "G4", "G5")
DEPTH_SETS = ("G1", "G2", "G3", "G4")
STD_METHODS = ("R4", "R5", "R6", "PO")
EPS = 1e-12


# ----------------------------------------------------------------------------- data
def load_world(g, seed):
    if g in VARIANTS:
        D1 = sg.load(DATA / f"G1_seed{seed}.npz")
        return sg.g1_count_variant(D1, sg.G1_VARIANTS[g], seed)
    return sg.load(DATA / f"{g}_seed{seed}.npz")


def truth_extras(g, seed):
    """Cell offset and depth-standardized truth, regenerated from the generator (P checked equal)."""
    g0 = "G1" if g in VARIANTS else g
    if g0 not in DEPTH_SETS:
        return None
    G = sg.generate(g0, seed)
    return dict(cell_off=G["cell_off"], P_std=G["P_std"], P=G["P"])


# ----------------------------------------------------------------------------- metrics
def bern_kl(p, q):
    from scipy.special import xlogy
    q = np.clip(q, EPS, 1 - EPS)
    p = np.clip(p, 0.0, 1.0)
    return xlogy(p, p) - xlogy(p, q) + xlogy(1 - p, 1 - p) - xlogy(1 - p, 1 - q)


def prob_metrics(P_hat, D):
    from sklearn.metrics import roc_auc_score
    te, va = D["split"] == 2, D["split"] == 1
    p = P_hat[te]; y = D["Y"][te].astype(float); pt = D["P"][te]
    pc = np.clip(p, EPS, 1 - EPS)
    pv = np.clip(P_hat[va], EPS, 1 - EPS); yv = D["Y"][va].astype(float)
    return dict(val_nll=float(-np.mean(yv * np.log(pv) + (1 - yv) * np.log1p(-pv))),
                test_nll=float(-np.mean(y * np.log(pc) + (1 - y) * np.log1p(-pc))),
                test_auc=float(roc_auc_score(y, p)), test_brier=float(np.mean((p - y) ** 2)),
                kl=float(np.mean(bern_kl(pt, p))), rmse=float(np.sqrt(np.mean((p - pt) ** 2))))


def calibration(P_hat, D):
    te = D["split"] == 2
    p = P_hat[te]; y = D["Y"][te].astype(float); pt = D["P"][te]
    b = np.clip(np.digitize(p, np.linspace(0, 1, 11)) - 1, 0, 9)
    cnt = np.bincount(b, minlength=10)
    f = lambda w: np.bincount(b, w, 10) / np.maximum(cnt, 1)  # noqa: E731
    mp, my, mt = f(p), f(y), f(pt)
    return [dict(bin=i, mean_pred=mp[i], frac_pos=my[i], mean_ptrue=mt[i], count=int(cnt[i])) for i in range(10)]


def embed_metrics(E, D, g):
    """kNN balanced accuracy, k-means ARI and best Leiden ARI vs the known groups (+ latent R^2 on G4); as v3."""
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


def depth_metrics(c, TX):
    """Task 5. p~ (cell offset -> median) vs the true depth-standardized P, and p^ vs the true observed P.

    All n x m entries (p~ is a structural quantity, not a held-out predictor). Reported:
      kl_std    mean KL(P_std_true || p~)      rmse_std   RMSE(p~ - P_std_true)
      kl_obs    mean KL(P_true || p^)          rmse_obs   RMSE(p^ - P_true)
      kl_std_nostd  mean KL(P_std_true || p^)  (what you get if you do NOT standardize)
      kl_truth_gap  mean KL(P_std_true || P_true)  (how far apart the two truths are)
      rho_off   Spearman(fitted cell offset c_i, true cell offset)
      rho_resid_std / rho_resid_true  Spearman(true offset, row mean of logit p~ / of logit P_std_true)
    """
    from scipy.stats import spearmanr
    Pt, Ps, off = TX["P"], TX["P_std"], TX["cell_off"]
    lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))  # noqa: E731
    return dict(kl_std=float(np.mean(bern_kl(Ps, c["std"]["P_std"]))),
                rmse_std=float(np.sqrt(np.mean((c["std"]["P_std"] - Ps) ** 2))),
                kl_obs=float(np.mean(bern_kl(Pt, c["P"]))), rmse_obs=float(np.sqrt(np.mean((c["P"] - Pt) ** 2))),
                kl_std_nostd=float(np.mean(bern_kl(Ps, c["P"]))),
                rmse_std_nostd=float(np.sqrt(np.mean((c["P"] - Ps) ** 2))),
                kl_truth_gap=float(np.mean(bern_kl(Ps, Pt))),
                rho_off=float(spearmanr(c["std"]["c"], off)[0]),
                rho_resid_std=float(spearmanr(off, lg(c["std"]["P_std"]).mean(1))[0]),
                rho_resid_obs=float(spearmanr(off, lg(c["P"]).mean(1))[0]),
                rho_resid_true=float(spearmanr(off, lg(Ps).mean(1))[0]))


# ----------------------------------------------------------------------------- tasks
def task_path(g, seed, method, ci=None):
    return RUNS / (f"{g}_s{seed}_{method}.json" if ci is None else f"{g}_s{seed}_{method}_c{ci}.json")


def score_cands(g, seed, method, res, D, embed_all=False):
    """Score every candidate; return per-config rows + embedding / depth / calibration records."""
    cands = res["cands"]
    rows = []
    for c in cands:
        r = dict(dataset=g, seed=seed, method=method, config=c["config"], seconds=c["seconds"],
                 iterations=c.get("iterations"), converged=c.get("converged"), termination=c.get("termination"),
                 error=c.get("error"))
        if c.get("P") is not None:
            r.update(prob_metrics(c["P"], D))
        rows.append(r)
    ok = [i for i, r in enumerate(rows) if np.isfinite(r.get("val_nll", np.inf))]
    sel = min(ok, key=lambda i: rows[i]["val_nll"]) if ok else None
    ora = min(ok, key=lambda i: rows[i]["kl"]) if ok else None
    for i, r in enumerate(rows):
        r["selected"] = i == sel
        r["oracle"] = i == ora
    embeds, depth, cal = [], [], []
    todo = sorted({i for i in ok if embed_all} | ({sel, ora} - {None}))
    if g in EMBED_SETS:
        for i in todo:
            if cands[i].get("emb") is not None:
                embeds.append(dict(dataset=g, seed=seed, method=method, config=cands[i]["config"],
                                   selected=i == sel, oracle=i == ora, rep="model",
                                   **embed_metrics(cands[i]["emb"], D, g)))
    if (g in DEPTH_SETS or g in VARIANTS) and method in STD_METHODS:
        TX = truth_extras(g, seed)
        for i in sorted({sel, ora} - {None}):
            if cands[i].get("std") is not None:
                depth.append(dict(dataset=g, seed=seed, method=method, config=cands[i]["config"],
                                  selected=i == sel, oracle=i == ora, **depth_metrics(cands[i], TX)))
    if sel is not None:
        cal = [dict(dataset=g, seed=seed, method=method, **x) for x in calibration(cands[sel]["P"], D)]
        if cands[sel].get("phi") is not None and D.get("phi") is not None:   # dispersion recovery (R6)
            from scipy.stats import spearmanr
            ph, tp = cands[sel]["phi"], D["phi"]
            rows[sel]["phi_spearman"] = float(spearmanr(ph, tp)[0])
            rows[sel]["phi_median_log_ratio"] = float(np.median(np.log(ph / tp)))
    return rows, embeds, depth, cal


def run_task(g, seed, method, ci=None, force=False):
    import syn_methods as sm
    f = task_path(g, seed, method, ci)
    if f.exists() and not force:
        return f"skip {f.name}"
    D = load_world(g, seed)
    t0 = time.time()
    try:
        if method == "BASE":
            rows, embeds, cal = [], [], []
            for code in ("B0", "B1"):
                r, _, _, c = score_cands(g, seed, code, sm.RUNNERS[code](D), D)
                rows += r; cal += c
            orc = dict(dataset=g, seed=seed, method="oracle", config="P_true", selected=True, oracle=True,
                       seconds=0.0, **prob_metrics(D["P"], D))
            rows.append(orc)
            if g in EMBED_SETS:
                for code, fn in (("R1", sm.embed_R1), ("R2", sm.embed_R2)):
                    for masked in (False, True):
                        t1 = time.time()
                        E = fn(D, masked=masked)
                        embeds.append(dict(dataset=g, seed=seed, method=code, config="PCA-50",
                                           rep="train" if masked else "full", selected=True, oracle=False,
                                           seconds=time.time() - t1, **embed_metrics(E, D, g)))
            depth = []
        elif method == "R8":
            d, lv = sm.R8_GRID[ci]
            rows, embeds, depth, cal = score_cands(g, seed, "R8", sm.run_R8_one(D, d, lv), D, embed_all=True)
        else:
            rows, embeds, depth, cal = score_cands(g, seed, method, sm.RUNNERS[method](D), D)
    except Exception:
        err = traceback.format_exc()
        f.with_suffix(".error.txt").write_text(err)
        return f"FAILED {g} s{seed} {method} {ci}: {err.splitlines()[-1]}"
    f.write_text(json.dumps(dict(rows=rows, embeds=embeds, depth=depth, calibration=cal, wall=time.time() - t0),
                            default=lambda o: None if o is None else float(o)))
    s = [r for r in rows if r.get("selected")]
    s = s[0] if s else rows[0]
    o = [r for r in rows if r.get("oracle")]
    return (f"done {g} s{seed} {method}{'' if ci is None else f' c{ci}'} {time.time() - t0:7.1f}s  "
            f"sel={s.get('config')} KL={s.get('kl', np.nan):.4f}  oracle={o[0]['config'] if o else '-'}"
            f" it={s.get('iterations')}")


def _star(args):
    t = time.time()
    msg = run_task(*args)
    return f"[{time.strftime('%H:%M:%S')}] {msg}  (task wall {time.time() - t:.0f}s)"


def task_list(methods=None, datasets=None):
    import syn_methods as sm
    want = lambda m: methods is None or m in methods  # noqa: E731
    ws = datasets or WORLDS
    tasks = []
    # R8 configs on its own world and the cell-type world first, then the fast methods, then the rest of R8
    r8 = [(g, s, "R8", ci) for g in ("G4", "G3", "G1", "G2", "G5") if g in ws for s in sg.SEEDS
          for ci in range(len(sm.R8_GRID))]
    first = [t for t in r8 if t[0] in ("G4", "G3")]
    later = [t for t in r8 if t[0] not in ("G4", "G3")]
    if want("R8"):
        tasks += first
    for m in ["R6", "R7", "R5", "PO", "R4", "R3", "R9", "BASE"]:
        if want(m):
            tasks += [(g, s, m, None) for g in ws for s in sg.SEEDS]
    for m in ["R6", "PO"]:
        if want(m) and (datasets is None or any(v in datasets for v in VARIANTS)):
            tasks += [(v, s, m, None) for v in VARIANTS if datasets is None or v in datasets for s in sg.SEEDS]
    if want("R8"):
        tasks += later
    return [t for t in tasks if not task_path(*t).exists()]


def run_all(jobs=4, methods=None, datasets=None):
    from multiprocessing import get_context
    RUNS.mkdir(parents=True, exist_ok=True)
    tasks = task_list(methods, datasets)
    print(f"{len(tasks)} tasks, {jobs} workers", flush=True)
    with get_context("spawn").Pool(jobs, maxtasksperchild=4) as pool:
        for msg in pool.imap_unordered(_star, tasks, chunksize=1):
            print(msg, flush=True)


# ----------------------------------------------------------------------------- collect
def collect():
    rows, embeds, depth, cals = [], [], [], []
    for f in sorted(RUNS.glob("*.json")):
        z = json.loads(f.read_text())
        rows += z["rows"]; embeds += z["embeds"]; depth += z["depth"]; cals += z["calibration"]
    grid = pd.DataFrame(rows)
    # R8 configs are separate tasks: redo selected / oracle flags per (dataset, seed, method)
    for key, idx in grid.groupby(["dataset", "seed", "method"]).groups.items():
        sub = grid.loc[idx]
        okv = sub["val_nll"].astype(float)
        if okv.notna().any():
            grid.loc[idx, "selected"] = sub.index == okv.idxmin()
            grid.loc[idx, "oracle"] = sub.index == sub["kl"].astype(float).idxmin()
    E = pd.DataFrame(embeds)
    if len(E):
        r8 = E.method == "R8"
        flag = grid.set_index(["dataset", "seed", "method", "config"])[["selected", "oracle"]]
        for i in E.index[r8]:
            k = (E.at[i, "dataset"], E.at[i, "seed"], "R8", E.at[i, "config"])
            E.at[i, "selected"], E.at[i, "oracle"] = bool(flag.loc[k, "selected"]), bool(flag.loc[k, "oracle"])
    OUT.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / "grid_all_configs.csv", index=False)
    E.to_csv(OUT / "embeddings.csv", index=False)
    pd.DataFrame(depth).to_csv(OUT / "depthstd.csv", index=False)
    pd.DataFrame(cals).to_csv(OUT / "calibration.csv", index=False)
    return grid, E, pd.DataFrame(depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "one", "collect", "list"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        run_all(a.jobs, a.methods, a.datasets)
    elif a.cmd == "list":
        for t in task_list(a.methods, a.datasets):
            print(t)
    elif a.cmd == "one":
        RUNS.mkdir(parents=True, exist_ok=True)
        ci = int(a.args[3]) if len(a.args) > 3 else None
        print(run_task(a.args[0], int(a.args[1]), a.args[2], ci, force=a.force))
    else:
        collect()


if __name__ == "__main__":
    main()
