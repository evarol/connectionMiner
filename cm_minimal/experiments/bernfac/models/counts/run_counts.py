"""Poisson / NB count factorization on the full motor preMN 24h matrix (16,460 x 19,543).

  python run_counts.py bench FAMILY K           # time one sweep (and one phi update for NB)
  python run_counts.py cv FAMILY K LAM...       # lambda path strong -> weak, warm-started, resumable
  python run_counts.py queue                    # the whole grid in priority order (resumable)
  python run_counts.py depth_total FAMILY K LAM # refit one config with the total-UMI depth offset
  python run_counts.py extend K LAM...         # NB at larger k, warm-started from padded k=80 fits
  python run_counts.py cloglog_ref              # score the earlier cloglog BernFac fits on val/test
  python run_counts.py collect                  # selection.csv, best_<family>.json, embeddings

Train entries are ~heldout; every fit excludes all held-out entries. Configurations are
selected on the validation half of heldout; the test half is only reported.
Depth offset (primary): s_i = sum of cell i's counts over TRAIN entries (leak-free).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import countfac as cf  # noqa: E402

ROOT = HERE.parents[4]
COUNTS = ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h/G_prime.npz"
DATA = ROOT / "outputs/bernfac_20261020/full/data"
OUT = ROOT / "outputs/bernfac_20261020/models/counts"
for d in ("tables", "fits", "logs", "configs", "figures"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
GRID_K, GRID_LAM = (20, 40, 80), (100, 30, 10, 3)
LAM0 = 1e-3


def load_counts():
    keep = np.load(DATA / "genes_kept.npy")
    return sp.load_npz(COUNTS).tocsc()[:, keep].tocsr()


def load(depth="train"):
    Xs = load_counts()
    H = np.load(DATA / "heldout.npy")
    s_total = np.asarray(Xs.sum(1), np.float64).ravel()
    D = cf.CountData(Xs.toarray().astype(np.float32), H, depth=depth, s_total=s_total)
    return D, Xs, H


def masks():
    return {"val": np.load(DATA / "val.npy"), "test": np.load(DATA / "test.npy")}


def bench(family, k):
    D, _, _ = load()
    U, V, a, b = cf.init_offsets(D, k)
    phi = cf.phi_init(D) if family == "nb" else None
    t = time.time(); cf.objective(D, U, V, a, b, 10.0, LAM0, phi); t0 = time.time() - t
    t = time.time(); cf.factor_iteration(D, U, V, a, b, 10.0, LAM0, phi); t1 = time.time() - t
    t2 = 0.0
    if family == "nb":
        t = time.time(); cf.update_phi(D, U, V, a, b, phi); t2 = time.time() - t
    print(f"{family} k={k}: objective {t0:.1f}s, factor sweep {t1:.1f}s, phi update {t2:.1f}s", flush=True)


def tag(family, k, lam, depth="train"):
    return f"{family}_k{k}_lam{lam:g}" + ("" if depth == "train" else f"_{depth}")


MAX_ITER = int(os.environ.get("COUNTS_MAX_ITER", 60))
TOL = float(os.environ.get("COUNTS_TOL", 1e-5))
LAMS = {"poisson": GRID_LAM, "nb": tuple(float(x) for x in os.environ.get("COUNTS_NB_LAMS", "100,30,10").split(","))}


def run_config(D, Xs, family, k, lam, init, phi, log, depth="train", max_iter=MAX_ITER, tol=TOL):
    t0 = time.time()
    U, V, a, b, phi, tr = cf.fit(D, k, lam, family=family, lam0=LAM0, init=init, phi=phi, max_iter=max_iter,
                                 tol=tol, log=log)
    secs = time.time() - t0
    sc, cal = cf.heldout_scores(np.load(DATA / "Y.npy", mmap_mode="r"), Xs, D.o, U, V, a, b, phi, masks())
    res = dict(family=family, k=k, lam=lam, depth=depth,
               val_nll=sc["val"]["nll"], val_auc=sc["val"]["auc"], val_brier=sc["val"]["brier"],
               val_count_ll=sc["val"]["count_ll"],
               test_nll=sc["test"]["nll"], test_auc=sc["test"]["auc"], test_brier=sc["test"]["brier"],
               test_count_ll=sc["test"]["count_ll"], n_val=sc["val"]["n"], n_test=sc["test"]["n"],
               J=tr["obj"][-1], train_ll_per_entry=tr["loglik"][-1] / D.n_train,
               iterations=tr["iterations"], converged=tr["converged"],
               monotone=bool(np.all(np.diff(tr["obj"]) >= -1e-9 * abs(tr["obj"][0]))),
               seconds=secs, warm_start=init is not None,
               phi_median=None if phi is None else float(np.median(phi)), calibration=cal)
    t = tag(family, k, lam, depth)
    np.savez(OUT / f"fits/{t}.npz", U=U.astype(np.float32), V=V.astype(np.float32), a=a, b=b, o=D.o,
             phi=np.array([]) if phi is None else phi, obj=np.array(tr["obj"]), loglik=np.array(tr["loglik"]),
             seconds=np.array(tr["seconds"]))
    (OUT / f"configs/{t}.json").write_text(json.dumps(res, indent=1))
    print({k_: v for k_, v in res.items() if k_ != "calibration"}, flush=True)
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "calibration"}), file=log, flush=True)
    return (U, V, a, b), phi


def cv(family, k, lams, D=None, Xs=None):
    if D is None:
        D, Xs, _ = load()
    init = phi = None
    with open(OUT / f"logs/cv_{family}_k{k}.log", "a") as log:
        for lam in sorted(lams, reverse=True):
            t = tag(family, k, lam)
            if (OUT / f"configs/{t}.json").exists():
                z = np.load(OUT / f"fits/{t}.npz")
                init = tuple(z[x].astype(np.float64) for x in ("U", "V", "a", "b"))
                phi = z["phi"] if family == "nb" else None
                continue
            init, phi = run_config(D, Xs, family, k, lam, init, phi, log)


def queue():
    order = os.environ.get("COUNTS_QUEUE", "poisson:20,nb:20,poisson:40,nb:40,poisson:80,nb:80")
    D, Xs, _ = load()
    for item in order.split(","):
        fam, k = item.split(":")
        cv(fam, int(k), LAMS[fam], D, Xs)
    print("queue done", flush=True)


def depth_total(family, k, lam):
    """Refit one config with s_i = total UMI over all genes (includes held-out counts: minor leak)."""
    D, Xs, _ = load(depth="total")
    z = np.load(OUT / f"fits/{tag(family, k, lam)}.npz")
    init = tuple(z[x].astype(np.float64) for x in ("U", "V", "a", "b"))
    init = (init[0], init[1], init[2] + z["o"] - D.o, init[3])          # same mu at the start
    with open(OUT / f"logs/depth_total.log", "a") as log:
        run_config(D, Xs, family, k, lam, init, z["phi"] if family == "nb" else None, log, depth="total")


def extend(k, lams, k_from=80, pad_scale=0.01):
    """NB rank extension. Each (k, lam) is warm-started from the NB k_from fit at the same lam:
    U, V get k - k_from extra columns drawn from N(0, pad_scale^2); a, b, phi are copied."""
    D, Xs, _ = load()
    rng = np.random.default_rng(k)
    with open(OUT / f"logs/extend_nb_k{k}.log", "a") as log:
        for lam in lams:
            if (OUT / f"configs/{tag('nb', k, lam)}.json").exists():
                continue
            z = np.load(OUT / f"fits/{tag('nb', k_from, lam)}.npz")
            U = np.c_[z["U"], pad_scale * rng.standard_normal((D.n, k - k_from))].astype(np.float64)
            V = np.c_[z["V"], pad_scale * rng.standard_normal((D.m, k - k_from))].astype(np.float64)
            run_config(D, Xs, "nb", k, lam, (U, V, z["a"], z["b"]), z["phi"], log)


def cloglog_ref():
    """Score earlier cloglog BernFac fits with bernfac_large.heldout_scores_cloglog on val / test."""
    import bernfac_large as bl
    Y = np.load(DATA / "Y.npy")
    rows = []
    fits = ROOT / "outputs/bernfac_20261020/full/fits"
    for name, note in [("best_full_cloglog", "refit on ALL entries incl. val/test (leaky)"),
                       ("cv_cll_k40_lam30", "train mask only (clean); CV-selected config"),
                       ("cv_cll_k40_lam10", "train mask only (clean)")]:
        z = np.load(fits / f"{name}.npz")
        r = dict(fit=name, note=note)
        for split, M in masks().items():
            sc = bl.heldout_scores_cloglog(Y, z["U"], z["V"], z["a"], z["b"], M)
            r.update({f"{split}_{m}": sc[m] for m in ("nll", "auc", "brier")})
        rows.append(r); print(r, flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "tables/cloglog_reference.csv", index=False)


def collect():
    import pandas as pd
    recs, cal = [], []
    for p in sorted((OUT / "configs").glob("*.json")):
        r = json.loads(p.read_text())
        for c in r.pop("calibration"):
            cal.append(dict(family=r["family"], k=r["k"], lam=r["lam"], depth=r["depth"], **c))
        recs.append(r)
    df = pd.DataFrame(recs)
    cols = ["family", "k", "lam", "val_nll", "val_auc", "val_brier", "test_nll", "test_auc", "test_brier",
            "iterations", "converged", "monotone", "seconds"]
    main = df[df.depth == "train"].sort_values(["family", "k", "lam"], ascending=[True, True, False])
    main[cols].to_csv(OUT / "tables/selection.csv", index=False)
    main.to_csv(OUT / "tables/selection_full.csv", index=False)
    if (df.depth != "train").any():
        df[df.depth != "train"].to_csv(OUT / "tables/depth_total.csv", index=False)
    pd.DataFrame(cal).to_csv(OUT / "tables/calibration.csv", index=False)
    for fam, g in main.groupby("family"):
        best = g.loc[g.val_nll.idxmin()].to_dict()
        best = {k_: (v.item() if hasattr(v, "item") else v) for k_, v in best.items()}
        best["selection"] = "lowest validation Bernoulli NLL over k x lambda; test reported only"
        (OUT / f"best_{fam}.json").write_text(json.dumps(best, indent=1))
        z = np.load(OUT / f"fits/{tag(fam, int(best['k']), best['lam'])}.npz")
        np.savez(OUT / f"fits/best_{fam}.npz", U=z["U"], V=z["V"], a=z["a"], b=z["b"], o=z["o"], phi=z["phi"],
                 k=best["k"], lam=best["lam"])
        E, s = cf.embedding(z["U"].astype(np.float64), z["V"].astype(np.float64))
        np.save(OUT / f"embedding_{fam}.npy", E.astype(np.float32))
        print(fam, {k_: best[k_] for k_ in cols}, "top singular values", np.round(s[:5], 2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "bench":
        bench(sys.argv[2], int(sys.argv[3]))
    elif cmd == "cv":
        cv(sys.argv[2], int(sys.argv[3]), [float(x) for x in sys.argv[4:]])
    elif cmd == "queue":
        queue()
    elif cmd == "depth_total":
        depth_total(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
    elif cmd == "extend":
        extend(int(sys.argv[2]), [float(x) for x in sys.argv[3:]])
    elif cmd == "cloglog_ref":
        cloglog_ref()
    elif cmd == "collect":
        collect()
