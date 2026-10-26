"""BernFac on the FULL motor preMN 24h APF matrix (16,460 cells x 19,543 detected genes).

  python run_full.py prep            # binarize, gene index, held-out mask
  python run_full.py bench K         # time one PG-EM iteration at rank K
  python run_full.py cv K LAM...     # lambda path (large -> small, warm-started) on the train mask
  python run_full.py final K LAM     # refit the selected (K, LAM) on all entries -> fits/best_full.npz

Held-out CV: a fixed random 5% of entries (16 M) is excluded from training and
scored (NLL, Brier, AUC). With this many held-out entries a single split has
negligible Monte-Carlo error, so one split replaces the 5 folds used on the subset.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import bernfac_large as bl

ROOT = Path(__file__).resolve().parents[3]
COUNTS = ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h/G_prime.npz"
OUT = ROOT / "outputs/bernfac_20261020/full"
HELDOUT_FRAC, SEED = 0.05, 2026


def prep():
    X = sp.load_npz(COUNTS).tocsc()
    keep_path = OUT / "data/genes_kept.npy"
    if keep_path.exists():
        keep = np.load(keep_path)
    else:
        keep = np.flatnonzero(np.asarray((X > 0).sum(0)).ravel() >= 1)
        np.save(keep_path, keep)
    Y = (X[:, keep] > 0).toarray().astype(np.uint8)
    np.save(OUT / "data/Y.npy", Y)
    H = np.random.default_rng(SEED).random(Y.shape, dtype=np.float32) < HELDOUT_FRAC
    np.save(OUT / "data/heldout.npy", H)
    print(f"Y {Y.shape} density {Y.mean():.4f}; held-out {H.mean():.4f} ({H.sum():,} entries)")


def load():
    return np.load(OUT / "data/Y.npy"), np.load(OUT / "data/heldout.npy")


def bench(k):
    Y, H = load()
    U, V = bl.svd_init(Y, k, H)
    t = time.time(); bl.pgem_iteration(Y, U, V, 10.0, H); t1 = time.time() - t
    t = time.time(); bl.objective(Y, U, V, 10.0, H); t2 = time.time() - t
    print(f"k={k}: iteration {t1:.1f}s, objective {t2:.1f}s")


def cv(k, lams, max_iter=300, tol=1e-6):
    """Lambda path from strong to weak shrinkage, each warm-started from the previous."""
    Y, H = load()
    lams = sorted(lams, reverse=True)
    U = V = None
    with open(OUT / f"logs/cv_k{k}.log", "a") as log:
        for lam in lams:
            tag = f"k{k}_lam{lam:g}"
            if (OUT / f"tables/cv_{tag}.json").exists():
                z = np.load(OUT / f"fits/cv_{tag}.npz"); U, V = z["U"], z["V"]
                continue
            t0 = time.time()
            U, V, tr = bl.fit(Y, k, lam, H, U0=U, V0=V, max_iter=max_iter, tol=tol, log=log)
            sc = bl.heldout_scores(Y, U, V, H)
            res = dict(k=k, lam=lam, **sc, train_ll_per_entry=tr["loglik"][-1] / float((~H).sum()),
                       J=tr["obj"][-1], iterations=tr["iterations"], converged=bool(tr["converged"]),
                       monotone=bool(np.all(np.diff(tr["obj"]) > -1e-7 * abs(tr["obj"][0]))),
                       seconds=time.time() - t0, warm_start=U is not None)
            np.savez(OUT / f"fits/cv_{tag}.npz", U=U, V=V, obj=np.array(tr["obj"]), loglik=np.array(tr["loglik"]))
            (OUT / f"tables/cv_{tag}.json").write_text(json.dumps(res, indent=1))
            print(json.dumps(res), file=log, flush=True)
            print(res, flush=True)


def final(k, lam, max_iter=500, tol=1e-7):
    Y, _ = load()
    z = np.load(OUT / f"fits/cv_k{k}_lam{lam:g}.npz")             # warm start from the CV fit
    U, V, tr = bl.fit(Y, k, lam, None, U0=z["U"], V0=z["V"], max_iter=max_iter, tol=tol,
                      log=open(OUT / "logs/final.log", "a"))
    np.savez(OUT / "fits/best_full.npz", U=U, V=V, obj=np.array(tr["obj"]), loglik=np.array(tr["loglik"]),
             k=k, lam=lam, iterations=tr["iterations"], converged=tr["converged"])
    print("saved best_full.npz", tr["iterations"], tr["converged"])


def cv_offsets(k, lams, max_iter=250, tol=1e-6):
    """Offset model Psi = a1' + 1b' + UV': lambda path, strong -> weak, warm-started."""
    Y, H = load()
    init = None
    with open(OUT / f"logs/cvoff_k{k}.log", "a") as log:
        for lam in sorted(lams, reverse=True):
            tag = f"off_k{k}_lam{lam:g}"
            if (OUT / f"tables/cv_{tag}.json").exists():
                z = np.load(OUT / f"fits/cv_{tag}.npz"); init = (z["U"], z["V"], z["a"], z["b"])
                continue
            t0 = time.time()
            U, V, a, b, tr = bl.fit_offsets(Y, k, lam, heldout=H, init=init, max_iter=max_iter, tol=tol, log=log)
            init = (U, V, a, b)
            sc = bl.heldout_scores_offsets(Y, U, V, a, b, H)
            res = dict(model="offsets", k=k, lam=lam, **sc, J=tr["obj"][-1], iterations=tr["iterations"],
                       converged=bool(tr["converged"]),
                       monotone=bool(np.all(np.diff(tr["obj"]) > -1e-7 * abs(tr["obj"][0]))),
                       seconds=time.time() - t0)
            np.savez(OUT / f"fits/cv_{tag}.npz", U=U, V=V, a=a, b=b, obj=np.array(tr["obj"]))
            (OUT / f"tables/cv_{tag}.json").write_text(json.dumps(res, indent=1))
            print(res, flush=True)


def final_offsets(k, lam, max_iter=500, tol=1e-7):
    Y, _ = load()
    z = np.load(OUT / f"fits/cv_off_k{k}_lam{lam:g}.npz")
    U, V, a, b, tr = bl.fit_offsets(Y, k, lam, init=(z["U"], z["V"], z["a"], z["b"]), max_iter=max_iter,
                                    tol=tol, log=open(OUT / "logs/final_offsets.log", "a"))
    np.savez(OUT / "fits/best_full_offsets.npz", U=U, V=V, a=a, b=b, obj=np.array(tr["obj"]), k=k, lam=lam,
             iterations=tr["iterations"], converged=tr["converged"])
    print("saved best_full_offsets.npz", tr["iterations"], tr["converged"])


def cv_cloglog(k, lams, max_iter=250, tol=1e-6):
    """cloglog model log(-log(1-p)) = a1' + 1b' + UV': lambda path, strong -> weak, warm-started."""
    Y, H = load()
    init = None
    with open(OUT / f"logs/cvcll_k{k}.log", "a") as log:
        for lam in sorted(lams, reverse=True):
            tag = f"cll_k{k}_lam{lam:g}"
            if (OUT / f"tables/cv_{tag}.json").exists():
                z = np.load(OUT / f"fits/cv_{tag}.npz"); init = (z["U"], z["V"], z["a"], z["b"])
                continue
            t0 = time.time()
            U, V, a, b, tr = bl.fit_cloglog(Y, k, lam, heldout=H, init=init, max_iter=max_iter, tol=tol, log=log)
            init = (U, V, a, b)
            sc = bl.heldout_scores_cloglog(Y, U, V, a, b, H)
            res = dict(model="cloglog", k=k, lam=lam, **sc, J=tr["obj"][-1], iterations=tr["iterations"],
                       converged=bool(tr["converged"]),
                       monotone=bool(np.all(np.diff(tr["obj"]) >= -1e-9 * abs(tr["obj"][0]))),
                       seconds=time.time() - t0)
            np.savez(OUT / f"fits/cv_{tag}.npz", U=U, V=V, a=a, b=b, obj=np.array(tr["obj"]))
            (OUT / f"tables/cv_{tag}.json").write_text(json.dumps(res, indent=1))
            print(res, flush=True)


def final_cloglog(k, lam, max_iter=500, tol=1e-7):
    Y, _ = load()
    z = np.load(OUT / f"fits/cv_cll_k{k}_lam{lam:g}.npz")
    U, V, a, b, tr = bl.fit_cloglog(Y, k, lam, init=(z["U"], z["V"], z["a"], z["b"]), max_iter=max_iter,
                                    tol=tol, log=open(OUT / "logs/final_cloglog.log", "a"))
    np.savez(OUT / "fits/best_full_cloglog.npz", U=U, V=V, a=a, b=b, obj=np.array(tr["obj"]), k=k, lam=lam,
             iterations=tr["iterations"], converged=tr["converged"])
    print("saved best_full_cloglog.npz", tr["iterations"], tr["converged"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
    elif cmd == "bench":
        bench(int(sys.argv[2]))
    elif cmd == "cv":
        cv(int(sys.argv[2]), [float(x) for x in sys.argv[3:]])
    elif cmd == "final":
        final(int(sys.argv[2]), float(sys.argv[3]))
