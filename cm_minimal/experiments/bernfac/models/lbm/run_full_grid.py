"""Full-data LBM model selection (resumable; one JSON per configuration).

Train = ~heldout. Restart chosen by training ELBO; configurations compared on VALIDATION;
TEST is reported only. Usage:  python run_full_grid.py [--max-hours 7.5]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi

from full_common import CELLS, OUT, lbm, load, spectral_svd

KR = [10, 23, 50, 100, 200]
KC = [10, 25, 50, 100, 200]
LAMS = [0.1, 1.0]
SETTINGS = dict(alpha_r=1.1, alpha_c=1.1, max_iter=300, tol=1e-6, label_sweeps=1,
                block_inner_steps=5, block_solver="pg", n_restarts=3, init="mixed", seed=1)


def cfg_name(Kr, Kc, lam):
    return f"Kr{Kr}_Kc{Kc}_lam{lam:g}"


def configs():
    out = [(1, 1, 1.0)]
    grid = [(kr, kc, lam) for kr in KR for kc in KC for lam in LAMS]
    grid.sort(key=lambda c: (c[0] * c[1], c[0], c[1], c[2]))  # cheap first
    return out + grid


def best_val_so_far(res_dir):
    best = np.inf
    for f in res_dir.glob("*.json"):
        r = json.loads(f.read_text())
        if r["Kr"] > 1 and r["val"]["nll"] < best:
            best = r["val"]["nll"]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, default=7.5)
    ap.add_argument("--only", default=None, help="comma list of config names")
    a = ap.parse_args()
    res_dir = OUT / "results"; res_dir.mkdir(parents=True, exist_ok=True)
    (OUT / "fits").mkdir(exist_ok=True); (OUT / "labels").mkdir(exist_ok=True)
    t_start = time.time()
    Y, val, test, data = load()
    svd = spectral_svd(data)
    cells = pd.read_csv(CELLS)
    hemi = cells["hemilineage"].to_numpy(); lib = cells["library"].to_numpy()
    ok = pd.notna(hemi)
    print(f"loaded in {time.time() - t_start:.1f}s; n={data.n} m={data.m} train obs={data.n_obs:.0f}",
          flush=True)
    todo = configs()
    if a.only:
        keep = set(a.only.split(",")); todo = [c for c in todo if cfg_name(*c) in keep]
    for Kr, Kc, lam in todo:
        name = cfg_name(Kr, Kc, lam)
        f = res_dir / f"{name}.json"
        if f.exists():
            continue
        if (time.time() - t_start) / 3600 > a.max_hours:
            print("time budget reached; stopping", flush=True); break
        print(f"== {name}", flush=True)
        monitor = lambda R, C, B: lbm.heldout_scores(  # noqa: E731  (diagnostic only, never used for stopping)
            lbm.LBMModel(R, C, B, None, None, lam, 1.1, 1.1), Y, val)["nll"]
        st = dict(SETTINGS)
        if Kr == 1 and Kc == 1:
            st.update(n_restarts=1, init="random")
        t0 = time.time()
        mdl = lbm.fit_lbm(data, Kr, Kc, lam=lam, spectral_svd=svd, monitor=monitor, monitor_every=50,
                          mono_tol=1e-6, **st)
        fit_time = time.time() - t0
        t1 = time.time()
        sv = lbm.heldout_scores(mdl, Y, val); stt = lbm.heldout_scores(mdl, Y, test)
        score_time = time.time() - t1
        lab = mdl.R.argmax(1).astype(np.int16)
        np.save(OUT / "labels" / f"{name}_rows.npy", lab)
        np.save(OUT / "labels" / f"{name}_cols.npy", mdl.C.argmax(1).astype(np.int16))
        np.savez(OUT / "fits" / f"{name}_globals.npz", B=mdl.B, pi=mdl.pi, rho=mdl.rho)
        rec = dict(name=name, Kr=Kr, Kc=Kc, lam=lam, settings=st,
                   train_elbo=mdl.info["elbo"], elbo_parts=mdl.info["parts"],
                   n_iter=mdl.info["n_iter"], status=mdl.info["status"],
                   restart_chosen=mdl.info["restart"], init_chosen=mdl.info["init"],
                   restarts=mdl.info["restarts"],
                   n_empty_rows=mdl.info["n_empty_rows"], n_empty_cols=mdl.info["n_empty_cols"],
                   n_rows_groups_used=int(len(np.unique(lab))),
                   n_col_groups_used=int(len(np.unique(mdl.C.argmax(1)))),
                   occupancy_rows=mdl.info["occupancy_rows"].tolist(),
                   occupancy_cols=mdl.info["occupancy_cols"].tolist(),
                   val=sv, test=stt, fit_time_s=fit_time, score_time_s=score_time,
                   ari_hemilineage=float(ari(hemi[ok], lab[ok])), nmi_hemilineage=float(nmi(hemi[ok], lab[ok])),
                   ari_library=float(ari(lib, lab)), nmi_library=float(nmi(lib, lab)))
        if Kr > 1 and sv["nll"] < best_val_so_far(res_dir):
            np.savez(OUT / "fits" / "best.npz", R=mdl.R.astype(np.float32), C=mdl.C.astype(np.float32),
                     B=mdl.B, pi=mdl.pi, rho=mdl.rho, name=name, lam=lam)
        f.write_text(json.dumps(rec, default=float))
        print(f"   val nll {sv['nll']:.5f} auc {sv['auc']:.4f}  test nll {stt['nll']:.5f}  "
              f"ARI hemi {rec['ari_hemilineage']:.3f}  iters {rec['n_iter']}  "
              f"fit {fit_time:.0f}s  total {(time.time() - t_start) / 3600:.2f}h", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
