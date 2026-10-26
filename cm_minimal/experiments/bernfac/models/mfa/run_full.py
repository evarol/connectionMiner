"""Full-data model selection for the Bernoulli mixture of local factor models.

Data: outputs/bernfac_20261020/full/data (16,460 cells x 19,543 genes, binary detection).
Training cells = ~heldout. Validation (val.npy) selects; test (test.npy) is reported only.
Both are disjoint halves of heldout, so neither ever enters a fit.

Usage (3 BLAS / numba threads; long runs under nohup):
    python run_full.py probe            # time + memory for one config
    python run_full.py stage1 ...       # run a stage (resumable: finished configs are skipped)
    python run_full.py summarize        # selection table, best.json, embedding, responsibilities

Each finished configuration writes runs/<tag>.json (scores, traces, timings) and
fits/<tag>.npz (globals, float32) + fits/<tag>_local.npz (responsibilities, coordinate
means). xi is never written to disk (RAM only), so an interrupted configuration restarts
from scratch; finished configurations are never recomputed.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
import mfa  # noqa: E402

DATA = ROOT / "outputs/bernfac_20261020/full/data"
OUT = ROOT / "outputs/bernfac_20261020/models/mfa"
RUNS, FITS, LOGS, TABLES, CACHE = (OUT / s for s in ("runs", "fits", "logs", "tables", "cache"))
for p in (RUNS, FITS, LOGS, TABLES, CACHE):
    p.mkdir(parents=True, exist_ok=True)

DEFAULTS = dict(max_iter=40, tol=1e-5, init="kmeans", seed=1, lam=1.0, t=0.25, warm=None, suffix=None,
                diag_every=10)


def tag_of(c):
    s = f"{c['mode']}_K{c['K']}_d{c['d']}_lam{c['lam']:g}"
    if c["mode"] == "iso":
        s += f"_t{c['t']:g}"
    s += f"_s{c['seed']}"
    if c.get("suffix"):
        s += c["suffix"]
    elif c.get("warm"):
        s += "_warm"  # warm-started from the best stage-1 base fit (recorded in config["warm"])
    return s


def load_data():
    Y = np.load(DATA / "Y.npy")
    H = np.load(DATA / "heldout.npy")
    return Y, H


def get_pcs(Y, H):
    f = CACHE / "train_pcs30.npy"
    if f.exists():
        return np.load(f)
    pcs = mfa.training_pcs(Y, H, 30, seed=0)
    np.save(f, pcs.astype(np.float32))
    return pcs


def options(c):
    mode = {"iso": "isotropic", "isoL": "isotropic", "none": "none"}[c["mode"]]
    return mfa.MFAOptions(
        residual_mode=mode, residual_variance=c["t"] if c["mode"] == "iso" else 0.25,
        learn_residual_variance=c["mode"] == "isoL", lambda_mu=c["lam"], lambda_B=c["lam"],
        max_iter=c["max_iter"], tol=c["tol"], seed=c["seed"], init=c["init"],
        big_dtype="float32", xi_dtype="float16", chunk_budget_mb=400.0, mono_tol=1e-7, px=True)


def warm_model(c, opts):
    z = np.load(FITS / f"{c['warm']}.npz")
    t = np.full(c["K"], c["t"] if c["mode"] == "iso" else (0.25 if c["mode"] == "isoL" else 0.0))
    if c["mode"] == "isoL" and "t" in z and np.all(z["t"] > 0):
        t = z["t"].astype(float)
    return mfa.MFAModel(z["pi"].astype(float), z["mu"].astype(float), z["B"].astype(float), t, opts)


def run_config(c, Y, H, pcs):
    c = {**DEFAULTS, **c}
    diag_every = c["diag_every"]
    tag = tag_of(c)
    if (RUNS / f"{tag}.json").exists():
        print(f"skip {tag} (done)", flush=True)
        return json.loads((RUNS / f"{tag}.json").read_text())
    opts = options(c)
    est = mfa.memory_estimate(*Y.shape, c["K"], c["d"], opts)
    log = open(LOGS / f"{tag}.log", "a")
    msg = f"=== {tag}  memory estimate {est['total']:.2f} GB (+0.64 GB data)  {time.ctime()}"
    print(msg, flush=True); print(msg, file=log, flush=True)
    val = np.load(DATA / "val.npy", mmap_mode="r")
    diag = []

    def cb(it, model, state, info):
        if diag_every and it > 0 and it % diag_every == 0:
            s = mfa.score_cells(Y, val, model, state, auc_sample=200_000)
            diag.append(dict(it=it, val_nll=s["nll"]))
            print(f"    [diagnostic only, not used for stopping] it={it} val_nll={s['nll']:.5f}", file=log, flush=True)

    model0 = warm_model(c, opts) if c.get("warm") else None
    t0 = time.time()
    model, state, info = mfa.fit(Y, c["K"], c["d"], excluded=H, opts=opts, model0=model0, pcs=pcs,
                                 log=log, callback=cb)
    fit_s = time.time() - t0
    t1 = time.time()
    sv = mfa.score_cells(Y, val, model, state)
    st = mfa.score_cells(Y, np.load(DATA / "test.npy", mmap_mode="r"), model, state)
    score_s = time.time() - t1
    np.savez(FITS / f"{tag}.npz", pi=model.pi, mu=model.mu.astype(np.float32), B=model.B.astype(np.float32),
             t=model.t, opts=json.dumps(mfa.asdict(model.opts)))
    np.savez(FITS / f"{tag}_local.npz", r=state.r.astype(np.float32), m=state.m.astype(np.float16))
    rec = dict(tag=tag, config=c, val=sv, test=st, bound=info["bound"][-1], bound_trace=info["bound"],
               rel_trace=info["rel"], iterations=info["iterations"], converged=info["converged"],
               monotone=info["monotone"], max_rel_decrease=info["max_rel_decrease"],
               occupancy=info["occupancy"][-1], t_final=model.t.tolist(), t_trace=info["t"],
               fit_seconds=fit_s, score_seconds=score_s, init_seconds=info["init_seconds"],
               seconds_per_pass=float(np.mean(np.diff(info["seconds"]))) if len(info["seconds"]) > 1 else None,
               memory_estimate_gb=est, maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
               val_nll_diagnostic=diag, pi=model.pi.tolist())
    (RUNS / f"{tag}.json").write_text(json.dumps(rec, indent=1))
    msg = (f"--- {tag} done: B={rec['bound']:.6e} it={info['iterations']} conv={info['converged']} "
           f"mono={info['monotone']} val_nll={sv['nll']:.5f} test_nll={st['nll']:.5f} val_auc={sv['auc']:.4f} "
           f"fit={fit_s / 60:.1f} min")
    print(msg, flush=True); print(msg, file=log, flush=True)
    log.close()
    del model, state
    return rec


def done_records():
    return [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]


def best_by_val(recs, pred=lambda r: True):
    recs = [r for r in recs if pred(r)]
    return min(recs, key=lambda r: r["val"]["nll"]) if recs else None


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
def _base(recs):
    """iso, t = 0.25, lambda = 1, fresh (not warm-started) seed-1 runs."""
    return [r for r in recs if r["config"]["mode"] == "iso" and r["config"]["t"] == 0.25
            and r["config"]["lam"] == 1.0 and not r["config"].get("warm") and r["config"]["seed"] == 1]


def stage_configs(name, recs):
    """Staged, validation-driven grid (pruned to fit ~10 h on a shared machine).

    stage1 : iso t=0.25, lambda=1, 30 iterations: K in {1,4,8,16} at d=10, and d in {5,20} at K in {1,4}.
    stage2a: at the best K of stage1 (if not 1 or 4), the d that looked best at K=4 (or d=20 if
             d=10 won there), 30 iterations.
    stage2b: K=1 comparisons, fresh, 30 iterations: residual 'none', learned t, t=0.1, t=1, lambda=10.
             Warm-started variants of the best base fit (K*, d*), 15 further iterations each:
             iso t=0.25 (control), t=0.1, t=1, learned t, residual 'none', lambda=10.
    stage3 : second restart (seed 2) of the best base configuration and of the best K=1 base
             configuration; the restart is chosen by the training bound, never by validation.
    """
    cs = []
    if name == "stage1":
        for K in (1, 4, 8, 16):
            cs.append(dict(mode="iso", K=K, d=10, max_iter=30))
        for K in (1, 4):
            for d in (5, 20):
                cs.append(dict(mode="iso", K=K, d=d, max_iter=30))
    elif name == "stage2a":
        b = best_by_val(_base(recs))["config"]
        k4 = best_by_val(_base(recs), lambda r: r["config"]["K"] == 4)
        dk4 = k4["config"]["d"] if k4 else 10
        if b["K"] not in (1, 4):
            cs.append(dict(mode="iso", K=b["K"], d=dk4 if dk4 != 10 else 20, max_iter=30))
    elif name == "stage2b":
        b = best_by_val(_base(recs))
        K, d = b["config"]["K"], b["config"]["d"]
        d1 = best_by_val(_base(recs), lambda r: r["config"]["K"] == 1)["config"]["d"]
        for v in (dict(mode="none"), dict(mode="isoL"), dict(t=0.1), dict(t=1.0), dict(lam=10.0)):
            cs.append({**dict(mode="iso", K=1, d=d1, max_iter=30), **v})
        for v in (dict(), dict(t=0.1), dict(t=1.0), dict(mode="isoL"), dict(mode="none"), dict(lam=10.0)):
            cs.append({**dict(mode="iso", K=K, d=d, max_iter=15, warm=b["tag"]), **v})
    elif name == "stage3":
        b = best_by_val(_base(recs))["config"]
        cs.append({**b, "seed": 2})
        k1 = best_by_val(_base(recs), lambda r: r["config"]["K"] == 1)["config"]
        cs.append({**k1, "seed": 2})
    elif name == "stage4a":
        # added after stage 3 because residual 'none' won at K=1 and K=4: fresh 'none' fits over K
        for K in (4, 8, 16):
            cs.append(dict(mode="none", K=K, d=20, max_iter=30))
    elif name == "stage4b":
        nones = [r for r in recs if r["config"]["mode"] == "none" and not r["config"].get("warm")
                 and r["config"]["seed"] == 1 and r["config"]["lam"] == 1.0]
        bn = best_by_val(nones)
        c = bn["config"]
        cs.append({**c, "lam": 10.0})
        cs.append({**c, "seed": 2})
        # convergence check: continue the best fresh 'none' and best fresh iso fits for 70 more
        # iterations (100 in total) from their 30-iteration fits
        cs.append({**c, "max_iter": 70, "warm": bn["tag"], "suffix": "_cont70"})
        bi = best_by_val(_base(recs))
        cs.append({**bi["config"], "max_iter": 70, "warm": bi["tag"], "suffix": "_cont70"})
    elif name == "stage5":
        # 'none' K=1 over d: d in {5, 10} completes the grid; d=40 is an out-of-grid boundary
        # check (reported, excluded from selection) because d=20 was the largest grid value
        for d in (5, 10, 40):
            cs.append(dict(mode="none", K=1, d=d, max_iter=30))
    elif name == "stage6":
        # CSO follow-up (22 Oct): formal extension of the d grid, run to tol 1e-5 or 150 iterations
        for K, d in ((1, 40), (1, 80), (4, 40), (4, 80)):
            cs.append(dict(mode="none", K=K, d=d, max_iter=150, suffix="_it150", diag_every=25))
        cs.append(dict(mode="iso", K=1, d=40, max_iter=150, suffix="_it150", diag_every=25))
    return cs


def summarize():
    import pandas as pd
    recs = done_records()
    rows = []
    for r in recs:
        c = r["config"]
        rows.append(dict(tag=r["tag"], residual_mode=c["mode"], K=c["K"], d=c["d"], lam=c["lam"],
                         t_init=c["t"] if c["mode"] != "none" else 0.0,
                         t_final=float(np.mean(r["t_final"])), seed=c["seed"], warm=bool(c.get("warm")),
                         variant=c.get("suffix") or ("_warm" if c.get("warm") else "fresh"),
                         warm_from=c.get("warm") or "", total_iterations=r["iterations"] + (
                             next((q["iterations"] for q in recs if q["tag"] == c.get("warm")), 0)),
                         val_nll=r["val"]["nll"], val_auc=r["val"]["auc"], val_brier=r["val"]["brier"],
                         test_nll=r["test"]["nll"], test_auc=r["test"]["auc"], test_brier=r["test"]["brier"],
                         bound=r["bound"], iterations=r["iterations"], converged=r["converged"],
                         monotone=r["monotone"], max_rel_decrease=r["max_rel_decrease"],
                         fit_seconds=r["fit_seconds"], seconds_per_pass=r["seconds_per_pass"],
                         maxrss_gb=r["maxrss_gb"], min_occupancy=min(r["occupancy"])))
    df = pd.DataFrame(rows)
    # restart choice within a configuration: largest TRAINING bound
    key = ["residual_mode", "K", "d", "lam", "t_init", "variant"]
    df["chosen_restart"] = False
    for _, g in df.groupby(key):
        df.loc[g["bound"].idxmax(), "chosen_restart"] = True
    df = df.sort_values("val_nll")
    df.to_csv(TABLES / "selection.csv", index=False)
    df["in_grid"] = df.d.isin([5, 10, 20, 40, 80])  # d grid extended to 40, 80 on 22 Oct (CSO)
    df.to_csv(TABLES / "selection.csv", index=False)
    chosen = df[df.chosen_restart & df.in_grid]
    best = chosen.iloc[0]
    kgt1 = chosen[chosen.K > 1].iloc[0]
    iso = chosen[chosen.residual_mode != "none"].iloc[0]
    oog = df[~df.in_grid]
    k1 = chosen[chosen.K == 1].iloc[0] if (chosen.K == 1).any() else None
    none = chosen[chosen.residual_mode == "none"].iloc[0] if (chosen.residual_mode == "none").any() else None
    rec = next(r for r in recs if r["tag"] == best.tag)
    out = dict(best_tag=best.tag, config=rec["config"], val=rec["val"], test=rec["test"], bound=rec["bound"],
               iterations=rec["iterations"], converged=rec["converged"], monotone=rec["monotone"],
               pi=rec["pi"], t_final=rec["t_final"],
               best_K1=None if k1 is None else k1.to_dict(), best_none=None if none is None else none.to_dict(),
               best_K_gt_1=kgt1.to_dict(), best_isotropic=iso.to_dict(),
               out_of_grid_boundary_check=oog.to_dict(orient="records"),
               selection_rule="lowest validation NLL among configurations; within a configuration the restart "
                              "with the largest final training bound",
               embedding="sum_l r_il (mu_l + B_l m_il), centred over cells, exact SVD of its rank <= K(d+1) "
                         "factorisation, top 50 (or fewer) components scaled by singular values")
    # global parameters, responsibilities, coordinate means and embedding of the selected model
    z = np.load(FITS / f"{best.tag}.npz")
    loc = np.load(FITS / f"{best.tag}_local.npz")
    np.savez(OUT / "fits/best.npz", pi=z["pi"], mu=z["mu"], B=z["B"], t=z["t"], opts=z["opts"])
    np.save(OUT / "responsibilities.npy", loc["r"].astype(np.float32))
    np.save(OUT / "coordinate_means.npy", loc["m"].astype(np.float32))
    model = mfa.MFAModel(z["pi"].astype(float), z["mu"].astype(float), z["B"].astype(float), z["t"].astype(float))
    st = mfa._empty_state(loc["r"].shape[0], model.K, model.d)
    st.r, st.m = loc["r"].astype(float), loc["m"].astype(float)
    lk = np.load(FITS / f"{kgt1.tag}_local.npz")
    np.save(OUT / "responsibilities_best_K_gt_1.npy", lk["r"].astype(np.float32))
    out["responsibilities_best_K_gt_1_tag"] = kgt1.tag
    out["pi_best_K_gt_1"] = np.load(FITS / f"{kgt1.tag}.npz")["pi"].tolist()
    emb, s, frac = mfa.logit_mean_embedding(model, st, 50)
    np.save(OUT / "embedding.npy", emb)
    out["embedding_dims"] = int(emb.shape[1])
    out["embedding_variance_fraction"] = frac
    (OUT / "best.json").write_text(json.dumps(out, indent=1, default=float))
    print(df.head(30).to_string())
    print(json.dumps({k: out[k] for k in ("best_tag", "embedding_dims")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--mode", default="iso")
    ap.add_argument("--iters", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "summarize":
        summarize()
        return
    Y, H = load_data()
    pcs = get_pcs(Y, H)
    if a.cmd == "probe":
        c = {**DEFAULTS, **dict(mode=a.mode, K=a.K, d=a.d, max_iter=a.iters, tol=0.0)}
        opts = options(c)
        print("memory estimate (GB):", mfa.memory_estimate(*Y.shape, a.K, a.d, opts), flush=True)
        t0 = time.time()
        model, state, info = mfa.fit(Y, a.K, a.d, excluded=H, opts=opts, pcs=pcs, log=sys.stdout)
        print(f"init {info['init_seconds']:.1f}s, seconds per pass {np.diff([0.0] + info['seconds']).round(1)}")
        t1 = time.time()
        s = mfa.score_cells(Y, np.load(DATA / "val.npy", mmap_mode="r"), model, state)
        print(f"val scoring {time.time() - t1:.1f}s nll={s['nll']:.5f} auc={s['auc']:.4f}")
        print(f"maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB, total {time.time() - t0:.0f}s")
        return
    for st in a.cmd.split(","):
        cs = stage_configs(st, done_records())
        print(f"##### {st}: {len(cs)} configs", flush=True)
        for c in cs:
            try:
                run_config(c, Y, H, pcs)
            except Exception as e:  # keep the grid going; the failure is logged
                import traceback
                print(f"!!! config {c} failed: {e!r}", flush=True)
                traceback.print_exc()


if __name__ == "__main__":
    main()
