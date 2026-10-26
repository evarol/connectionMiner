"""Full-data model selection for the neural-manifold auto-decoder (16,460 cells x 19,543 genes).

Train = ~heldout. Hyperparameters are chosen on VALIDATION; TEST is scored once per finished
fit and is never used for any decision (restart choice uses the training objective F).
float32 on CPU for these runs (float64 is the reference for the tests and synthetic demo);
all objective sums are accumulated in float64.

Usage:
  python run_full.py --time                       # one outer iteration per mode, memory estimate
  python run_full.py --configs NAME [NAME ...]    # run named configs (resumable; skips finished)
  python run_full.py --table                      # rebuild tables/selection.csv from results/
Config name: {opt}_{head}_d{d}_w{w}_lv{lambda_output}_r{restart}, e.g. pg_affine_d10_w64_lv10_r0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "3")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bernoulli_neural import (MaskedBinary, NeuralConfig, NeuralManifoldModel, heldout_scores,  # noqa: E402
                              initialize, save_model)
from bernoulli_neural.decoder import invert_link  # noqa: E402
from bernoulli_neural.fit import fit_single  # noqa: E402

torch.set_num_threads(3)
ROOT = Path(__file__).resolve().parents[5]
DATA = ROOT / "outputs/bernfac_20261020/full/data"
OUT = ROOT / "outputs/bernfac_20261020/models/neural"
RES, FITS, LOGS, TAB = OUT / "results", OUT / "fits", OUT / "logs", OUT / "tables"
for p in (RES, FITS, LOGS, TAB):
    p.mkdir(parents=True, exist_ok=True)

# defaults for the full-data runs (documented in README / best.json)
FULL_DEFAULTS = dict(dtype="float32", lambda_z=1.0, lambda_hidden=0.01, lambda_bias=0.01, max_outer=200,
                     tol=2e-5, inner_steps=10, inner_rounds=1, head_steps=2, resolve_head=True, n_restarts=1,
                     chunk_rows=2048, val_every=1, max_trials=12)
# gradient-only inner loops (nonlinear head, direct_bce) need a full streamed pass per trial, so they
# take fewer inner steps per outer iteration
GRAD_MODE_OVERRIDES = dict(inner_steps=3, head_steps=2)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # macOS: bytes


def parse(name):
    m = re.fullmatch(r"(pg|direct)_(affine|nonlinear)_d(\d+)_w(\d+)_lv([\d.]+)_r(\d+)(?:_(\w+))?", name)
    if not m:
        raise ValueError(name)
    opt, head, d, w, lv, r, tag = m.groups()
    return dict(optimizer_mode="pg_gem" if opt == "pg" else "direct_bce", head_mode=head, latent_dim=int(d),
                hidden_widths=(int(w), int(w)), lambda_output=float(lv), restart=int(r), tag=tag)


_CACHE = {}


def load_data():
    if "data" in _CACHE:
        return _CACHE["data"]
    Y = np.load(DATA / "Y.npy")
    H = np.load(DATA / "heldout.npy")
    V = np.load(DATA / "val.npy")
    T = np.load(DATA / "test.npy")
    data = MaskedBinary.from_trusted(Y, ~H)
    del H
    _CACHE["data"] = (Y, V, T, data)
    return _CACHE["data"]


def svd_init(Y, data, dmax=20):
    """Training-only start: top singular vectors of 4(Y - 1/2) with held-out cells set to 0,
    scaled to unit variance per coordinate (matches the N(0, 1/lambda_z) prior scale at lambda_z=1)."""
    f = FITS / f"svd_init_d{dmax}.npy"
    if f.exists():
        return np.load(f)
    from sklearn.utils.extmath import randomized_svd
    Zp = np.where(data.mask.numpy(), 4.0 * (Y.astype(np.float32) - 0.5), np.float32(0))
    Us, s, _ = randomized_svd(Zp, dmax, n_iter=4, random_state=0)
    del Zp
    Z0 = (Us * np.sqrt(Us.shape[0])).astype(np.float64)
    np.save(f, Z0)
    np.save(FITS / f"svd_init_d{dmax}_singular_values.npy", s)
    return Z0


def make_model(spec, data, Y, overrides=None):
    kw = dict(FULL_DEFAULTS)
    kw.update(dict(latent_dim=spec["latent_dim"], hidden_widths=spec["hidden_widths"], head_mode=spec["head_mode"],
                   optimizer_mode=spec["optimizer_mode"], lambda_output=spec["lambda_output"],
                   seed=1 + spec["restart"]))
    if spec["head_mode"] == "nonlinear" or spec["optimizer_mode"] == "direct_bce":
        kw.update(GRAD_MODE_OVERRIDES)
    kw.update(overrides or {})
    cfg = NeuralConfig(**kw)
    model = NeuralManifoldModel(cfg, data.n, data.m)
    Z0 = svd_init(Y, data)[:, :cfg.latent_dim]
    initialize(model, data, seed=cfg.seed, init_Z=Z0)
    return model


def warm_start_nonlinear(model, src_path):
    """Initialise a nonlinear-head model from a fitted affine model (training-only information):
    Z and hidden layers copied; t = b + V h chosen so that t + rho tanh(t) matches the affine logits
    at the column mean of h is not linear, so we use the simple map V/(1+rho), b -> invert_link(b)."""
    from bernoulli_neural import load_model
    src, _, _ = load_model(src_path)
    rho = model.config.nonlinear_head_rho
    with torch.no_grad():
        model.Z.copy_(src.Z)
        for p, q in zip(model.decoder.hidden.parameters(), src.decoder.hidden.parameters()):
            p.copy_(q)
        model.decoder.V.copy_(src.decoder.V / (1 + rho))
        model.decoder.b.copy_(invert_link(src.decoder.b.to(torch.float64), rho).to(model.Z.dtype))


def run_config(name, max_seconds=None, overrides=None, warm=None, force=False):
    out_json = RES / f"{name}.json"
    if out_json.exists() and not force:
        print(f"skip {name} (done)")
        return json.loads(out_json.read_text())
    spec = parse(name)
    Y, V, T, data = load_data()
    ov = dict(overrides or {})
    if max_seconds is not None:
        ov["max_seconds"] = max_seconds
    model = make_model(spec, data, Y, ov)
    if warm:
        warm_start_nonlinear(model, FITS / f"{warm}.pt")
    logf = open(LOGS / f"{name}.log", "a")

    def log(msg):
        print(msg, flush=True)
        print(msg, file=logf, flush=True)

    log(f"== {name}  cfg={model.config.to_dict()}  warm={warm}")
    eval_sets = {"val": (torch.from_numpy(Y), torch.from_numpy(V))}
    t0 = time.time()
    hist = fit_single(model, data, model.config, eval_sets=eval_sets, log=log, grad_norm_every=10)
    fit_sec = time.time() - t0
    t1 = time.time()
    sv = heldout_scores(model, Y, V)
    st = heldout_scores(model, Y, T)
    score_sec = time.time() - t1
    res = dict(name=name, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()},
               warm_start=warm, val_nll=sv["nll"], val_brier=sv["brier"], val_auc=sv["auc"],
               test_nll=st["nll"], test_brier=st["brier"], test_auc=st["auc"],
               final_F=hist["final_F"], final_nll_train=hist["original_nll"][-1],
               iterations=hist["iteration"][-1], termination=hist["termination"],
               monotone=hist["monotone_F"], rejected_outer=hist["rejected_outer"],
               z_unit_accepted=int(np.sum(hist["z_unit_accepted"])), z_unit_rejected=int(np.sum(hist["z_unit_rejected"])),
               hidden_accepted=int(np.sum(hist["hidden_accepted"])), hidden_rejected=int(np.sum(hist["hidden_rejected"])),
               head_unit_accepted=int(np.sum(hist["head_unit_accepted"])),
               head_unit_rejected=int(np.sum(hist["head_unit_rejected"])),
               head_solves=int(np.sum(hist["head_solves"])), seconds=fit_sec, score_seconds=score_sec,
               max_rss_gb=rss_gb(), val_calibration=sv["calibration"], test_calibration=st["calibration"],
               config=model.config.to_dict())
    save_model(FITS / f"{name}.pt", model, hist, extra=dict(val=sv, test=st))
    out_json.write_text(json.dumps(res, indent=1, default=float))
    log(f"done {name}: val nll {sv['nll']:.5f} auc {sv['auc']:.4f} | test nll {st['nll']:.5f} "
        f"| F {hist['final_F']:.6e} it {hist['iteration'][-1]} ({hist['termination']}) {fit_sec / 60:.1f} min "
        f"rss {rss_gb():.1f} GB")
    logf.close()
    return res


def build_table():
    import pandas as pd
    rows = [json.loads(f.read_text()) for f in sorted(RES.glob("*.json"))]
    if not rows:
        return None
    d = pd.DataFrame(rows).drop(columns=["val_calibration", "test_calibration", "config"], errors="ignore")
    d["hidden_widths"] = d["hidden_widths"].map(lambda w: "x".join(map(str, w)))
    d["config"] = d["name"].str.replace(r"_r\d+$", "", regex=True)
    # restart choice within a configuration: lowest training objective F (never a held-out score)
    d["chosen_restart"] = False
    d.loc[d.groupby("config").final_F.idxmin(), "chosen_restart"] = True
    cols = ["config", "name", "optimizer_mode", "head_mode", "latent_dim", "hidden_widths", "lambda_output",
            "restart", "chosen_restart", "warm_start", "val_nll", "val_auc", "val_brier", "test_nll", "test_auc",
            "test_brier", "final_F", "final_nll_train", "iterations", "termination", "monotone", "rejected_outer",
            "z_unit_accepted", "z_unit_rejected", "hidden_accepted", "hidden_rejected", "head_unit_accepted",
            "head_unit_rejected", "head_solves", "seconds", "max_rss_gb"]
    d = d[[c for c in cols if c in d.columns]].sort_values("val_nll")
    d.to_csv(TAB / "selection.csv", index=False)
    return d


def time_modes():
    """Time one outer iteration of each mode at (64,64), d=10; print memory/runtime estimates."""
    Y, V, T, data = load_data()
    eval_sets = {"val": (torch.from_numpy(Y), torch.from_numpy(V))}
    print(f"n={data.n} m={data.m} train cells={data.n_obs:,}  rss {rss_gb():.2f} GB", flush=True)
    rows = []
    for name in ["pg_affine_d10_w64_lv10_r0", "pg_affine_d10_w128_lv10_r0", "pg_nonlinear_d10_w64_lv10_r0",
                 "direct_affine_d10_w64_lv10_r0"]:
        spec = parse(name)
        model = make_model(spec, data, Y, dict(max_outer=2, tol=0.0))
        t = time.time()
        h = fit_single(model, data, model.config, eval_sets=eval_sets, log=print, grad_norm_every=0)
        per = (time.time() - t) / 2
        rows.append((name, per, rss_gb()))
        print(f"TIMING {name}: {per:.1f} s per outer iteration (incl. fused F/S/val pass); "
              f"F {h['original_map_loss']}; max rss so far {rss_gb():.1f} GB", flush=True)
        del model
    for name, per, r in rows:
        print(f"{name:34s} {per:7.1f} s/iter -> 60 iters {per * 60 / 60:.0f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--warm", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    a = ap.parse_args()
    ov = {}
    for kv in a.set:
        k, v = kv.split("=")
        ov[k] = type(FULL_DEFAULTS.get(k, 1.0))(v) if k in FULL_DEFAULTS else float(v)
    if a.time:
        time_modes()
    for c in a.configs:
        run_config(c, a.max_seconds, ov, a.warm)
        build_table()
    if a.table:
        print(build_table().to_string())
