"""Synthetic nonlinear-manifold demonstration (handoff 8.2, 8.4).

Generator (d = 2):  z_i ~ N(0, I_2),
    g_j(z) = b_j + a_j sin(q_j'z) + c_j cos(r_j'z) + e_j tanh(s_j'z),
with random per-column coefficients, rescaled so the logit standard deviation is about 2.5
(no universal saturation). The logit matrix has rank well above d. n = 250, m = 100,
all cells observed, entries split 70/15/15 into train / validation / test (seeded).

Models (float64, joint MAP, 3 restarts each, restart chosen by training objective F):
  neural affine head    (64, 64), d = 2, pg_gem, lambda_output in {1, 10} (chosen on validation)
  neural nonlinear head (64, 64), d = 2, pg_gem, lambda_output in {1, 10}
  linear factor model   hidden_widths=() i.e. psi = b + V z (the linear-network limit), rank
                        k in {1, 2, 3, 5, 8}, lambda_output in {1, 10}
  column-rate baseline  training-only smoothed column rates
  oracle                true logits (diagnostic only; never used for fitting)
Also a matched pg_gem vs direct_bce run (same init) for the selected affine model on seed 0.
Outputs: outputs/bernfac_20261020/models/neural/synthetic/
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bernoulli_neural import (MaskedBinary, NeuralConfig, NeuralManifoldModel, fit_bernoulli_neural_manifold,  # noqa: E402
                              initialize, predict_logits, scores_from_logit_fn)
from bernoulli_neural.fit import fit_single  # noqa: E402

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "outputs/bernfac_20261020/models/neural/synthetic"
OUT.mkdir(parents=True, exist_ok=True)
N, M, D = 250, 100, 2
SEEDS = [0, 1, 2, 3, 4]


def generate(seed):
    rng = np.random.default_rng(1000 + seed)
    z = rng.standard_normal((N, D))
    q, r, s = (rng.standard_normal((M, D)) * 1.3 for _ in range(3))
    a, c, e = (rng.standard_normal(M) for _ in range(3))
    b = rng.normal(-0.5, 0.7, M)
    psi = b + a * np.sin(z @ q.T) + c * np.cos(z @ r.T) + e * np.tanh(z @ s.T)
    psi = b + (psi - b) * (2.5 / (psi - b).std())
    Y = (rng.random((N, M)) < 1 / (1 + np.exp(-psi))).astype(np.uint8)
    u = rng.random((N, M))
    train, val, test = u < 0.70, (u >= 0.70) & (u < 0.85), u >= 0.85
    return dict(z=z, psi=psi, Y=Y, train=train, val=val, test=test)


def score(Y, mask, logits):
    return scores_from_logit_fn(Y, mask, lambda sl: logits[sl], block=10_000)


def fit(S, **kw):
    base = dict(dtype="float64", max_outer=400, tol=1e-7, n_restarts=3, seed=1, inner_steps=10, head_steps=5)
    base.update(kw)
    cfg = NeuralConfig(**base)
    t = time.time()
    model, hist = fit_bernoulli_neural_manifold(S["Y"].astype(float), cfg.latent_dim, cfg, observed_mask=S["train"])
    L = predict_logits(model).astype(np.float64)
    return dict(model=model, hist=hist, logits=L, val=score(S["Y"], S["val"], L), seconds=time.time() - t, cfg=cfg)


def main():
    rows, spectra, best_models = [], {}, {}
    for seed in SEEDS:
        S = generate(seed)
        np.savez_compressed(OUT / f"data_seed{seed}.npz", **S)
        cand = {}
        for lv in (1.0, 10.0):
            cand[("neural affine", lv, D)] = fit(S, latent_dim=D, hidden_widths=(64, 64), head_mode="affine",
                                                 lambda_output=lv)
            cand[("neural nonlinear", lv, D)] = fit(S, latent_dim=D, hidden_widths=(64, 64), head_mode="nonlinear",
                                                    lambda_output=lv)
            for k in (1, 2, 3, 5, 8):
                cand[("linear factor", lv, k)] = fit(S, latent_dim=k, hidden_widths=(), head_mode="affine",
                                                     lambda_output=lv)
        Ytr = S["Y"]
        rate = (np.where(S["train"], Ytr, 0).sum(0) + 0.5) / (S["train"].sum(0) + 1.0)
        base = np.broadcast_to(np.log(rate / (1 - rate)), (N, M))
        extra = {"column rate": base, "oracle (true logits)": S["psi"]}
        for (fam, lv, k), r in cand.items():
            rows.append(dict(seed=seed, family=fam, lambda_output=lv, latent_dim=k, val_nll=r["val"]["nll"],
                             final_F=r["hist"]["final_F"], iterations=r["hist"]["iteration"][-1],
                             termination=r["hist"]["termination"], monotone=r["hist"]["monotone_F"],
                             seconds=r["seconds"], selected=False))
        sel = {}
        for fam in ("neural affine", "neural nonlinear", "linear factor"):
            keys = [k for k in cand if k[0] == fam]
            kbest = min(keys, key=lambda k: cand[k]["val"]["nll"])         # validation selection only
            sel[fam] = kbest
            for rrow in rows:
                if rrow["seed"] == seed and (rrow["family"], rrow["lambda_output"], rrow["latent_dim"]) == kbest:
                    rrow["selected"] = True
        for fam, kbest in sel.items():
            r = cand[kbest]
            te = score(S["Y"], S["test"], r["logits"])
            s = np.linalg.svd(r["logits"], compute_uv=False)
            rows.append(dict(seed=seed, family=fam + " [selected]", lambda_output=kbest[1], latent_dim=kbest[2],
                             val_nll=r["val"]["nll"], test_nll=te["nll"], test_brier=te["brier"], test_auc=te["auc"],
                             logit_rmse_vs_truth=float(np.sqrt(np.mean((r["logits"] - S["psi"]) ** 2))),
                             eff_rank_99=int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), 0.99) + 1),
                             final_F=r["hist"]["final_F"], seconds=r["seconds"], selected=True))
            if seed == 0:
                spectra[fam] = s.tolist()
                best_models[fam] = r
        for name, L in extra.items():
            va, te = score(S["Y"], S["val"], L), score(S["Y"], S["test"], L)
            s = np.linalg.svd(L, compute_uv=False)
            rows.append(dict(seed=seed, family=name + " [selected]", val_nll=va["nll"], test_nll=te["nll"],
                             test_brier=te["brier"], test_auc=te["auc"],
                             logit_rmse_vs_truth=float(np.sqrt(np.mean((L - S["psi"]) ** 2))),
                             eff_rank_99=int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), 0.99) + 1),
                             selected=True))
            if seed == 0 and name.startswith("oracle"):
                spectra["true logits"] = s.tolist()
        print(pd.DataFrame([r for r in rows if r["seed"] == seed and r.get("test_nll") is not None])
              [["family", "lambda_output", "latent_dim", "val_nll", "test_nll", "test_auc", "test_brier"]].to_string(),
              flush=True)
        if seed == 0:
            # embeddings and matched optimizer comparison on seed 0
            np.save(OUT / "seed0_true_z.npy", S["z"])
            for fam in ("neural affine", "neural nonlinear"):
                np.save(OUT / f"seed0_Z_{fam.split()[1]}.npy", best_models[fam]["model"].Z.detach().numpy())
            kb = sel["neural affine"]
            traces = {}
            for opt in ("pg_gem", "direct_bce"):
                cfg = NeuralConfig(latent_dim=D, hidden_widths=(64, 64), head_mode="affine", lambda_output=kb[1],
                                   optimizer_mode=opt, dtype="float64", max_outer=3000, tol=1e-9, max_seconds=60,
                                   inner_steps=10, head_steps=5, seed=1, n_restarts=1)
                data = MaskedBinary(S["Y"].astype(float), S["train"])
                model = NeuralManifoldModel(cfg, N, M)
                initialize(model, data, seed=1)
                h = fit_single(model, data, cfg, grad_norm_every=0)
                L = predict_logits(model).astype(np.float64)
                traces[opt] = dict(seconds=h["seconds"], F=h["original_map_loss"], S_start=h["surrogate_start"],
                                   S_end=h["surrogate_end"], val_nll=score(S["Y"], S["val"], L)["nll"],
                                   test_nll=score(S["Y"], S["test"], L)["nll"], final_F=h["final_F"],
                                   iterations=h["iteration"][-1], termination=h["termination"])
                print(f"matched {opt}: F {h['final_F']:.4f} it {h['iteration'][-1]} {h['seconds'][-1]:.1f}s "
                      f"val {traces[opt]['val_nll']:.4f}", flush=True)
            (OUT / "seed0_optimizer_traces.json").write_text(json.dumps(traces))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "synthetic_all_fits.csv", index=False)
    s = d[d.family.str.endswith("[selected]")].groupby("family")[["test_nll", "test_brier", "test_auc",
                                                                  "logit_rmse_vs_truth", "eff_rank_99"]]
    summ = s.agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.sort_values("test_nll_mean")
    summ.to_csv(OUT / "synthetic_summary.csv")
    (OUT / "seed0_singular_values.json").write_text(json.dumps(spectra))
    print(summ.to_string())


if __name__ == "__main__":
    main()
