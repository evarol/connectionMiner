"""New-row experiment for the selected configuration (handoff 6.1, optional task).

1. Refit the selected configuration on the training rows only (the 1,646 new-row test cells are
   removed entirely; ~heldout mask on the remaining rows; same budget and settings).
2. Freeze the decoder; for each test cell infer z from its context columns (newrow_context.npy,
   50% of genes) with infer_neural_rows (pg_gem, 2 starts: z = 0 and one prior draw).
3. Score the remaining (non-context) columns of those cells: NLL, Brier, AUC (2M subsample), with
   the gene-rate baseline (training rows) and the z = 0 mode plug-in as references.
Output: results/newrow.json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

import run_full as rf
from bernoulli_neural import MaskedBinary, infer_neural_rows, scores_from_logit_fn
from bernoulli_neural.fit import fit_single


def main(budget=1500.0):
    best = json.loads((rf.OUT / "best.json").read_text())
    name = best["selected"]
    spec = rf.parse(name)
    Y = np.load(rf.DATA / "Y.npy")
    H = np.load(rf.DATA / "heldout.npy")
    rows = np.load(rf.DATA / "newrow_test_rows.npy")
    ctx = np.load(rf.DATA / "newrow_context.npy")
    keep = np.setdiff1d(np.arange(Y.shape[0]), rows)
    Ytr = np.ascontiguousarray(Y[keep])
    data = MaskedBinary.from_trusted(Ytr, ~H[keep])
    del H
    from bernoulli_neural import NeuralConfig, NeuralManifoldModel, initialize
    cfg = NeuralConfig.from_dict({**best["config"], "max_seconds": budget})
    model = NeuralManifoldModel(cfg, data.n, data.m)
    # training-row-only SVD start (the cached full-data start would use the test rows' cells)
    from sklearn.utils.extmath import randomized_svd
    Zp = np.where(data.mask.numpy(), 4.0 * (Ytr.astype(np.float32) - 0.5), np.float32(0))
    Us, _, _ = randomized_svd(Zp, cfg.latent_dim, n_iter=4, random_state=0)
    del Zp
    Z0 = Us * np.sqrt(Us.shape[0])
    initialize(model, data, seed=cfg.seed, init_Z=Z0)
    log = open(rf.LOGS / "newrow.log", "a")
    t0 = time.time()
    hist = fit_single(model, data, cfg, log=lambda m: (print(m, flush=True), print(m, file=log, flush=True)),
                      grad_norm_every=0)
    fit_sec = time.time() - t0
    del data
    Ynew = Y[rows].astype(np.float32)
    t1 = time.time()
    Znew, diag = infer_neural_rows(model, Ynew, ctx, method="pg_gem", n_starts=2, max_iter=80, tol=1e-6)
    inf_sec = time.time() - t1
    ev = ~ctx
    Yn = Y[rows]
    with torch.no_grad():
        Lz = model.decoder(torch.as_tensor(Znew, dtype=model.Z.dtype)).numpy()
        L0 = model.decoder(torch.zeros(1, cfg.latent_dim, dtype=model.Z.dtype)).numpy()
    tr_rate = (Ytr.sum(0) + 0.5) / (Ytr.shape[0] + 1.0)
    Lg = np.log(tr_rate / (1 - tr_rate)).astype(np.float32)
    out = dict(selected=name, n_test_rows=int(len(rows)), context_fraction=float(ctx.mean()),
               fit_seconds=fit_sec, inference_seconds=inf_sec, fit_iterations=hist["iteration"][-1],
               fit_termination=hist["termination"], inference=dict(method=diag["method"], n_starts=2,
                                                                     rows_without_context=diag["rows_without_context"]))
    for k, L in (("neural_inferred_z", Lz), ("neural_z0_mode_plugin", np.broadcast_to(L0, Lz.shape)),
                 ("gene_rate_training_rows", np.broadcast_to(Lg, Lz.shape))):
        s = scores_from_logit_fn(Yn, ev, lambda sl, L=L: np.asarray(L[sl]))
        s.pop("calibration")
        out[k] = s
        print(k, s, flush=True)
    np.save(rf.OUT / "newrow_Z.npy", Znew.astype(np.float32))
    (rf.RES / "newrow.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1500.0)
