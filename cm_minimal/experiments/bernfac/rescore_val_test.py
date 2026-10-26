"""Rescore every earlier full-data fit on the validation / test halves of the held-out set.

Earlier BernFac fits (logit, logit + offsets, cloglog + offsets) were trained on the same
train mask (~heldout), so their saved CV fits can be rescored without refitting.
Selection is redone on VALIDATION only; TEST is reported once for the selected configs.
Output: outputs/bernfac_20261020/models/rescored/selection.csv
"""
import glob
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd

import bernfac_large as bl
from bernfac import sigmoid

FULL = Path(__file__).resolve().parents[3] / "outputs/bernfac_20261020/full"
OUT = FULL.parent / "models/rescored"


def scores_from_logit(Y, M, logit_rows, block=2048, link="logit"):
    """NLL / Brier / AUC on mask M for a row-block logit function."""
    from sklearn.metrics import roc_auc_score
    nll = brier = 0.0; cnt = 0; ys, ps = [], []
    for s in range(0, Y.shape[0], block):
        Mb = M[s:s + block]
        if not Mb.any():
            continue
        eta = logit_rows(slice(s, s + block))[Mb].astype(np.float64)
        y = Y[s:s + block][Mb].astype(np.float64)
        if link == "logit":
            ll = y * eta - np.logaddexp(0, eta); p = sigmoid(eta)
        else:
            ll, _, _ = bl.cll_terms(eta, y); ll = ll.astype(np.float64)
            p = -np.expm1(-np.exp(np.clip(eta, bl.ETA_MIN, bl.ETA_MAX)))
        nll -= ll.sum(); brier += np.sum((p - y) ** 2); cnt += y.size
        ys.append(y.astype(np.int8)); ps.append(p.astype(np.float32))
    y, p = np.concatenate(ys), np.concatenate(ps)
    idx = np.random.default_rng(0).choice(y.size, min(2_000_000, y.size), replace=False)
    return dict(nll=nll / cnt, brier=brier / cnt, auc=float(roc_auc_score(y[idx], p[idx])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Y = np.load(FULL / "data/Y.npy"); V = np.load(FULL / "data/val.npy"); T = np.load(FULL / "data/test.npy")
    H = np.load(FULL / "data/heldout.npy")
    rows = []
    for f in sorted(glob.glob(str(FULL / "fits/cv_*.npz"))):
        tag = Path(f).stem[3:]
        z = np.load(f)
        if tag.startswith("off_"):
            model, link = "logit + offsets", "logit"
        elif tag.startswith("cll_"):
            model, link = "cloglog + offsets", "cloglog"
        else:
            model, link = "logit (no offsets)", "logit"
        U, Vf = z["U"].astype(np.float32), z["V"].astype(np.float32)
        a = z["a"].astype(np.float32) if "a" in z.files else np.zeros(len(U), np.float32)
        b = z["b"].astype(np.float32) if "b" in z.files else np.zeros(len(Vf), np.float32)
        fn = lambda r: U[r] @ Vf.T + a[r, None] + b[None, :]
        mt = re.search(r"k(\d+)_lam([0-9.]+)$", tag)
        k, lam = int(mt.group(1)), float(mt.group(2))
        rv, rt = scores_from_logit(Y, V, fn, link=link), scores_from_logit(Y, T, fn, link=link)
        rows.append(dict(model=model, k=k, lam=lam, **{f"val_{x}": rv[x] for x in rv}, **{f"test_{x}": rt[x] for x in rt}))
        print(rows[-1], flush=True)
    # baselines: global rate, gene rate (training entries only)
    Tr = ~H
    g = np.clip((Y * Tr).sum(0) / Tr.sum(0), 1e-6, 1 - 1e-6); lg = np.log(g / (1 - g)).astype(np.float32)
    p0 = np.clip((Y * Tr).sum() / Tr.sum(), 1e-6, 1 - 1e-6); l0 = np.float32(np.log(p0 / (1 - p0)))
    for name, fn in [("constant rate", lambda r: np.full((len(range(*r.indices(Y.shape[0]))), Y.shape[1]), l0, np.float32)),
                     ("gene rate", lambda r: np.broadcast_to(lg, (len(range(*r.indices(Y.shape[0]))), Y.shape[1])))]:
        rv, rt = scores_from_logit(Y, V, fn), scores_from_logit(Y, T, fn)
        rows.append(dict(model=name, k=0, lam=np.nan, **{f"val_{x}": rv[x] for x in rv}, **{f"test_{x}": rt[x] for x in rt}))
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "selection.csv", index=False)
    best = d.loc[d.groupby("model").val_nll.idxmin()].sort_values("val_nll")
    best.to_csv(OUT / "best_per_model.csv", index=False)
    print(best[["model", "k", "lam", "val_nll", "test_nll", "test_auc", "test_brier"]].to_string())


if __name__ == "__main__":
    main()
