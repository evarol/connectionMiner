"""Held-out baselines on the full matrix (same 5% held-out entries as run_full.py).

  gene rate          p_ij = detection rate of gene j on training entries
  additive logistic  logit p_ij = a_i + b_j  (cell depth + gene abundance), fit by
                     alternating 1-D Newton steps on the training entries
"""
import json

import numpy as np

import bernfac_large as bl
from bernfac import sigmoid
from run_full import OUT, load


def scores(Y, H, logit_fn):
    nll, brier, cnt, ys, ps = 0.0, 0.0, 0, [], []
    for s in range(0, Y.shape[0], 2048):
        Hb = H[s:s + 2048]
        psi = logit_fn(slice(s, s + 2048))[Hb].astype(np.float64)
        y = Y[s:s + 2048][Hb].astype(np.float64)
        p = sigmoid(psi)
        nll += float(np.sum(np.logaddexp(0, psi) - y * psi)); brier += float(np.sum((p - y) ** 2)); cnt += y.size
        ys.append(y.astype(np.int8)); ps.append(p.astype(np.float32))
    from sklearn.metrics import roc_auc_score
    y, p = np.concatenate(ys), np.concatenate(ps)
    idx = np.random.default_rng(0).choice(y.size, 2_000_000, replace=False)
    return dict(nll=nll / cnt, brier=brier / cnt, auc=float(roc_auc_score(y[idx], p[idx])))


def main():
    Y, H = load()
    T = ~H
    rate = (Y * T).sum(0) / T.sum(0)
    lg = np.log(np.clip(rate, 1e-6, 1 - 1e-6) / (1 - np.clip(rate, 1e-6, 1 - 1e-6)))
    out = {"gene_rate": scores(Y, H, lambda r: np.broadcast_to(lg, (Y[r].shape[0], Y.shape[1])).astype(np.float32))}

    a, b = np.zeros(Y.shape[0], np.float32), lg.astype(np.float32)
    for _ in range(25):                                   # alternating Newton on cell / gene offsets
        for axis in (1, 0):
            g, h = np.zeros(Y.shape[1 - axis] if axis == 0 else Y.shape[0]), None
            num = np.zeros(Y.shape[0] if axis == 1 else Y.shape[1]); den = np.zeros_like(num)
            for s in range(0, Y.shape[0], 2048):
                psi = a[s:s + 2048, None] + b[None, :]
                p = sigmoid(psi); t = T[s:s + 2048]
                r = (Y[s:s + 2048] - p) * t; w = p * (1 - p) * t
                if axis == 1:
                    num[s:s + 2048] = r.sum(1); den[s:s + 2048] = w.sum(1)
                else:
                    num += r.sum(0); den += w.sum(0)
            if axis == 1:
                a += (num / np.maximum(den, 1e-9)).astype(np.float32)
            else:
                b += (num / np.maximum(den, 1e-9)).astype(np.float32)
    out["additive_logistic"] = scores(Y, H, lambda r: (a[r, None] + b[None, :]).astype(np.float32))
    (OUT / "tables/baselines.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
