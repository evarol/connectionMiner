"""Synthetic demonstration for the Bernoulli MFA (handoff section 8.1).

Generator: n=250 rows, m=100 columns, K=3 components with unequal weights (0.5, 0.3, 0.2),
d=2, distinct component means, independent (hence non-parallel) loading subspaces, residual
logit variance t=0.25. Two cases: 'balanced' (component means around 0) and 'sparse'
(means around -2.5, so most cells are 0). Each cell is observed with probability 0.9; the
observed cells are split 70/15/15 into train/validation/test. Three data seeds per case.

Fits: K in {1, 3} x {isotropic fixed t=0.25, isotropic learned t, residual 'none'};
3 random restarts each, keeping the largest training bound. True logits are used only for
diagnostics (oracle NLL, component recovery after Hungarian alignment).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import mfa  # noqa: E402

OUT = HERE.parents[4] / "outputs/bernfac_20261020/models/mfa"
SYN = OUT / "synthetic"
SYN.mkdir(parents=True, exist_ok=True)

N, M, K_TRUE, D, T = 250, 100, 3, 2, 0.25
PI = np.array([0.5, 0.3, 0.2])
MODELS = [("iso t=0.25", dict(residual_mode="isotropic", residual_variance=0.25)),
          ("iso learned t", dict(residual_mode="isotropic", residual_variance=0.25, learn_residual_variance=True)),
          ("none (t=0)", dict(residual_mode="none"))]


def generate(case, seed):
    rng = np.random.default_rng(seed)
    c = rng.choice(K_TRUE, N, p=PI)
    shift = 0.0 if case == "balanced" else -2.5
    mu = rng.normal(shift, 1.0 if case == "sparse" else 1.5, (K_TRUE, M))
    B = rng.normal(0, 1.0, (K_TRUE, M, D))
    z = rng.normal(size=(N, D))
    psi = mu[c] + np.einsum("imd,id->im", B[c], z) + np.sqrt(T) * rng.normal(size=(N, M))
    Y = (rng.random((N, M)) < mfa.sigmoid(psi)).astype(np.uint8)
    obs = rng.random((N, M)) < 0.9
    u = rng.random((N, M))
    train, val, test = obs & (u < 0.70), obs & (u >= 0.70) & (u < 0.85), obs & (u >= 0.85)
    return dict(Y=Y, c=c, psi=psi, train=train, val=val, test=test)


def aligned(r, c, K):
    """Hungarian alignment of fitted components to true labels; returns accuracy and ARI."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score
    hard = r.argmax(1)
    C = np.zeros((K_TRUE, K))
    for a, b in zip(c, hard):
        C[a, b] += 1
    ri, ci = linear_sum_assignment(-C)
    return C[ri, ci].sum() / len(c), adjusted_rand_score(c, hard), dict(zip(ci.tolist(), ri.tolist()))


def main():
    rows, traces, resp = [], {}, {}
    for case in ("balanced", "sparse"):
        for seed in range(3):
            g = generate(case, seed)
            Y = g["Y"]
            ps = mfa.sigmoid(g["psi"])
            te = g["test"]
            oracle = -np.mean(np.where(Y[te] > 0, np.log(ps[te]), np.log1p(-ps[te])))
            for K in (1, 3):
                for name, kw in MODELS:
                    best = None
                    bounds = []
                    for rs in range(3):
                        o = mfa.MFAOptions(max_iter=1000, tol=1e-7, seed=100 * seed + rs, init="random", **kw)
                        res = mfa.fit(Y, K, D, mask=g["train"], opts=o)
                        bounds.append(res[2]["bound"][-1])
                        if best is None or bounds[-1] > best[2]["bound"][-1]:
                            best = res
                    model, state, info = best
                    sv = mfa.score_cells(Y, g["val"], model, state)
                    st = mfa.score_cells(Y, g["test"], model, state)
                    acc, ari, perm = aligned(state.r, g["c"], K) if K == 3 else (np.nan, np.nan, None)
                    rows.append(dict(case=case, seed=seed, K=K, model=name, val_nll=sv["nll"], test_nll=st["nll"],
                                     val_auc=sv["auc"], test_auc=st["auc"], val_brier=sv["brier"],
                                     test_brier=st["brier"], oracle_test_nll=oracle, bound=info["bound"][-1],
                                     restart_bounds=bounds, iterations=info["iterations"],
                                     converged=info["converged"], monotone=info["monotone"],
                                     max_rel_decrease=info["max_rel_decrease"], t_final=float(np.mean(model.t)),
                                     occupancy=np.round(model.pi, 3).tolist(), label_accuracy=acc, ARI=ari))
                    if seed == 0:
                        traces[f"{case}|K{K}|{name}"] = info["bound"]
                        if K == 3:
                            order = np.argsort(g["c"], kind="stable")
                            cols = [k for k, _ in sorted(perm.items(), key=lambda kv: kv[1])]
                            resp[f"{case}|{name}"] = dict(r=state.r[order][:, cols].tolist(), c=g["c"][order].tolist())
                    print(f"{case} seed={seed} K={K} {name:14s} val={sv['nll']:.4f} test={st['nll']:.4f} "
                          f"oracle={oracle:.4f} it={info['iterations']} mono={info['monotone']} ARI={ari:.3f}",
                          flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(SYN / "synthetic_results.csv", index=False)
    (SYN / "synthetic_traces.json").write_text(json.dumps(dict(traces=traces, resp=resp)))
    summ = df.groupby(["case", "K", "model"])[["val_nll", "test_nll", "test_auc", "test_brier", "oracle_test_nll",
                                                "ARI", "label_accuracy", "t_final"]].agg(["mean", "std"])
    summ.to_csv(SYN / "synthetic_summary.csv")
    print(summ.round(4).to_string())
    print("all monotone:", bool(df.monotone.all()))


if __name__ == "__main__":
    main()
