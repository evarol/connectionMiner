"""Synthetic block demo (handoff 8.3): K_r=3, K_c=4, rectangular 250 x 100, masks 70/15/15.

Fits (1,1), the true (3,4) and an over-specified (6,8) model on 5 data seeds and reports
held-out NLL/Brier, permutation-aligned ARI and the recovered block-probability table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score as ari

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import lbm  # noqa: E402
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = HERE.parents[4] / "outputs/bernfac_20261020/models/lbm"
N, M = 250, 100
PI = np.array([0.5, 0.3, 0.2]); RHO = np.array([0.4, 0.3, 0.2, 0.1])
PTRUE = np.array([[0.05, 0.60, 0.30, 0.85],
                  [0.70, 0.15, 0.45, 0.10],
                  [0.35, 0.90, 0.08, 0.55]])
FITS = [(1, 1), (3, 4), (6, 8)]


def generate(seed):
    rng = np.random.default_rng(seed)
    zr = rng.choice(3, N, p=PI); zc = rng.choice(4, M, p=RHO)
    P = PTRUE[zr][:, zc]
    Y = (rng.random((N, M)) < P).astype(float)
    u = rng.random((N, M))
    train, val, test = u < 0.70, (u >= 0.70) & (u < 0.85), u >= 0.85
    return Y, P, zr, zc, train, val, test


def aligned_table(mdl, zr, zc):
    """sigma(B) rows/cols reordered to the true groups by Hungarian matching on hard labels."""
    lr, lc = mdl.R.argmax(1), mdl.C.argmax(1)
    mr, _ = lbm.align_labels(zr, lr, 3, mdl.B.shape[0]); mc, _ = lbm.align_labels(zc, lc, 4, mdl.B.shape[1])
    inv_r = {v: k for k, v in mr.items()}; inv_c = {v: k for k, v in mc.items()}
    S = lbm.sigmoid(mdl.B)
    return np.array([[S[inv_r[a], inv_c[b]] for b in range(4)] for a in range(3)])


def main():
    fs.apply_publication_style()
    rows, keep = [], {}
    for seed in range(5):
        Y, P, zr, zc, train, val, test = generate(seed)
        d = lbm.BinaryData(Y, train)
        oracle = {k: lbm.score_probs(Y[msk], P[msk]) for k, msk in [("val", val), ("test", test)]}
        for Kr, Kc in FITS:
            mdl = lbm.fit_lbm(d, Kr, Kc, lam=1.0, max_iter=500, tol=1e-10, n_restarts=5, init="mixed",
                              seed=seed)
            pv = lbm.predict_matrix(mdl)
            sv = lbm.score_probs(Y[val], pv[val]); st = lbm.score_probs(Y[test], pv[test])
            rec = dict(seed=seed, Kr=Kr, Kc=Kc, elbo=mdl.info["elbo"], n_iter=mdl.info["n_iter"],
                       n_decrease=sum(r["n_decrease"] for r in mdl.info["restarts"]),
                       val_nll=sv["nll"], val_brier=sv["brier"], test_nll=st["nll"], test_brier=st["brier"],
                       test_auc=st["auc"], oracle_test_nll=oracle["test"]["nll"],
                       oracle_test_brier=oracle["test"]["brier"],
                       ari_rows=ari(zr, mdl.R.argmax(1)), ari_cols=ari(zc, mdl.C.argmax(1)),
                       low_occ_rows=int((mdl.info["occupancy_rows"] < 2).sum()),
                       low_occ_cols=int((mdl.info["occupancy_cols"] < 2).sum()),
                       mae_P=float(np.mean(np.abs(pv - P))))
            if (Kr, Kc) == (3, 4):
                rec["table"] = aligned_table(mdl, zr, zc).tolist()
                rec["table_max_abs_err"] = float(np.max(np.abs(np.array(rec["table"]) - PTRUE)))
            rows.append(rec)
            keep[(seed, Kr, Kc)] = (mdl, Y, zr, zc)
            print(f"seed {seed} ({Kr},{Kc}) test NLL {st['nll']:.4f} (oracle {oracle['test']['nll']:.4f}) "
                  f"ARI r/c {rec['ari_rows']:.3f}/{rec['ari_cols']:.3f} decreases {rec['n_decrease']}")
    (OUT / "synthetic").mkdir(parents=True, exist_ok=True)
    (OUT / "synthetic" / "synthetic_results.json").write_text(json.dumps(rows, indent=1, default=float))
    import pandas as pd
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "table"} for r in rows])
    df.to_csv(OUT / "synthetic" / "synthetic_results.csv", index=False)
    summ = df.groupby(["Kr", "Kc"]).agg(["mean", "std"])[
        ["test_nll", "oracle_test_nll", "test_brier", "ari_rows", "ari_cols", "low_occ_rows", "low_occ_cols"]]
    print(summ.to_string())
    summ.to_csv(OUT / "synthetic" / "synthetic_summary.csv")

    # ---------------- figure
    mdl, Y, zr, zc = keep[(0, 3, 4)]
    fig = plt.figure(figsize=(17, 9.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.15, 1.15, 1, 1], hspace=0.5, wspace=0.42)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(Y[np.argsort(zr, kind="stable")][:, np.argsort(zc, kind="stable")], cmap=fs.CMAP_Y,
              aspect="auto")
    ax.set_title("Data $Y$, true group order"); ax.set_xlabel("columns (genes)"); ax.set_ylabel("rows (cells)")
    fs.panel_label(ax, "a", x=-0.18, y=1.01)
    ax = fig.add_subplot(gs[:, 1])
    lr, lc = mdl.R.argmax(1), mdl.C.argmax(1)
    ax.imshow(Y[np.argsort(lr, kind="stable")][:, np.argsort(lc, kind="stable")], cmap=fs.CMAP_Y, aspect="auto")
    ax.set_title("Data $Y$, fitted group order"); ax.set_xlabel("columns (genes)"); ax.set_yticks([])
    fs.panel_label(ax, "b", x=-0.08, y=1.01)
    tab = np.array([r["table"] for r in rows if (r["Kr"], r["Kc"]) == (3, 4) and r["seed"] == 0][0])
    for k, (T, title) in enumerate([(PTRUE, "True block probabilities"),
                                    (tab, r"Fitted $\sigma(B)$, aligned")]):
        ax = fig.add_subplot(gs[0, 2 + k])
        ax.imshow(T, cmap=fs.CMAP_PROB, vmin=0, vmax=1)
        for a in range(3):
            for b in range(4):
                ax.text(b, a, f"{T[a, b]:.2f}", ha="center", va="center", fontsize=fs.FS_SMALL,
                        color="white" if T[a, b] < 0.6 else "black")
        ax.set_xticks(range(4), [f"$d={b+1}$" for b in range(4)]); ax.set_yticks(range(3), [f"$c={a+1}$" for a in range(3)])
        ax.set_title(title); fs.panel_label(ax, "cd"[k], x=-0.3, y=1.04)
    # NLL by model across seeds
    ax = fig.add_subplot(gs[1, 2])
    labels = [f"({a},{b})" for a, b in FITS]
    for x, (a, b) in enumerate(FITS):
        v = df[(df.Kr == a) & (df.Kc == b)]["test_nll"].to_numpy()
        ax.bar(x, v.mean(), color=[fs.PALETTE["neutral"], fs.PALETTE["blue_main"], fs.PALETTE["teal"]][x], width=0.65)
        ax.scatter(np.full(len(v), x) + np.linspace(-0.15, 0.15, len(v)), v, color="k", s=14, zorder=3)
    orc = df.groupby("seed").oracle_test_nll.first().mean()
    ax.axhline(orc, color=fs.PALETTE["red_strong"], ls="--", lw=1.8, label="true $P$")
    ax.set_xticks(range(3), labels); ax.set_xlabel(r"$(K_r, K_c)$"); ax.set_ylabel("test NLL / entry")
    ax.set_ylim(0.4, 0.72); ax.legend(loc="upper right"); fs.panel_label(ax, "e", x=-0.3, y=1.04)
    # ELBO traces for (3,4) restarts, seed 0
    ax = fig.add_subplot(gs[1, 3])
    for r in mdl.info["restarts"]:
        tr = np.array(r["elbo_trace"])
        ax.plot(np.arange(len(tr)), tr, color=fs.PALETTE["blue_main"] if r["init"] == "spectral" else fs.PALETTE["grey_mid"],
                lw=1.8, label=r["init"])
    h, l_ = ax.get_legend_handles_labels(); u = dict(zip(l_, h))
    ax.legend(u.values(), u.keys(), loc="lower right")
    ax.set_xscale("symlog", linthresh=1); ax.set_xlabel("iteration"); ax.set_ylabel(r"ELBO $\mathcal{F}$")
    ax.set_title("5 restarts, $K_r{=}3, K_c{=}4$"); fs.panel_label(ax, "f", x=-0.3, y=1.04)
    fs.finalize_figure(fig, OUT / "figures" / "lbm_synthetic_demo")


if __name__ == "__main__":
    main()
