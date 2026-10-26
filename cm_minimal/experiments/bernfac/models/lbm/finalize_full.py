"""Collect full-data LBM results: selection table, best.json, embedding, figures."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from full_common import HERE, OUT, lbm

sys.path.insert(0, str(HERE.parents[1]))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402


def main():
    fs.apply_publication_style()
    recs = [json.loads(f.read_text()) for f in sorted((OUT / "results").glob("*.json"))]
    rows = []
    for r in recs:
        rows.append(dict(name=r["name"], Kr=r["Kr"], Kc=r["Kc"], lam=r["lam"],
                         val_nll=r["val"]["nll"], val_brier=r["val"]["brier"], val_auc=r["val"]["auc"],
                         val_ece=r["val"]["ece"], test_nll=r["test"]["nll"], test_brier=r["test"]["brier"],
                         test_auc=r["test"]["auc"], test_ece=r["test"]["ece"],
                         train_elbo=r["train_elbo"], n_iter=r["n_iter"], status=r["status"],
                         restart_chosen=r["restart_chosen"], init_chosen=r["init_chosen"],
                         restart_elbos=";".join(f"{x['elbo']:.6e}" for x in r["restarts"]),
                         max_rel_elbo_decrease=max(x["max_rel_decrease"] for x in r["restarts"]),
                         row_groups_used=r["n_rows_groups_used"], col_groups_used=r["n_col_groups_used"],
                         n_low_occ_rows=r["n_empty_rows"], n_low_occ_cols=r["n_empty_cols"],
                         ari_hemilineage=r["ari_hemilineage"], nmi_hemilineage=r["nmi_hemilineage"],
                         ari_library=r["ari_library"], nmi_library=r["nmi_library"],
                         fit_time_s=r["fit_time_s"], n_params_B=r["Kr"] * r["Kc"]))
    df = pd.DataFrame(rows).sort_values(["Kr", "Kc", "lam"])
    (OUT / "tables").mkdir(exist_ok=True)
    df.to_csv(OUT / "tables" / "selection.csv", index=False)
    grid = df[df.Kr > 1]
    best = grid.loc[grid.val_nll.idxmin()]
    z = np.load(OUT / "fits" / "best.npz")
    zb = grid[grid.name == str(z["name"])]
    if len(zb) and zb.val_nll.iloc[0] <= best.val_nll:  # exact tie: keep the saved fit
        best = zb.iloc[0]
    rb = next(r for r in recs if r["name"] == best["name"])
    assert str(z["name"]) == best["name"], (str(z["name"]), best["name"])
    R, C, B = z["R"].astype(np.float64), z["C"].astype(np.float64), z["B"]
    emb = (R @ B).astype(np.float32)          # expected row log-odds profile over column groups
    np.save(OUT / "embedding.npy", emb)
    np.save(OUT / "row_responsibilities.npy", R.astype(np.float32))
    total_fit_h = df.fit_time_s.sum() / 3600
    bj = dict(model="Bernoulli latent block model (mean-field VB labels, MAP B/pi/rho, PG block updates)",
              selected=dict(Kr=int(best.Kr), Kc=int(best.Kc), lam_block=float(best.lam),
                            alpha_r=1.1, alpha_c=1.1),
              selection_rule="lowest validation NLL across configurations; restart within a "
                             "configuration chosen by final training ELBO",
              val=rb["val"], test=rb["test"], train_elbo=rb["train_elbo"], elbo_parts=rb["elbo_parts"],
              n_iter=rb["n_iter"], status=rb["status"], restart_chosen=rb["restart_chosen"],
              init_chosen=rb["init_chosen"], restarts=[{k: v for k, v in x.items() if k != "elbo_trace"}
                                                       for x in rb["restarts"]],
              settings=rb["settings"],
              ari_hemilineage=rb["ari_hemilineage"], nmi_hemilineage=rb["nmi_hemilineage"],
              ari_library=rb["ari_library"], nmi_library=rb["nmi_library"],
              row_groups_used=rb["n_rows_groups_used"], col_groups_used=rb["n_col_groups_used"],
              baseline_K1=dict(val=next(r for r in recs if r["Kr"] == 1)["val"]["nll"],
                               test=next(r for r in recs if r["Kr"] == 1)["test"]["nll"]),
              n_configs=len(df), total_fit_hours=total_fit_h,
              prediction="P = R sigmoid(B) C^T (mean-field posterior predictive); probabilities clipped "
                         "to [1e-12, 1-1e-12] only for the log in NLL",
              embedding=dict(file="embedding.npy", definition="E = R B (n x K_c): expected log-odds of "
                             "each cell for each column (gene) group under q(c_i); float32",
                             shape=list(emb.shape)),
              files=dict(fit="fits/best.npz (R, C float32; B, pi, rho float64)",
                         row_responsibilities="row_responsibilities.npy (n x K_r float32)"))
    (OUT / "best.json").write_text(json.dumps(bj, indent=1, default=float))
    print(df[["name", "val_nll", "test_nll", "val_auc", "ari_hemilineage", "n_iter", "fit_time_s"]]
          .sort_values("val_nll").head(12).to_string())
    print("best", best["name"])

    figdir = OUT / "figures"
    # ---- selection heatmaps
    Krs = sorted(grid.Kr.unique()); Kcs = sorted(grid.Kc.unique())
    lams = sorted(grid.lam.unique())
    fig, axes = plt.subplots(1, len(lams), figsize=(6.6 * len(lams), 5.6), squeeze=False)
    vmin, vmax = grid.val_nll.min(), grid.val_nll.max()
    for k, lam in enumerate(lams):
        ax = axes[0, k]
        A = np.full((len(Krs), len(Kcs)), np.nan)
        for _, r in grid[grid.lam == lam].iterrows():
            A[Krs.index(r.Kr), Kcs.index(r.Kc)] = r.val_nll
        im = ax.imshow(A, cmap=fs.CMAP_SEQ_R, vmin=vmin, vmax=vmax, origin="lower")
        for i in range(len(Krs)):
            for j in range(len(Kcs)):
                if np.isfinite(A[i, j]):
                    isb = (Krs[i], Kcs[j], lam) == (best.Kr, best.Kc, best.lam)
                    ax.text(j, i, f"{A[i, j]:.4f}", ha="center", va="center", fontsize=fs.FS_TINY,
                            color="white" if A[i, j] > (vmin + vmax) / 2 else "black",
                            fontweight="bold" if isb else "normal")
                    if isb:
                        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=fs.PALETTE["highlight"], lw=3))
        ax.set_xticks(range(len(Kcs)), Kcs); ax.set_yticks(range(len(Krs)), Krs)
        ax.set_xlabel(r"gene groups $K_c$"); ax.set_ylabel(r"cell groups $K_r$")
        ax.set_title(rf"validation NLL, $\lambda_{{\rm block}}={lam:g}$")
        fs.panel_label(ax, "ab"[k], x=-0.2)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85); cb.set_label("NLL / held-out entry")
    fs.finalize_figure(fig, figdir / "lbm_full_selection_heatmap")

    # ---- fitted block-probability table, reordered
    S = lbm.sigmoid(B)
    from scipy.cluster.hierarchy import leaves_list, linkage
    ro = leaves_list(linkage(B, "average")) if B.shape[0] > 2 else np.arange(B.shape[0])
    co = leaves_list(linkage(B.T, "average")) if B.shape[1] > 2 else np.arange(B.shape[1])
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.22], height_ratios=[0.22, 1], hspace=0.05, wspace=0.03)
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(S[ro][:, co], cmap=fs.CMAP_PROB, vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel(rf"gene group (reordered, $K_c={B.shape[1]}$)"); ax.set_ylabel(rf"cell group (reordered, $K_r={B.shape[0]}$)")
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axt.bar(range(len(co)), np.bincount(C.argmax(1), minlength=B.shape[1])[co], color=fs.PALETTE["teal"], width=0.9)
    axt.set_yscale("log"); axt.set_ylabel("genes"); plt.setp(axt.get_xticklabels(), visible=False)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    axr.barh(range(len(ro)), np.bincount(R.argmax(1), minlength=B.shape[0])[ro], color=fs.PALETTE["blue_main"], height=0.9)
    axr.set_xscale("log"); axr.set_xlabel("cells"); plt.setp(axr.get_yticklabels(), visible=False)
    cax = fig.add_axes([0.93, 0.12, 0.012, 0.5]); cb = fig.colorbar(im, cax=cax); cb.set_label(r"$\sigma(B_{ab})$")
    axt.set_title(f"Fitted block detection probabilities ({best['name']})")
    fs.finalize_figure(fig, figdir / "lbm_full_block_table")

    # ---- ELBO traces (selected config, all restarts) + ARI vs K_r
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    ax = axes[0]
    Fbest = max(max(x["elbo_trace"]) for x in rb["restarts"])
    for x in rb["restarts"]:
        tr = np.array(x["elbo_trace"])
        ax.plot(np.arange(len(tr)), np.maximum(Fbest - tr, 1.0), lw=2,
                color=fs.PALETTE["blue_main"] if x["init"] == "spectral" else fs.PALETTE["grey_mid"],
                ls="-" if x["restart"] == rb["restart_chosen"] else "--",
                label=f"{x['init']} (restart {x['restart']})" + (", chosen" if x["restart"] == rb["restart_chosen"] else ""))
    ax.set_yscale("log"); ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\mathcal{F}_{\rm best} - \mathcal{F}$ (nats, floor 1)")
    ax.legend(loc="upper right", fontsize=fs.FS_LEGEND - 1)
    ax.set_title(f"ELBO traces, {best['name']}"); fs.panel_label(ax, "a")
    ax = axes[1]
    for x in rb["restarts"]:
        if x["monitor"]:
            it, v = zip(*x["monitor"])
            ax.plot(it, v, "o-", color=fs.PALETTE["blue_main"] if x["init"] == "spectral" else fs.PALETTE["grey_mid"], lw=2)
    ax.axhline(rb["val"]["nll"], color=fs.PALETTE["red_strong"], ls="--", lw=1.5, label="final (selected)")
    ax.set_xlabel("iteration"); ax.set_ylabel("validation NLL (monitor only)"); ax.legend()
    ax.set_title("validation NLL during fitting"); fs.panel_label(ax, "b")
    ax = axes[2]
    cols = fs.k_ramp(len(Kcs))
    for k, kc in enumerate(Kcs):
        sub = grid[(grid.Kc == kc)].sort_values("Kr")
        # for each Kr take the lam with lower val NLL
        sub = sub.loc[sub.groupby("Kr").val_nll.idxmin()]
        ax.plot(sub.Kr, sub.ari_hemilineage, "o-", color=cols[k], label=rf"$K_c={kc}$")
    sub = grid.loc[grid.groupby("Kr").val_nll.idxmin()]
    ax.plot(sub.Kr, sub.ari_library, "s--", color=fs.PALETTE["red_strong"], label="library (batch), best $K_c$")
    ax.axvline(23, color=fs.PALETTE["neutral"], lw=1.5, zorder=0)
    ax.set_xscale("log"); ax.set_xticks(Krs, Krs); ax.xaxis.set_minor_formatter(plt.NullFormatter()); ax.set_xlabel(r"cell groups $K_r$")
    ax.set_ylabel("ARI of argmax $R$"); ax.set_title("agreement with hemilineage (23)")
    ax.legend(fontsize=fs.FS_LEGEND - 1, loc="upper right"); fs.panel_label(ax, "c")
    fig.tight_layout()
    fs.finalize_figure(fig, figdir / "lbm_full_traces_ari")


if __name__ == "__main__":
    main()
