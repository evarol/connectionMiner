"""Cross-model comparison of held-out prediction (validation-selected, test reported).

Every model was fit on the same training entries (~heldout), selected on the same validation
half and scored once on the same test half. Rows: best configuration per model family.
Outputs: models/comparison.csv and figures/fig_model_comparison.{png,pdf}
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figstyle import PALETTE, apply_publication_style, finalize_figure

M = Path(__file__).resolve().parents[3] / "outputs/bernfac_20261020/models"


def rows():
    out = []
    r = pd.read_csv(M / "rescored/best_per_model.csv")
    names = {"logit (no offsets)": ("BernFac logit", "binary"), "logit + offsets": ("BernFac logit + offsets", "binary"),
             "cloglog + offsets": ("BernFac cloglog + offsets", "binary"), "gene rate": ("gene rate", "baseline"),
             "constant rate": ("constant rate", "baseline")}
    for _, x in r.iterrows():
        n, fam = names[x.model]
        cfg = f"k={int(x.k)}, λ={x.lam:g}" if x.k > 0 else ""
        out.append(dict(model=n, family=fam, config=cfg, val_nll=x.val_nll, test_nll=x.test_nll,
                        test_auc=x.test_auc, test_brier=x.test_brier))
    for fam_key, label in [("nb", "NB count factorization"), ("poisson", "Poisson count factorization")]:
        b = json.loads((M / f"counts/best_{fam_key}.json").read_text())
        out.append(dict(model=label, family="counts", config=f"k={int(b['k'])}, λ={b['lam']:g}",
                        val_nll=b["val_nll"], test_nll=b["test_nll"], test_auc=b["test_auc"], test_brier=b["test_brier"]))
    b = json.loads((M / "mfa/best.json").read_text())
    c = b["config"]
    out.append(dict(model="mixture of local factors", family="binary",
                    config=f"K={c['K']}, d={c['d']}, residual {c['mode']}",
                    val_nll=b["val"]["nll"], test_nll=b["test"]["nll"], test_auc=b["test"]["auc"],
                    test_brier=b["test"]["brier"]))
    s = pd.read_csv(M / "neural/tables/selection.csv")
    x = s[s.chosen_restart == True].sort_values("val_nll").iloc[0]  # noqa: E712
    out.append(dict(model="neural manifold", family="binary",
                    config=f"{x.head_mode} head, d={int(x.latent_dim)}, {x.hidden_widths}",
                    val_nll=x.val_nll, test_nll=x.test_nll, test_auc=x.test_auc, test_brier=x.test_brier))
    b = json.loads((M / "lbm/best.json").read_text())
    sel = b["selected"]
    out.append(dict(model="latent block model", family="binary", config=f"Kr={sel['Kr']}, Kc={sel['Kc']}",
                    val_nll=b["val"]["nll"], test_nll=b["test"]["nll"], test_auc=b["test"]["auc"],
                    test_brier=b["test"]["brier"]))
    return pd.DataFrame(out).sort_values("test_nll").reset_index(drop=True)


def main():
    d = rows()
    d.to_csv(M / "comparison.csv", index=False)
    print(d.to_string())
    d = d[d.family != "baseline"].iloc[::-1]
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1.3, 1]})
    col = {"counts": PALETTE["red_strong"], "binary": PALETTE["blue_main"]}
    y = np.arange(len(d))
    for ax, m, lab in [(axes[0], "test_nll", "test NLL per entry (lower is better)"),
                       (axes[1], "test_auc", "test AUC (higher is better)")]:
        ax.barh(y, d[m], color=[col[f] for f in d.family], edgecolor="k", lw=0.6)
        lo = d[m].min() - 0.002 if m == "test_nll" else d[m].min() - 0.001
        hi = d[m].max() + 0.0015 if m == "test_nll" else d[m].max() + 0.0005
        ax.set_xlim(lo, hi)
        for yi, v in zip(y, d[m]):
            ax.text(v, yi, f" {v:.5f}" if m == "test_nll" else f" {v:.4f}", va="center", fontsize=10)
        ax.set_xlabel(lab)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    axes[0].set_yticks(y, [f"{a}\n({c})" for a, c in zip(d.model, d.config)], fontsize=10)
    axes[1].set_yticks(y, [""] * len(y))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=col["binary"], label="models of binary detection Y"),
                        Patch(color=col["counts"], label="models of UMI counts (scored on detection)")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.04))
    fig.suptitle("Held-out detection prediction on 8.0 M test entries (each model selected on validation)",
                 fontsize=14)
    for i, ax in enumerate(axes):
        ax.text(-0.02 if i else -0.5, 1.02, "ab"[i], transform=ax.transAxes, fontsize=16, weight="bold")
    finalize_figure(fig, M / "figures/fig_model_comparison")


if __name__ == "__main__":
    main()
