"""Figures for the Bernoulli MFA study (house style from ../../figstyle.py).

    python make_figures.py synthetic   # synthetic demo (needs synthetic_demo.py output)
    python make_figures.py full        # full-data selection, convergence, occupancy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from figstyle import PALETTE, apply_publication_style, finalize_figure, k_ramp, panel_label  # noqa: E402

OUT = HERE.parents[4] / "outputs/bernfac_20261020/models/mfa"
FIG = OUT / "figures"
MODEL_COLORS = {"iso t=0.25": PALETTE["blue_main"], "iso learned t": PALETTE["teal"],
                "none (t=0)": PALETTE["red_strong"]}
MODE_COLORS = {"iso": PALETTE["blue_main"], "isoL": PALETTE["violet"], "none": PALETTE["red_strong"]}
MODE_LABEL = {"iso": r"isotropic, fixed $t$", "isoL": r"isotropic, learned $t$", "none": r"none ($t=0$)"}


def synthetic():
    df = pd.read_csv(OUT / "synthetic/synthetic_results.csv")
    tr = json.loads((OUT / "synthetic/synthetic_traces.json").read_text())
    apply_publication_style()
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.2), gridspec_kw=dict(width_ratios=[1.2, 0.9, 1, 1]))
    # a: bound traces (balanced, seed 0)
    ax = axes[0]
    for key, b in tr["traces"].items():
        case, K, name = key.split("|")
        if case != "balanced":
            continue
        b = np.asarray(b)
        ax.plot(np.arange(len(b)), b, color=MODEL_COLORS[name], ls="-" if K == "K3" else "--", lw=2,
                label=f"{K.replace('K', '$K=')}$, {name}")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("outer iteration")
    ax.set_ylabel(r"variational bound $\mathcal{B}(\Theta,\xi)$")
    ax.set_title("bound never decreases (balanced, seed 0)")
    lo = min(v[min(3, len(v) - 1)] for k, v in tr["traces"].items() if k.startswith("balanced"))
    hi = max(np.max(v) for k, v in tr["traces"].items() if k.startswith("balanced"))
    ax.set_ylim(lo, hi + 0.04 * (hi - lo))
    ax.legend(fontsize=10.5, loc="lower right")
    panel_label(ax, "a")
    # b: responsibilities (balanced, iso t=0.25, aligned components, rows sorted by true label)
    ax = axes[1]
    rr = tr["resp"]["balanced|iso t=0.25"]
    R = np.asarray(rr["r"])
    ax.imshow(R, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    c = np.asarray(rr["c"])
    for b in np.flatnonzero(np.diff(c)) + 0.5:
        ax.axhline(b, color=PALETTE["red_strong"], lw=1.5)
    ax.set_xticks(range(R.shape[1]), [f"comp {k + 1}" for k in range(R.shape[1])])
    ax.set_ylabel("rows (sorted by true component)")
    ax.set_title(r"responsibilities $r_{i\ell}$ ($K=3$)")
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, fraction=0.06, pad=0.03)
    panel_label(ax, "b")
    # c, d: test NLL per case
    for ax, case, lab in zip(axes[2:], ("balanced", "sparse"), ("c", "d")):
        sub = df[df.case == case]
        names = list(MODEL_COLORS)
        x = np.arange(2)
        w = 0.26
        for j, nm in enumerate(names):
            g = sub[sub.model == nm].groupby("K").test_nll
            ax.bar(x + (j - 1) * w, g.mean().values, w, yerr=g.std().values, color=MODEL_COLORS[nm],
                   capsize=3, label=nm, edgecolor="white")
        orc = sub.oracle_test_nll.mean()
        ax.axhline(orc, color=PALETTE["grey"], ls="--", lw=1.6, label="true-probability NLL")
        ax.set_xticks(x, [r"$K=1$", r"$K=3$"])
        lo = min(orc, sub.test_nll.min())
        ax.set_ylim(lo - 0.03, sub.test_nll.max() + 0.02)
        ax.set_ylabel("test NLL / cell")
        ax.set_title(f"{case} case (3 data seeds)")
        if case == "balanced":
            ax.legend(fontsize=10.5, loc="upper right")
        panel_label(ax, lab)
    fig.tight_layout()
    finalize_figure(fig, FIG / "synthetic_demo")


def _runs():
    return [json.loads(p.read_text()) for p in sorted((OUT / "runs").glob("*.json"))]


def full():
    runs = _runs()
    sel = pd.read_csv(OUT / "tables/selection.csv")
    best = json.loads((OUT / "best.json").read_text())
    base_nll = None
    resc = OUT.parent / "rescored/best_per_model.csv"
    if resc.exists():
        base_nll = pd.read_csv(resc)
    apply_publication_style()

    # --- selection figure
    fig, axes = plt.subplots(1, 4, figsize=(25, 5.6), gridspec_kw=dict(width_ratios=[1, 1, 1.15, 0.95]))
    v = sel[sel.chosen_restart].copy()
    fresh = v[v.variant == "fresh"]
    base = fresh[(fresh.residual_mode == "iso") & (fresh.t_init == 0.25) & (fresh.lam == 1.0)]
    ref = None
    if base_nll is not None:
        r = base_nll[base_nll.model == "logit + offsets"]
        ref = r.val_nll.iloc[0] if len(r) else None
    # a: K x d for the isotropic base grid
    ax = axes[0]
    ds = sorted(base.d.unique())
    for col, d in zip(k_ramp(len(ds) + 1)[1:], ds):
        s_ = base[base.d == d].sort_values("K")
        ax.plot(s_.K, s_.val_nll, "o-", color=col, lw=2.2, ms=7, label=rf"$d={d}$")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16], ["1", "4", "8", "16"])
    ax.set_xlabel(r"components $K$")
    ax.set_ylabel("validation NLL / cell")
    ax.set_title(r"isotropic, $t=0.25$, $\lambda=1$ (30 it.)")
    ax.legend(fontsize=11, loc="upper right")
    panel_label(ax, "a")
    # b: residual mode across K at d = 20, fresh fits
    ax = axes[1]
    for mode, mk in (("iso", "o-"), ("none", "s-")):
        s_ = fresh[(fresh.residual_mode == mode) & (fresh.d == 20) & (fresh.lam == 1.0)
                   & ((fresh.t_init == 0.25) | (mode == "none"))].sort_values("K")
        ax.plot(s_.K, s_.val_nll, mk, color=MODE_COLORS[mode], lw=2.2, ms=7, label=MODE_LABEL[mode].replace("fixed $t$", "$t=0.25$"))
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16], ["1", "4", "8", "16"])
    ax.set_xlabel(r"components $K$")
    ax.set_title(r"residual mode, $d=20$ (30 it.)")
    if ref is not None:
        for a_ in axes[:2]:
            a_.axhline(ref, color=PALETTE["grey"], ls="--", lw=1.4)
        ax.text(1, ref + 0.00005, "logit + offsets ($k=40$)", fontsize=10.5, color=PALETTE["grey"], va="bottom")
    ax.legend(fontsize=11, loc="upper right")
    panel_label(ax, "b")
    # c: t / lambda / mode variants (K=1 fresh; K=4 warm-started +15 it.; 100-it. continuations)
    ax = axes[2]
    labels, vals, cols = [], [], []
    def nm_of(r):
        nm = {"iso": rf"$t={r.t_init:g}$", "isoL": r"learned $t$", "none": r"$t=0$"}[r.residual_mode]
        if r.lam != 1.0:
            nm += rf", $\lambda={r.lam:g}$"
        return nm
    groups = [("K=1, 30 it.", v[(v.K == 1) & (v.variant == "fresh") & (v.d == 20)]),
              ("K=4, 30+15 it.", v[(v.K == 4) & (v.variant == "_warm")]),
              ("100 it.", v[v.variant == "_cont70"])]
    for gname, sub in groups:
        for _, r in sub.sort_values(["residual_mode", "t_init", "lam"]).iterrows():
            labels.append(f"{gname}: " + (nm_of(r) if gname != "100 it." else f"K={r.K}, " + nm_of(r)))
            vals.append(r.val_nll)
            cols.append(MODE_COLORS[r.residual_mode])
    y = np.arange(len(labels))
    ax.barh(y, vals, color=cols, edgecolor="white")
    ax.set_yticks(y, labels, fontsize=10.5)
    ax.invert_yaxis()
    if vals:
        ax.set_xlim(min(vals) - 0.0015, max(vals) + 0.0012)
    for i, val in enumerate(vals):
        ax.text(val + 0.00005, i, f"{val:.4f}", va="center", fontsize=9.5)
    ax.set_xlabel("validation NLL / cell")
    ax.set_title(r"residual variance $t$ and $\lambda$ ($d=20$)")
    panel_label(ax, "c", x=-0.5)
    # d: validation vs test
    ax = axes[3]
    for mode, g in v.groupby("residual_mode"):
        ax.scatter(g.val_nll, g.test_nll, s=26 + 6 * g.K, color=MODE_COLORS[mode], alpha=0.85, label=MODE_LABEL[mode],
                   edgecolor="white")
    lim = [v.val_nll.min() - 0.0005, v.val_nll.max() + 0.0005]
    ax.plot(lim, lim, color=PALETTE["grey_mid"], lw=1, ls="--")
    bb = v[v.tag == best["best_tag"]]
    ax.scatter(bb.val_nll, bb.test_nll, s=220, facecolor="none", edgecolor=PALETTE["highlight"], lw=2.5, label="selected")
    ax.set_xlabel("validation NLL / cell")
    ax.set_ylabel("test NLL / cell")
    ax.set_title("all configurations (size ~ K)")
    ax.legend(fontsize=10)
    panel_label(ax, "d")
    fig.tight_layout()
    finalize_figure(fig, FIG / "full_selection")

    # --- convergence figure
    fig, axes = plt.subplots(1, 3, figsize=(22, 5.4))
    rf_ = [r for r in runs if not r["config"].get("warm")]
    Ks = sorted({r["config"]["K"] for r in rf_ if r["config"]["mode"] == "iso"})
    colK = dict(zip(Ks, k_ramp(len(Ks) + 1)[1:]))
    nK = sorted({r["config"]["K"] for r in rf_ if r["config"]["mode"] == "none"})
    colN = dict(zip(nK, plt.get_cmap("Reds")(np.linspace(0.45, 0.95, len(nK)))))
    for r in rf_:
        c = r["config"]
        b = np.asarray(r["bound_trace"])
        ls = {5: ":", 10: "-", 20: "--"}.get(c["d"], "-")
        col = {"iso": colK.get(c["K"]), "none": colN.get(c["K"]), "isoL": MODE_COLORS["isoL"]}[c["mode"]]
        axes[0].plot(np.arange(len(b)), (b - b[0]) / abs(b[0]), color=col, ls=ls, lw=1.6, alpha=0.9)
        rel = np.asarray(r["rel_trace"])
        axes[1].semilogy(np.arange(1, len(rel) + 1), np.maximum(rel, 1e-9), color=col, ls=ls, lw=1.6, alpha=0.9)
    for K in Ks:
        axes[0].plot([], [], color=colK[K], lw=2, label=rf"iso, $K={K}$")
    for K in nK:
        axes[0].plot([], [], color=colN[K], lw=2, label=rf"none, $K={K}$")
    axes[0].plot([], [], color=MODE_COLORS["isoL"], lw=2, label=r"learned $t$")
    for d, ls in ((5, ":"), (10, "-"), (20, "--")):
        axes[0].plot([], [], color=PALETTE["grey"], ls=ls, label=rf"$d={d}$")
    axes[0].set_xlabel("outer iteration")
    axes[0].set_ylabel(r"$(\mathcal{B}_k-\mathcal{B}_0)/|\mathcal{B}_0|$")
    axes[0].set_title("training bound (fresh fits)")
    axes[0].legend(fontsize=9.5, ncol=2)
    axes[1].set_xlabel("outer iteration")
    axes[1].set_ylabel("relative bound increase per iteration")
    axes[1].set_title(r"no iteration decreased the bound; tol $10^{-5}$")
    axes[1].axhline(1e-5, color=PALETTE["grey"], ls="--", lw=1.2)
    # c: validation NLL along the two 100-iteration chains (diagnostic only)
    ax = axes[2]
    for r in runs:
        if r["config"].get("suffix") != "_cont70":
            continue
        basetag = r["config"]["warm"]
        b0 = next(q for q in runs if q["tag"] == basetag)
        its = [p["it"] for p in b0["val_nll_diagnostic"]] + [30 + p["it"] for p in r["val_nll_diagnostic"]]
        vv = [p["val_nll"] for p in b0["val_nll_diagnostic"]] + [p["val_nll"] for p in r["val_nll_diagnostic"]]
        its.append(30 + r["iterations"]); vv.append(r["val"]["nll"])
        c = r["config"]
        ax.plot(its, vv, "o-", color=MODE_COLORS[c["mode"]], lw=2, label=f"{MODE_LABEL[c['mode']]}, K={c['K']}, d={c['d']}")
    if ref is not None:
        ax.axhline(ref, color=PALETTE["grey"], ls="--", lw=1.4, label="logit + offsets ($k=40$)")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("validation NLL / cell")
    ax.set_title("prediction keeps improving (diagnostic only)")
    ax.legend(fontsize=10)
    for a_, l_ in zip(axes, "abc"):
        panel_label(a_, l_)
    fig.tight_layout()
    finalize_figure(fig, FIG / "full_convergence")

    # --- d sweep (all fits incl. the d = 40, 80 extension)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    vv = sel[sel.chosen_restart & (sel.lam == 1.0) & ((sel.residual_mode == "none") | (sel.t_init == 0.25))
             & (sel.residual_mode != "isoL")]
    for (mode, K), g in vv.groupby(["residual_mode", "K"]):
        if K not in (1, 4):
            continue
        pts = g.sort_values("val_nll").groupby("d").head(1).sort_values("d")
        col = MODE_COLORS[mode]
        ls = "-" if K == 1 else "--"
        ax.plot(pts.d, pts.val_nll, ls, color=col, lw=2, label=f"{MODE_LABEL[mode].replace('fixed $t$', '$t=0.25$')}, $K={K}$")
        for _, r in pts.iterrows():
            ax.plot(r.d, r.val_nll, "o", ms=8, color=col, mfc=col if r.converged else "white", mew=2)
    if ref is not None:
        ax.axhline(ref, color=PALETTE["grey"], ls=":", lw=1.4, label="logit + offsets ($k=40$)")
    ax.plot([], [], "o", color=PALETTE["grey"], mfc=PALETTE["grey"], label="converged (tol $10^{-5}$)")
    ax.plot([], [], "o", color=PALETTE["grey"], mfc="white", mew=2, label="iteration cap reached")
    ax.set_xscale("log", base=2)
    ax.set_xticks([5, 10, 20, 40, 80], ["5", "10", "20", "40", "80"])
    ax.set_xlabel(r"local dimension $d$")
    ax.set_ylabel("validation NLL / cell")
    ax.set_title("latent dimension is the main lever (best run per point)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    finalize_figure(fig, FIG / "full_d_sweep")

    # --- occupancy figure (selected model)
    rsp = np.load(OUT / "responsibilities_best_K_gt_1.npy")
    best = {**best, "best_tag": best["responsibilities_best_K_gt_1_tag"], "pi": best["pi_best_K_gt_1"]}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    occ = rsp.mean(0)
    order = np.argsort(-occ)
    ax = axes[0]
    ax.bar(np.arange(len(occ)), occ[order], color=PALETTE["blue_main"])
    ax.bar(np.arange(len(occ)), np.asarray(best["pi"])[order], color="none", edgecolor=PALETTE["red_strong"], lw=1.5,
           label=r"$\pi_\ell$")
    ax.set_xticks(np.arange(len(occ)), [str(i + 1) for i in order], fontsize=10)
    ax.set_xlabel("component (sorted by occupancy)")
    ax.set_ylabel(r"mean responsibility $\bar r_\ell$")
    ax.set_title(f"occupancy, best K>1 fit ({best['best_tag']})", fontsize=12.5)
    ax.legend()
    ax = axes[1]
    ax.hist(rsp.max(1), bins=40, color=PALETTE["teal"], edgecolor="white")
    ax.set_xlabel(r"$\max_\ell r_{i\ell}$ per cell")
    ax.set_ylabel("cells")
    ax.set_title("assignment certainty")
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    fig.tight_layout()
    finalize_figure(fig, FIG / "full_occupancy")


if __name__ == "__main__":
    {"synthetic": synthetic, "full": full}[sys.argv[1]]()
