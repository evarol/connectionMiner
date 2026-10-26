"""W2 (v4): revised comparison table and selection figure, adding the v4 refits.

Every family is selected on VALIDATION NLL only (test is reported for the selected config).
Pools: v3 grids (models/*) plus the v4 refits (v4/configs/*.json, v4/tables/neural_v4.csv).
Grid-edge flags: a selected hyperparameter equal to the largest or smallest value actually run.
Outputs
    v4/tables/comparison_v4.csv     one row per family (+ baselines)
    v4/tables/grid_v4.csv           every configuration in every pool (val / test, start, converged)
    v4/figures/fig_select_all_v4.{png,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import figstyle as fs  # noqa: E402
import v3style as vs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

fs.apply_publication_style()
ST = vs.STUDY
MOD = ST / "models"
V4 = ST / "v4"
FIG, TAB = V4 / "figures", V4 / "tables"
POIS_COL = "#4D4D4D"
LAMS = [1, 3, 10, 30, 100, 300]
LAM_COL = dict(zip(LAMS, [plt.get_cmap("viridis")(x) for x in np.linspace(0.85, 0.0, len(LAMS))]))
EDGE_COL = fs.PALETTE["red_strong"]


def col(code):
    return POIS_COL if code == "PO" else vs.colour(code)


def cname(code):
    return "Poisson counts" if code == "PO" else vs.name(code)


# ----------------------------------------------------------------------------- pools
def bf_pool():
    resc = pd.read_csv(MOD / "rescored/selection.csv")
    rows = []
    for code, m, pre in [("R3", "logit (no offsets)", "cv_k"), ("R4", "logit + offsets", "cv_off_k"),
                         ("R5", "cloglog + offsets", "cv_cll_k")]:
        for r in resc[resc.model == m].itertuples():
            js = ST / f"full/tables/{pre}{r.k}_lam{r.lam:g}.json"
            j = json.loads(js.read_text()) if js.exists() else {}
            rows.append(dict(code=code, k=r.k, lam=r.lam, val_nll=r.val_nll, val_auc=r.val_auc,
                             val_brier=r.val_brier, test_nll=r.test_nll, test_auc=r.test_auc,
                             test_brier=r.test_brier, iterations=j.get("iterations"),
                             converged=j.get("converged"), start="v3 path", source="v3", seconds=j.get("seconds")))
    for f in sorted((V4 / "configs").glob("off_*.json")):
        j = json.loads(f.read_text())
        rows.append(dict(code="R4", k=j["k"], lam=j["lam"], **{x: j[x] for x in (
            "val_nll", "val_auc", "val_brier", "test_nll", "test_auc", "test_brier", "iterations", "converged",
            "seconds")}, start=j["init"], source=f.stem))
    return pd.DataFrame(rows)


def cnt_pool():
    c = pd.read_csv(MOD / "counts/tables/selection_full.csv")
    c = c[c.depth == "train"]
    rows = []
    for r in c.itertuples():
        # k=120 was padded from k=80; the v3 k=160 trace is reproduced bit-for-bit by a k=120-padded start
        # (v4 nb_k160_lam10_from120), so it was padded from k=120, not k=80 as run_counts.extend's default says
        start = "v3 path" if r.k <= 80 else ("v3 extend (k=80 padded)" if r.k == 120 else "v3 extend (k=120 padded)")
        rows.append(dict(code="R6" if r.family == "nb" else "PO", k=r.k, lam=r.lam, val_nll=r.val_nll,
                         val_auc=r.val_auc, val_brier=r.val_brier, test_nll=r.test_nll, test_auc=r.test_auc,
                         test_brier=r.test_brier, iterations=r.iterations, converged=r.converged,
                         phi_median=r.phi_median, start=start, source="v3", seconds=r.seconds))
    for f in sorted((V4 / "configs").glob("*.json")):
        if not f.stem.startswith(("nb_", "poisson_")):
            continue
        j = json.loads(f.read_text())
        rows.append(dict(code="R6" if j["family"] == "nb" else "PO", k=j["k"], lam=j["lam"],
                         **{x: j[x] for x in ("val_nll", "val_auc", "val_brier", "test_nll", "test_auc",
                                              "test_brier", "iterations", "converged", "phi_median", "seconds")},
                         start=f.stem.split("_")[-1], source=f.stem))
    return pd.DataFrame(rows)


def neural_pool():
    n = pd.read_csv(MOD / "neural/tables/selection.csv")
    n = n[n.chosen_restart | n.config.duplicated(keep=False).eq(False)]
    n = n.assign(source="v3", budget="1500 s wall clock")
    f = V4 / "tables/neural_v4.csv"
    if f.exists():
        m = pd.read_csv(f)
        m["hidden_widths"] = m["hidden_widths"].map(lambda w: "x".join(str(x) for x in json.loads(w))
                                                    if isinstance(w, str) and w.startswith("[") else w)
        m = m.assign(source="neural/" + m["name"], budget="250 outer its / tol 2e-5")
        n = pd.concat([n, m], ignore_index=True)
    n["converged"] = n.termination.eq("tol")
    n["code"] = "R8"
    return n


# ----------------------------------------------------------------------------- selection
def edge_flags(sel, pool, keys):
    notes = []
    for k in keys:
        vals = sorted(pool[k].dropna().unique())
        v = sel[k]
        if len(vals) > 1 and v == vals[-1]:
            notes.append(f"{k}={v:g} is the largest run")
        elif len(vals) > 1 and v == vals[0]:
            notes.append(f"{k}={v:g} is the smallest run")
    return notes


def build():
    rows, grid = [], []
    bf, cn, ne = bf_pool(), cnt_pool(), neural_pool()
    v3comp = pd.read_csv(MOD / "comparison.csv")
    for code, pool in [("R3", bf), ("R4", bf), ("R5", bf), ("R6", cn), ("PO", cn)]:
        P = pool[(pool.code == code) & (pool.k > 0)]
        grid.append(P)
        s = P.loc[P.val_nll.idxmin()]
        # edge on k within the selected lambda's column, lambda within the selected k's row
        notes = edge_flags(s, P[P.lam == s.lam], ["k"]) + edge_flags(s, P[P.k == s.k], ["lam"])
        rows.append(dict(code=code, model=cname(code), config=f"k={int(s.k)}, λ={s.lam:g}", source=s.source,
                         start=s.start, **{x: s[x] for x in ("val_nll", "val_auc", "val_brier", "test_nll",
                                                             "test_auc", "test_brier", "iterations")},
                         converged=bool(s.converged), grid_edge=bool(notes), edge_note="; ".join(notes),
                         n_configs=len(P)))
    # R7 mixture of local factors (unchanged; v3 best.json)
    mb = json.loads((MOD / "mfa/best.json").read_text())
    mfa = pd.read_csv(MOD / "mfa/tables/selection.csv")
    r7 = mfa[mfa.tag == mb["best_tag"]].iloc[0]
    notes = edge_flags(r7, mfa[mfa.K == r7.K], ["d"])
    r7_edge = bool(notes)
    notes.append("K=1 is the smallest K run (no mixture; not counted as an edge)")
    rows.append(dict(code="R7", model=cname("R7"), config=f"K={r7.K}, d={r7.d}, residual {r7.residual_mode}",
                     source="v3", start="kmeans", **{x: r7[x] for x in ("val_nll", "val_auc", "val_brier",
                                                                         "test_nll", "test_auc", "test_brier",
                                                                         "iterations")},
                     converged=bool(r7.converged), grid_edge=r7_edge, edge_note="; ".join(notes), n_configs=len(mfa)))
    # R8 neural
    grid.append(ne.assign(k=ne.latent_dim, lam=ne.lambda_output))
    s = ne.loc[ne.val_nll.idxmin()]
    base = ne[(ne.head_mode == "affine") & (ne.optimizer_mode == "pg_gem") & (ne.hidden_widths == s.hidden_widths)]
    notes = (edge_flags(s, base[base.lambda_output == s.lambda_output], ["latent_dim"])
             + edge_flags(s, base[base.latent_dim == s.latent_dim], ["lambda_output"]))
    rows.append(dict(code="R8", model=cname("R8"),
                     config=f"{s.head_mode} head, d={int(s.latent_dim)}, {s.hidden_widths}, λV={s.lambda_output:g}",
                     source=s.source, start=f"SVD init, seed {int(s.restart) + 1}; {s.budget}",
                     **{x: s[x] for x in ("val_nll", "val_auc", "val_brier", "test_nll", "test_auc", "test_brier",
                                          "iterations")},
                     converged=bool(s.converged), grid_edge=bool(notes),
                     edge_note="; ".join(notes) + f"; termination={s.termination}", n_configs=len(ne)))
    # R9 LBM
    lb = pd.read_csv(MOD / "lbm/tables/selection.csv")
    L = lb[lb.Kr > 1]
    s = L.loc[L.val_nll.idxmin()]
    notes = edge_flags(s, L, ["Kr", "Kc"])
    rows.append(dict(code="R9", model=cname("R9"), config=f"Kr={s.Kr}, Kc={s.Kc}", source="v3", start="restarts",
                     **{x: s[x] for x in ("val_nll", "val_auc", "val_brier", "test_nll", "test_auc", "test_brier")},
                     iterations=s.n_iter, converged=s.status == "converged", grid_edge=bool(notes),
                     edge_note="; ".join(notes), n_configs=len(L)))
    d = pd.DataFrame(rows)
    old = {"R3": "BernFac logit", "R4": "BernFac logit + offsets", "R5": "BernFac cloglog + offsets",
           "R6": "NB count factorization", "PO": "Poisson count factorization", "R7": "mixture of local factors",
           "R8": "neural manifold", "R9": "latent block model"}
    d["v3_config"] = d.code.map(lambda c: v3comp.set_index("model").loc[old[c], "config"])
    d["v3_val_nll"] = d.code.map(lambda c: v3comp.set_index("model").loc[old[c], "val_nll"])
    d["v3_test_nll"] = d.code.map(lambda c: v3comp.set_index("model").loc[old[c], "test_nll"])
    d["selection_changed"] = d.source != "v3"
    resc = pd.read_csv(MOD / "rescored/selection.csv")
    for m in ("gene rate", "constant rate"):
        r = resc[resc.model == m].iloc[0]
        d = pd.concat([d, pd.DataFrame([dict(code="base", model=m, config="", source="v3", val_nll=r.val_nll,
                                             val_auc=r.val_auc, val_brier=r.val_brier, test_nll=r.test_nll,
                                             test_auc=r.test_auc, test_brier=r.test_brier)])], ignore_index=True)
    d = d.sort_values("val_nll").reset_index(drop=True)
    d.to_csv(TAB / "comparison_v4.csv", index=False)
    g = pd.concat(grid, ignore_index=True)
    keep = ["code", "k", "lam", "val_nll", "test_nll", "val_auc", "test_auc", "val_brier", "test_brier",
            "iterations", "converged", "start", "source", "seconds", "phi_median", "name", "termination", "budget"]
    g[[c for c in keep if c in g.columns]].to_csv(TAB / "grid_v4.csv", index=False)
    return d, bf, cn, ne


# ----------------------------------------------------------------------------- figure
def fig(d, bf, cn, ne):
    YLO, YHI = 0.1762, 0.1895
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.6), sharey=True)
    axes = axes.ravel()
    best_val = d[d.code != "base"].val_nll.min()
    sel = d.set_index("code")

    def clip_plot(ax, x, y, **kw):
        y = np.asarray(y, float)
        ax.plot(x, np.minimum(y, YHI - 0.0004), **kw)
        over = y > YHI
        if over.any():
            ax.scatter(np.asarray(x)[over], np.full(over.sum(), YHI - 0.0004), marker="^", s=70,
                       color=kw.get("color", "k"), zorder=5, clip_on=False)

    def star(ax, x, y, code):
        r = sel.loc[code]
        ax.scatter([x], [y], marker="*", s=420, color=col(code), edgecolor="k", linewidth=1.2, zorder=10)
        if r.grid_edge:
            ax.scatter([x], [y], marker="o", s=900, facecolor="none", edgecolor=EDGE_COL, linewidth=2.0, zorder=9)

    def title(ax, code):
        ax.set_title(f"{code}  {cname(code)}", color=col(code), fontweight="bold", fontsize=fs.FS_TITLE)

    def sel_text(ax, code, txt, x=0.03, y=0.92, ha="left"):
        r = sel.loc[code]
        conv = "" if r.converged else ", not converged"
        ax.text(x, y, f"selected {txt}{conv}", transform=ax.transAxes, va="top", ha=ha, fontsize=fs.FS_SMALL)
        if r.grid_edge:
            ax.text(x, y - 0.07, "grid edge: " + r.edge_note.split(";")[0].replace("lam", "λ"),
                    transform=ax.transAxes, va="top", ha=ha, fontsize=fs.FS_SMALL - 1, color=EDGE_COL)

    def kx(ax, ks):
        ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(int(k)) for k in ks])
        ax.set_xlabel("rank $k$")

    def k_lam_panel(ax, P, code, txt_kw=None):
        old, new = P[P.source == "v3"], P[P.source != "v3"]
        for lam, g in P.groupby("lam"):
            g = g.groupby("k").val_nll.min().sort_index()     # best start per (k, lambda)
            clip_plot(ax, g.index, g.values, color=LAM_COL[int(lam)], marker="o", lw=2, ms=6)
        if len(new):
            ax.scatter(new.k, np.minimum(new.val_nll, YHI - 0.0004), marker="D", s=95, facecolor="white",
                       edgecolor=[LAM_COL[int(l)] for l in new.lam], linewidth=2.2, zorder=8)
        r = sel.loc[code]
        s = P[P.source == r.source] if r.source != "v3" else P.loc[[P.val_nll.idxmin()]]
        s = s.loc[s.val_nll.idxmin()]
        star(ax, s.k, s.val_nll, code)
        kx(ax, sorted(P.k.unique()))
        title(ax, code)
        sel_text(ax, code, f"$k$={int(s.k)}, $\\lambda$={s.lam:g}", **(txt_kw or {}))

    for ax, c in zip(axes[:3], ["R3", "R4", "R5"]):
        k_lam_panel(ax, bf[(bf.code == c) & (bf.k > 0)], c)
    for ax, c in zip([axes[3], axes[4]], ["R6", "PO"]):
        k_lam_panel(ax, cn[cn.code == c], c, dict(x=0.97, ha="right"))
    hs = [Line2D([], [], color=LAM_COL[l], marker="o", lw=2, label=f"{l:g}") for l in LAMS]
    hs.append(Line2D([], [], color="k", marker="D", mfc="white", lw=0, ms=8, label="v4 refit"))
    axes[1].legend(handles=hs, title="penalty $\\lambda$ (panels a–e)", ncol=3, loc="upper right",
                   fontsize=fs.FS_SMALL - 1, bbox_to_anchor=(1.0, 0.80), columnspacing=0.8, handlelength=1.4)
    # R7
    ax = axes[5]
    m = pd.read_csv(MOD / "mfa/tables/selection.csv")
    main = m[(m.lam == 1.0) & (((m.residual_mode == "iso") & (m.t_init == 0.25)) | (m.residual_mode == "none"))]
    Ks = [1, 4, 8, 16]
    kc = dict(zip(Ks, fs.k_ramp(len(Ks) + 1)[1:]))
    for (mode, K), g in main.groupby(["residual_mode", "K"]):
        g = g.groupby("d").val_nll.min().sort_index()
        clip_plot(ax, g.index, g.values, color=kc[K], marker="o" if mode == "none" else "s",
                  ls="-" if mode == "none" else "--", lw=2, ms=6)
    other = m.drop(main.index)
    ax.scatter(other.d * 1.08, np.minimum(other.val_nll, YHI - 0.0004), s=22, color=fs.PALETTE["neutral"],
               edgecolor=fs.PALETTE["grey_mid"], zorder=3)
    r7 = sel.loc["R7"]
    d7 = int(r7.config.split("d=")[1].split(",")[0])
    ax.scatter([d7], [r7.val_nll], marker="*", s=420, color=col("R7"), edgecolor="k", linewidth=1.2, zorder=10)
    ax.set_xscale("log", base=2); ax.set_xticks([5, 10, 20, 40, 80]); ax.set_xticklabels(["5", "10", "20", "40", "80"])
    ax.set_xlabel("local dimension $d$")
    title(ax, "R7")
    hs = [Line2D([], [], color=kc[K], lw=2, label=f"$K$={K}") for K in Ks] + [
        Line2D([], [], color="k", lw=2, marker="o", label="no residual"),
        Line2D([], [], color="k", lw=2, ls="--", marker="s", label="isotropic")]
    ax.legend(handles=hs, loc="upper right", fontsize=fs.FS_SMALL - 2, ncol=2, columnspacing=0.6, handlelength=1.4)
    ax.text(0.03, 0.04, f"selected $K$=1, $d$={d7}; $K$>1 fits mostly not converged",
            transform=ax.transAxes, fontsize=fs.FS_SMALL - 1)
    # R8
    ax = axes[6]
    n = ne
    for budget, mk, ls, lab in [("1500 s wall clock", "o", ":", "v3: 1500 s wall clock"),
                                ("250 outer its / tol 2e-5", "D", "-", "v4: iteration / tol budget")]:
        b = n[(n.budget == budget) & (n.optimizer_mode == "pg_gem") & (n.head_mode == "affine")
              & (n.hidden_widths == "64x64")]
        for lv, g in b.groupby("lambda_output"):
            g = g.groupby("latent_dim").val_nll.min().sort_index()
            cc = {1.0: col("R8"), 10.0: "#8E3F72", 30.0: "#E3A9CC"}[lv]
            ax.plot(g.index, np.minimum(g.values, YHI - 0.0004), color=cc, ls=ls, marker=mk, lw=2, ms=7,
                    mfc="white" if mk == "D" else cc, mew=2)
    oth = n[(n.hidden_widths != "64x64") | (n.head_mode != "affine") | (n.optimizer_mode != "pg_gem")]
    ax.scatter(oth.latent_dim * 1.06, oth.val_nll, marker="X", s=60, color=fs.PALETTE["grey_mid"], zorder=5)
    r8 = sel.loc["R8"]
    s = n.loc[n.val_nll.idxmin()]
    star(ax, s.latent_dim, s.val_nll, "R8")
    ax.set_xscale("log", base=2); ax.set_xticks([5, 10, 20, 40]); ax.set_xticklabels(["5", "10", "20", "40"])
    ax.set_xlim(4, 52)
    ax.set_xlabel("latent dimension $d$")
    title(ax, "R8")
    hs = [Line2D([], [], color=col("R8"), ls=":", marker="o", lw=2, label="1500 s budget (v3)"),
          Line2D([], [], color=col("R8"), ls="-", marker="D", mfc="white", mew=2, lw=2, label="iteration budget (v4)"),
          Line2D([], [], color=col("R8"), lw=4, label="$\\lambda_V$=1"),
          Line2D([], [], color="#8E3F72", lw=4, label="$\\lambda_V$=10"),
          Line2D([], [], color="#E3A9CC", lw=4, label="$\\lambda_V$=30"),
          Line2D([], [], color=fs.PALETTE["grey_mid"], marker="X", lw=0, label="other head / width / optimiser")]
    ax.legend(handles=hs, loc="center right", fontsize=fs.FS_SMALL - 2, handlelength=1.6,
              bbox_to_anchor=(1.0, 0.45))
    conv = "converged (tol)" if r8.converged else f"not converged ({s.termination})"
    ax.text(0.03, 0.92, f"selected $d$={int(s.latent_dim)}, $\\lambda_V$={s.lambda_output:g}: {conv}",
            transform=ax.transAxes, fontsize=fs.FS_SMALL - 1, va="top")
    if r8.grid_edge:
        ax.text(0.03, 0.85, "grid edge: " + r8.edge_note.split(";")[0].replace("lambda_output", "λV")
                .replace("latent_dim", "d"), transform=ax.transAxes, fontsize=fs.FS_SMALL - 1, color=EDGE_COL, va="top")
    # R9
    ax = axes[7]
    lb = pd.read_csv(MOD / "lbm/tables/selection.csv")
    L = lb[lb.Kr > 1]
    Krs = sorted(L.Kr.unique())
    krc = dict(zip(Krs, fs.k_ramp(len(Krs) + 1)[1:]))
    for Kr in Krs:
        g = L[L.Kr == Kr].groupby("Kc").val_nll.min().sort_index()
        clip_plot(ax, g.index, g.values, color=krc[Kr], marker="o", lw=2, ms=6)
    r9 = sel.loc["R9"]
    kc9 = int(r9.config.split("Kc=")[1])
    star(ax, kc9, r9.val_nll, "R9")
    ax.set_xscale("log"); kcs = sorted(L.Kc.unique()); ax.set_xticks(kcs); ax.set_xticklabels([str(k) for k in kcs])
    ax.set_xlabel("gene blocks $K_c$")
    title(ax, "R9")
    ax.legend(handles=[Line2D([], [], color=krc[k], lw=2, marker="o", label=str(k)) for k in Krs],
              title="cell blocks $K_r$", loc="upper right", fontsize=fs.FS_SMALL - 2, ncol=2,
              columnspacing=0.6, handlelength=1.2)
    ax.text(0.03, 0.04, f"selected {r9.config}: grid edge", transform=ax.transAxes, fontsize=fs.FS_SMALL,
            color=EDGE_COL)

    for i, ax in enumerate(axes):
        ax.axhline(best_val, color=fs.PALETTE["grey_dark"], lw=1.3, ls=":", zorder=1)
        ax.set_ylim(YLO, YHI)
        ax.grid(axis="y", color="#EEEEEE", lw=0.8, zorder=0)
        fs.panel_label(ax, "abcdefgh"[i], x=-0.12 if i % 4 == 0 else -0.06)
    for ax in axes[[0, 4]]:
        ax.set_ylabel("validation NLL (nats / entry)")
    gene = d[d.model == "gene rate"].val_nll.iloc[0]
    fig.text(0.5, -0.01, f"★ = selected configuration (lowest validation NLL in the family); red ring = selected value "
             f"sits at the edge of the grid that was run; ◇ = v4 refit.  Dotted line = best overall validation NLL "
             f"{best_val:.4f}.  Gene-rate baseline = {gene:.4f} (off scale).", ha="center", fontsize=fs.FS_SMALL)
    fig.tight_layout(w_pad=1.0, h_pad=2.0)
    fs.finalize_figure(fig, FIG / "fig_select_all_v4")


if __name__ == "__main__":
    d, bf, cn, ne = build()
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(d[["code", "model", "config", "source", "val_nll", "test_nll", "test_auc", "test_brier",
                 "converged", "grid_edge", "edge_note", "v3_config", "selection_changed"]].to_string())
    fig(d, bf, cn, ne)
