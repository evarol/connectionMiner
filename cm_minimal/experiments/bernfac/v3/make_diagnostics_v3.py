"""Worker D figures for the v3 deck: model selection, convergence, runtime, scorecard, tests.

Reads only precomputed tables, traces and logs (no refitting).  Outputs go to
outputs/bernfac_20261020/v3/{figures,tables}.
    fig_select_all, fig_converge_all, fig_runtime, fig_scorecard, fig_tests
"""
from __future__ import annotations

import json
import re
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
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib import cm  # noqa: E402

fs.apply_publication_style()
ST = vs.STUDY
MOD = ST / "models"
FIG = vs.FIGS
TAB = vs.V3 / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

POIS_COL = "#4D4D4D"
LAMS = [1, 3, 10, 30, 100, 300]
LAM_COL = dict(zip(LAMS, [plt.get_cmap("viridis")(x) for x in np.linspace(0.85, 0.0, len(LAMS))]))


def col(code):
    return POIS_COL if code == "Pois" else vs.colour(code)


def label(code):
    return "Poisson counts" if code == "Pois" else f"{code}  {vs.name(code)}"


# ----------------------------------------------------------------------------- data
resc = pd.read_csv(MOD / "rescored/selection.csv")
cnt = pd.read_csv(MOD / "counts/tables/selection_full.csv")
mfa = pd.read_csv(MOD / "mfa/tables/selection.csv")
neu = pd.read_csv(MOD / "neural/tables/selection.csv")
lbm = pd.read_csv(MOD / "lbm/tables/selection.csv")
comp = pd.read_csv(MOD / "comparison.csv")
umap = pd.read_csv(ST / "full/umap/metrics.csv").set_index("rep")
BEST_VAL = comp["val_nll"].min()
GENE_RATE_VAL = float(comp.loc[comp.model == "gene rate", "val_nll"].iloc[0])

BF = {"R3": ("logit (no offsets)", "cv_k{k}_lam{lam}"),
      "R4": ("logit + offsets", "cv_off_k{k}_lam{lam}"),
      "R5": ("cloglog + offsets", "cv_cll_k{k}_lam{lam}")}
BF_SEL = {}
for c, (m, _) in BF.items():
    d = resc[resc.model == m]
    r = d.loc[d.val_nll.idxmin()]
    BF_SEL[c] = (int(r.k), float(r.lam))
CNT_SEL = {"R6": json.load(open(MOD / "counts/best_nb.json")),
           "Pois": json.load(open(MOD / "counts/best_poisson.json"))}
MFA_BEST = json.load(open(MOD / "mfa/best.json"))
NEU_BEST = json.load(open(MOD / "neural/best.json"))
LBM_BEST = json.load(open(MOD / "lbm/best.json"))


def _lamkey(x):
    return f"{x:g}"


# ============================================================================ 1. selection
def fig_select_all():
    YLO, YHI = 0.1762, 0.1895
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.6), sharey=True)
    axes = axes.ravel()

    def clip_plot(ax, x, y, **kw):
        y = np.asarray(y, float)
        yc = np.minimum(y, YHI - 0.0004)
        ax.plot(x, yc, **kw)
        over = y > YHI
        if over.any():
            ax.scatter(np.asarray(x)[over], np.full(over.sum(), YHI - 0.0004), marker="^", s=70,
                       color=kw.get("color", "k"), zorder=5, clip_on=False)

    def star(ax, x, y, c):
        ax.scatter([x], [y], marker="*", s=420, color=c, edgecolor="k", linewidth=1.2, zorder=10)

    def title(ax, code, extra=""):
        ax.set_title(f"{code}  {vs.name(code) if code != 'Pois' else 'Poisson counts'}{extra}",
                     color=col(code), fontweight="bold", fontsize=fs.FS_TITLE)

    # BernFac links
    for ax, c in zip(axes[:3], ["R3", "R4", "R5"]):
        d = resc[(resc.model == BF[c][0]) & (resc.k > 0)]
        for lam, g in d.groupby("lam"):
            g = g.sort_values("k")
            clip_plot(ax, g.k, g.val_nll, color=LAM_COL[int(lam)], marker="o", lw=2, ms=6)
        k, lam = BF_SEL[c]
        star(ax, k, d[(d.k == k) & (d.lam == lam)].val_nll.iloc[0], col(c))
        ax.set_xscale("log", base=2)
        ax.set_xticks([10, 20, 40, 80]); ax.set_xticklabels(["10", "20", "40", "80"])
        ax.set_xlabel("rank $k$")
        title(ax, c)
        ax.text(0.03, 0.92, f"selected $k$={k}, $\\lambda$={lam:g}", transform=ax.transAxes,
                va="top", fontsize=fs.FS_SMALL)
        if c == "R5":
            k0 = resc[(resc.model == BF[c][0]) & (resc.k == 0)].val_nll.iloc[0]
            ax.text(0.03, 0.84, f"$k$=0 (offsets only): {k0:.4f}", transform=ax.transAxes, va="top",
                    fontsize=fs.FS_SMALL - 1, color=fs.PALETTE["grey"])
        if c == "R3":
            ax.text(0.97, 0.80, "$\\lambda$=300 ≈ 0.195\n(all $k$, ▲ = off scale)", transform=ax.transAxes,
                    ha="right", va="top", fontsize=fs.FS_SMALL - 1, color=fs.PALETTE["grey"])
    # NB and Poisson
    for ax, (c, fam) in zip([axes[3], axes[4]], [("R6", "nb"), ("Pois", "poisson")]):
        d = cnt[(cnt.family == fam) & (cnt.depth == "train")]
        for lam, g in d.groupby("lam"):
            g = g.sort_values("k")
            clip_plot(ax, g.k, g.val_nll, color=LAM_COL[int(lam)], marker="o", lw=2, ms=6)
        b = CNT_SEL[c]
        star(ax, b["k"], b["val_nll"], col(c))
        ax.set_xscale("log", base=2)
        ks = sorted(d.k.unique()); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("rank $k$")
        title(ax, c)
        ax.text(0.97, 0.92, f"selected $k$={b['k']}, $\\lambda$={b['lam']:g}", transform=ax.transAxes,
                va="top", ha="right", fontsize=fs.FS_SMALL)
    # lambda legend (shared by the five k x lambda panels)
    hs = [Line2D([], [], color=LAM_COL[l], marker="o", lw=2, label=f"{l:g}") for l in LAMS]
    axes[1].legend(handles=hs, title="penalty $\\lambda$ (panels a–e)", ncol=3, loc="upper right",
                   fontsize=fs.FS_SMALL - 1, bbox_to_anchor=(1.0, 0.86), columnspacing=0.8, handlelength=1.4)
    # MFA
    ax = axes[5]
    m = mfa.copy()
    main = m[(m.lam == 1.0) & (((m.residual_mode == "iso") & (m.t_init == 0.25)) | (m.residual_mode == "none"))]
    other = m.drop(main.index)
    Ks = [1, 4, 8, 16]
    kc = dict(zip(Ks, fs.k_ramp(len(Ks) + 1)[1:]))
    for (mode, K), g in main.groupby(["residual_mode", "K"]):
        g = g.groupby("d").val_nll.min().sort_index()
        clip_plot(ax, g.index, g.values, color=kc[K], marker="o" if mode == "none" else "s",
                  ls="-" if mode == "none" else "--", lw=2, ms=6)
    ax.scatter(other.d * (1.08), np.minimum(other.val_nll, YHI - 0.0004), s=22, color=fs.PALETTE["neutral"],
               edgecolor=fs.PALETTE["grey_mid"], zorder=3)
    bb = MFA_BEST["best_K1"]
    star(ax, bb["d"], bb["val_nll"], col("R7"))
    ax.set_xscale("log", base=2); ax.set_xticks([5, 10, 20, 40, 80]); ax.set_xticklabels(["5", "10", "20", "40", "80"])
    ax.set_xlabel("local dimension $d$")
    title(ax, "R7")
    hs = [Line2D([], [], color=kc[K], lw=2, label=f"$K$={K}") for K in Ks] + [
        Line2D([], [], color="k", lw=2, marker="o", label="no residual"),
        Line2D([], [], color="k", lw=2, ls="--", marker="s", label="isotropic"),
        Line2D([], [], color=fs.PALETTE["neutral"], marker="o", lw=0, mec=fs.PALETTE["grey_mid"],
               label="other ($\\lambda$=10, $t$, iso-L)")]
    ax.legend(handles=hs, loc="upper right", fontsize=fs.FS_SMALL - 2, ncol=2, columnspacing=0.6,
              handlelength=1.4)
    ax.text(0.03, 0.04, f"selected $K$=1, $d$={bb['d']}", transform=ax.transAxes,
            fontsize=fs.FS_SMALL)
    # Neural
    ax = axes[6]
    n = neu.copy()
    base = n[(n.optimizer_mode == "pg_gem") & (n.head_mode == "affine") & (n.hidden_widths == "64x64")
             & (n.lambda_output == 1.0)]
    g = base.groupby("latent_dim").val_nll.min().sort_index()
    ax.plot(g.index, g.values, color=col("R8"), marker="o", lw=2, ms=7, label="affine head, 64×64, $\\lambda_V$=1")
    ax.scatter(base.latent_dim, base.val_nll, color=col("R8"), s=25, zorder=4)
    specs = [((n.lambda_output == 10.0), "s", "#E3A9CC", "$\\lambda_V$=10"),
             ((n.lambda_output == 30.0), "s", "#8E3F72", "$\\lambda_V$=30"),
             ((n.hidden_widths == "128x128"), "D", fs.PALETTE["grey"], "width 128×128"),
             ((n.head_mode == "nonlinear"), "^", fs.PALETTE["violet"], "nonlinear head (warm)"),
             ((n.optimizer_mode == "direct_bce"), "X", fs.PALETTE["grey_mid"], "direct BCE optimiser")]
    for mask, mk, cc, lab in specs:
        r = n[mask]
        ax.scatter(r.latent_dim * 1.06, r.val_nll, marker=mk, s=80, color=cc, edgecolor="k", lw=0.6,
                   label=lab, zorder=6)
    sel = n[n.name == NEU_BEST["selected"]].iloc[0]
    star(ax, sel.latent_dim, sel.val_nll, col("R8"))
    ax.set_xscale("log", base=2); ax.set_xticks([5, 10, 20]); ax.set_xticklabels(["5", "10", "20"])
    ax.set_xlim(4, 26)
    ax.set_xlabel("latent dimension $d$")
    title(ax, "R8")
    ax.legend(loc="upper right", fontsize=fs.FS_SMALL - 2, handlelength=1.2)
    ax.text(0.03, 0.04, "all runs hit the 1500 s limit", transform=ax.transAxes, fontsize=fs.FS_SMALL - 1,
            color=fs.PALETTE["grey"])
    # LBM
    ax = axes[7]
    L = lbm[lbm.Kr > 1]
    Krs = sorted(L.Kr.unique())
    krc = dict(zip(Krs, fs.k_ramp(len(Krs) + 1)[1:]))
    for Kr in Krs:
        g = L[L.Kr == Kr].groupby("Kc").val_nll.min().sort_index()
        clip_plot(ax, g.index, g.values, color=krc[Kr], marker="o", lw=2, ms=6)
    s = LBM_BEST["selected"]
    star(ax, s["Kc"], LBM_BEST["val"]["nll"], col("R9"))
    ax.set_xscale("log"); kcs = sorted(L.Kc.unique()); ax.set_xticks(kcs); ax.set_xticklabels([str(k) for k in kcs])
    ax.set_xlabel("gene blocks $K_c$")
    title(ax, "R9")
    ax.legend(handles=[Line2D([], [], color=krc[k], lw=2, marker="o", label=str(k)) for k in Krs],
              title="cell blocks $K_r$", loc="upper right", fontsize=fs.FS_SMALL - 2, ncol=2,
              columnspacing=0.6, handlelength=1.2)
    ax.text(0.03, 0.04, "selected 200×200: grid edge", transform=ax.transAxes, fontsize=fs.FS_SMALL,
            color=fs.PALETTE["red_strong"])

    for i, ax in enumerate(axes):
        ax.axhline(BEST_VAL, color=fs.PALETTE["grey_dark"], lw=1.3, ls=":", zorder=1)
        ax.set_ylim(YLO, YHI)
        ax.grid(axis="y", color="#EEEEEE", lw=0.8, zorder=0)
        fs.panel_label(ax, "abcdefgh"[i], x=-0.12 if i % 4 == 0 else -0.06)
    for ax in axes[[0, 4]]:
        ax.set_ylabel("validation NLL (nats / entry)")
    axes[1].text(0.98, (BEST_VAL - YLO) / (YHI - YLO) + 0.012, "best overall (R6)", transform=axes[1].transAxes,
                 ha="right", va="bottom", fontsize=fs.FS_SMALL - 1, color=fs.PALETTE["grey_dark"])
    fig.text(0.5, -0.01, f"★ = selected configuration (lowest validation NLL in the family).  Dotted line = best overall "
             f"validation NLL {BEST_VAL:.4f}.  Gene-rate baseline = {GENE_RATE_VAL:.4f} (off scale).",
             ha="center", fontsize=fs.FS_SMALL)
    fig.tight_layout(w_pad=1.0, h_pad=2.0)
    fs.finalize_figure(fig, FIG / "fig_select_all")


# ============================================================================ 2. convergence
def _parse_counts_log(path, fam, k, lam):
    pat = re.compile(rf"\[{fam}\] k={k} lam={lam:g} it=\s*(\d+) J=([-\d.e+]+) .*t=(\d+)s")
    blocks, cur = [], []
    for line in open(path):
        mt = pat.search(line)
        if mt:
            it = int(mt.group(1))
            if cur and it <= cur[-1][0]:
                blocks.append(cur); cur = []
            cur.append((it, -float(mt.group(2)), float(mt.group(3))))
    if cur:
        blocks.append(cur)
    return np.array(blocks[-1])


def traces():
    """Return {code: dict(it, F, t, F_final, iters, seconds, warm, note)} with F minimised."""
    out = {}
    for c, (_, pat) in BF.items():
        k, lam = BF_SEL[c]
        tag = pat.format(k=k, lam=int(lam))
        obj = np.load(ST / f"full/fits/{tag}.npz")["obj"].astype(float)
        js = json.load(open(ST / f"full/tables/{tag}.json"))
        F = -obj
        it = np.arange(len(F))
        out[c] = dict(it=it, F=F, t=js["seconds"] * it / it[-1], F_final=F[-1], iters=js["iterations"],
                      seconds=js["seconds"], warm=True, monotone_flag=js["monotone"], time_note="uniform")
    for c, fam, logf in [("R6", "nb", "extend_nb_k120.log"), ("Pois", "poisson", "cv_poisson_k80.log")]:
        b = CNT_SEL[c]
        a = _parse_counts_log(MOD / "counts/logs" / logf, fam, b["k"], b["lam"])
        Ff = -b["J"]
        it = np.r_[a[:, 0], b["iterations"]]
        F = np.r_[a[:, 1], Ff]
        t = np.r_[a[:, 2], b["seconds"]]
        out[c] = dict(it=it, F=F, t=t, F_final=Ff, iters=b["iterations"], seconds=b["seconds"],
                      warm=b["warm_start"], monotone_flag=b["monotone"], time_note="logged")
    r = json.load(open(MOD / f"mfa/runs/{MFA_BEST['best_tag']}.json"))
    F = -np.asarray(r["bound_trace"], float)
    it = np.arange(len(F))
    out["R7"] = dict(it=it, F=F, t=r["init_seconds"] + r["seconds_per_pass"] * it, F_final=F[-1],
                     iters=r["iterations"], seconds=r["fit_seconds"], warm=False, monotone_flag=r["monotone"],
                     time_note="per-pass", maxrss=r["maxrss_gb"])
    nr = json.load(open(MOD / f"neural/results/{NEU_BEST['selected']}.json"))
    pat = re.compile(r"it\s+(\d+) F=([\d.e+]+) .* t=(\d+)s")
    rows = [tuple(map(float, m.groups())) for m in map(pat.search, open(MOD / f"neural/logs/{NEU_BEST['selected']}.log")) if m]
    a = np.array(rows)
    out["R8"] = dict(it=np.r_[a[:, 0], nr["iterations"]], F=np.r_[a[:, 1], nr["final_F"]],
                     t=np.r_[a[:, 2], nr["seconds"]], F_final=nr["final_F"], iters=nr["iterations"],
                     seconds=nr["seconds"], warm=False, monotone_flag=nr["monotone"], time_note="logged",
                     maxrss=nr["max_rss_gb"], term=nr["termination"])
    s = LBM_BEST["selected"]
    lr = json.load(open(MOD / f"lbm/results/Kr{s['Kr']}_Kc{s['Kc']}_lam{s['lam_block']:g}.json"))
    rs = lr["restarts"][lr["restart_chosen"]]
    F = -np.asarray(rs["elbo_trace"], float)
    it = np.arange(len(F))
    out["R9"] = dict(it=it, F=F, t=rs["wall"] * it / it[-1], F_final=F[-1], iters=rs["n_iter"],
                     seconds=lr["fit_time_s"], restart_wall=rs["wall"], warm=False,
                     monotone_flag=rs["n_decrease"] == 0, time_note="uniform", n_restarts=len(lr["restarts"]))
    for c, d in out.items():
        dF = np.diff(d["F"])
        d["max_rel_rise"] = float(max(0.0, (dF / np.abs(d["F_final"])).max()))
    return out


def fig_converge_all(T):
    order = ["R3", "R4", "R5", "R6", "Pois", "R7", "R8", "R9"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6))
    for c in order:
        d = T[c]
        g = (d["F"] - d["F_final"]) / abs(d["F_final"])
        keep = g > 0
        mk = "o" if d["time_note"] == "logged" else None
        for ax, x in zip(axes, [d["it"], d["t"] / 60]):
            ax.plot(np.asarray(x)[keep], g[keep], color=col(c), lw=2.2, marker=mk, ms=5,
                    ls="--" if c == "Pois" else "-")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("$(F - F_\\mathrm{final}) / |F_\\mathrm{final}|$")
        ax.grid(color="#EEEEEE", lw=0.8)
    axes[0].set_xlabel("iteration"); axes[0].set_xscale("log")
    axes[1].set_xlabel("wall time (min)"); axes[1].set_xscale("log")
    handles = []
    for c in order:
        d = T[c]
        mono = "monotone" if d["max_rel_rise"] == 0 else f"rise ≤ {d['max_rel_rise']:.0e}"
        extra = []
        if d.get("warm"):
            extra.append("warm")
        if d.get("term") == "time_limit":
            extra.append("time limit")
        lab = f"{c if c != 'Pois' else 'Pois.'} {vs.name(c) if c != 'Pois' else 'Poisson counts'}: {d['iters']} it, {mono}"
        if extra:
            lab += " (" + ", ".join(extra) + ")"
        handles.append(Line2D([], [], color=col(c), lw=2.5, ls="--" if c == "Pois" else "-", label=lab))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=fs.FS_LEGEND, ncol=2,
               title="selected config  (F = minimised objective)", columnspacing=2.0)
    fs.panel_label(axes[0], "a"); fs.panel_label(axes[1], "b")
    fig.text(0.5, -0.24, "warm = started from the previous λ in the grid (gap is relative to that start). Markers = "
             "sparsely logged iterations (R6, Poisson, R8); BernFac and LBM wall time assumes uniform time per "
             "iteration. Final point (gap 0) omitted on log scale.", ha="center", fontsize=fs.FS_SMALL - 1, wrap=True)
    fig.tight_layout()
    fs.finalize_figure(fig, FIG / "fig_converge_all")


# ============================================================================ 3. runtime
def runtime_table(T):
    rows = []
    for c, (m, pat) in BF.items():
        grid = list((ST / "full/tables").glob(pat.split("_k")[0] + "_k*.json"))
        if c == "R3":
            grid = [p for p in (ST / "full/tables").glob("cv_k*.json")]
        tot = sum(json.load(open(p))["seconds"] for p in grid)
        k, lam = BF_SEL[c]
        rows.append(dict(code=c, config=f"k={k}, λ={lam:g}", n_configs=len(grid), grid_hours=tot / 3600))
    for c, fam in [("R6", "nb"), ("Pois", "poisson")]:
        d = cnt[(cnt.family == fam) & (cnt.depth == "train")]
        b = CNT_SEL[c]
        rows.append(dict(code=c, config=f"k={b['k']}, λ={b['lam']:g}", n_configs=len(d), grid_hours=d.seconds.sum() / 3600))
    bb = MFA_BEST["best_K1"]
    rows.append(dict(code="R7", config=f"K=1, d={bb['d']}, none", n_configs=len(mfa), grid_hours=mfa.fit_seconds.sum() / 3600))
    rows.append(dict(code="R8", config="affine, d=20, 64×64", n_configs=len(neu), grid_hours=neu.seconds.sum() / 3600))
    s = LBM_BEST["selected"]
    rows.append(dict(code="R9", config=f"{s['Kr']}×{s['Kc']}, λ={s['lam_block']:g}", n_configs=len(lbm),
                     grid_hours=lbm.fit_time_s.sum() / 3600))
    df = pd.DataFrame(rows).set_index("code")
    for c in df.index:
        d = T[c]
        df.loc[c, "fit_seconds"] = d["seconds"]
        df.loc[c, "iterations"] = d["iters"]
        df.loc[c, "sec_per_iter"] = (d.get("restart_wall", d["seconds"]) / d["iters"]) if c != "R7" else \
            json.load(open(MOD / f"mfa/runs/{MFA_BEST['best_tag']}.json"))["seconds_per_pass"]
        df.loc[c, "peak_mem_gb"] = d.get("maxrss", np.nan)
        df.loc[c, "warm_start"] = bool(d.get("warm"))
        df.loc[c, "termination"] = d.get("term", "converged" if c != "R7" else "converged")
    df.loc["R9", "note"] = f"fit time covers {T['R9']['n_restarts']} restarts; chosen restart {T['R9']['restart_wall']:.0f} s"
    df.loc["R8", "note"] = "1500 s budget; max RSS is per process (shared across sequential runs)"
    df.loc["R6", "note"] = "warm start from neighbouring λ; time excludes predecessor"
    df.loc["Pois", "note"] = df.loc["R6", "note"]
    for c in ["R3", "R4", "R5"]:
        df.loc[c, "note"] = "λ-chain warm start; time excludes predecessor"
    df.loc["R7", "note"] = "cold start (k-means init)"
    df = df.reindex(["R3", "R4", "R5", "R6", "Pois", "R7", "R8", "R9"])
    df.to_csv(TAB / "runtime.csv", float_format="%.4g")
    return df


def fig_runtime(df):
    codes = list(df.index)
    y = np.arange(len(codes))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), sharey=True, gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    cols = [col(c) for c in codes]
    specs = [("fit_seconds", "fit time, selected config (min)", 1 / 60),
             ("sec_per_iter", "time per iteration (s)", 1),
             ("grid_hours", "total grid compute (h)", 1)]
    for ax, (key, xl, sc) in zip(axes, specs):
        v = df[key].values * sc
        ax.barh(y, v, color=cols, edgecolor="k", lw=0.6, height=0.7)
        ax.set_xscale("linear" if key == "grid_hours" else "log"); ax.set_xlabel(xl)
        ax.grid(axis="x", color="#EEEEEE", lw=0.8, zorder=0); ax.set_axisbelow(True)
        for yi, vi, c in zip(y, v, codes):
            r = df.loc[c]
            if key == "fit_seconds":
                t = f"{vi:.1f} min · {int(r.iterations)} it"
                if np.isfinite(r.peak_mem_gb):
                    t += f" · {r.peak_mem_gb:.1f} GB"
            elif key == "sec_per_iter":
                t = f"{vi:.1f} s"
            else:
                t = f"{vi:.1f} h · {int(r.n_configs)} cfg"
            ax.text(vi + 0.25 if key == "grid_hours" else vi * 1.12, yi, t, va="center", fontsize=fs.FS_SMALL - 1)
        ax.set_xlim(right=ax.get_xlim()[1] * (1.7 if key == "grid_hours" else 9))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([("Pois. Poisson counts" if c == "Pois" else f"{c} {vs.name(c)}") + f"\n{df.loc[c, 'config']}"
                             for c in codes], fontsize=fs.FS_SMALL)
    for t, c in zip(axes[0].get_yticklabels(), codes):
        t.set_color(col(c))
    for i, ax in enumerate(axes):
        fs.panel_label(ax, "abc"[i], x=-0.05 if i else -0.52)
    fig.text(0.5, -0.05, "CPU, ≤4 threads. Memory shown where recorded (R7, R8). R3–R6 and Poisson were warm-started "
             "along the λ grid, so their selected-fit times exclude predecessors. R8 fits stop at a 1500 s budget. "
             "R9 time covers 3 restarts.  R1/R2 (PCA) not timed.", ha="center", fontsize=fs.FS_SMALL - 1, wrap=True)
    fig.tight_layout()
    fs.finalize_figure(fig, FIG / "fig_runtime")


# ============================================================================ 5. tests
TESTS = [
    ("R3–R5", "test_bernfac.py", "PG mean, stable σ/softplus, finite-diff gradient, block optimality, monotone J, masks, "
     "empty rows, prior shrinkage, zero-init saddle, synthetic recovery (logit core only)"),
    ("R6 + Pois.", "test_counts.py", "score/Fisher/φ finite differences, Poisson & NB recovery + monotone J, "
     "held-out mask invariance, NB→Poisson limit, held-out NLL & embedding"),
    ("R7", "test_mfa.py", "Gaussian reference, bound normalisation, ξ moments, M-step optimality, monotone bound, "
     "empty components, quadrature, gauge invariance, chunk/sparse/float32 parity, new rows, I/O"),
    ("R8", "test_neural.py", "PG mean & bound, autodiff finite-diff, frozen weights, head solve, monotone + rollback, "
     "linear-network limit, rank bound, new-row inference, matched optimisers, I/O"),
    ("R9", "test_lbm.py", "sufficient stats, PG vs Newton, assignment updates, exact enumeration, full schedule, "
     "predictive mixture, permutation invariance, one-block limit, new rows/cols, float32"),
]


def parse_tests():
    res = {}
    for _, f, _ in TESTS:
        txt = (TAB / "test_logs" / f.replace(".py", ".txt")).read_text()
        npass = len(re.findall(r"^PASS ", txt, re.M))
        nfail = len(re.findall(r"^FAIL", txt, re.M))
        if f == "test_counts.py":   # prints 'ok' per check; 5 test functions, all passed
            src = (vs.ROOT / "cm_minimal/experiments/bernfac/models/counts/tests/test_counts.py").read_text()
            ntot = len(re.findall(r"^def test_", src, re.M))
            npass = ntot if "all tests passed" in txt else 0
        else:
            ntot = npass + nfail
        real = re.search(r"([\d.]+) real", txt)
        res[f] = (npass, ntot, float(real.group(1)) if real else np.nan)
    return res


def fig_tests(res):
    fig, ax = plt.subplots(figsize=(17, 5.6))
    ax.axis("off")
    hdr = ["methods", "suite", "passed", "time", "what the tests cover"]
    xs = [0.0, 0.10, 0.235, 0.305, 0.37]
    ax.text(0, 1.0, "", transform=ax.transAxes)
    for x, h in zip(xs, hdr):
        ax.text(x, 0.97, h, transform=ax.transAxes, fontweight="bold", fontsize=fs.FS, va="top")
    ax.plot([0, 1], [0.89, 0.89], transform=ax.transAxes, color="k", lw=1.2)
    rowcol = {"R3–R5": "R3", "R6 + Pois.": "R6", "R7": "R7", "R8": "R8", "R9": "R9"}
    import textwrap
    for i, (m, f, cov) in enumerate(TESTS):
        yy = 0.84 - i * 0.165
        p, n, t = res[f]
        ax.text(xs[0], yy, m, transform=ax.transAxes, va="top", fontsize=fs.FS, fontweight="bold",
                color=col(rowcol[m]))
        ax.text(xs[1], yy, f, transform=ax.transAxes, va="top", fontsize=fs.FS_SMALL, family="monospace")
        ax.text(xs[2], yy, f"{p}/{n}", transform=ax.transAxes, va="top", fontsize=fs.FS,
                color="#1B7837" if p == n else fs.PALETTE["red_strong"], fontweight="bold")
        ax.text(xs[3], yy, f"{t:.0f} s" if t >= 1 else f"{t:.1f} s", transform=ax.transAxes, va="top", fontsize=fs.FS_SMALL)
        ax.text(xs[4], yy, textwrap.fill(cov, 95), transform=ax.transAxes, va="top", fontsize=fs.FS_SMALL)
    ax.plot([0, 1], [0.03, 0.03], transform=ax.transAxes, color="k", lw=0.8)
    ax.text(0, -0.02, "R1/R2 (PCA) have no model-specific tests.  R4 (offsets) and R5 (cloglog) are fitted by "
            "bernfac_large.fit_offsets / fit_cloglog, which test_bernfac.py does not exercise; "
            "their monotonicity is enforced at run time (fit aborts on any decrease).",
            transform=ax.transAxes, va="top", fontsize=fs.FS_SMALL - 1, color=fs.PALETTE["grey"], wrap=True)
    fs.finalize_figure(fig, FIG / "fig_tests")


# ============================================================================ 4. scorecard
def scorecard(T, rt, tres):
    codes = vs.ORDER + ["Pois"]
    cmap_name = {"R3": "BernFac logit", "R4": "BernFac logit + offsets", "R5": "BernFac cloglog + offsets",
                 "R6": "NB count factorization", "R7": "mixture of local factors", "R8": "neural manifold",
                 "R9": "latent block model", "Pois": "Poisson count factorization"}
    cidx = comp.set_index("model")
    S = pd.DataFrame(index=codes)
    for c in codes:
        if c in cmap_name:
            r = cidx.loc[cmap_name[c]]
            S.loc[c, "test NLL"] = r.test_nll; S.loc[c, "test AUC"] = r.test_auc; S.loc[c, "test Brier"] = r.test_brier
        if c in umap.index:
            u = umap.loc[c]
            S.loc[c, "HL kNN bal. acc."] = u.hl_knn_balacc
            S.loc[c, "HL Leiden ARI"] = u.hl_leiden_best_ARI
            S.loc[c, "HL NN purity"] = u.hl_nn_purity
            S.loc[c, "library kNN bal. acc."] = u.lib_knn_balacc
            S.loc[c, "depth |ρ| (PC1–2)"] = u.depth_max_abs_top2
        if c in rt.index:
            S.loc[c, "fit time (min)"] = rt.loc[c, "fit_seconds"] / 60
    tmap = {"R3": "test_bernfac.py", "R4": "test_bernfac.py", "R5": "test_bernfac.py", "R6": "test_counts.py",
            "Pois": "test_counts.py", "R7": "test_mfa.py", "R8": "test_neural.py", "R9": "test_lbm.py"}
    tests_txt, mono, ident = {}, {}, {}
    for c in codes:
        if c in tmap:
            p, n, _ = tres[tmap[c]]
            tests_txt[c] = f"{p}/{n}" + ("*" if c in ("R4", "R5") else "")
        else:
            tests_txt[c] = "—"
    mono.update({"R1": "n/a", "R2": "n/a", "R3": "yes (PG-EM MM)", "R4": "yes (PG-EM MM)",
                 "R5": "yes (line search)", "R6": "yes (line search)", "Pois": "yes (line search)",
                 "R7": "yes (EM bound)", "R8": "yes (accept/rollback)", "R9": ("VB ascent (monotone)" if T["R9"]["max_rel_rise"] == 0 else "VB ascent; rise ≤ {:.0e}".format(T["R9"]["max_rel_rise"]))})
    ident.update({"R1": "unique up to sign", "R2": "unique up to sign", "R3": "UVᵀ up to GL(k); SVD fixes gauge",
                  "R4": "UVᵀ up to GL(k); offsets centred", "R5": "UVᵀ up to GL(k); offsets centred",
                  "R6": "UVᵀ up to GL(k); φ→∞ (≈Poisson)", "Pois": "UVᵀ up to GL(k)",
                  "R7": "K=1: B up to rotation (gauge test)", "R8": "z up to reparam.; time-limited",
                  "R9": "label permutation; grid edge"})
    S["tests passed"] = pd.Series(tests_txt); S["monotone guarantee"] = pd.Series(mono)
    S["identifiability"] = pd.Series(ident)
    S.index.name = "code"
    S.insert(0, "name", [("Poisson counts" if c == "Pois" else vs.name(c)) for c in codes])
    S.to_csv(TAB / "scorecard.csv", float_format="%.5g")

    num_cols = [("test NLL", -1, "{:.4f}"), ("test AUC", 1, "{:.4f}"), ("test Brier", -1, "{:.4f}"),
                ("HL kNN bal. acc.", 1, "{:.3f}"), ("HL Leiden ARI", 1, "{:.3f}"), ("HL NN purity", 1, "{:.3f}"),
                ("library kNN bal. acc.", -1, "{:.3f}"), ("depth |ρ| (PC1–2)", -1, "{:.2f}"),
                ("fit time (min)", -1, "{:.1f}")]
    txt_cols = [("tests passed", 0.95), ("monotone guarantee", 2.5), ("identifiability", 3.4)]
    groups = [("prediction (test)", 3), ("biology", 3), ("batch / depth", 2),
              ("practicality", 1 + len(txt_cols))]
    widths = [1.15] * len(num_cols) + [w for _, w in txt_cols]
    x0 = np.r_[0, np.cumsum(widths)]
    nrow = len(codes)
    fig, ax = plt.subplots(figsize=(20, 8.6))
    ax.set_xlim(-3.3, x0[-1]); ax.set_ylim(nrow + 2.0, -0.05); ax.axis("off")
    cmap = plt.get_cmap("YlGn")
    for j, (cn, sgn, fmt) in enumerate(num_cols):
        v = S[cn].astype(float)
        good = sgn * v
        lo, hi = np.nanmin(good), np.nanmax(good)
        best = good.idxmax()
        for i, c in enumerate(codes):
            val = v[c]
            if np.isnan(val):
                ax.add_patch(Rectangle((x0[j], i + 2), widths[j], 1, fc="#F4F4F4", ec="white", lw=1.5))
                ax.text(x0[j] + widths[j] / 2, i + 2.5, "n/a", ha="center", va="center", fontsize=fs.FS_TINY,
                        color=fs.PALETTE["grey_mid"])
                continue
            z = (sgn * val - lo) / (hi - lo) if hi > lo else 0.5
            fc = cmap(0.08 + 0.72 * z)
            ax.add_patch(Rectangle((x0[j], i + 2), widths[j], 1, fc=fc, ec="white", lw=1.5))
            ax.text(x0[j] + widths[j] / 2, i + 2.5, fmt.format(val), ha="center", va="center",
                    fontsize=fs.FS_TINY + 0.5, color="white" if z > 0.75 else "k")
            if c == best:
                ax.add_patch(Rectangle((x0[j] + 0.04, i + 2.06), widths[j] - 0.08, 0.88, fc="none", ec="k", lw=2.4))
        ax.text(x0[j] + widths[j] / 2, 1.92, cn + (" ↓" if sgn < 0 else " ↑"), ha="center", va="bottom",
                fontsize=fs.FS_TINY + 0.5, rotation=0, wrap=True)
    for jj, (cn, w) in enumerate(txt_cols):
        j = len(num_cols) + jj
        for i, c in enumerate(codes):
            ax.add_patch(Rectangle((x0[j], i + 2), widths[j], 1, fc="white" if i % 2 else "#FAFAFA", ec="white"))
            t = str(S.loc[c, cn])
            ax.text(x0[j] + 0.08, i + 2.5, t, ha="left", va="center", fontsize=fs.FS_TINY)
        ax.text(x0[j] + widths[j] / 2, 1.92, cn, ha="center", va="bottom", fontsize=fs.FS_TINY + 0.5)
    # header label wrapping: put long numeric headers on two lines
    for t in ax.texts:
        s = t.get_text()
        if t.get_position()[1] == 1.92 and len(s) > 6 and "\n" not in s:
            parts = s.split(" ")
            h = len(parts) // 2
            t.set_text(" ".join(parts[:h]) + "\n" + " ".join(parts[h:]))
    gx = 0
    for gname, n in groups:
        a, b = x0[gx], x0[gx + n]
        ax.plot([a + 0.05, b - 0.05], [0.95, 0.95], color="k", lw=1.5)
        ax.text((a + b) / 2, 0.85, gname, ha="center", va="bottom", fontsize=fs.FS, fontweight="bold")
        gx += n
    for i, c in enumerate(codes):
        ax.add_patch(Rectangle((-3.25, i + 2.2), 0.18, 0.6, fc=col(c), ec="none"))
        ax.text(-3.0, i + 2.5, ("Pois." if c == "Pois" else c) + "  " + S.loc[c, "name"], ha="left", va="center",
                fontsize=fs.FS_SMALL, fontweight="bold" if c != "Pois" else "normal")
    ax.text(-3.25, nrow + 2.35, "Colour = within-column min–max scaling, darker = better; black box = best in column. "
            "Test NLL/AUC/Brier from models/comparison.csv; biology and batch metrics from full/umap/metrics.csv "
            "(R1–R9 embeddings). * = shared logit-core suite; the offsets/cloglog fit paths have no unit tests.",
            fontsize=fs.FS_TINY, va="top", color=fs.PALETTE["grey"], wrap=True)
    fs.finalize_figure(fig, FIG / "fig_scorecard")
    return S


def consistency_checks(T, rt):
    notes = []
    for c, (m, _) in BF.items():
        k, lam = BF_SEL[c]
        cm_ = comp[comp.model == {"R3": "BernFac logit", "R4": "BernFac logit + offsets",
                                  "R5": "BernFac cloglog + offsets"}[c]].iloc[0]
        notes.append(f"{c}: rescored selection k={k}, λ={lam:g}; comparison.csv config '{cm_.config}'")
    p = CNT_SEL["Pois"]
    notes.append(f"Poisson val/test NLL gap {p['val_nll']:.4f} vs {p['test_nll']:.4f} (others differ by <2e-4)")
    notes.append(f"NB selected phi_median={CNT_SEL['R6']['phi_median']:.3g} (Poisson limit)")
    n = neu[neu.config == "pg_affine_d20_w64_lv1"]
    notes.append("R8 restarts val: " + ", ".join(f"{r.name}:{r.val_nll:.6f} chosen={r.chosen_restart}" for r in n.itertuples()))
    notes.append(f"R9 selected at grid edge {LBM_BEST['selected']}; LBM grid total {lbm.fit_time_s.sum()/3600:.2f} h vs grid.log 4.85 h")
    (TAB / "worker_D_consistency_notes.txt").write_text("\n".join(notes) + "\n")
    print("\n".join(notes))


if __name__ == "__main__":
    T = traces()
    fig_select_all()
    fig_converge_all(T)
    rt = runtime_table(T)
    print(rt.to_string())
    fig_runtime(rt)
    tres = parse_tests()
    print(tres)
    fig_tests(tres)
    S = scorecard(T, rt, tres)
    print(S.to_string())
    consistency_checks(T, rt)
