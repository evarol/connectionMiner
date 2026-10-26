"""Figures for the neural-manifold study (house style: ../../figstyle.py, mathtext 'cm').

fig_neural_synthetic     synthetic demo: test NLL over seeds, fitted-logit spectra, embeddings,
                         PG vs direct BCE traces (seed 0)
fig_neural_selection     full-data validation selection curves (+ test, baselines)
fig_neural_traces        full data: F, surrogate and validation NLL, PG vs direct BCE vs wall time
fig_neural_spectrum      singular values of fitted logits (2,000-row subsample), affine vs nonlinear
Output: outputs/bernfac_20261020/models/neural/figures/{name}.png (dpi 200) + .pdf
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
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

fs.apply_publication_style()
ROOT = HERE.parents[4]
OUT = ROOT / "outputs/bernfac_20261020/models/neural"
FIG = OUT / "figures"
SYN = OUT / "synthetic"
P = fs.PALETTE
FAM_COL = {"neural affine": P["blue_main"], "neural nonlinear": P["teal"], "linear factor": P["red_strong"],
           "column rate": P["grey_mid"], "oracle (true logits)": P["grey_dark"], "true logits": P["grey_dark"]}


def fig_synthetic():
    d = pd.read_csv(SYN / "synthetic_all_fits.csv")
    sel = d[d.family.str.endswith("[selected]")].copy()
    sel["fam"] = sel.family.str.replace(" [selected]", "", regex=False)
    order = ["oracle (true logits)", "neural nonlinear", "neural affine", "linear factor", "column rate"]
    order = [o for o in order if o in set(sel.fam)]
    spec = json.loads((SYN / "seed0_singular_values.json").read_text())
    tr = json.loads((SYN / "seed0_optimizer_traces.json").read_text())
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8), gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1]))
    a = ax[0]
    rng = np.random.default_rng(0)
    for k, f in enumerate(order):
        v = sel.loc[sel.fam == f, "test_nll"].to_numpy()
        a.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=28, color=FAM_COL[f], alpha=0.75, zorder=3)
        a.hlines(v.mean(), k - 0.3, k + 0.3, color=FAM_COL[f], lw=3)
    a.set_xticks(range(len(order)))
    a.set_xticklabels([o.replace(" (true logits)", "\n(true logits)").replace("neural ", "neural\n")
                       .replace("linear factor", "linear\nfactor").replace("column rate", "column\nrate")
                       for o in order], fontsize=fs.FS_TICK - 1)
    a.set_ylabel("test NLL per cell")
    a.set_title("Held-out cells, 5 data seeds")
    fs.panel_label(a, "a", x=-0.2)
    a = ax[1]
    for f in ["true logits", "neural nonlinear", "neural affine", "linear factor"]:
        if f in spec:
            s = np.asarray(spec[f])
            a.semilogy(np.arange(1, len(s) + 1), s / s[0], color=FAM_COL[f], label=f, lw=2)
    a.set_xlabel("index")
    a.set_ylabel(r"$s_k / s_1$ of fitted logits")
    a.set_ylim(1e-6, 1.5)
    a.set_xlim(0, 40)
    a.legend(loc="lower left")
    a.set_title("Logit spectra (seed 0)")
    fs.panel_label(a, "b")
    a = ax[2]
    z = np.load(SYN / "seed0_true_z.npy")
    Zf = np.load(SYN / "seed0_Z_nonlinear.npy") if (SYN / "seed0_Z_nonlinear.npy").exists() else None
    ang = np.arctan2(z[:, 1], z[:, 0])
    if Zf is not None:
        a.scatter(Zf[:, 0], Zf[:, 1], c=ang, cmap="twilight", s=16)
        a.set_xlabel(r"fitted $z_1$")
        a.set_ylabel(r"fitted $z_2$")
        a.set_title("Fitted Z (nonlinear head),\ncoloured by true angle")
    fs.panel_label(a, "c")
    a = ax[3]
    cols = {"pg_gem": P["blue_main"], "direct_bce": P["red_strong"]}
    Fmin = min(min(v["F"]) for v in tr.values())
    for k, v in tr.items():
        F = np.asarray(v["F"])
        a.semilogy(np.asarray(v["seconds"]) + 1e-3, F - Fmin + 1e-1, color=cols[k],
                   label=f"{'PG-GEM' if k == 'pg_gem' else 'direct BCE'} (val {v['val_nll']:.3f})")
    a.set_xscale("log")
    a.set_xlabel("wall time (s)")
    a.set_ylabel(r"$F - F_{\min} + 0.1$")
    a.legend(loc="upper right")
    a.set_title("Same init, same model")
    fs.panel_label(a, "d")
    fig.tight_layout()
    fs.finalize_figure(fig, FIG / "fig_neural_synthetic")


def _baselines():
    f = ROOT / "outputs/bernfac_20261020/models/rescored/best_per_model.csv"
    if not f.exists():
        return {}
    b = pd.read_csv(f).set_index("model")
    return {"logistic factor rank 40 (+offsets)": b.loc["logit + offsets", "val_nll"]}


def fig_selection():
    d = pd.read_csv(OUT / "tables/selection.csv")
    d = d[d.optimizer_mode == "pg_gem"]
    base = _baselines()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    a = ax[0]
    s = d[(d.head_mode == "affine") & (d.latent_dim == 10) & (d.hidden_widths == "64x64") & (d.restart == 0)]
    s = s.sort_values("lambda_output")
    a.plot(s.lambda_output, s.val_nll, "o-", color=P["blue_main"], label="validation")
    a.plot(s.lambda_output, s.test_nll, "s--", color=P["blue_secondary"], alpha=0.7, label="test (reported only)")
    a.set_xscale("log")
    a.set_xlabel(r"$\lambda_V$ (output-weight precision)")
    a.set_ylabel("NLL per held-out cell")
    a.set_title(r"Affine head, $d=10$, widths 64")
    a.legend()
    fs.panel_label(a, "a")
    a = ax[1]
    if len(s):
        lv = s.loc[s.val_nll.idxmin(), "lambda_output"]
        s2 = d[(d.head_mode == "affine") & (d.lambda_output == lv) & (d.hidden_widths == "64x64") & (d.restart == 0)]
        s2 = s2.sort_values("latent_dim")
        a.plot(s2.latent_dim, s2.val_nll, "o-", color=P["blue_main"], label="widths 64")
        s3 = d[(d.head_mode == "affine") & (d.lambda_output == lv) & (d.hidden_widths == "128x128")]
        a.plot(s3.latent_dim, s3.val_nll, "D", ms=9, color=P["violet"], label="widths 128")
        a.set_xticks(sorted(set(s2.latent_dim)))
        a.set_xlabel("latent dimension $d$")
        a.set_title(rf"Affine head, $\lambda_V={lv:g}$")
        a.legend()
    fs.panel_label(a, "b")
    a = ax[2]
    rows = d[d.chosen_restart].sort_values("val_nll").head(8)
    lab = [f"{r.head_mode[:3]} d{r.latent_dim} w{r.hidden_widths.split('x')[0]} " + rf"$\lambda_V$={r.lambda_output:g}"
           for r in rows.itertuples()]
    y = np.arange(len(rows))
    a.barh(y, rows.val_nll, color=[P["teal"] if h == "nonlinear" else P["blue_main"] for h in rows.head_mode])
    for k, (nm, v) in enumerate(base.items()):
        a.axvline(v, color=[P["red_strong"], P["grey_mid"]][k], ls="--", lw=1.8, label=nm)
    a.set_yticks(y)
    a.set_yticklabels(lab, fontsize=fs.FS_SMALL)
    a.invert_yaxis()
    lo = min(rows.val_nll.min(), min(base.values()) if base else 1) - 0.0003
    a.set_xlim(lo, max(rows.val_nll.max(), max(base.values()) if base else 0) + 0.0003)
    a.ticklabel_format(axis="x", useOffset=False)
    a.set_xlabel("validation NLL")
    a.legend(loc="lower right", fontsize=fs.FS_LEGEND - 1)
    a.set_title("Best configurations (blue affine, teal nonlinear)")
    fs.panel_label(a, "c", x=-0.45)
    fig.tight_layout()
    fs.finalize_figure(fig, FIG / "fig_neural_selection")


def fig_traces(pg_name, direct_name):
    import torch
    hs = {}
    for k, nm in (("PG-GEM", pg_name), ("direct BCE", direct_name)):
        ck = torch.load(OUT / f"fits/{nm}.pt", map_location="cpu", weights_only=True)
        hs[k] = json.loads(ck["history"])
    cols = {"PG-GEM": P["blue_main"], "direct BCE": P["red_strong"]}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    Fmin = min(min(h["original_map_loss"]) for h in hs.values())
    for k, h in hs.items():
        t = np.asarray(h["seconds"]) / 60
        ax[0].semilogy(t, np.asarray(h["original_map_loss"]) - Fmin + 1e3, "-o", ms=3, color=cols[k], label=k)
        v = np.asarray([np.nan if x is None else x for x in h["val_nll"]], float)
        ax[2].plot(t, v, "-o", ms=3, color=cols[k], label=k)
    ax[0].set_xlabel("wall time (min)")
    ax[0].set_ylabel(r"$F - F_{\min} + 10^3$")
    ax[0].set_title("Original MAP objective")
    ax[0].legend()
    h = hs["PG-GEM"]
    it = np.asarray(h["iteration"][1:])
    S0 = np.asarray(h["surrogate_start"][1:], float)
    S1 = np.asarray(h["surrogate_end"][1:], float)
    F = np.asarray(h["original_map_loss"], float)
    ax[1].semilogy(it, S0 - S1, "o-", ms=3, color=P["blue_main"], label=r"surrogate drop $S_{start}-S_{end}$")
    ax[1].semilogy(it, F[:-1] - F[1:], "s-", ms=3, color=P["green_3"], label=r"objective drop $F_{q-1}-F_q$")
    ax[1].set_xlabel("outer iteration (PG-GEM)")
    ax[1].set_ylabel("decrease per iteration")
    ax[1].set_title(r"$F$ drop $\geq$ surrogate drop")
    ax[1].legend()
    ax[2].set_xlabel("wall time (min)")
    ax[2].set_ylabel("validation NLL")
    ax[2].set_title("Validation (monitored, not used to stop)")
    lo = np.nanmin([np.nanmin(np.asarray([np.nan if x is None else x for x in hh["val_nll"]], float))
                    for hh in hs.values()])
    ax[2].set_ylim(lo - 0.001, lo + 0.012)
    for a, l in zip(ax, "abc"):
        fs.panel_label(a, l)
    fig.tight_layout()
    fs.finalize_figure(fig, FIG / "fig_neural_traces")


def fig_spectrum():
    f = OUT / "tables/logit_spectra.json"
    sp = json.loads(f.read_text())
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8))
    col = {"affine": P["blue_main"], "nonlinear": P["teal"]}
    for k, v in sp["spectra"].items():
        s = np.asarray(v["s"])
        ax.semilogy(np.arange(1, len(s) + 1), s / s[0], color=col[v["head_mode"]], label=f"{v['head_mode']} head ({k})")
    H = sp["H"]
    ax.axvline(H + 1, color=P["grey_mid"], ls="--", lw=1.5)
    ax.text(H + 3, 1e-6, f"$H+1={H + 1}$", color=P["grey"], fontsize=fs.FS_SMALL)
    ax.set_xlim(0, 3 * (H + 1))
    ax.set_ylim(1e-8, 1.5)
    ax.set_xlabel("index $k$")
    ax.set_ylabel(r"$s_k/s_1$")
    ax.set_title(f"Fitted logits, {sp['n_rows']:,} random cells x all genes")
    ax.legend(fontsize=fs.FS_LEGEND - 1, loc="center right")
    fig.tight_layout()
    fs.finalize_figure(fig, FIG / "fig_neural_spectrum")


if __name__ == "__main__":
    what = sys.argv[1:] or ["synthetic"]
    if "synthetic" in what:
        fig_synthetic()
    if "selection" in what:
        fig_selection()
    if "traces" in what:
        fig_traces(*[a for a in sys.argv if a.startswith("pg_") or a.startswith("direct_")][:2])
    if "spectrum" in what:
        fig_spectrum()
