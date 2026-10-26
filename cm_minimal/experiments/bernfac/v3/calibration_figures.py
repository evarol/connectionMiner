"""Calibration and stratified held-out NLL for all probability models on the TEST mask.

Inputs: v3/cache/test_p_<code>.npy, test_nll_<code>.npy, test_y.npy (row-major order of the test
mask; written by validate_predictors.py), cell depth and train gene detection rate.
Outputs: fig_calibration_all, fig_test_by_depth (v3/figures);
         calibration_bins.csv, calibration_summary.csv, test_nll_by_depth_decile.csv,
         test_nll_by_gene_rate_decile.csv (v3/tables).
"""
import os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit
import predictors as pr
import figstyle as fs
import v3style as vs

fs.apply_publication_style()
CACHE, FIGS, TABS = pr.CACHE, vs.FIGS, vs.V3 / "tables"
CODES = vs.PROB_MODELS + ["PO"]
COL = {c: vs.colour(c) for c in vs.PROB_MODELS} | {"PO": "#4D4D4D"}
NAME = {c: vs.name(c) for c in vs.PROB_MODELS} | {"PO": "Poisson counts"}
LS = {c: "-" for c in vs.PROB_MODELS} | {"PO": (0, (4, 2))}
NB = 20

y = np.load(CACHE / "test_y.npy").astype(np.float64)
T = np.load(pr.DATA / "test.npy")
ri, ci = np.nonzero(T)
del T
depth = np.load(CACHE / "cell_depth.npy")
grate = np.load(CACHE / "gene_train_rate.npy")


def decile(v):
    q = np.quantile(v, np.linspace(0, 1, 11))
    return np.clip(np.searchsorted(q, v, side="right") - 1, 0, 9), q


cell_dec, cell_q = decile(depth)          # deciles over cells
gene_dec, gene_q = decile(grate)          # deciles over genes
e_cd, e_gd = cell_dec[ri], gene_dec[ci]

bins, summ, by_d, by_g = [], [], [], []
for c in CODES:
    p = np.load(CACHE / f"test_p_{c}.npy").astype(np.float64)
    le = np.load(CACHE / f"test_nll_{c}.npy").astype(np.float64)
    qs = np.quantile(p, np.linspace(0, 1, NB + 1))
    b = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, NB - 1)
    cnt = np.bincount(b, minlength=NB)
    mp = np.bincount(b, p, NB) / np.maximum(cnt, 1)
    my = np.bincount(b, y, NB) / np.maximum(cnt, 1)
    for k in range(NB):
        bins.append(dict(code=c, bin=k, p_lo=qs[k], p_hi=qs[k + 1], n=int(cnt[k]), mean_p=mp[k], frac_pos=my[k]))
    ece = float(np.sum(cnt * np.abs(mp - my)) / cnt.sum())
    # 10 equal-width bins too (matches the per-model reports)
    bw = np.clip((p * 10).astype(int), 0, 9)
    cw = np.bincount(bw, minlength=10)
    ece_w = float(np.sum(np.abs(np.bincount(bw, p, 10) - np.bincount(bw, y, 10))) / cw.sum())
    summ.append(dict(code=c, name=NAME[c], test_nll=le.mean(), test_brier=float(np.mean((p - y) ** 2)),
                     ece_quantile20=ece, ece_width10=ece_w, mean_p=p.mean(), mean_y=y.mean(),
                     frac_p_below_0p01=float(np.mean(p < 0.01)), frac_p_above_0p99=float(np.mean(p > 0.99)),
                     max_entry_nll=float(le.max()), n_entries_nll_gt_20=int(np.sum(le > 20)),
                     nll_share_from_entries_gt_20=float(le[le > 20].sum() / le.sum())))
    for lab, e, rows in (("cell depth", e_cd, by_d), ("gene rate", e_gd, by_g)):
        n = np.bincount(e, minlength=10)
        s = np.bincount(e, le, 10)
        pp = np.bincount(e, p, 10)
        yy = np.bincount(e, y, 10)
        for k in range(10):
            rows.append(dict(code=c, decile=k + 1, n=int(n[k]), test_nll=s[k] / n[k], mean_p=pp[k] / n[k],
                             mean_y=yy[k] / n[k]))
    print(c, f"nll {le.mean():.6f}  ece20 {ece:.5f}", flush=True)
    globals()[f"hist_{c}"] = np.histogram(p, bins=np.linspace(0, 1, 51))[0]
    globals()[f"histl_{c}"] = np.histogram(logit(np.clip(p, 1e-7, 1 - 1e-7)), bins=np.linspace(-16, 16, 65))[0]
    del p, le

bins, summ = pd.DataFrame(bins), pd.DataFrame(summ)
by_d, by_g = pd.DataFrame(by_d), pd.DataFrame(by_g)
for df, q in ((by_d, cell_q), (by_g, gene_q)):
    df["edge_lo"] = df.decile.map(lambda k: q[k - 1]); df["edge_hi"] = df.decile.map(lambda k: q[k])
bins.to_csv(TABS / "calibration_bins.csv", index=False)
summ.to_csv(TABS / "calibration_summary.csv", index=False)
by_d.to_csv(TABS / "test_nll_by_depth_decile.csv", index=False)
by_g.to_csv(TABS / "test_nll_by_gene_rate_decile.csv", index=False)
print(summ.round(5).to_string())

# --------------------------------------------------------------------------- calibration figure
fig, axs = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw=dict(width_ratios=[1, 1, 1.05], wspace=0.3))
ax = axs[0]
ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=1.5, zorder=0)
for c in CODES:
    d = bins[bins.code == c]
    e3 = summ.set_index("code").loc[c, "ece_quantile20"] * 1e3
    ax.plot(d.mean_p, d.frac_pos, color=COL[c], ls=LS[c], marker="o", ms=4, lw=1.8, label=f"{c if c != 'PO' else ''}"
            f"{'  ' if c != 'PO' else ''}{NAME[c]}  ({e3:.2f})")
ax.set_xlabel("mean predicted $\\hat p$ (20 quantile bins)")
ax.set_ylabel("observed detection rate")
ax.set_title("reliability, test entries")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=10.5, loc="upper left", handlelength=1.6, title="(ECE, 20 quantile bins, $\\times10^{-3}$)",
          title_fontsize=10.5)
fs.panel_label(ax, "a", x=-0.16)

ax = axs[1]
ax.axhline(0, color="#BBBBBB", lw=1.5, zorder=0)
for c in CODES:
    d = bins[bins.code == c]
    ax.plot(logit(d.mean_p), (d.frac_pos - d.mean_p) * 100, color=COL[c], ls=LS[c], marker="o", ms=4, lw=1.8)
ax.set_xlabel("logit of mean predicted $\\hat p$")
ax.set_ylabel("observed $-$ predicted (percentage points)")
ax.set_title("calibration error per bin")
yl = np.abs(bins.frac_pos - bins.mean_p).max() * 115
ax.set_ylim(-yl, yl)
fs.panel_label(ax, "b", x=-0.2)

ax = axs[2]
edges = np.linspace(-16, 16, 65)
ctr = (edges[1:] + edges[:-1]) / 2
for c in CODES:
    ax.step(ctr, globals()[f"histl_{c}"] / len(y), where="mid", color=COL[c], ls=LS[c], lw=1.6)
ax.set_yscale("log")
ax.set_xlabel("logit $\\hat p$ (clipped to $10^{-7}$)")
ax.set_ylabel("fraction of test entries")
ax.set_title("distribution of predictions")
for v, t in ((-4.595, "$\\hat p$=0.01"), (4.595, "0.99")):
    ax.axvline(v, color="#999999", lw=1, ls=":")
    ax.text(v + 0.4, 0.97, t, transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=10, color="#555555")
fs.panel_label(ax, "c", x=-0.18)
fig.suptitle("Calibration on 8.04 M test entries (cells and genes seen in training; entries held out)", y=1.04,
             fontsize=15)
fs.finalize_figure(fig, FIGS / "fig_calibration_all")

# --------------------------------------------------------------------------- stratified NLL figure
fig, axs = plt.subplots(2, 2, figsize=(17, 9.6), gridspec_kw=dict(height_ratios=[1, 1.25], hspace=0.42, wspace=0.25),
                        sharex="col")
for col, (df, xl, q) in enumerate(((by_d, "cell depth decile (total UMI)", cell_q),
                                    (by_g, "gene detection-rate decile (train)", gene_q))):
    ax = axs[0, col]
    for c in CODES:
        d = df[df.code == c]
        ax.plot(d.decile, d.test_nll, color=COL[c], ls=LS[c], marker="o", ms=4.5, lw=1.8,
                label=f"{c if c != 'PO' else ''}{'  ' if c != 'PO' else ''}{NAME[c]}")
    ax.set_ylabel("test NLL per entry")
    ax.set_title(f"test NLL by {xl.split(' (')[0]}")
    fs.panel_label(ax, "ab"[col], x=-0.13)
    ax = axs[1, col]
    ref = df[df.code == "R6"].set_index("decile").test_nll
    ax.axhline(0, color=COL["R6"], lw=2)
    for c in CODES:
        if c == "R6":
            continue
        d = df[df.code == c].set_index("decile")
        ax.plot(d.index, (d.test_nll - ref) * 1e3, color=COL[c], ls=LS[c], marker="o", ms=4.5, lw=1.8, clip_on=True)
    others = np.concatenate([(df[df.code == c].set_index("decile").test_nll - ref).values * 1e3
                             for c in CODES if c not in ("R6", "PO")])
    top = max(others.max() * 1.35, 3)
    ax.set_ylim(min(-0.5, others.min() * 1.2), top)
    dpo = (df[df.code == "PO"].set_index("decile").test_nll - ref) * 1e3
    for k, v in dpo.items():
        if v > top:
            ax.annotate(f"Poisson {v:.0f} (off scale)", (k, top), xytext=(k - 1.4, top * 0.9), fontsize=10.5, color=COL["PO"],
                        ha="right", arrowprops=dict(arrowstyle="->", color=COL["PO"], lw=1.2))
    ax.set_ylabel("$\\Delta$NLL vs R6 NB ($\\times10^{-3}$)")
    ax.set_xlabel(xl)
    ax.set_xticks(range(1, 11))
    if col == 0:
        lab = [f"{int(q[k])}" for k in range(10)]
        ax.set_xticklabels([f"{k + 1}" for k in range(10)])
        ax.text(0.0, -0.28, f"UMI edges: {int(q[0])}, {int(q[1])}, {int(q[5])} (median), {int(q[9])}, {int(q[10])}",
                transform=ax.transAxes, fontsize=10.5, color="#444444")
    else:
        ax.text(0.0, -0.28, f"rate edges: {q[1]:.4f}, {q[5]:.3f} (median), {q[9]:.3f}",
                transform=ax.transAxes, fontsize=10.5, color="#444444")
    ax.set_title("difference from the best model (R6)")
    fs.panel_label(ax, "cd"[col], x=-0.13)
axs[0, 0].legend(fontsize=10.5, ncol=2, loc="upper left", handlelength=1.6)
fs.finalize_figure(fig, FIGS / "fig_test_by_depth")
