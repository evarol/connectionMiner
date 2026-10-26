"""W1 (v4): NB vs Poisson count factorization at matched (k, lambda).

Inputs (nothing is refit):
  models/counts/tables/selection_full.csv    val/test NLL of every NB / Poisson config
  models/counts/fits/{nb,poisson}_k*_lam*.npz  every config's fit; NB fits store phi_j (bounded to
                                               [1e-3, 1e6] by countfac.LOGPHI_MIN/MAX)
  v3/cache/gene_train_rate.npy                 gene detection rate on train entries
  v3/cache/test_nll_PO.npy                     per-entry test NLL of the selected Poisson (k=80, lam=10)
Computes per-entry test NLL of NB k=80, lam=10 (matched to the selected Poisson) and NB k=120,
lam=10 (= selected R6, from cache), then per-gene NB - Poisson gains.

Outputs: v4/tables/nb_vs_poisson_grid.csv, nb_phi_by_config.csv, nb_phi_by_detection.csv,
         nb_vs_poisson_per_gene.csv (one row per gene), nb_vs_poisson_attribution.csv
         v4/figures/fig_nb_vs_poisson.{png,pdf}
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v3"))
sys.path.insert(0, str(HERE.parent))
import predictors as pr  # noqa: E402
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

STUDY = pr.STUDY
CNT = STUDY / "models/counts"
C3 = STUDY / "v3/cache"
C4 = STUDY / "v4/cache"
TAB = STUDY / "v4/tables"
FIG = STUDY / "v4/figures"
PHI_LO, PHI_HI = 1e-3, 1e6


def test_index():
    T = np.load(pr.DATA / "test.npy", mmap_mode="r")
    I, J = [], []
    for s in range(0, T.shape[0], 2048):
        r, c = np.nonzero(np.asarray(T[s:s + 2048]))
        I.append((r + s).astype(np.int32)); J.append(c.astype(np.int32))
    return np.concatenate(I), np.concatenate(J)


def nb_test_nll(tag, I, J, y):
    f = C4 / f"test_nll_{tag}.npy"
    if not f.exists():
        z = np.load(CNT / f"fits/{tag}.npz")
        p = pr.Counts(z["U"], z["V"], z["a"], z["b"], z["o"], z["phi"])
        out = np.empty(len(I), np.float32)
        for s in range(0, len(I), 1_000_000):
            lP, lQ = p.logpairs(I[s:s + 1_000_000], J[s:s + 1_000_000])
            out[s:s + 1_000_000] = -np.where(y[s:s + 1_000_000] > 0, lP, lQ)
        np.save(f, out)
    return np.load(f).astype(np.float64)


def main():
    C4.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(CNT / "tables/selection_full.csv")
    sel = sel[sel.depth == "train"].copy()
    grate = np.load(C3 / "gene_train_rate.npy")
    m = len(grate)

    # ---- grid at matched (k, lambda)
    piv = sel.pivot_table(index=["k", "lam"], columns="family", values=["val_nll", "test_nll"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv["val_gap_nb_minus_po"] = piv.val_nll_nb - piv.val_nll_poisson
    piv["test_gap_nb_minus_po"] = piv.test_nll_nb - piv.test_nll_poisson
    piv.to_csv(TAB / "nb_vs_poisson_grid.csv", index=False)
    print(piv.to_string())

    # ---- phi per NB config, overall and by detection-rate bin
    dbins = [0, 1e-3, 1e-2, 0.05, 0.2, 1.0001]
    dlab = ["<0.1%", "0.1-1%", "1-5%", "5-20%", ">20%"]
    db = np.digitize(grate, dbins) - 1
    rows, rows_d = [], []
    phis = {}
    for _, r in sel[sel.family == "nb"].iterrows():
        tag = f"nb_k{int(r.k)}_lam{int(r.lam) if float(r.lam).is_integer() else r.lam}"
        phi = np.load(CNT / f"fits/{tag}.npz")["phi"]
        phis[(int(r.k), float(r.lam))] = phi
        q = np.percentile(phi, [5, 25, 50, 75, 95])
        rows.append(dict(config=tag, k=int(r.k), lam=r.lam, val_nll=r.val_nll, n_genes=m,
                         phi_p05=q[0], phi_p25=q[1], phi_median=q[2], phi_p75=q[3], phi_p95=q[4],
                         frac_phi_lt1=(phi < 1).mean(), frac_phi_lt10=(phi < 10).mean(),
                         frac_phi_gt1e4=(phi > 1e4).mean(),
                         frac_at_lower_bound=(phi <= PHI_LO * 1.01).mean(),
                         frac_at_upper_bound=(phi >= PHI_HI * 0.99).mean(),
                         median_phi_genes_detect_ge_1pct=np.median(phi[grate >= 0.01]),
                         median_detect_rate_phi_gt1e4=np.median(grate[phi > 1e4]) if (phi > 1e4).any() else np.nan))
        for b in range(len(dlab)):
            s = db == b
            rows_d.append(dict(config=tag, k=int(r.k), lam=r.lam, detection_bin=dlab[b], n_genes=int(s.sum()),
                               phi_median=np.median(phi[s]), frac_phi_lt10=(phi[s] < 10).mean(),
                               frac_phi_gt1e4=(phi[s] > 1e4).mean(),
                               frac_at_lower_bound=(phi[s] <= PHI_LO * 1.01).mean()))
    dphi = pd.DataFrame(rows).sort_values(["k", "lam"])
    dphi.to_csv(TAB / "nb_phi_by_config.csv", index=False)
    pd.DataFrame(rows_d).sort_values(["k", "lam"]).to_csv(TAB / "nb_phi_by_detection.csv", index=False)
    print(dphi.to_string())

    # ---- per-gene gain on test entries
    I, J = test_index()
    y = np.load(C3 / "test_y.npy").astype(np.float64)
    L_po = np.load(C3 / "test_nll_PO.npy").astype(np.float64)
    L_nb80 = nb_test_nll("nb_k80_lam10", I, J, y)
    L_nb120 = np.load(C3 / "test_nll_R6.npy").astype(np.float64)
    ng = np.bincount(J, minlength=m)
    pos = y > 0
    phi80 = phis[(80, 10.0)]
    phi120 = phis[(120, 10.0)]
    pg = pd.DataFrame(dict(gene=np.arange(m), detect_rate_train=grate, n_test=ng,
                           n_test_pos=np.bincount(J, pos.astype(float), minlength=m).astype(int),
                           phi_nb_k80_lam10=phi80, phi_nb_k120_lam10=phi120,
                           sum_nll_po=np.bincount(J, L_po, minlength=m),
                           sum_nll_nb80=np.bincount(J, L_nb80, minlength=m),
                           sum_nll_nb120=np.bincount(J, L_nb120, minlength=m)))
    # per-gene clipped Poisson (exclude the >20-nat tail) so the gain is not driven by the outliers
    L_po_c = np.minimum(L_po, 20.0)
    pg["sum_nll_po_cap20"] = np.bincount(J, L_po_c, minlength=m)
    pg["gain_nb80_vs_po_per_entry"] = (pg.sum_nll_po - pg.sum_nll_nb80) / np.maximum(ng, 1)
    pg["gain_nb80_vs_po_cap20_per_entry"] = (pg.sum_nll_po_cap20 - pg.sum_nll_nb80) / np.maximum(ng, 1)
    try:
        genes = np.load(pr.DATA / "genes_kept.npy", allow_pickle=True)
        pg["gene_name"] = genes
    except Exception:
        pass
    pg.to_csv(TAB / "nb_vs_poisson_per_gene.csv", index=False)

    N = len(y)
    tot = (L_po - L_nb80).sum()
    tot_c = (L_po_c - L_nb80).sum()
    att = []
    phibin = np.digitize(phi80, [1, 10, 100, 1e4]); plab = ["<1", "1-10", "10-100", "100-1e4", ">1e4"]
    for name, lab, gb in [("phi_k80", plab, phibin), ("detection", dlab, db)]:
        for b, l in enumerate(lab):
            e = gb[J] == b
            att.append(dict(split=name, bin=l, n_genes=int((gb == b).sum()), n_test=int(e.sum()),
                            gain_per_entry_all=(L_po - L_nb80).sum() / N,
                            share_of_gain=(L_po[e] - L_nb80[e]).sum() / tot,
                            share_of_gain_po_cap20=(L_po_c[e] - L_nb80[e]).sum() / tot_c,
                            gain_on_pos=(L_po[e & pos] - L_nb80[e & pos]).sum() / N,
                            gain_on_zero=(L_po[e & ~pos] - L_nb80[e & ~pos]).sum() / N))
    att = pd.DataFrame(att)
    att.to_csv(TAB / "nb_vs_poisson_attribution.csv", index=False)
    print(att.to_string())
    print("test gain NB80 - PO per entry", tot / N, "capped", tot_c / N,
          "NB120 vs NB80", (L_nb80 - L_nb120).mean(),
          "on y=1", (L_po[pos] - L_nb80[pos]).sum() / N, "on y=0", (L_po[~pos] - L_nb80[~pos]).sum() / N)

    # ---- figure
    fs.apply_publication_style()
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.8), gridspec_kw=dict(wspace=0.38))
    lams = sorted(sel.lam.unique())
    cols = fs.lambda_ramp(len(lams))
    a = ax[0]
    for c, lam in zip(cols, lams):
        for fam, ls, mk in [("nb", "-", "o"), ("poisson", "--", "s")]:
            s = sel[(sel.family == fam) & (sel.lam == lam)].sort_values("k")
            if len(s):
                a.plot(s.k, s.val_nll, ls=ls, marker=mk, color=c, ms=7)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, lw=3, label=f"λ = {lam:g}") for c, lam in zip(cols, lams)]
    h += [Line2D([], [], color="k", ls="-", marker="o", label="NB"),
          Line2D([], [], color="k", ls="--", marker="s", label="Poisson")]
    a.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=fs.FS_LEGEND)
    a.set_xscale("log"); a.set_xticks([20, 40, 80, 120, 160]); a.set_xticklabels(["20", "40", "80", "120", "160"])
    a.minorticks_off()
    a.set_xlabel("rank k"); a.set_ylabel("validation NLL (nats / entry)")
    a.set_title("NB beats Poisson at every matched (k, λ)")
    a.set_ylim(0.175, 0.1915)
    fs.panel_label(a, "a")

    a = ax[1]
    ks10 = sorted(k for (k, lam) in phis if lam == 10.0)
    hi, lo = grate >= 0.05, grate < 1e-3
    series = [
        ("all genes: φ < 10", fs.PALETTE["blue_main"], "-", lambda p: (p < 10).mean()),
        ("all genes: φ > 10⁴", fs.PALETTE["red_strong"], "-", lambda p: (p > 1e4).mean()),
        (f"detected ≥ 5% ({hi.sum():,}): φ < 10", fs.PALETTE["blue_main"], "--", lambda p: (p[hi] < 10).mean()),
        (f"detected < 0.1% ({lo.sum():,}): φ > 10⁴", fs.PALETTE["red_strong"], "--",
         lambda p: (p[lo] > 1e4).mean()),
    ]
    for lab, col, ls, f in series:
        a.plot(ks10, [f(phis[(k, 10.0)]) for k in ks10], color=col, ls=ls, marker="o", ms=7, label=lab)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=fs.FS_LEGEND)
    a.set_xscale("log"); a.set_xticks([20, 40, 80, 120, 160]); a.set_xticklabels(["20", "40", "80", "120", "160"])
    a.minorticks_off()
    a.set_ylim(0, 1)
    a.set_xlabel("rank k (λ = 10)"); a.set_ylabel("fraction of genes")
    a.set_title("φ is bimodal: rare genes run to 10⁶")
    fs.panel_label(a, "b")

    a = ax[2]
    g = pg[pg.n_test > 0]
    x = np.maximum(g.detect_rate_train, 1e-5)
    yv = g.gain_nb80_vs_po_cap20_per_entry * 1e3
    lp = np.log10(np.clip(g.phi_nb_k80_lam10, PHI_LO, PHI_HI))
    sc = a.scatter(x, yv, c=lp, cmap="viridis", s=4, alpha=0.6, rasterized=True, vmin=-3, vmax=6)
    # binned mean
    bins = np.logspace(-5, 0, 21)
    bi = np.digitize(x, bins)
    mx = [np.sum(g.sum_nll_po_cap20[bi == b] - g.sum_nll_nb80[bi == b]) / max(1, g.n_test[bi == b].sum()) * 1e3
          for b in range(1, len(bins))]
    cx = np.sqrt(bins[:-1] * bins[1:])
    ok = np.array([np.any(bi == b) for b in range(1, len(bins))])
    a.plot(cx[ok], np.array(mx)[ok], color=fs.PALETTE["red_strong"], lw=3, label="mean per bin")
    a.axhline(0, color="k", lw=1)
    a.set_xscale("log"); a.set_yscale("symlog", linthresh=1)
    a.set_xlabel("gene detection rate (train)")
    a.set_ylabel("NB − Poisson gain\n(10⁻³ nats / test entry)")
    a.set_title("Gain comes from frequently detected genes", loc="right")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
    cb = fig.colorbar(sc, ax=a, pad=0.02)
    cb.set_label("log₁₀ φ (NB, k = 80)")
    fs.panel_label(a, "c")
    FIG.mkdir(parents=True, exist_ok=True)
    fs.finalize_figure(fig, FIG / "fig_nb_vs_poisson")


if __name__ == "__main__":
    main()
