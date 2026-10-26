"""Tables for W5 (depth-standardization rigor) from v4/cache/std_rigor.npz, std_rigor_r2.csv, std_rigor_gaps.csv.

v4/tables/std_rigor.csv            residual depth per source (task 1): CV R^2 of log UMI from 10 PCs (total and
                                   partial beyond hemilineage x birth-time strata, observed and permuted depth);
                                   median |rho| with depth within strata (observed, permuted, Bernoulli draw)
v4/tables/std_retention.csv        p~ vs p^ retention of hemilineage and birth-time structure (task 4)
v4/tables/std_depthgenes_heldout.csv  depth gap of genes chosen on one half, evaluated on the other (task 2)
v4/tables/std_reference.csv        reference sensitivity c_bar = 10/50/90th percentile (task 3)
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v3"))
import numpy as np
import pandas as pd
import predictors as pr

V4 = pr.STUDY / "v4"
CACHE, TABS = V4 / "cache", V4 / "tables"
TABS.mkdir(parents=True, exist_ok=True)
Z = np.load(CACHE / "std_rigor.npz", allow_pickle=True)
R2 = pd.read_csv(CACHE / "std_rigor_r2.csv")
GAP = pd.read_csv(CACHE / "std_rigor_gaps.csv")
SOURCES = list(Z["sources"])
I_rand, I_mark, I_heat = Z["rand_idx"], Z["marker_idx"], Z["heat_marker_idx"]
REFS = ["p10", "p50", "p90"]
MODELS = ["R4", "R5", "R6"]


def g(k, s):
    return Z[f"{k}__{s}"].astype(np.float64)


def med_abs(v):
    return float(np.nanmedian(np.abs(v)))


def describe(s):
    if s == "Y":
        return "binary Y (data)"
    if s == "PC":
        return "processed counts log1p(1e4 x / UMI)"
    if s == "R1":
        return "R1 HVG PCA embedding (top 10)"
    if s == "R4_UV":
        return "R4 u_i'v_j (= centred logit p~ of R4)"
    if "~" in s:
        return f"{s[:2]} p~ (c_bar = {s.split('~')[1][1:]}th percentile of c_i)"
    return f"{s} p^ (observed depth)"


# ---------------------------------------------------------------- task 1
rows = []
for src in list(dict.fromkeys(list(R2.source))):
    r = {"source": src, "description": describe(src)}
    for scale in ("logit", "p"):
        d = R2[(R2.source == src) & (R2.scale == scale)]
        if d.empty:
            continue
        o = d[d.target == "logUMI"].iloc[0]; q = d[d.target != "logUMI"].iloc[0]
        r[f"r2_total_{scale}"] = o.r2_cv_total
        r[f"r2_partial_{scale}"] = o.r2_cv_partial_beyond_strata
        r[f"r2_total_{scale}_perm"] = q.r2_cv_total
        r[f"r2_partial_{scale}_perm"] = q.r2_cv_partial_beyond_strata
        r["r2_strata_only"] = o.r2_cv_strata_only
    if src in SOURCES:
        r["med_abs_rho_hlbt"] = med_abs(g("rho_hlbt", src)[I_rand])
        r["med_abs_rho_hlbt_perm"] = med_abs(g("rho_hlbt_perm", src)[I_rand])
        r["med_abs_rho_hl"] = med_abs(g("rho_hl", src))
        r["mean_rho_hlbt"] = float(np.nanmean(g("rho_hlbt", src)[I_rand]))
        if f"rho_hlbt_draw__{src}" in Z:
            r["med_abs_rho_hlbt_bernoulli_draw"] = med_abs(g("rho_hlbt_draw", src))
    rows.append(r)
T1 = pd.DataFrame(rows)
T1.to_csv(TABS / "std_rigor.csv", index=False)

# ---------------------------------------------------------------- task 4 retention
rows = []
for c in MODELS:
    for ref in REFS:
        t, h = f"{c}~{ref}", c
        r = {"model": c, "ref": ref}
        for lab, I in (("markers115", I_mark), ("heat20", I_heat), ("random3000", I_rand)):
            bt, bh = g("between", t)[I], g("between", h)[I]
            et, eh = bt / g("total", t)[I], bh / g("total", h)[I]
            Mt, Mh = g("hlmean", t)[:, I], g("hlmean", h)[:, I]
            pg = [np.corrcoef(Mt[:, k], Mh[:, k])[0, 1] for k in range(len(I))]
            Mtc, Mhc = Mt - Mt.mean(0), Mh - Mh.mean(0)
            r[f"{lab}_median_ratio_between_hl_var"] = float(np.median(bt / bh))
            r[f"{lab}_ratio_sum_between_hl_var"] = float(bt.sum() / bh.sum())
            r[f"{lab}_median_eta2_hl_ptilde"] = float(np.nanmedian(et))
            r[f"{lab}_median_eta2_hl_phat"] = float(np.nanmedian(eh))
            r[f"{lab}_median_gene_corr_hl_profile"] = float(np.nanmedian(pg))
            r[f"{lab}_corr_hl_profiles_all"] = float(np.corrcoef(Mtc.ravel(), Mhc.ravel())[0, 1])
            wt, wh = g("wbt", t)[I], g("wbt", h)[I]
            r[f"{lab}_ratio_sum_birthtime_within_hl_var"] = float(wt.sum() / wh.sum())
            r[f"{lab}_median_ratio_birthtime_within_hl_var"] = float(np.median(wt / wh))
        # versus the data: between-HL variance of p^ / p~ relative to the empirical Y on markers
        bY = g("between", "Y")[I_mark]
        r["markers115_ratio_sum_between_hl_var_vs_Y_ptilde"] = float(g("between", t)[I_mark].sum() / bY.sum())
        r["markers115_ratio_sum_between_hl_var_vs_Y_phat"] = float(g("between", h)[I_mark].sum() / bY.sum())
        rows.append(r)
T4 = pd.DataFrame(rows)
T4.to_csv(TABS / "std_retention.csv", index=False)

# ---------------------------------------------------------------- task 2
G = GAP.copy()
G["model"] = G.source.str[:2]
G["kind"] = np.where(G.source.isin(["Y", "PC"]), G.source,
                     np.where(G.source.str.contains("~"), "p_tilde_" + G.source.str.split("~").str[1], "p_hat"))
T2 = (G.groupby(["source", "kind", "ntop", "half"])
       .agg(gap_mean=("gap", "mean"), gap_sd=("gap", "std"), gap_hlbt_mean=("gap_hlbt", "mean"),
            gap_hlbt_sd=("gap_hlbt", "std"), n_splits=("gap", "size")).reset_index())
T2["ntop"] = T2.ntop.map({0: "300 random broad genes", 8: "top 8", 50: "top 50"})
T2.to_csv(TABS / "std_depthgenes_heldout.csv", index=False)

# ---------------------------------------------------------------- task 3
rows = []
for c in MODELS:
    cobs = Z[f"c__{c}"]
    for ref, q in zip(REFS, (10, 50, 90)):
        t = f"{c}~{ref}"
        r = {"model": c, "ref": ref, "c_bar": float(np.percentile(cobs, q)),
             "mean_p_random_genes": float(g("mean_p", t).mean()),
             "mean_p_random_genes_phat": float(g("mean_p", c).mean()),
             "median_cell_detection": float(np.median(g("cell_mean", t))),
             "sd_cell_detection": float(np.std(g("cell_mean", t))),
             "max_abs_diff_rho_vs_p50": float(np.nanmax(np.abs(g("rho_hlbt", t) - g("rho_hlbt", f"{c}~p50")))),
             "med_abs_rho_hlbt": med_abs(g("rho_hlbt", t)[I_rand]),
             "corr_hl_profiles_vs_p50_markers": float(np.corrcoef(
                 (g("hlmean", t)[:, I_mark] - g("hlmean", t)[:, I_mark].mean(0)).ravel(),
                 (g("hlmean", f"{c}~p50")[:, I_mark] - g("hlmean", f"{c}~p50")[:, I_mark].mean(0)).ravel())[0, 1]),
             "spearman_mean_p_vs_p50": float(pd.Series(g("mean_p", t)).corr(pd.Series(g("mean_p", f"{c}~p50")),
                                                                            method="spearman"))}
        for scale in ("logit", "p"):
            d = R2[(R2.source == t) & (R2.scale == scale) & (R2.target == "logUMI")].iloc[0]
            r[f"r2_total_{scale}"] = d.r2_cv_total; r[f"r2_partial_{scale}"] = d.r2_cv_partial_beyond_strata
        d = T2[(T2.source == t) & (T2.ntop == "top 8") & (T2.half == "heldout_half")].iloc[0]
        r["heldout_gap_top8"] = d.gap_mean
        rr = T4[(T4.model == c) & (T4.ref == ref)].iloc[0]
        r["markers115_ratio_sum_between_hl_var"] = rr.markers115_ratio_sum_between_hl_var
        r["markers115_corr_hl_profiles_all"] = rr.markers115_corr_hl_profiles_all
        rows.append(r)
T3 = pd.DataFrame(rows)
T3.to_csv(TABS / "std_reference.csv", index=False)

pd.set_option("display.width", 250)
for name, T in (("std_rigor", T1), ("std_retention", T4), ("std_depthgenes_heldout", T2), ("std_reference", T3)):
    print("==", name); print(T.round(4).to_string())

# ---------------------------------------------------------------- birth-time share of variance (over-correction check)
rows = []
for src in SOURCES:
    for lab, I in (("random3000", I_rand), ("markers", I_mark)):
        w, t, b = g("wbt", src)[I], g("total", src)[I], g("between", src)[I]
        rows.append(dict(source=src, genes=lab, frac_var_birthtime_within_hl=float(w.sum() / t.sum()),
                         frac_var_hemilineage=float(b.sum() / t.sum())))
T5 = pd.DataFrame(rows)
T5.to_csv(TABS / "std_birthtime_share.csv", index=False)
print("== std_birthtime_share"); print(T5.round(4).to_string())
