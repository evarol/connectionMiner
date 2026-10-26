"""W3 step 3: data-level confound facts (no embeddings).

(a) roX1 / roX2 / Y-linked detection by annotated sex, per library (mixed vs single-sex libraries).
(b) Could the Sex column have been called from roX? Per-cell roX UMI fraction vs annotation:
    AUC, best-threshold balanced accuracy, and the same in single-sex libraries (ambient check).
(c) Depth composition: eta^2 of log10 UMI by birth time, hemilineage, hemilineage x birth time,
    library, sex; per-birth-time depth quantiles.
(d) Library composition by hemilineage (chi-square), the reason library kNN must be read within hemilineage.
(e) Are roX1 / roX2 among the R1 HVGs (i.e. visible to R1, R10, R11)?
Output: v4/cache/w3_confounds_raw.json, v4/cache/w3_rox_cells.csv.gz (per-cell roX counts, sex, library)
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from w3_common import CACHE, gene_names, hvg_index, load_cells, load_counts

Y_GENES = ["kl-2", "kl-3", "kl-5", "ORY", "PRY", "CCY", "Ppr-Y", "WDY", "FDY", "ARY"]


def eta2(x, g):
    df = pd.DataFrame({"x": x, "g": g})
    m = df.groupby("g")["x"].transform("mean")
    return float(((m - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum())


def main():
    X, kept = load_counts()
    genes = gene_names(kept)
    c = load_cells(X)
    sex, lib = c["Sex"].values, c["library"].values
    tot = np.asarray(X.sum(1)).ravel()
    out = {}
    gi = {g: int(np.flatnonzero(genes == g)[0]) for g in ["lncRNA:roX1", "lncRNA:roX2"] + Y_GENES if (genes == g).any()}
    cnt = {g: np.asarray(X[:, j].todense()).ravel() for g, j in gi.items()}
    hv = set(hvg_index(X).tolist())
    out["rox_in_R1_hvgs"] = {g: (gi[g] in hv) for g in gi if "roX" in g}
    out["y_genes_present"] = {g: int((cnt[g] > 0).sum()) for g in Y_GENES if g in cnt}
    vc = pd.crosstab(lib, sex)
    mixed = vc.index[(vc.min(1) >= 50)]
    kind = np.where(np.isin(lib, mixed), "mixed", "single-sex")
    rows = []
    for g in ["lncRNA:roX1", "lncRNA:roX2"]:
        for k in ("mixed", "single-sex", "all"):
            for s in ("F", "M"):
                m = (sex == s) & ((kind == k) if k != "all" else True)
                if m.sum() == 0:
                    continue
                rows.append(dict(gene=g.split(":")[1], libraries=k, sex=s, n=int(m.sum()),
                                 detected_frac=float((cnt[g][m] > 0).mean()),
                                 mean_umi=float(cnt[g][m].mean()),
                                 mean_frac_of_cell_umi=float((cnt[g][m] / tot[m]).mean())))
    out["rox_detection"] = rows
    rox = cnt["lncRNA:roX1"] + cnt["lncRNA:roX2"]
    score = np.log1p(1e4 * rox / tot)
    y = (sex == "M").astype(int)
    for k in ("mixed", "single-sex", "all"):
        m = (kind == k) if k != "all" else np.ones(len(y), bool)
        if len(set(y[m])) < 2:
            continue
        thr = np.unique(np.quantile(score[m], np.linspace(0.01, 0.99, 197)))
        ba = [balanced_accuracy_score(y[m], score[m] > t) for t in thr]
        out[f"rox_call_{k}"] = dict(auc=float(roc_auc_score(y[m], score[m])), best_balacc=float(max(ba)),
                                    best_threshold_log1p_cp10k=float(thr[int(np.argmax(ba))]),
                                    any_rox_rule_balacc=float(balanced_accuracy_score(y[m], rox[m] > 0)))
    # female roX in single-sex female libraries vs mixed (ambient RNA check)
    fem_only = vc.index[(vc.get("M", 0) == 0)]
    per_lib = []
    for L in vc.index:
        for s in ("F", "M"):
            m = (lib == L) & (sex == s)
            if m.sum() < 20:
                continue
            per_lib.append(dict(library=L, sex=s, n=int(m.sum()), male_frac_in_library=float((sex[lib == L] == "M").mean()),
                                roX1_detected=float((cnt["lncRNA:roX1"][m] > 0).mean()),
                                roX2_detected=float((cnt["lncRNA:roX2"][m] > 0).mean()),
                                rox_cp10k_median=float(np.median(1e4 * rox[m] / tot[m]))))
    out["rox_per_library"] = per_lib
    out["female_only_libraries"] = list(map(str, fem_only))
    out["mixed_libraries"] = list(map(str, mixed))
    # depth composition
    x = c["log10_umi"].values
    out["eta2_log_umi"] = {v: eta2(x, c[v].astype(str).values) for v in
                           ["birthtime", "hemilineage", "stratum", "library", "Sex"]}
    out["log_umi_by_birthtime"] = c.groupby("birthtime")["log10_umi"].describe().round(4).reset_index().to_dict("records")
    q = np.quantile(x, [0.1, 0.9])
    out["primary_frac_by_depth"] = dict(overall=float((c.birthtime == "primary").mean()),
                                        top_decile=float((c.birthtime[x >= q[1]] == "primary").mean()),
                                        bottom_decile=float((c.birthtime[x <= q[0]] == "primary").mean()))
    ct = pd.crosstab(c.hemilineage, c.library)
    chi = chi2_contingency(ct)
    out["library_by_hemilineage_chi2"] = dict(chi2=float(chi[0]), dof=int(chi[2]), p=float(chi[1]),
                                              cramers_v=float(np.sqrt(chi[0] / (ct.values.sum() * (min(ct.shape) - 1)))))
    (CACHE / "w3_confounds_raw.json").write_text(json.dumps(out, indent=1, default=float))
    pd.DataFrame(dict(cell_id=c.cell_id, sex=sex, library=lib, kind=kind, roX1=cnt["lncRNA:roX1"],
                      roX2=cnt["lncRNA:roX2"], total_umi=tot)).to_csv(CACHE / "w3_rox_cells.csv.gz", index=False)
    print(json.dumps({k: v for k, v in out.items() if k not in ("rox_per_library",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
