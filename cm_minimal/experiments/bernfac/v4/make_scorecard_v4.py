"""Scorecard for deck v4, built only from the v4 tables (one definition per column).

prediction : test NLL / Brier of the validation-selected fit (tables/comparison_v4.csv) and the
             paired two-way bootstrap ΔNLL vs R6 with 95% CI (tables/test_ci.csv, test_ci_v4refits.csv)
biology    : at matched dimension d = 40 (R8 only has 20), centred SVD axes, whitened:
             hemilineage linear probe, 15-NN balanced accuracy, Leiden ARI at 23 clusters (5 seeds)
nuisance   : R² of log UMI from the top-10 axes within hemilineage × birth-time strata
             (tables/depth_axes_v4.csv) and library kNN within hemilineage (whitened, d = 40)
Shading is within-column rank (darker = better); the best value is printed bold among
representations whose hemilineage probe is ≥ 0.9 (R9 is excluded from "best" because its
embedding barely encodes hemilineage).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle as fs  # noqa: E402
import v3style as st  # noqa: E402

plt = fs.plt
T = st.STUDY / "v4/tables"
OUT = st.STUDY / "v4/figures"
EXTRA = {"PO": ("Poisson counts", "#4D4D4D"), "R10": ("Pearson residuals", "#6B6B6B"),
         "R11": ("scVI", "#3B3B3B")}
ROWS = ["R1", "R2", "R3", "R4", "R5", "R6", "PO", "R7", "R8", "R9", "R10", "R11"]


def nm(c):
    return EXTRA[c][0] if c in EXTRA else st.name(c)


def col(c):
    return EXTRA[c][1] if c in EXTRA else st.colour(c)


def build():
    comp = pd.read_csv(T / "comparison_v4.csv").set_index("code")
    ci = pd.concat([pd.read_csv(T / "test_ci.csv"), pd.read_csv(T / "test_ci_v4refits.csv")])
    ci = ci[(ci.stratum == "all") & (ci.metric == "nll") & (ci.ref == "R6")]
    rep = pd.read_csv(T / "rep_metrics_v4.csv")
    dep = pd.read_csv(T / "depth_axes_v4.csv").set_index("rep")

    def rm(code, metric, scaling, d="40"):
        r = rep[(rep.rep == code) & (rep.metric == metric) & (rep.d_requested.astype(str) == d)
                & (rep.scaling == scaling)]
        if r.empty and d == "40":
            r = rep[(rep.rep == code) & (rep.metric == metric) & (rep.d_requested.astype(str) == "20")
                    & (rep.scaling == scaling)]
        return (r.value.iloc[0], r.sd.iloc[0]) if len(r) else (np.nan, np.nan)

    rows = []
    for c in ROWS:
        r = {"code": c}
        if c in comp.index:
            r["test NLL"] = comp.loc[c, "test_nll"]
            r["test Brier"] = comp.loc[c, "test_brier"]
            cfg = str(comp.loc[c, "config"])
            m = ci[ci.model == c]
            # the CI table keys v4 refits by tag; prefer a row whose config matches the selected fit
            v4tag = {"R8": "R8_d40l10", "PO": "PO_k120l30"}   # v4-selected refits (comparison_v4.csv)
            if c in v4tag and "d=40" in cfg or (c == "PO" and "k=120" in cfg):
                m = ci[ci.model == v4tag[c]]
            r["ΔNLL vs R6 (×10⁻³)"] = m.delta.iloc[0] * 1e3 if len(m) else (0.0 if c == "R6" else np.nan)
            r["_ci"] = (m.ci_lo.iloc[0] * 1e3, m.ci_hi.iloc[0] * 1e3) if len(m) else None
            r["_cfg"] = cfg
        r["HL probe"] = rm(c, "hl_probe_balacc", "invariant")[0]
        r["HL 15-NN"] = rm(c, "hl_knn_balacc", "whitened")[0]
        r["HL Leiden ARI (23)"], r["_ari_sd"] = rm(c, "hl_leiden23_ari", "whitened")
        r["depth R² within strata ↓"] = dep.loc[c, "depth_r2_top10_within_stratum"] if c in dep.index else np.nan
        r["library kNN within HL ↓"] = rm(c, "lib_knn_within_hl_balacc", "whitened")[0]
        rows.append(r)
    return pd.DataFrame(rows).set_index("code")


COLS = [("test NLL", False, "{:.4f}"), ("ΔNLL vs R6 (×10⁻³)", False, "{:+.2f}"), ("test Brier", False, "{:.4f}"),
        ("HL probe", True, "{:.3f}"), ("HL 15-NN", True, "{:.3f}"), ("HL Leiden ARI (23)", True, "{:.2f}"),
        ("depth R² within strata ↓", False, "{:.2f}"), ("library kNN within HL ↓", False, "{:.3f}")]
HEAD = {"test NLL": "test\nNLL ↓", "ΔNLL vs R6 (×10⁻³)": "ΔNLL vs R6\n(×10⁻³, 95% CI)", "test Brier": "test\nBrier ↓",
        "HL probe": "hemilineage\nlinear probe", "HL 15-NN": "hemilineage\n15-NN", "HL Leiden ARI (23)": "Leiden ARI\n(23 clusters)",
        "depth R² within strata ↓": "depth R²\nwithin strata ↓", "library kNN within HL ↓": "library kNN\nwithin HL ↓"}
GROUPS = [("held-out prediction (test entries)", 0, 3), ("biology (d = 40, whitened)", 3, 6),
          ("nuisance", 6, 8)]


def figure(df):
    fs.apply_publication_style()
    n, m = len(df), len(COLS)
    fig, ax = plt.subplots(figsize=(19, 0.62 * n + 2.6))
    ax.set_xlim(-3.6, m); ax.set_ylim(n, -1.8); ax.axis("off")
    cmap = plt.get_cmap("Greens")
    eligible = df["HL probe"] >= 0.9
    for j, (c, hi, fmt) in enumerate(COLS):
        v = df[c].astype(float)
        ok = v.notna()
        ranks = v[ok].rank(ascending=not hi)
        best = v[ok & eligible].idxmax() if hi else v[ok & eligible].idxmin()
        for i, code in enumerate(df.index):
            x = v[code]
            if np.isnan(x):
                ax.text(j + 0.5, i + 0.5, "—", ha="center", va="center", color="#999999", fontsize=12)
                continue
            shade = 1 - (ranks[code] - 1) / max(ok.sum() - 1, 1)
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.06), 0.94, 0.88, color=cmap(0.12 + 0.6 * shade), lw=0))
            txt = fmt.format(x)
            if c.startswith("ΔNLL") and df["_ci"][code] is not None and code != "R6":
                lo, hh = df["_ci"][code]; txt += f"\n[{lo:+.2f}, {hh:+.2f}]"
            if c.startswith("HL Leiden") and not np.isnan(df["_ari_sd"][code]):
                txt += f" ± {df['_ari_sd'][code]:.2f}"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=10.5 if "\n" in txt else 12,
                    fontweight="bold" if code == best else "normal",
                    color="white" if shade > 0.75 else "#1A1A1A")
        ax.text(j + 0.5, -0.12, HEAD[c], ha="center", va="bottom", fontsize=11.5)
    for g, a, b in GROUPS:
        ax.plot([a + 0.05, b - 0.05], [-1.2, -1.2], color="#444444", lw=1.2)
        ax.text((a + b) / 2, -1.3, g, ha="center", va="bottom", fontsize=13, fontweight="bold")
    for i, code in enumerate(df.index):
        ax.add_patch(plt.Rectangle((-3.55, i + 0.2), 0.12, 0.6, color=col(code), lw=0))
        ax.text(-3.35, i + 0.38, f"{code}  {nm(code)}", va="center", fontsize=12, color=col(code))
        cfg = df["_cfg"].get(code)
        if isinstance(cfg, str) and cfg != "nan":
            cfg = cfg.replace("affine head, ", "").replace(", residual none", "").replace(", 64x64", "")
            ax.text(-3.35, i + 0.72, cfg, va="center", fontsize=9.5, color="#666666")
    ax.text(-3.55, n + 0.35, "Shading: within-column rank, darker = better. Bold: best among embeddings with hemilineage "
            "probe ≥ 0.9 (R9 excluded). Leiden: 5 seeds, mean ± sd. Depth strata: hemilineage × birth time. ΔNLL: paired two-way bootstrap 95% CI. R11 (scVI) biology at d = 20, its native size. "
            "R10, R11: representation-only comparators.", fontsize=10, color="#555555", va="top")
    fs.finalize_figure(fig, OUT / "fig_scorecard_v4")


if __name__ == "__main__":
    df = build()
    df.drop(columns=["_ci", "_cfg", "_ari_sd"]).to_csv(T / "scorecard_v4.csv")
    print(df.drop(columns=["_ci", "_cfg"]).round(4).to_string())
    figure(df)
