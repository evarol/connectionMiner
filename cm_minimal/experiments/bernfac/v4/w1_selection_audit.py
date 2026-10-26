"""W1 (v4): selection audit.

For every family: is the reported config (models/comparison.csv) the validation-NLL argmin of its
selection table (after the family's own restart rule)? How many configs had test NLL computed? Which
config would test have chosen, and what is the val/test regret? For R3-R5 also: which config the
Round-1 CV on the full 5% held-out mask (val U test, full/tables/cv_summary_all.csv, before val/test
existed) chose, and whether the val-only rescoring (models/rescored/selection.csv) changes it.
Output: v4/tables/selection_audit.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

S = Path(__file__).resolve().parents[4] / "outputs/bernfac_20261020"
M = S / "models"


def fam_tables():
    out = {}
    r = pd.read_csv(M / "rescored/selection.csv")
    for code, name in [("R3", "logit (no offsets)"), ("R4", "logit + offsets"), ("R5", "cloglog + offsets")]:
        d = r[r.model == name].copy()
        d["config"] = d.apply(lambda x: f"k={int(x.k)}, λ={x.lam:g}", axis=1)
        out[code] = (d, "models/rescored/selection.csv", "all rows (one fit per config)")
    c = pd.read_csv(M / "counts/tables/selection.csv")
    for code, fam in [("R6", "nb"), ("PO", "poisson")]:
        d = c[c.family == fam].copy()
        d["config"] = d.apply(lambda x: f"k={int(x.k)}, λ={x.lam:g}", axis=1)
        out[code] = (d, "models/counts/tables/selection.csv", "all rows (depth = train UMI variant only)")
    d = pd.read_csv(M / "mfa/tables/selection.csv")
    d = d[d.chosen_restart & d.in_grid].copy()
    d["config"] = d.tag
    out["R7"] = (d, "models/mfa/tables/selection.csv", "chosen_restart & in_grid (restart by training bound)")
    d = pd.read_csv(M / "neural/tables/selection.csv")
    d = d[d.chosen_restart].copy()
    d["config"] = d.name
    out["R8"] = (d, "models/neural/tables/selection.csv", "chosen_restart (restart by training objective F)")
    d = pd.read_csv(M / "lbm/tables/selection.csv")
    d["config"] = d.name
    out["R9"] = (d, "models/lbm/tables/selection.csv", "all rows (restart within config by ELBO)")
    return out


REPORTED = {"R3": "BernFac logit", "R4": "BernFac logit + offsets", "R5": "BernFac cloglog + offsets",
            "R6": "NB count factorization", "PO": "Poisson count factorization",
            "R7": "mixture of local factors", "R8": "neural manifold", "R9": "latent block model"}
ROUND1 = {"R3": "no offsets", "R4": "offsets", "R5": "cloglog"}


def main():
    comp = pd.read_csv(M / "comparison.csv").set_index("model")
    r1 = pd.read_csv(S / "full/tables/cv_summary_all.csv")
    rows = []
    for code, (d, src, rule) in fam_tables().items():
        rep = comp.loc[REPORTED[code]]
        iv, it = d.val_nll.idxmin(), d.test_nll.idxmin()
        match = np.isclose(d.loc[iv, "val_nll"], rep.val_nll, rtol=0, atol=1e-12)
        srt = d.val_nll.sort_values()
        row = dict(code=code, model=REPORTED[code], source=src, rows_used=rule, n_configs=len(d),
                   n_configs_with_test_nll=int(d.test_nll.notna().sum()),
                   reported_config=rep.config, reported_val_nll=rep.val_nll, reported_test_nll=rep.test_nll,
                   val_argmin_config=d.loc[iv, "config"], reported_is_val_argmin=bool(match),
                   val_runner_up=d.loc[srt.index[1], "config"], val_margin_to_runner_up=srt.iloc[1] - srt.iloc[0],
                   test_argmin_config=d.loc[it, "config"],
                   test_would_choose_differently=bool(it != iv),
                   test_regret_of_val_choice=d.loc[iv, "test_nll"] - d.loc[it, "test_nll"])
        if code in ROUND1:
            q = r1[r1.model == ROUND1[code]].sort_values("nll")
            row.update(round1_heldout_n=int(q.n_heldout.iloc[0]),
                       round1_heldout_argmin=f"k={int(q.k.iloc[0])}, λ={q.lam.iloc[0]:g}",
                       round1_runner_up=f"k={int(q.k.iloc[1])}, λ={q.lam.iloc[1]:g}",
                       round1_margin=q.nll.iloc[1] - q.nll.iloc[0],
                       val_only_reselection_changes_choice=bool(
                           f"k={int(q.k.iloc[0])}, λ={q.lam.iloc[0]:g}" != d.loc[iv, "config"]))
        rows.append(row)
    a = pd.DataFrame(rows)
    a.to_csv(S / "v4/tables/selection_audit.csv", index=False)
    pd.set_option("display.width", 250)
    print(a.T.to_string())


if __name__ == "__main__":
    main()
