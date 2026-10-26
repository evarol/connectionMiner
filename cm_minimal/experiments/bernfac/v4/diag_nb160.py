"""Is the NB k=160 validation jump (0.17646 -> 0.17832) a tail / optimisation artefact?
Per-entry val/test NLL for NB fits (v3 predictors conventions); summarise the tail.
  python diag_nb160.py FIT.npz [FIT.npz ...]   -> v4/tables/nb160_diag.csv
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v3"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import predictors as P  # noqa: E402

V4 = P.STUDY / "v4"


def summarise(path):
    z = np.load(path)
    pr = P.Counts(z["U"], z["V"], z["a"], z["b"], z["o"], z["phi"])
    Y = np.load(P.DATA / "Y.npy", mmap_mode="r")
    rows = []
    for split in ("val", "test"):
        M = np.load(P.DATA / f"{split}.npy", mmap_mode="r")
        nll, n, y, p, le = P.nll_on_mask(pr, Y, M, per_entry=True)
        le = le.astype(np.float64)
        srt = np.sort(le)[::-1]
        r = dict(fit=Path(path).stem, split=split, nll=nll, n=n, n_gt20=int((le > 20).sum()),
                 share_gt20=float(le[le > 20].sum() / le.sum()), nll_without_gt20=float(le[le <= 20].sum() / n),
                 top10_sum_share=float(srt[:10].sum() / le.sum()), max_entry=float(srt[0]),
                 nll_trim_top100=float(srt[100:].sum() / n),
                 phi_median=float(np.median(z["phi"])), phi_q05=float(np.quantile(z["phi"], 0.05)),
                 phi_q95=float(np.quantile(z["phi"], 0.95)), max_abs_eta_proxy=float(np.abs(z["U"]).max()))
        if split == "val":
            ri, ci = np.nonzero(np.asarray(M))
            top = np.argsort(le)[::-1][:5]
            r["top5_entries"] = ";".join(f"({ri[t]},{ci[t]}) y={y[t]} p={p[t]:.3g} nll={le[t]:.1f}" for t in top)
        rows.append(r); print(r, flush=True)
    return rows


if __name__ == "__main__":
    rows = sum((summarise(f) for f in sys.argv[1:]), [])
    out = V4 / "tables/nb160_diag.csv"
    old = pd.read_csv(out) if out.exists() else None
    d = pd.DataFrame(rows)
    if old is not None:
        d = pd.concat([old[~old.fit.isin(d.fit)], d])
    d.to_csv(out, index=False); print(d.drop(columns=["top5_entries"], errors="ignore").to_string())
