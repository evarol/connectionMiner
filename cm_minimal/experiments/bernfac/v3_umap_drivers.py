"""Worker U: what drives each of the nine embeddings (kNN predictability + depth correlations)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from v3style import ROOT, STUDY, V3, ORDER

CACHE = STUDY / "full/umap/cache"
OUT = V3 / "tables"; OUT.mkdir(parents=True, exist_ok=True)


def load_meta():
    c = pd.read_csv(ROOT / "outputs/connectionminer_ready_inputs/bundles/motor/premn_24h/cells.csv.gz")
    G = sp.load_npz(ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h/G_prime.npz")
    c["log10_umi"] = np.log10(np.asarray(G.sum(1)).ravel())
    Y = np.load(STUDY / "full/data/Y.npy", mmap_mode="r")
    c["detect"] = np.concatenate([Y[i:i + 2000].mean(1) for i in range(0, Y.shape[0], 2000)])
    c["birthtime"] = c["birthtime"].fillna("precursor")
    return c


def knn_pred(E, y, seed=0):
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(E, y):
        pred[te] = KNeighborsClassifier(15, n_jobs=4).fit(E[tr], y[tr]).predict(E[te])
    return pred


if __name__ == "__main__":
    c = load_meta()
    c[["cell_id", "log10_umi", "detect"]].to_csv(OUT / "cell_depth.csv.gz", index=False)
    vars_ = ["hemilineage", "library", "Sex", "birthtime", "neurotransmitter", "maturation"]
    hl = c["hemilineage"].values
    rows, preds = [], {}
    for r in ORDER:
        E = np.load(CACHE / f"{r}_embedding.npz")["E"].astype(np.float64)
        row = {"rep": r}
        for v in vars_:
            y = c[v].astype(str).values
            p = knn_pred(E, y)
            row[f"{v}_knn_balacc"] = balanced_accuracy_score(y, p)
            preds[f"{r}_{v}"] = p
            # within-hemilineage version: does the embedding resolve v beyond what hemilineage gives?
            if v not in ("hemilineage",):
                accs, ws = [], []
                for h in np.unique(hl):
                    m = hl == h
                    yy = y[m]
                    vc = pd.Series(yy).value_counts()
                    if len(vc) < 2 or vc.iloc[1] < 10:
                        continue
                    accs.append(balanced_accuracy_score(yy, p[m])); ws.append(m.sum())
                row[f"{v}_within_hl_balacc"] = np.average(accs, weights=ws) if accs else np.nan
                row[f"{v}_n_hl_informative"] = len(accs)
        var = E.var(0); top = np.argsort(var)[::-1][:5]
        row["depth_abs_rho_top5"] = max(abs(spearmanr(E[:, j], c["log10_umi"]).statistic) for j in top)
        row["detect_abs_rho_top5"] = max(abs(spearmanr(E[:, j], c["detect"]).statistic) for j in top)
        row["depth_abs_rho_pc1"] = abs(spearmanr(E[:, top[0]], c["log10_umi"]).statistic)
        row["detect_abs_rho_pc1"] = abs(spearmanr(E[:, top[0]], c["detect"]).statistic)
        rows.append(row); print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "umap_drivers_full.csv", index=False)
    pd.DataFrame(preds).assign(cell_id=c["cell_id"]).to_csv(OUT / "knn_predictions.csv.gz", index=False)
